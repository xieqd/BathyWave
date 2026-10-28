"""Masked reconstruction pretraining and supervised echo-localization fine-tuning."""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .config import WAVEFORM_LENGTH, ModelConfig, TrainingConfig
from .data import WaveformDataset, assert_disjoint
from .model import BathyWave, MaskedWaveformAutoencoder, patch_mask
from .utils import (
    atomic_checkpoint,
    capture_rng,
    distinct_output,
    export_weights,
    get_device,
    load_encoder,
    load_state_strict,
    new_output,
    restore_rng,
    set_seed,
    sha256,
    write_json,
)


def reconstruction_loss(
    reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average Smooth L1 only over True mask positions; target is the original waveform."""
    loss = F.smooth_l1_loss(reconstruction, target, beta=1.0, reduction="none")
    return (loss * mask).sum() / mask.sum()


def gaussian_targets(
    positions: torch.Tensor, length: int = WAVEFORM_LENGTH, sigma: float = 2.0
) -> torch.Tensor:
    """Convert (N, 2) integer sample indices to (N, 2, T) Gaussian soft targets."""
    grid = torch.arange(length, device=positions.device).float()[None, None, :]
    return torch.exp(-0.5 * ((grid - positions.float()[..., None]) / sigma) ** 2)


def localization_loss(
    logits: torch.Tensor, targets: torch.Tensor, peak_weight: float = 20.0
) -> torch.Tensor:
    """Mean BCEWithLogits weighted elementwise by 1 + peak_weight * target."""
    element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (element_loss * (1.0 + peak_weight * targets)).mean()


@dataclass
class TrainingData:
    source: WaveformDataset
    validation_source: WaveformDataset | None
    training: WaveformDataset | Subset
    validation: WaveformDataset | Subset | None
    split: dict


def prepare_data(
    data_root: str | Path, validation_root: str | Path | None, config: TrainingConfig
) -> TrainingData:
    source = WaveformDataset(data_root, labeled=config.stage == "finetune")
    indices = list(range(len(source)))
    validation_source = None
    if validation_root:
        if config.val_ratio:
            raise ValueError("Use either an explicit validation dataset or val_ratio.")
        validation_source = WaveformDataset(validation_root, labeled=config.stage == "finetune")
        assert_disjoint(source, validation_source)
        training_data, validation_data = source, validation_source
        split = {
            "train": indices,
            "validation": list(range(len(validation_source))),
            "separate_validation": True,
        }
    elif config.val_ratio:
        random.Random(config.seed).shuffle(indices)
        validation_count = max(1, round(len(indices) * config.val_ratio))
        if validation_count >= len(indices):
            raise ValueError("Not enough samples for nonempty training and validation sets.")
        train_indices = sorted(indices[validation_count:])
        validation_indices = sorted(indices[:validation_count])
        training_data = Subset(source, train_indices)
        validation_data = Subset(source, validation_indices)
        split = {
            "train": train_indices,
            "validation": validation_indices,
            "separate_validation": False,
        }
    else:
        training_data, validation_data = source, None
        split = {"train": indices, "validation": [], "separate_validation": False}
    if validation_data is None and (
        config.scheduler == "plateau" or config.selection == "validation"
    ):
        raise ValueError("Validation-based scheduling/selection requires validation samples.")
    return TrainingData(source, validation_source, training_data, validation_data, split)


def batch_loss(
    model: nn.Module, batch: dict, config: TrainingConfig, device: torch.device
) -> torch.Tensor:
    waveforms = batch["waveform"].to(device, non_blocking=True)
    if config.stage == "pretrain":
        mask = patch_mask(
            len(waveforms), WAVEFORM_LENGTH, config.patch_length, config.mask_ratio, device
        )
        return reconstruction_loss(model(waveforms, mask), waveforms, mask)
    targets = gaussian_targets(batch["positions"].to(device, non_blocking=True), sigma=config.sigma)
    return localization_loss(model(waveforms), targets, config.peak_weight)


def run_epoch(
    model: nn.Module, loader: DataLoader, optimizer, config: TrainingConfig, device: torch.device
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, sample_count, batch_count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            loss = batch_loss(model, batch, config, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = len(batch["waveform"])
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
            batch_count += 1
    if sample_count == 0:
        raise ValueError("The data loader contains no samples.")
    return {
        "loss": total_loss / sample_count,
        "samples": sample_count,
        "batches": batch_count,
    }


def run_validation(model, loader, config, device) -> dict:
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 100000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed + 100000)
        return run_epoch(model, loader, None, config, device)


def resume_identity(data: TrainingData, config: TrainingConfig, device: torch.device) -> dict:
    return {
        "model_config": asdict(ModelConfig()),
        "configuration": {key: value for key, value in asdict(config).items() if key != "epochs"},
        "data": data.source.fingerprint(),
        "split": data.split,
        "validation": data.validation_source.fingerprint()
        if data.validation_source is not None
        else None,
        "device": str(device),
    }


def restore_training(path, identity, model, optimizer, scheduler, generator, epochs):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "history",
        "identity",
        "best_loss",
        "best_epoch",
        "encoder_sha256",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Incomplete training state. Use a weight export to start a new run.")
    if state["identity"] != identity:
        raise ValueError("Resume requires matching data, model, training settings, and device.")
    if len(state["history"]) > epochs:
        raise ValueError("Requested epochs precede the saved state.")
    load_state_strict(model, state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"], generator)
    return state


def fit(
    data_root: str | Path,
    output: str | Path,
    config: TrainingConfig,
    device_name: str = "cuda",
    encoder: str | Path | None = None,
    validation: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """Train a stage and return its selected weight export.

    Resume restores the full saved model; the original encoder file is not needed.
    """
    config.validate()
    device = get_device(device_name)
    if encoder and config.stage != "finetune":
        raise ValueError("Encoder transfer is only used for fine-tuning.")
    output = distinct_output(output, data_root, validation, encoder)
    data = prepare_data(data_root, validation, config)
    set_seed(config.seed)
    # Keep seeded initialization in the same order for both training stages.
    model = BathyWave()
    if config.stage == "pretrain":
        model = MaskedWaveformAutoencoder(model.encoder)
    elif encoder and not resume:
        load_encoder(model, encoder)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.scheduler_patience, factor=0.5
        )
        if config.scheduler == "plateau"
        else None
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    train_loader = DataLoader(data.training, shuffle=True, generator=generator, **loader_options)
    validation_loader = (
        DataLoader(data.validation, shuffle=False, **loader_options)
        if data.validation is not None
        else None
    )
    identity = resume_identity(data, config, device)
    history, best_loss, best_epoch = [], math.inf, 0
    state_path = output / "last_training_state.pt"
    if resume:
        state = restore_training(
            state_path, identity, model, optimizer, scheduler, generator, config.epochs
        )
        history, best_loss, best_epoch = state["history"], state["best_loss"], state["best_epoch"]
        encoder_digest = state["encoder_sha256"]
    else:
        encoder_digest = sha256(encoder) if encoder else None
        new_output(output)
    write_json(output / "config.json", asdict(config))
    best_filename = "best_encoder.pt" if config.stage == "pretrain" else "best_model.pt"
    for epoch in range(len(history) + 1, config.epochs + 1):
        if config.stage == "pretrain":
            torch.manual_seed(config.seed + epoch)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed + epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        train_stats = run_epoch(model, train_loader, optimizer, config, device)
        validation_stats = None
        if validation_loader is not None:
            validation_stats = run_validation(model, validation_loader, config, device)
            if scheduler:
                scheduler.step(validation_stats["loss"])
            if validation_stats["loss"] < best_loss:
                best_loss, best_epoch = validation_stats["loss"], epoch
                export_weights(output / best_filename, model, config, epoch)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_stats,
                "validation": validation_stats,
            }
        )
        atomic_checkpoint(
            state_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "rng": capture_rng(generator),
                "history": history,
                "identity": identity,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "encoder_sha256": encoder_digest,
            },
        )
        write_json(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
    final_filename = "encoder.pt" if config.stage == "pretrain" else "final_model.pt"
    export_weights(output / final_filename, model, config, len(history))
    selected = best_filename if config.selection == "validation" else final_filename
    write_json(
        output / "summary.json",
        {
            "stage": config.stage,
            "epochs_completed": len(history),
            "train_samples": len(data.training),
            "validation_samples": len(data.validation) if data.validation is not None else 0,
            "selected_checkpoint": selected,
            "selected_epoch": best_epoch if config.selection == "validation" else len(history),
            "encoder_initialized": encoder_digest is not None,
            "note": "Loss logs are not DEM/MBES accuracy measurements.",
        },
    )
    return output / selected
