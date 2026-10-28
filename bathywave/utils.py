"""Weight loading, output files, reproducibility, and depth conversion."""

import hashlib
import json
import random
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import MODEL_NAME, PREPROCESSING, TrainingConfig, require_integer
from .model import BathyWave, WaveletConvEncoder


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a new empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def distinct_output(output, *inputs):
    output = Path(output).resolve()
    for source in inputs:
        if source is None:
            continue
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must be separate from input data/checkpoint paths.")
    return output


@contextmanager
def staged_output(path: str | Path):
    """Publish an output directory only after all files have been written."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a new empty output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temporary:
        staging = Path(temporary)
        yield staging
        if path.exists():
            path.rmdir()  # Fails safely if another writer populated the directory.
        staging.rename(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(name: str) -> torch.device:
    if name not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu for local tests.")
    return torch.device(name)


def capture_rng(generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {
        "loader": generator.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    generator.set_state(state["loader"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )


def read_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary.")
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError("Expected a BathyWave checkpoint.")
    actual = checkpoint.get("preprocessing", checkpoint.get("config", {}))
    for key in ("baseline_method", "baseline_quantile", "normalization", "clip_negative"):
        if actual.get(key) != PREPROCESSING[key]:
            raise ValueError(f"Checkpoint preprocessing mismatch: {key}")
    if actual.get("length", PREPROCESSING["length"]) != PREPROCESSING["length"]:
        raise ValueError("Checkpoint waveform length mismatch.")
    require_integer("checkpoint epoch", checkpoint.get("epoch"), minimum=0)
    return checkpoint


def load_state_strict(module: nn.Module, state: dict) -> None:
    """Check every key, shape and dtype before changing any model parameter."""
    if not isinstance(state, dict):
        raise ValueError("Missing state dictionary.")
    expected = module.state_dict()
    if set(state) != set(expected):
        missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
        raise ValueError(f"State keys differ: missing={missing}, unexpected={extra}.")
    for key, reference in expected.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
        ):
            raise ValueError(f"State tensor shape or dtype mismatch: {key}.")
    module.load_state_dict(state, strict=True)


def _check_model_config(model, checkpoint: dict) -> None:
    if asdict(model.config) != checkpoint["model_kwargs"]:
        raise ValueError("Model configuration does not match the checkpoint.")


def load_encoder(model: BathyWave | WaveletConvEncoder, path: str | Path) -> int:
    checkpoint = read_checkpoint(path)
    _check_model_config(model, checkpoint)
    if checkpoint.get("pretrain", {}).get("method") != "masked_waveform_autoencoder":
        raise ValueError("Encoder transfer requires a reconstruction-pretrained encoder export.")
    state = checkpoint.get("encoder_state_dict")
    encoder = model.encoder if isinstance(model, BathyWave) else model
    load_state_strict(encoder, state)
    return checkpoint["epoch"]


def load_localizer(model: BathyWave, path: str | Path) -> dict:
    checkpoint = read_checkpoint(path)
    restore_localizer(model, checkpoint)
    return checkpoint


def restore_localizer(model: BathyWave, checkpoint: dict) -> None:
    """Restore an already validated localization export."""
    _check_model_config(model, checkpoint)
    if checkpoint.get("pretrain") or "model_state_dict" not in checkpoint:
        raise ValueError("Inference requires a supervised localization checkpoint.")
    state = checkpoint["model_state_dict"]
    load_state_strict(model, state)


def export_weights(path: str | Path, model: nn.Module, config: TrainingConfig, epoch: int) -> None:
    checkpoint = {
        "model_name": MODEL_NAME,
        "model_kwargs": asdict(model.config),
        "preprocessing": PREPROCESSING,
        "config": asdict(config),
        "epoch": epoch,
        "selection": config.selection,
    }
    if config.stage == "pretrain":
        checkpoint["encoder_state_dict"] = {
            key: value.cpu() for key, value in model.encoder.state_dict().items()
        }
        checkpoint["pretrain"] = {
            "method": "masked_waveform_autoencoder",
            "loss": "smooth_l1",
            "patch_len": config.patch_length,
            "mask_ratio": config.mask_ratio,
            "mask_input": "learnable",
            "decoder_channels": 64,
            "decoder_layers": 2,
        }
    else:
        checkpoint["model_state_dict"] = {
            key: value.cpu() for key, value in model.state_dict().items()
        }
    atomic_checkpoint(path, checkpoint)


def echo_positions_to_depth(surface, bottom, incidence_angle_deg, sampling_interval_ns=1.0):
    """Return vertical depth (m); invalid order/indices yield NaN, never zero depth.

    The angle is mandatory because zero incidence must not silently substitute
    for measured beam geometry. This function does not georeference a point cloud.
    """
    s, b, angle, dt = np.broadcast_arrays(
        np.asarray(surface, dtype=float),
        np.asarray(bottom, dtype=float),
        np.asarray(incidence_angle_deg, dtype=float),
        np.asarray(sampling_interval_ns, dtype=float),
    )
    if not np.isfinite(angle).all() or np.any(np.abs(angle) >= 90):
        raise ValueError("Incidence angles must be finite and strictly between -90 and 90 degrees.")
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Sampling intervals must be finite and positive.")
    theta_water = np.arcsin(np.sin(np.deg2rad(angle)) / 1.33)
    depth = (b - s) * dt * 1e-9 * 299792458.0 / (2 * 1.33) * np.cos(theta_water)
    valid = np.isfinite(s) & np.isfinite(b) & (s >= 0) & (b < 400) & (b > s)
    return np.where(valid, depth, np.nan)


def paired_elevation_metrics(prediction, reference):
    """Metrics on already aligned DEM cells, with pairwise finite-value masking."""
    prediction, reference = np.asarray(prediction, float), np.asarray(reference, float)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must share the same aligned grid/shape.")
    valid = np.isfinite(prediction) & np.isfinite(reference)
    if not valid.any():
        raise ValueError("No paired finite elevations.")
    p, r = prediction[valid], reference[valid]
    error = p - r
    denominator = np.square(r - r.mean()).sum()
    return {
        "paired_cells": int(valid.sum()),
        "mae_m": float(np.abs(error).mean()),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else None,
    }
