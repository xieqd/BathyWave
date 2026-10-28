"""Validated model and training settings."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

WAVEFORM_LENGTH = 400
MODEL_NAME = "BathyWave"
PREPROCESSING = {
    "baseline_method": "quantile",
    "baseline_quantile": 5.0,
    "normalization": "minmax",
    "clip_negative": True,
    "length": WAVEFORM_LENGTH,
}


def require_integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    out_channels: int = 2
    base_channels: int = 32
    kernel_size: int = 7
    blocks_per_stage: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "in_channels",
            "out_channels",
            "base_channels",
            "kernel_size",
            "blocks_per_stage",
        ):
            require_integer(name, getattr(self, name))
        if self.in_channels != 1 or self.out_channels != 2:
            raise ValueError("BathyWave uses one input channel and two localization channels.")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3.")
        require_finite("dropout", self.dropout)
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


@dataclass
class TrainingConfig:
    stage: str = "finetune"
    epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    val_ratio: float = 0.0
    scheduler: str = "none"
    scheduler_patience: int = 5
    selection: str = "final"
    patch_length: int = 16
    mask_ratio: float = 0.5
    sigma: float = 2.0
    peak_weight: float = 20.0

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "patch_length"):
            require_integer(name, getattr(self, name))
        for name in ("seed", "num_workers", "scheduler_patience"):
            require_integer(name, getattr(self, name), minimum=0)
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32.")
        for name in (
            "learning_rate",
            "weight_decay",
            "val_ratio",
            "mask_ratio",
            "sigma",
            "peak_weight",
        ):
            require_finite(name, getattr(self, name))
        if self.stage not in ("pretrain", "finetune"):
            raise ValueError("stage must be pretrain or finetune.")
        if self.scheduler not in ("none", "plateau") or self.selection not in (
            "final",
            "validation",
        ):
            raise ValueError("Use scheduler none/plateau and selection final/validation.")
        if not 0 <= self.val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1).")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.sigma <= 0
            or self.peak_weight < 0
        ):
            raise ValueError("Invalid optimizer or loss configuration.")
        if WAVEFORM_LENGTH % self.patch_length or not 0 < self.mask_ratio < 1:
            raise ValueError("patch_length must divide 400 and mask_ratio must be in (0, 1).")
        if self.stage == "finetune" and self.val_ratio:
            raise ValueError("Supply an explicit --validation dataset for supervised validation.")


def load_config(path: str | Path) -> TrainingConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig(**values)
