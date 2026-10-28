"""Portable, memory-mapped raw waveform arrays and explicit label validation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import WAVEFORM_LENGTH
from .utils import sha256


def normalize_waveform(raw: np.ndarray) -> np.ndarray:
    """Normalize one validated waveform, returning float32 values of shape (400,)."""
    x = np.asarray(raw, dtype=np.float32)
    x = np.maximum(x - float(np.percentile(x, 5.0)), 0.0)
    low, high = float(x.min()), float(x.max())
    return np.zeros_like(x) if high - low <= 1e-8 else ((x - low) / (high - low)).astype(np.float32)


def normalize_waveforms(raw: np.ndarray) -> np.ndarray:
    """Normalize a validated batch of shape (N, 400)."""
    waveforms = np.asarray(raw, dtype=np.float32)
    baseline = np.percentile(waveforms, 5.0, axis=1, keepdims=True).astype(np.float32)
    corrected = np.maximum(waveforms - baseline, 0.0)
    minimum = corrected.min(axis=1, keepdims=True)
    span = corrected.max(axis=1, keepdims=True) - minimum
    return np.divide(corrected - minimum, span, out=np.zeros_like(corrected), where=span > 1e-8)


class WaveformDataset(Dataset):
    """Directory with waveforms.npy (N,400) and optional surface/bottom.npy (N,)."""

    def __init__(self, root: str | Path, labeled: bool = False):
        self.root = Path(root)
        self.waveforms = np.load(self.root / "waveforms.npy", mmap_mode="r", allow_pickle=False)
        x = self.waveforms
        if x.ndim != 2 or x.shape[1] != WAVEFORM_LENGTH or len(x) == 0 or x.dtype.kind not in "uif":
            raise ValueError("waveforms.npy must be a nonempty real numeric (N, 400) array.")
        for offset in range(0, len(x), 8192):
            if not np.isfinite(x[offset : offset + 8192]).all():
                raise ValueError("Waveforms contain NaN or infinite amplitudes.")
        self.positions = None
        self.label_values = None
        if labeled:
            labels = [
                np.load(self.root / f"{name}.npy", allow_pickle=False)
                for name in ("surface", "bottom")
            ]
            if any(a.shape != (len(x),) or a.dtype.kind not in "uif" for a in labels):
                raise ValueError("Each label array must have shape (N,) and real numeric values.")
            # Round float32 labels consistently before constructing soft targets.
            positions = np.stack(labels, axis=1).astype(np.float32)
            if (
                not np.isfinite(positions).all()
                or np.any(positions < 0)
                or np.any(positions >= 400)
                or np.any(positions[:, 0] >= positions[:, 1])
            ):
                raise ValueError("Labels must satisfy 0 <= surface < bottom < 400, with no NaN.")
            self.label_values = positions
            self.positions = np.clip(np.rint(positions), 0, 399).astype(np.int64)
        ids_path = self.root / "sample_ids.npy"
        self.ids = np.load(ids_path, allow_pickle=False) if ids_path.exists() else None
        if self.ids is not None:
            if self.ids.shape != (len(x),) or self.ids.dtype.kind not in "US":
                raise ValueError("sample_ids.npy must contain N unique fixed-width string IDs.")
            self.ids = np.asarray(
                [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in self.ids
                ]
            )
            if len(set(self.ids.tolist())) != len(x) or any(not value for value in self.ids):
                raise ValueError("Sample IDs must be nonempty and unique.")

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, index: int) -> dict:
        result = {
            "waveform": torch.from_numpy(normalize_waveform(self.waveforms[index])[None]),
            "row_index": index,
        }
        if self.positions is not None:
            result["positions"] = torch.from_numpy(self.positions[index].copy())
        return result

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Normalize a fetched batch once, preserving requested order and repeats."""
        if not indices:
            return []
        waveforms = normalize_waveforms(self.waveforms[indices])
        records = []
        for row, index in enumerate(indices):
            record = {"waveform": torch.from_numpy(waveforms[row, None]), "row_index": index}
            if self.positions is not None:
                record["positions"] = torch.from_numpy(self.positions[index].copy())
            records.append(record)
        return records

    def fingerprint(self) -> dict[str, str]:
        names = ["waveforms.npy"]
        if self.positions is not None:
            names += ["surface.npy", "bottom.npy"]
        if self.ids is not None:
            names.append("sample_ids.npy")
        return {name: sha256(self.root / name) for name in names}


def assert_disjoint(first, second):
    """Require stable sample IDs when using a separately supplied validation set."""
    if first.ids is None or second.ids is None:
        raise ValueError("Explicit train/validation datasets need sample_ids.npy to check overlap.")
    if set(first.ids.tolist()) & set(second.ids.tolist()):
        raise ValueError("Training and validation sample IDs overlap.")
