"""Streaming echo-position prediction and optional vertical depth conversion."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import require_integer
from .data import WaveformDataset
from .model import BathyWave
from .utils import (
    distinct_output,
    echo_positions_to_depth,
    get_device,
    read_checkpoint,
    restore_localizer,
    staged_output,
    write_json,
)


def predict(
    data_root: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device_name: str = "cuda",
    batch_size: int = 512,
    incidence_angle_deg: float | None = None,
    num_workers: int = 0,
) -> Path:
    """Stream argmax positions in input order and publish complete results together."""
    require_integer("batch_size", batch_size)
    require_integer("num_workers", num_workers, minimum=0)
    device = get_device(device_name)
    output = distinct_output(output, data_root, checkpoint)
    dataset = WaveformDataset(data_root)
    metadata = read_checkpoint(checkpoint)
    model = BathyWave(**metadata["model_kwargs"])
    restore_localizer(model, metadata)
    model.to(device).eval()
    if incidence_angle_deg is not None:
        echo_positions_to_depth([0], [1], incidence_angle_deg)  # Validate before writing.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    with staged_output(output) as staging:
        with ExitStack() as files:
            positions = []
            for name in ("surface", "bottom"):
                array = np.lib.format.open_memmap(
                    staging / f"{name}.npy", mode="w+", dtype="int64", shape=(len(dataset),)
                )
                files.callback(array._mmap.close)
                positions.append(array)
            surface, bottom = positions
            offset = 0
            with torch.inference_mode():
                for batch in loader:
                    logits = model(batch["waveform"].to(device, non_blocking=True))
                    peaks = logits.argmax(dim=-1).cpu().numpy()
                    end = offset + len(peaks)
                    surface[offset:end], bottom[offset:end] = peaks[:, 0], peaks[:, 1]
                    offset = end
            surface.flush()
            bottom.flush()
            valid = bottom > surface
            np.save(staging / "valid_order.npy", valid)
            if incidence_angle_deg is not None:
                np.save(
                    staging / "depth_m.npy",
                    echo_positions_to_depth(surface, bottom, incidence_angle_deg),
                )
            write_json(
                staging / "summary.json",
                {
                    "records": offset,
                    "valid_order": int(valid.sum()),
                    "checkpoint_epoch": metadata["epoch"],
                    "coordinate": "zero-based local waveform sample",
                    "incidence_angle_deg": incidence_angle_deg,
                    "sampling_interval_ns": 1.0,
                    "rows_dropped": 0,
                    "depth_invalid_order": "NaN",
                },
            )
    return output
