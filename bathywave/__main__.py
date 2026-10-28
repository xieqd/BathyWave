"""Command-line interface: python -m bathywave --help."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import WaveformDataset
from .inference import predict
from .training import fit
from .utils import paired_elevation_metrics, write_json


def main():
    parser = argparse.ArgumentParser(
        description="BathyWave: masked pretraining and full-waveform echo localization"
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "check-data", help="Validate array shapes, amplitudes, labels, and IDs"
    )
    audit.add_argument("--data", required=True)
    audit.add_argument("--labeled", action="store_true")
    for stage in ("pretrain", "finetune"):
        command = commands.add_parser(stage)
        command.add_argument("--config", required=True)
        command.add_argument("--data", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        command.add_argument("--validation")
        command.add_argument("--resume", action="store_true")
        command.add_argument(
            "--epochs", type=int, help="Override epoch budget (recorded in config)"
        )
        command.add_argument("--batch-size", type=int)
        if stage == "finetune":
            command.add_argument("--encoder", help="Omit for training from scratch")
    infer = commands.add_parser("infer")
    infer.add_argument("--data", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    infer.add_argument("--batch-size", type=int, default=512)
    infer.add_argument("--num-workers", type=int, default=0)
    infer.add_argument(
        "--incidence-angle-deg",
        type=float,
        help="Optional constant angle from vertical; omit unless this geometry applies",
    )
    metrics = commands.add_parser("evaluate", help="Evaluate already aligned NPY DEM grids")
    metrics.add_argument("--prediction", required=True)
    metrics.add_argument("--reference", required=True)
    metrics.add_argument("--output", required=True, help="New JSON file")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "check-data":
        WaveformDataset(args.data, args.labeled)
        print("Input validation passed.")
    elif args.command in ("pretrain", "finetune"):
        cfg = load_config(args.config)
        if cfg.stage != args.command:
            parser.error("The configuration stage does not match the subcommand")
        overrides = {
            k: getattr(args, k) for k in ("epochs", "batch_size") if getattr(args, k) is not None
        }
        cfg = replace(cfg, **overrides)
        print(
            fit(
                args.data,
                args.output,
                cfg,
                args.device,
                getattr(args, "encoder", None),
                args.validation,
                args.resume,
            )
        )
    elif args.command == "infer":
        print(
            predict(
                args.data,
                args.checkpoint,
                args.output,
                args.device,
                args.batch_size,
                args.incidence_angle_deg,
                args.num_workers,
            )
        )
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        result = paired_elevation_metrics(
            np.load(args.prediction, allow_pickle=False),
            np.load(args.reference, allow_pickle=False),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
