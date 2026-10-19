#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.data_utils import seed_worker
from data.datasets import BraTSEvalDataset
from data.transforms import Compose, NumpyType
from evaluation import evaluate_subset, summarize_rows, write_csv
from utils.checkpoint import load_model_checkpoint
from utils.masks import subset_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PMSM on fixed BraTS split files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--split_file", default="test.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subsets", default="all", help="all, full, representative, or comma list")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crop_size <= 0:
        raise ValueError("--crop_size must be positive.")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must lie in [0,1).")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model_checkpoint(args.checkpoint, device=device, strict=True)
    dataset = BraTSEvalDataset(
        root=args.datapath,
        split_file=args.split_file,
        transforms=Compose([NumpyType(("float32", "int64"))]),
        num_classes=4,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    selected = subset_names(args.subsets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for index, subset in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] evaluating {subset}", flush=True)
        all_rows.extend(
            evaluate_subset(
                model,
                loader,
                subset_name=subset,
                device=device,
                crop_size=args.crop_size,
                overlap=args.overlap,
                amp=args.amp,
            )
        )

    subset_rows, overall = summarize_rows(all_rows)
    write_csv(str(output_dir / "per_subject_all_subsets.csv"), all_rows)
    write_csv(str(output_dir / "subset_summary.csv"), subset_rows)
    with open(output_dir / "overall_summary.json", "w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "datapath": os.path.abspath(args.datapath),
        "split_file": args.split_file,
        "subsets": selected,
        "crop_size": args.crop_size,
        "overlap": args.overlap,
        "amp": bool(args.amp and device.type == "cuda"),
        "device": str(device),
        "empty_set_dice_convention": "both empty = 1; exactly one empty = 0",
        "postprocessing": "none",
    }
    with open(output_dir / "evaluation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(overall, indent=2))
    print("Results saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
