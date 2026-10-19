#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.datasets import BraTSEvalDataset
from data.transforms import Compose, NumpyType
from evaluation import evaluate_subset, summarize_rows, write_csv
from utils.checkpoint import load_model_checkpoint
from utils.masks import subset_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the final PMSM checkpoint without test-set access. "
            "The top screening checkpoints are re-evaluated on all 15 validation subsets."
        )
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--val_split", default="val_late15_v6.txt")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def discover_candidates(run_dir: Path) -> List[Tuple[int, float, Path]]:
    candidates: List[Tuple[int, float, Path]] = []
    validation_root = run_dir / "validation"
    checkpoint_root = run_dir / "validation_candidates"
    for summary_path in sorted(validation_root.glob("epoch_*/overall_summary.json")):
        epoch_text = summary_path.parent.name.replace("epoch_", "")
        try:
            epoch = int(epoch_text)
        except ValueError:
            continue
        checkpoint_path = checkpoint_root / f"model_val_epoch_{epoch:04d}.pth"
        if not checkpoint_path.is_file():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        score = float(data["macro_dice"])
        candidates.append((epoch, score, checkpoint_path))
    if not candidates:
        raise RuntimeError(
            "No validation candidates were found. Expected validation/epoch_XXXX/overall_summary.json "
            "and validation_candidates/model_val_epoch_XXXX.pth."
        )
    return sorted(candidates, key=lambda item: item[1], reverse=True)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")
    if args.crop_size <= 0 or not 0.0 <= args.overlap < 1.0:
        raise ValueError("Invalid crop size or overlap.")

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "all15_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    candidates = discover_candidates(run_dir)[: args.top_k]
    dataset = BraTSEvalDataset(
        args.datapath,
        args.val_split,
        transforms=Compose([NumpyType(("float32", "int64"))]),
        num_classes=4,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    all_subset_names = subset_names("all")
    ranking: List[Dict[str, object]] = []

    for index, (epoch, screening_score, checkpoint_path) in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] epoch {epoch}: screening={screening_score:.6f}",
            flush=True,
        )
        model, checkpoint = load_model_checkpoint(str(checkpoint_path), device=device, strict=True)
        rows = []
        for subset_index, subset in enumerate(all_subset_names, start=1):
            print(f"  subset {subset_index:02d}/15: {subset}", flush=True)
            rows.extend(
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
        summaries, overall = summarize_rows(rows)
        epoch_dir = output_dir / f"epoch_{epoch:04d}"
        write_csv(str(epoch_dir / "per_subject_all_subsets.csv"), rows)
        write_csv(str(epoch_dir / "subset_summary.csv"), summaries)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        (epoch_dir / "overall_summary.json").write_text(
            json.dumps(overall, indent=2), encoding="utf-8"
        )
        ranking.append(
            {
                "epoch": epoch,
                "screening_macro_dice": screening_score,
                "all15_macro_dice": float(overall["macro_dice"]),
                "all15_dice_WT": float(overall["dice_WT"]),
                "all15_dice_TC": float(overall["dice_TC"]),
                "all15_dice_ET": float(overall["dice_ET"]),
                "checkpoint": str(checkpoint_path),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranking.sort(key=lambda row: float(row["all15_macro_dice"]), reverse=True)
    fieldnames = list(ranking[0].keys())
    with open(output_dir / "checkpoint_ranking.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranking)

    best = ranking[0]
    source = Path(str(best["checkpoint"]))
    checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    checkpoint.update(
        {
            "best_epoch": int(best["epoch"]),
            "best_val_score": float(best["all15_macro_dice"]),
            "selection_protocol": {
                "screening_top_k": int(args.top_k),
                "final_subsets": all_subset_names,
                "split_file": args.val_split,
                "test_set_accessed": False,
            },
        }
    )
    final_path = run_dir / "model_best_all15.pth"
    torch.save(checkpoint, final_path)
    (output_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "selected_epoch": int(best["epoch"]),
                "selected_all15_macro_dice": float(best["all15_macro_dice"]),
                "final_checkpoint": str(final_path),
                "ranked_candidates": ranking,
                "test_set_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(ranking[0], indent=2))
    print("Final validation-selected checkpoint:", final_path)


if __name__ == "__main__":
    main()
