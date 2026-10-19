#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.checkpoint import load_model_checkpoint
from utils.masks import DISPLAY_NAMES, MODALITY_NAMES, SUBSETS, subset_names


SLOT_NAMES = ("BG", "NCR/NET", "ED", "ET")
REGION_NAMES = ("WT", "TC", "ET")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze corrected multi-stage PMSM assignment maps and state slots.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--split_file", default="val_late15_v6.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subsets", default="all,flair,t2")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--visualize_first_n", type=int, default=3)
    parser.add_argument("--visualize_subsets", default="all,flair,t2")
    parser.add_argument("--save_maps_npz", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def starts(length: int, window: int, overlap: float) -> List[int]:
    if length <= window:
        return [0]
    step = max(1, int(round(window * (1.0 - overlap))))
    count = int(math.ceil((length - window) / float(step)))
    values = [index * step for index in range(count)] + [length - window]
    return list(OrderedDict.fromkeys(values))


def coordinates(shape: Sequence[int], window: int, overlap: float):
    return [
        (h, w, d)
        for h in starts(shape[0], window, overlap)
        for w in starts(shape[1], window, overlap)
        for d in starts(shape[2], window, overlap)
    ]


def extract_patch(tensor: torch.Tensor, coord: Tuple[int, int, int], window: int):
    h, w, d = coord
    height, width, depth = tensor.shape[-3:]
    vh, vw, vd = min(window, height - h), min(window, width - w), min(window, depth - d)
    patch = tensor[..., h : h + vh, w : w + vw, d : d + vd]
    if (vh, vw, vd) != (window, window, window):
        patch = F.pad(patch, (0, window - vd, 0, window - vw, 0, window - vh), value=0.0)
    return patch, (vh, vw, vd)


def safe_cosine(a: np.ndarray, b: np.ndarray, axis: int = -1, eps: float = 1e-8):
    denominator = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return np.sum(a * b, axis=axis) / np.maximum(denominator, eps)


def dice_iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8):
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    pred_sum, target_sum = pred.sum(), target.sum()
    if pred_sum + target_sum == 0:
        return 1.0, 1.0
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    union = np.logical_or(pred, target).sum()
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def semantic_rows(subject: str, subset: str, stage: int, assignment: np.ndarray, label: np.ndarray, roi: np.ndarray):
    rows, activation_rows = [], []
    predicted = assignment.argmax(axis=0)
    for class_id, class_name in enumerate(SLOT_NAMES):
        dice, iou = dice_iou(np.logical_and(predicted == class_id, roi), np.logical_and(label == class_id, roi))
        rows.append({
            "subject": subject, "subset": subset, "stage": stage,
            "target": class_name, "target_type": "disjoint_class", "dice": dice, "iou": iou,
        })
        class_voxels = np.logical_and(label == class_id, roi)
        for slot in range(4):
            activation_rows.append({
                "subject": subject, "subset": subset, "stage": stage,
                "slot": slot, "slot_name": SLOT_NAMES[slot],
                "ground_truth_class": class_id, "ground_truth_class_name": class_name,
                "mean_activation": float(assignment[slot][class_voxels].mean()) if class_voxels.any() else np.nan,
            })

    pred_wt, true_wt = predicted > 0, label > 0
    pred_tc, true_tc = np.logical_or(predicted == 1, predicted == 3), np.logical_or(label == 1, label == 3)
    pred_et, true_et = predicted == 3, label == 3
    for name, pred_region, true_region in (
        ("WT", pred_wt, true_wt), ("TC", pred_tc, true_tc), ("ET", pred_et, true_et)
    ):
        dice, iou = dice_iou(np.logical_and(pred_region, roi), np.logical_and(true_region, roi))
        rows.append({
            "subject": subject, "subset": subset, "stage": stage,
            "target": name, "target_type": "nested_region", "dice": dice, "iou": iou,
        })
    return rows, activation_rows


def robust_normalize(image: np.ndarray):
    values = image[np.isfinite(image) & (np.abs(image) > 1e-8)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    return np.clip((image - low) / max(high - low, 1e-8), 0.0, 1.0)


def reference_modality(mask: Sequence[bool]) -> int:
    # Prefer FLAIR, then T2, T1ce, T1 for a readable tumor view.
    for index in (2, 3, 0, 1):
        if mask[index]:
            return index
    return int(np.argmax(mask))


def save_figure(path: Path, x: np.ndarray, y: np.ndarray, maps: Mapping[int, np.ndarray], mask, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tumor_area = (y > 0).sum(axis=(0, 1))
    z = int(np.argmax(tumor_area)) if tumor_area.max() > 0 else y.shape[2] // 2
    modality = reference_modality(mask)
    ref = robust_normalize(x[modality, :, :, z].T)
    gt = y[:, :, z].T
    stages = sorted(maps)
    fig, axes = plt.subplots(len(stages), 6, figsize=(17, 4.1 * len(stages)), squeeze=False)
    for row, stage in enumerate(stages):
        axes[row, 0].imshow(ref, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"{MODALITY_NAMES[modality]} | stage {stage}")
        axes[row, 1].imshow(gt, origin="lower", vmin=0, vmax=3)
        axes[row, 1].set_title("Ground truth")
        for slot in range(4):
            image = axes[row, slot + 2].imshow(
                maps[stage][slot, :, :, z].T,
                origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
            )
            axes[row, slot + 2].set_title(f"Slot {slot} ({SLOT_NAMES[slot]})")
            fig.colorbar(image, ax=axes[row, slot + 2], fraction=0.046, pad=0.04)
        for axis in axes[row]:
            axis.axis("off")
    fig.suptitle(f"{title} | axial z={z}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_csv(frame: pd.DataFrame, group_columns: Sequence[str], value_columns: Sequence[str], path: Path):
    if frame.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    summary = frame.groupby(list(group_columns), dropna=False)[list(value_columns)].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(v) for v in column if str(v)) if isinstance(column, tuple) else str(column) for column in summary.columns]
    summary.to_csv(path, index=False)


@torch.inference_mode()
def process_subset(model, x, y, subject, subset, coords, crop_size, amp, full_token_cache=None):
    mask_values = SUBSETS[subset]
    mask = torch.tensor(mask_values, dtype=torch.bool, device=x.device).unsqueeze(0)
    shape = tuple(int(v) for v in y.shape[-3:])
    map_sums = {stage: np.zeros((4,) + shape, dtype=np.float32) for stage in model.active_stages}
    weight = np.zeros(shape, dtype=np.float32)
    health_rows, token_rows, identity_rows, gate_rows, update_rows = [], [], [], [], []
    token_cache = {}
    amp_enabled = bool(amp and x.device.type == "cuda")

    for coord in coords:
        x_patch, valid = extract_patch(x, coord, crop_size)
        y_patch, _ = extract_patch(y.float(), coord, crop_size)
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(x_patch, mask, return_details=True, compute_auxiliary=False, return_debug=True)
        vh, vw, vd = valid
        h, w, d = coord
        tumor_patch = bool((y_patch[..., :vh, :vw, :vd] > 0).any().item())
        weight[h:h+vh, w:w+vw, d:d+vd] += 1.0

        stages = sorted(output["stage_details"])
        stage_tokens = {}
        for stage in stages:
            details = output["stage_details"][stage]
            assignment = details["mean_assignment"].float()
            up = F.interpolate(assignment, size=(crop_size,) * 3, mode="trilinear", align_corners=False)
            up_np = up[0, :, :vh, :vw, :vd].cpu().numpy()
            map_sums[stage][:, h:h+vh, w:w+vw, d:d+vd] += up_np

            probabilities = assignment.clamp_min(1e-8)
            entropy = -(probabilities * probabilities.log()).sum(dim=1) / math.log(4.0)
            state = details["state"].float()[0].cpu().numpy()
            stage_tokens[stage] = state
            normalized = state / np.maximum(np.linalg.norm(state, axis=1, keepdims=True), 1e-8)
            similarity = normalized @ normalized.T
            offdiag = similarity[~np.eye(4, dtype=bool)]
            health_rows.append({
                "subject": subject, "subset": subset, "stage": stage,
                "tumor_window": tumor_patch,
                "normalized_entropy": float(entropy.mean().item()),
                "mean_max_probability": float(probabilities.max(dim=1).values.mean().item()),
                "mean_abs_deviation_from_uniform": float((probabilities - 0.25).abs().mean().item()),
                "mean_offdiag_token_cosine": float(offdiag.mean()),
            })
            token_cache[(coord, stage)] = state

            fusion = details.get("fusion_debug")
            if fusion is not None:
                final_gates = fusion["final_gates"].float()[0].cpu().numpy()
                local_gates = fusion["local_gates"].float()[0].cpu().numpy()
                token_gates = fusion["token_gates"]
                token_gates = token_gates.float()[0].cpu().numpy() if token_gates is not None else None
                for modality in range(4):
                    if not mask_values[modality]:
                        continue
                    gate_rows.append({
                        "subject": subject, "subset": subset, "stage": stage,
                        "modality": MODALITY_NAMES[modality], "tumor_window": tumor_patch,
                        "local_gate_mean": float(local_gates[modality].mean()),
                        "local_gate_std": float(local_gates[modality].std()),
                        "token_gate_mean": float(token_gates[modality].mean()) if token_gates is not None else np.nan,
                        "token_gate_std": float(token_gates[modality].std()) if token_gates is not None else np.nan,
                        "final_gate_mean": float(final_gates[modality].mean()),
                        "final_gate_std": float(final_gates[modality].std()),
                    })

            state_input = details["state_input"].float()[0].cpu().numpy()
            update = state - state_input
            for slot in range(4):
                update_rows.append({
                    "subject": subject, "subset": subset, "stage": stage, "slot": slot,
                    "slot_name": SLOT_NAMES[slot], "tumor_window": tumor_patch,
                    "relative_update": float(np.linalg.norm(update[slot]) / max(np.linalg.norm(state_input[slot]), 1e-8)),
                })

        if stage_tokens:
            reference_stage = min(stage_tokens)
            base = stage_tokens[reference_stage]
            base_n = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
            for stage in sorted(s for s in stage_tokens if s != reference_stage):
                current = stage_tokens[stage]
                current_n = current / np.maximum(np.linalg.norm(current, axis=1, keepdims=True), 1e-8)
                matrix = current_n @ base_n.T
                for slot in range(4):
                    other = np.delete(matrix[slot], slot)
                    identity_rows.append({
                        "subject": subject, "subset": subset, "stage": stage,
                        "reference_stage": reference_stage,
                        "slot": slot, "slot_name": SLOT_NAMES[slot], "tumor_window": tumor_patch,
                        "same_slot_cosine_to_reference": float(matrix[slot, slot]),
                        "identity_match": int(np.argmax(matrix[slot]) == slot),
                        "diagonal_margin": float(matrix[slot, slot] - other.max()),
                    })

        if full_token_cache is not None:
            for stage, state in stage_tokens.items():
                full = full_token_cache[(coord, stage)]
                cosine = safe_cosine(state, full, axis=1)
                for slot in range(4):
                    token_rows.append({
                        "subject": subject, "subset": subset, "stage": stage,
                        "slot": slot, "slot_name": SLOT_NAMES[slot], "tumor_window": tumor_patch,
                        "cosine_to_full": float(cosine[slot]),
                    })

    maps = {stage: values / np.maximum(weight[None], 1e-8) for stage, values in map_sums.items()}
    return {
        "maps": maps, "health": health_rows, "token": token_rows,
        "identity": identity_rows, "gate": gate_rows, "update": update_rows,
        "token_cache": token_cache,
    }


def main():
    args = parse_args()
    if args.crop_size <= 0 or not 0.0 <= args.overlap < 1.0:
        raise ValueError("Invalid crop_size or overlap.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(args.checkpoint, device, strict=True)
    if not model.active_stages:
        raise ValueError("State analysis requires a model with active tumor-state stages.")

    root = Path(args.datapath)
    subjects = [line.strip() for line in (root / args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    selected = subset_names(args.subsets)
    if "all" not in selected:
        selected = ["all"] + selected
    visualize = set(subset_names(args.visualize_subsets))
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    map_dir = output_dir / "state_maps_npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {name: [] for name in ("semantic", "activation", "health", "token", "identity", "gate", "update")}
    for subject_index, subject in enumerate(subjects):
        print(f"[{subject_index + 1}/{len(subjects)}] {subject}", flush=True)
        x_hwdc = np.load(root / "vol" / f"{subject}_vol.npy", allow_pickle=False).astype(np.float32)
        y_np = np.load(root / "seg" / f"{subject}_seg.npy", allow_pickle=False).astype(np.int64)
        y_np[y_np == 4] = 3
        if y_np.min() < 0 or y_np.max() > 3:
            raise ValueError(f"Invalid labels for {subject}: {np.unique(y_np)}")
        x_np = np.transpose(x_hwdc, (3, 0, 1, 2)).copy()
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0).to(device)
        roi = np.logical_or(np.any(np.abs(x_np) > 1e-8, axis=0), y_np > 0)
        coords = coordinates(y_np.shape, args.crop_size, args.overlap)

        full = process_subset(model, x, y, subject, "all", coords, args.crop_size, args.amp)
        results = {"all": full}
        for subset in selected:
            if subset == "all":
                continue
            results[subset] = process_subset(
                model, x, y, subject, subset, coords, args.crop_size, args.amp,
                full_token_cache=full["token_cache"],
            )

        for subset, result in results.items():
            rows["health"].extend(result["health"])
            rows["token"].extend(result["token"])
            rows["identity"].extend(result["identity"])
            rows["gate"].extend(result["gate"])
            rows["update"].extend(result["update"])
            for stage, assignment in result["maps"].items():
                semantic, activation = semantic_rows(subject, subset, stage, assignment, y_np, roi)
                rows["semantic"].extend(semantic)
                rows["activation"].extend(activation)
            if subject_index < args.visualize_first_n and subset in visualize:
                save_figure(
                    figure_dir / f"{subject}_{subset}_assignment_maps.png",
                    x_np, y_np, result["maps"], SUBSETS[subset],
                    f"{subject} | {DISPLAY_NAMES[subset]}",
                )
                if args.save_maps_npz:
                    map_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        map_dir / f"{subject}_{subset}_assignment_maps.npz",
                        **{f"stage{stage}": value for stage, value in result["maps"].items()},
                    )
        del x, y, full, results
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frames = {name: pd.DataFrame(values) for name, values in rows.items()}
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}_raw.csv", index=False)
    summarize_csv(frames["semantic"], ["subset", "stage", "target_type", "target"], ["dice", "iou"], output_dir / "semantic_summary.csv")
    summarize_csv(frames["activation"], ["subset", "stage", "slot", "slot_name", "ground_truth_class", "ground_truth_class_name"], ["mean_activation"], output_dir / "slot_class_activation_summary.csv")
    summarize_csv(frames["health"], ["subset", "stage", "tumor_window"], ["normalized_entropy", "mean_max_probability", "mean_abs_deviation_from_uniform", "mean_offdiag_token_cosine"], output_dir / "state_health_summary.csv")
    summarize_csv(frames["identity"], ["subset", "reference_stage", "stage", "slot", "slot_name", "tumor_window"], ["same_slot_cosine_to_reference", "identity_match", "diagonal_margin"], output_dir / "cross_stage_identity_summary.csv")
    summarize_csv(frames["token"], ["subset", "stage", "slot", "slot_name", "tumor_window"], ["cosine_to_full"], output_dir / "full_missing_token_cosine_summary.csv")
    summarize_csv(frames["gate"], ["subset", "stage", "modality", "tumor_window"], ["local_gate_mean", "local_gate_std", "token_gate_mean", "token_gate_std", "final_gate_mean", "final_gate_std"], output_dir / "fusion_gate_summary.csv")
    summarize_csv(frames["update"], ["subset", "stage", "slot", "slot_name", "tumor_window"], ["relative_update"], output_dir / "state_update_summary.csv")

    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "split_file": args.split_file,
        "subjects": len(subjects),
        "subsets": selected,
        "crop_size": args.crop_size,
        "overlap": args.overlap,
        "semantic_stages": list(model.active_stages),
        "slot_order": list(SLOT_NAMES),
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Analysis saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
