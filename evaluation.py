from __future__ import annotations

import csv
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.masks import DISPLAY_NAMES, SUBSETS


def sliding_starts(length: int, window: int, overlap: float) -> List[int]:
    if length <= window:
        return [0]
    step = max(1, int(round(window * (1.0 - overlap))))
    count = int(math.ceil((length - window) / float(step)))
    starts = [index * step for index in range(count)] + [length - window]
    return list(OrderedDict.fromkeys(starts))


def _extract_padded_patch(
    volume: torch.Tensor, h: int, w: int, d: int, window: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    height, width, depth = volume.shape[-3:]
    valid_h = min(window, height - h)
    valid_w = min(window, width - w)
    valid_d = min(window, depth - d)
    patch = volume[..., h : h + valid_h, w : w + valid_w, d : d + valid_d]
    pad_h, pad_w, pad_d = window - valid_h, window - valid_w, window - valid_d
    if pad_h or pad_w or pad_d:
        patch = F.pad(patch, (0, pad_d, 0, pad_w, 0, pad_h), value=0.0)
    return patch, (valid_h, valid_w, valid_d)


@torch.inference_mode()
def sliding_window_probability(
    model: torch.nn.Module,
    volume: torch.Tensor,
    modality_mask: torch.Tensor,
    crop_size: int = 128,
    overlap: float = 0.5,
    amp: bool = False,
) -> torch.Tensor:
    """Uniformly blend overlapping patch probabilities for one or more volumes."""
    if volume.ndim != 5 or volume.shape[1] != 4:
        raise ValueError(f"Expected (B,4,H,W,D), got {tuple(volume.shape)}")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must lie in [0,1).")

    batch, _, height, width, depth = volume.shape
    device = volume.device
    probability_sum = torch.zeros(
        batch, 4, height, width, depth, dtype=torch.float32, device=device
    )
    weight = torch.zeros(batch, 1, height, width, depth, dtype=torch.float32, device=device)

    h_starts = sliding_starts(height, crop_size, overlap)
    w_starts = sliding_starts(width, crop_size, overlap)
    d_starts = sliding_starts(depth, crop_size, overlap)
    amp_enabled = bool(amp and device.type == "cuda")

    for h in h_starts:
        for w in w_starts:
            for d in d_starts:
                patch, valid = _extract_padded_patch(volume, h, w, d, crop_size)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    patch_probability = model(patch, modality_mask)
                patch_probability = patch_probability.float()
                vh, vw, vd = valid
                probability_sum[..., h : h + vh, w : w + vw, d : d + vd] += (
                    patch_probability[..., :vh, :vw, :vd]
                )
                weight[..., h : h + vh, w : w + vw, d : d + vd] += 1.0

    return probability_sum / weight.clamp_min(1.0)


def binary_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary Dice with an explicit, reproducible empty-mask convention.

    Both masks empty is scored as 1.  Exactly one empty mask is scored as 0.
    No epsilon is added to non-empty pairs, so the implementation matches the
    convention written to the evaluation metadata exactly.
    """
    dims = tuple(range(1, prediction.ndim))
    pred_volume = prediction.sum(dim=dims)
    target_volume = target.sum(dim=dims)
    intersection = (prediction * target).sum(dim=dims)
    denominator = pred_volume + target_volume
    both_empty = denominator.eq(0)
    exactly_one_empty = pred_volume.eq(0) ^ target_volume.eq(0)
    dice = torch.zeros_like(denominator, dtype=torch.float32)
    non_empty_pair = ~(both_empty | exactly_one_empty)
    dice[both_empty] = 1.0
    dice[non_empty_pair] = (2.0 * intersection[non_empty_pair]) / denominator[non_empty_pair]
    return dice


def brats_dice_from_labels(
    prediction: torch.Tensor, target: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Raw WT/TC/ET Dice with a declared empty-set convention."""
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("Expected label tensors with shape (B,H,W,D).")
    pred_ncr = prediction.eq(1)
    pred_ed = prediction.eq(2)
    pred_et = prediction.eq(3)
    target_ncr = target.eq(1)
    target_ed = target.eq(2)
    target_et = target.eq(3)

    pred_wt = pred_ncr | pred_ed | pred_et
    target_wt = target_ncr | target_ed | target_et
    pred_tc = pred_ncr | pred_et
    target_tc = target_ncr | target_et

    return {
        "WT": binary_dice(pred_wt.float(), target_wt.float()),
        "TC": binary_dice(pred_tc.float(), target_tc.float()),
        "ET": binary_dice(pred_et.float(), target_et.float()),
    }


@torch.inference_mode()
def evaluate_subset(
    model: torch.nn.Module,
    loader: Iterable,
    subset_name: str,
    device: torch.device,
    crop_size: int,
    overlap: float,
    amp: bool = False,
) -> List[Dict[str, object]]:
    if subset_name not in SUBSETS:
        raise ValueError(f"Unknown subset {subset_name!r}")
    rows: List[Dict[str, object]] = []
    model.eval()
    for batch in loader:
        volume, target, case_ids = batch[:3]
        volume = volume.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.long, non_blocking=True)
        mask = torch.tensor(
            SUBSETS[subset_name], dtype=torch.bool, device=device
        ).unsqueeze(0).expand(volume.shape[0], -1)
        probability = sliding_window_probability(
            model, volume, mask, crop_size=crop_size, overlap=overlap, amp=amp
        )
        labels = probability.argmax(dim=1)
        scores = brats_dice_from_labels(labels, target)
        for index, case_id in enumerate(case_ids):
            wt = float(scores["WT"][index].item())
            tc = float(scores["TC"][index].item())
            et = float(scores["ET"][index].item())
            rows.append(
                {
                    "subset": subset_name,
                    "subset_display": DISPLAY_NAMES[subset_name],
                    "subject": str(case_id),
                    "dice_WT": wt,
                    "dice_TC": tc,
                    "dice_ET": et,
                    "macro_dice": (wt + tc + et) / 3.0,
                }
            )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    if not rows:
        raise ValueError("No evaluation rows were provided.")
    subset_order = [name for name in SUBSETS if any(row["subset"] == name for row in rows)]
    summaries: List[Dict[str, object]] = []
    for subset in subset_order:
        selected = [row for row in rows if row["subset"] == subset]
        wt = float(np.mean([float(row["dice_WT"]) for row in selected]))
        tc = float(np.mean([float(row["dice_TC"]) for row in selected]))
        et = float(np.mean([float(row["dice_ET"]) for row in selected]))
        summaries.append(
            {
                "subset": subset,
                "subset_display": DISPLAY_NAMES[subset],
                "num_subjects": len(selected),
                "dice_WT": wt,
                "dice_TC": tc,
                "dice_ET": et,
                "macro_dice": (wt + tc + et) / 3.0,
            }
        )

    overall = {
        "dice_WT": float(np.mean([row["dice_WT"] for row in summaries])),
        "dice_TC": float(np.mean([row["dice_TC"] for row in summaries])),
        "dice_ET": float(np.mean([row["dice_ET"] for row in summaries])),
        "macro_dice": float(np.mean([row["macro_dice"] for row in summaries])),
        "num_subsets": float(len(summaries)),
    }
    return summaries, overall


def write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
