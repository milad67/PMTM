# coding: utf-8
"""
Robust BraTS preprocessing for PMSM.

Output format expected by datasets_nii.py:
    {out}/vol/{case_id}_vol.npy   float32, shape (H, W, D, 4)
    {out}/seg/{case_id}_seg.npy   uint8,   shape (H, W, D)

Canonical saved channel order:
    0: T1ce / T1c
    1: T1   / T1n
    2: FLAIR / T2f
    3: T2   / T2w

Examples:
    python preprocess_fixed.py --dataset brats2020 --src D:\\BraTS2020 --out D:\\BraTS2020_Processed
    python preprocess_fixed.py --dataset brats2023 --src D:\\BraTS2023-GLI --out D:\\BraTS2023_Processed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import medpy.io as medio
import numpy as np

CHANNEL_ORDER = ("t1ce", "t1", "flair", "t2")


def _case_modalities(dataset: str, case_id: str) -> Dict[str, str]:
    """Return modality suffixes for a BraTS case in canonical PMSM order."""
    dataset = dataset.lower()
    if dataset in {"brats2018", "brats2020", "brats2015"}:
        return {
            "t1ce": f"{case_id}_t1ce.nii",
            "t1": f"{case_id}_t1.nii",
            "flair": f"{case_id}_flair.nii",
            "t2": f"{case_id}_t2.nii",
        }
    if dataset in {"brats2023", "brats2024"}:
        # BraTS-GLI naming: t1c, t1n, t2f, t2w.
        return {
            "t1ce": f"{case_id}-t1c.nii.gz",
            "t1": f"{case_id}-t1n.nii.gz",
            "flair": f"{case_id}-t2f.nii.gz",
            "t2": f"{case_id}-t2w.nii.gz",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def _seg_name(dataset: str, case_id: str) -> str:
    dataset = dataset.lower()
    if dataset in {"brats2018", "brats2020", "brats2015"}:
        return f"{case_id}_seg.nii"
    if dataset in {"brats2023", "brats2024"}:
        return f"{case_id}-seg.nii.gz"
    raise ValueError(f"Unsupported dataset: {dataset}")


def _ensure_min_crop(lo: int, hi_excl: int, dim: int, min_size: int) -> Tuple[int, int]:
    """Expand [lo, hi_excl) to at least min_size while staying inside [0, dim]."""
    lo = int(max(0, lo))
    hi_excl = int(min(dim, hi_excl))
    size = hi_excl - lo
    if size >= min_size or dim <= min_size:
        return lo, hi_excl

    missing = min_size - size
    left = missing // 2
    right = missing - left
    lo -= left
    hi_excl += right

    if lo < 0:
        hi_excl += -lo
        lo = 0
    if hi_excl > dim:
        lo -= hi_excl - dim
        hi_excl = dim
    lo = max(0, lo)
    hi_excl = min(dim, hi_excl)
    return lo, hi_excl


def crop_bounds(vol: np.ndarray, min_size: int = 128) -> Tuple[int, int, int, int, int, int]:
    """Compute a foreground bounding box as exclusive slice bounds."""
    if vol.ndim == 4:
        foreground = np.any(vol != 0, axis=0)
    elif vol.ndim == 3:
        foreground = vol != 0
    else:
        raise ValueError(f"Expected 3D or 4D volume, got shape {vol.shape}")

    coords = np.where(foreground)
    if len(coords[0]) == 0:
        # Extremely defensive fallback: center-crop/preserve the volume.
        x, y, z = foreground.shape
        return 0, min(x, min_size), 0, min(y, min_size), 0, min(z, min_size)

    dims = foreground.shape
    bounds = []
    for ax_coords, dim in zip(coords, dims):
        lo = int(ax_coords.min())
        hi_excl = int(ax_coords.max()) + 1  # +1 is important because Python slices are exclusive.
        bounds.extend(_ensure_min_crop(lo, hi_excl, dim, min_size))
    return tuple(bounds)  # type: ignore[return-value]


def normalize_modalities(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-normalize each modality on the foreground mask, preserving zero background."""
    if vol.shape[0] != 4:
        raise ValueError(f"Expected channel-first volume with 4 modalities, got {vol.shape}")

    out = vol.astype(np.float32, copy=True)
    brain_mask = np.any(out != 0, axis=0)
    if not np.any(brain_mask):
        return out

    for k in range(4):
        x = out[k]
        vals = x[brain_mask]
        std = float(vals.std())
        mean = float(vals.mean())
        if std < eps:
            std = 1.0
        x_norm = (x - mean) / std
        x_norm[~brain_mask] = 0.0
        out[k] = x_norm.astype(np.float32)
    return out


def iter_cases(src: Path) -> Iterable[Path]:
    for p in sorted(src.iterdir()):
        if p.is_dir():
            yield p


def process_case(case_dir: Path, out_dir: Path, dataset: str, min_crop: int = 128) -> str:
    case_id = case_dir.name
    modality_files = _case_modalities(dataset, case_id)

    arrays = []
    for key in CHANNEL_ORDER:
        f = case_dir / modality_files[key]
        if not f.exists():
            raise FileNotFoundError(f"Missing {key} file for {case_id}: {f}")
        arr, _ = medio.load(str(f))
        arrays.append(arr.astype(np.float32))

    vol = np.stack(arrays, axis=0)  # (4, H, W, D), canonical order.
    x0, x1, y0, y1, z0, z1 = crop_bounds(vol, min_size=min_crop)
    vol = normalize_modalities(vol[:, x0:x1, y0:y1, z0:z1])
    vol = np.ascontiguousarray(vol.transpose(1, 2, 3, 0), dtype=np.float32)

    seg_file = case_dir / _seg_name(dataset, case_id)
    if not seg_file.exists():
        raise FileNotFoundError(f"Missing segmentation for {case_id}: {seg_file}")
    seg, _ = medio.load(str(seg_file))
    seg = seg.astype(np.uint8)[x0:x1, y0:y1, z0:z1]
    seg = seg.copy()
    seg[seg == 4] = 3

    (out_dir / "vol").mkdir(parents=True, exist_ok=True)
    (out_dir / "seg").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vol" / f"{case_id}_vol.npy", vol)
    np.save(out_dir / "seg" / f"{case_id}_seg.npy", seg)
    return case_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BraTS NIfTI files for PMSM.")
    parser.add_argument("--dataset", required=True, choices=["brats2015", "brats2018", "brats2020", "brats2023", "brats2024"])
    parser.add_argument("--src", required=True, type=Path, help="Raw BraTS training-data directory containing case folders.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for vol/ and seg/ npy files.")
    parser.add_argument("--min-crop", type=int, default=128)
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Source directory not found: {args.src}")

    count = 0
    for case_dir in iter_cases(args.src):
        case_id = process_case(case_dir, args.out, args.dataset, min_crop=args.min_crop)
        count += 1
        print(f"[{count:04d}] processed {case_id}")
    print(f"Done. Processed {count} cases into {args.out}")


if __name__ == "__main__":
    main()
