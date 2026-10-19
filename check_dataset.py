#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data.datasets import read_case_ids, validate_and_remap_label
from data.transforms import Compose, NumpyType, RandomCrop3D, RandomFlip3D, RandomIntensityChange, RandomRotation3D


def parse_args():
    parser = argparse.ArgumentParser(description="Validate PMSM data files, split isolation, and label-safe augmentation.")
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--train_split", default="train_late15_v6.txt")
    parser.add_argument("--val_split", default="val_late15_v6.txt")
    parser.add_argument("--test_split", default="test.txt")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--augmentation_trials", type=int, default=20)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.datapath)
    split_names = [args.train_split, args.val_split, args.test_split]
    split_ids = {name: read_case_ids(root, name) for name in split_names}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = sorted(set(split_ids[left]).intersection(split_ids[right]))
            if overlap:
                raise ValueError(f"Split leakage between {left} and {right}: {overlap[:10]}")

    all_ids = sorted(set().union(*(set(values) for values in split_ids.values())))
    if args.max_cases > 0:
        all_ids = all_ids[: args.max_cases]
    shapes = []
    label_values = set()
    for index, case_id in enumerate(all_ids, start=1):
        volume_path = root / "vol" / f"{case_id}_vol.npy"
        label_path = root / "seg" / f"{case_id}_seg.npy"
        if not volume_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing volume/label for {case_id}")
        volume = np.load(volume_path, allow_pickle=False)
        label = validate_and_remap_label(np.load(label_path, allow_pickle=False))
        if volume.ndim != 4 or volume.shape[-1] != 4:
            raise ValueError(f"{case_id}: expected HxWxDx4, got {volume.shape}")
        if tuple(volume.shape[:3]) != tuple(label.shape):
            raise ValueError(f"{case_id}: image/label mismatch {volume.shape[:3]} vs {label.shape}")
        if min(label.shape) < args.crop_size:
            raise ValueError(f"{case_id}: shape {label.shape} is smaller than crop {args.crop_size}")
        if not np.isfinite(volume).all():
            raise ValueError(f"{case_id}: non-finite image values")
        shapes.append(label.shape)
        label_values.update(np.unique(label).tolist())
        if index % 25 == 0:
            print(f"checked {index}/{len(all_ids)} cases", flush=True)

    # Repeatedly test the exact paired augmentation used for training.
    sample_id = split_ids[args.train_split][0]
    x = np.load(root / "vol" / f"{sample_id}_vol.npy", allow_pickle=False).astype(np.float32)
    y = validate_and_remap_label(np.load(root / "seg" / f"{sample_id}_seg.npy", allow_pickle=False))
    transform = Compose([
        RandomCrop3D((args.crop_size,) * 3),
        RandomRotation3D(10, 1.0),
        RandomIntensityChange(0.1, 0.1),
        RandomFlip3D(0.5),
        NumpyType((np.float32, np.int64)),
    ])
    for trial in range(args.augmentation_trials):
        x_aug, y_aug = transform([x[None], y[None]])
        values = set(np.unique(y_aug).tolist())
        if not values.issubset({0, 1, 2, 3}):
            raise ValueError(f"Unsafe augmented labels at trial {trial}: {sorted(values)}")
        if not np.isfinite(x_aug).all():
            raise ValueError(f"Non-finite augmented image values at trial {trial}")

    split_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in split_names
    }

    report = {
        "datapath": str(root.resolve()),
        "splits": {name: len(ids) for name, ids in split_ids.items()},
        "split_sha256": split_hashes,
        "split_overlap": 0,
        "checked_cases": len(all_ids),
        "shape_min": [int(min(shape[axis] for shape in shapes)) for axis in range(3)],
        "shape_max": [int(max(shape[axis] for shape in shapes)) for axis in range(3)],
        "label_values_after_remap": sorted(int(value) for value in label_values),
        "augmentation_trials": args.augmentation_trials,
        "augmentation_label_values_valid": True,
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
