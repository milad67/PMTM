from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Identity


CHANNEL_ORDER = ("T1ce", "T1", "FLAIR", "T2")


def read_case_ids(root: Union[str, os.PathLike], split_file: str) -> List[str]:
    path = Path(root) / split_file
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"No case identifiers were found in {path}")
    if len(ids) != len(set(ids)):
        duplicates = sorted({case for case in ids if ids.count(case) > 1})
        raise ValueError(f"Duplicate case identifiers in {path}: {duplicates[:10]}")
    return ids


def validate_and_remap_label(label: np.ndarray, num_classes: int = 4) -> np.ndarray:
    y = np.asarray(label, dtype=np.int64).copy()
    # BraTS2018/2020 use label 4 for enhancing tumor in the raw annotations.
    y[y == 4] = 3
    if y.ndim != 3:
        raise ValueError(f"Expected a 3-D label volume, got shape {y.shape}")
    y_min = int(y.min())
    y_max = int(y.max())
    if y_min < 0 or y_max >= num_classes:
        values = np.unique(y)
        raise ValueError(
            f"Invalid label values {values.tolist()}; expected integers in [0,{num_classes - 1}]."
        )
    return y


def one_hot_numpy(label: np.ndarray, num_classes: int = 4) -> np.ndarray:
    y = validate_and_remap_label(label, num_classes=num_classes)
    encoded = np.eye(num_classes, dtype=np.float32)[y.reshape(-1)]
    return encoded.reshape(*y.shape, num_classes)


class _BraTSBase(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        transforms: Optional[Callable] = None,
        num_classes: int = 4,
    ) -> None:
        self.root = Path(root)
        self.case_ids = read_case_ids(root, split_file)
        self.transforms = transforms if transforms is not None else Identity()
        if not callable(self.transforms):
            raise TypeError("transforms must be callable.")
        self.num_classes = int(num_classes)

        self.volume_paths = [self.root / "vol" / f"{case}_vol.npy" for case in self.case_ids]
        self.label_paths = [self.root / "seg" / f"{case}_seg.npy" for case in self.case_ids]
        missing = [str(path) for path in self.volume_paths + self.label_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} data files. First examples: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def _load(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        x = np.load(self.volume_paths[index], allow_pickle=False).astype(np.float32, copy=False)
        y = np.load(self.label_paths[index], allow_pickle=False)
        case_id = self.case_ids[index]
        if x.ndim != 4 or x.shape[-1] != 4:
            raise ValueError(
                f"Expected {self.volume_paths[index]} with shape (H,W,D,4), got {x.shape}"
            )
        y = validate_and_remap_label(y, num_classes=self.num_classes)
        if tuple(x.shape[:3]) != tuple(y.shape):
            raise ValueError(
                f"Image/label shape mismatch for {case_id}: {x.shape[:3]} versus {y.shape}"
            )
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite image values found for {case_id}")
        return x, y, case_id


class BraTSTrainDataset(_BraTSBase):
    """Return a four-channel crop and one-hot label."""

    def __getitem__(self, index: int):
        x, y, case_id = self._load(index)
        x_b = x[None, ...]
        y_b = y[None, ...]
        x_b, y_b = self.transforms([x_b, y_b])

        y_aug = validate_and_remap_label(np.asarray(y_b[0]), num_classes=self.num_classes)
        x_chwd = np.ascontiguousarray(x_b[0].transpose(3, 0, 1, 2), dtype=np.float32)
        y_one_hot = np.ascontiguousarray(
            one_hot_numpy(y_aug, self.num_classes).transpose(3, 0, 1, 2),
            dtype=np.float32,
        )
        return torch.from_numpy(x_chwd), torch.from_numpy(y_one_hot), case_id


class BraTSEvalDataset(_BraTSBase):
    """Return a full four-channel volume and integer label."""

    def __getitem__(self, index: int):
        x, y, case_id = self._load(index)
        x_b, y_b = self.transforms([x[None, ...], y[None, ...]])
        y_eval = validate_and_remap_label(np.asarray(y_b[0]), num_classes=self.num_classes)
        x_chwd = np.ascontiguousarray(x_b[0].transpose(3, 0, 1, 2), dtype=np.float32)
        y_eval = np.ascontiguousarray(y_eval, dtype=np.int64)
        return torch.from_numpy(x_chwd), torch.from_numpy(y_eval), case_id


# Backward-compatible names for users replacing files in the older repository.
Brats_loadall_nii = BraTSTrainDataset
Brats_loadall_test_nii = BraTSEvalDataset
