from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Sequence, Optional, Union

import torch

# Fixed channel order used by the preprocessed arrays and throughout the code.
MODALITY_NAMES = ("T1ce", "T1", "FLAIR", "T2")

SUBSETS = OrderedDict([
    ("t2",                 (False, False, False, True)),
    ("t1ce",               (True,  False, False, False)),
    ("t1",                 (False, True,  False, False)),
    ("flair",              (False, False, True,  False)),
    ("t1ce_t2",            (True,  False, False, True)),
    ("t1ce_t1",            (True,  True,  False, False)),
    ("flair_t1",           (False, True,  True,  False)),
    ("t1_t2",              (False, True,  False, True)),
    ("flair_t2",           (False, False, True,  True)),
    ("flair_t1ce",         (True,  False, True,  False)),
    ("flair_t1ce_t1",      (True,  True,  True,  False)),
    ("flair_t1_t2",        (False, True,  True,  True)),
    ("flair_t1ce_t2",      (True,  False, True,  True)),
    ("t1ce_t1_t2",         (True,  True,  False, True)),
    ("all",                (True,  True,  True,  True)),
])

DISPLAY_NAMES = OrderedDict([
    ("t2", "T2"),
    ("t1ce", "T1ce"),
    ("t1", "T1"),
    ("flair", "FLAIR"),
    ("t1ce_t2", "T1ce+T2"),
    ("t1ce_t1", "T1ce+T1"),
    ("flair_t1", "FLAIR+T1"),
    ("t1_t2", "T1+T2"),
    ("flair_t2", "FLAIR+T2"),
    ("flair_t1ce", "FLAIR+T1ce"),
    ("flair_t1ce_t1", "FLAIR+T1ce+T1"),
    ("flair_t1_t2", "FLAIR+T1+T2"),
    ("flair_t1ce_t2", "FLAIR+T1ce+T2"),
    ("t1ce_t1_t2", "T1ce+T1+T2"),
    ("all", "All four"),
])


def subset_names(spec: str) -> List[str]:
    """Parse ``all``, ``representative``, ``full``, or a comma list."""
    value = spec.strip().lower()
    if value == "all":
        return list(SUBSETS.keys())
    if value == "full":
        return ["all"]
    if value == "representative":
        return ["t2", "t1ce", "flair", "flair_t2", "flair_t1_t2", "all"]
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    names = ["all" if item == "full" else item for item in names]
    unknown = [item for item in names if item not in SUBSETS]
    if unknown:
        raise ValueError(f"Unknown modality subset(s): {unknown}. Valid names: {list(SUBSETS)}")
    if not names:
        raise ValueError("At least one modality subset must be selected.")
    return names


def masks_tensor(names: Sequence[str], device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    return torch.tensor([SUBSETS[name] for name in names], dtype=torch.bool, device=device)


def validate_mask(mask: torch.Tensor) -> None:
    if mask.ndim != 2 or mask.shape[1] != 4:
        raise ValueError(f"Expected mask with shape (B,4), got {tuple(mask.shape)}")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every sample must contain at least one available modality.")
