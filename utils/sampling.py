from __future__ import annotations

from typing import List

import numpy as np
import torch

from utils.masks import SUBSETS


ALL_NAMES = list(SUBSETS.keys())
SINGLE_NAMES = ["t2", "t1ce", "t1", "flair"]
NO_T1CE_NAMES = [name for name, mask in SUBSETS.items() if not mask[0]]
T1CE_RICH_NAMES = [
    "all",
    "t1ce_t1_t2",
    "flair_t1ce_t2",
    "flair_t1ce_t1",
    "flair_t1ce",
    "t1ce_t1",
    "t1ce_t2",
    "t1ce",
]
T1CE_RICH_PROBABILITY = np.asarray(
    [0.30, 0.16, 0.16, 0.13, 0.10, 0.07, 0.05, 0.03], dtype=np.float64
)
T1CE_RICH_PROBABILITY /= T1CE_RICH_PROBABILITY.sum()


def sample_modality_mask(
    epoch: int,
    batch_size: int,
    device: torch.device,
    mode: str = "uniform",
) -> torch.Tensor:
    selected: List[str] = []
    for _ in range(batch_size):
        if mode == "uniform":
            name = str(np.random.choice(ALL_NAMES))
        elif mode == "hard_balanced":
            if epoch < 80:
                name = (
                    str(np.random.choice(T1CE_RICH_NAMES, p=T1CE_RICH_PROBABILITY))
                    if np.random.rand() < 0.70
                    else str(np.random.choice(ALL_NAMES))
                )
            else:
                draw = np.random.rand()
                if draw < 0.40:
                    name = str(np.random.choice(ALL_NAMES))
                elif draw < 0.70:
                    name = str(np.random.choice(SINGLE_NAMES))
                else:
                    name = str(np.random.choice(NO_T1CE_NAMES))
        else:
            raise ValueError("mask sampling must be 'uniform' or 'hard_balanced'.")
        selected.append(name)
    return torch.tensor([SUBSETS[name] for name in selected], dtype=torch.bool, device=device)


def auxiliary_weight(epoch: int, num_epochs: int, initial: float) -> float:
    """Decay auxiliary supervision while preserving the original schedule."""
    current = epoch + 1
    if current <= 0.30 * num_epochs:
        return float(initial)
    if current <= 0.70 * num_epochs:
        fraction = (current - 0.30 * num_epochs) / max(1.0, 0.40 * num_epochs)
        return float(initial * (1.0 - 0.60 * fraction))
    fraction = (current - 0.70 * num_epochs) / max(1.0, 0.30 * num_epochs)
    return float(0.20 * (1.0 - fraction) + 0.10 * fraction)
