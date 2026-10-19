from __future__ import annotations

import random

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Deterministically seed NumPy and Python from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Backward-compatible alias.
init_fn = seed_worker
