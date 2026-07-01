"""
evaluation/purged_cv.py
Purged walk-forward splits to prevent label-horizon leakage.

Removes `purge_gap` samples before each validation fold (default: FORWARD_CANDLES).
"""
from __future__ import annotations

import numpy as np

from config import settings


def purged_time_series_split(
    n_samples: int,
    n_splits: int = 3,
    purge_gap: int = None,
    embargo: int = 0,
):
    """
    Yield (train_indices, val_indices) expanding-window splits.

    Train: [0, val_start - purge_gap)
    Val:   [val_start, val_end)
    """
    if n_samples < n_splits + 2:
        raise ValueError(f"Not enough samples ({n_samples}) for {n_splits} splits")

    purge_gap = purge_gap if purge_gap is not None else settings.FORWARD_CANDLES
    fold_size = n_samples // (n_splits + 1)

    for k in range(1, n_splits + 1):
        val_start = fold_size * k
        val_end   = fold_size * (k + 1) if k < n_splits else n_samples
        train_end = max(0, val_start - purge_gap - embargo)

        if train_end < 50 or val_end - val_start < 10:
            continue

        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(val_start, val_end)
        yield train_idx, val_idx
