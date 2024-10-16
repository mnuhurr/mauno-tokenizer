from pathlib import Path
import h5py
import numpy as np


def data_sizes(filename: str | Path) -> tuple[int, int]:
    with h5py.File(filename, 'r') as h5_file:
        n_files, n_mels, _ = h5_file['mels'].shape

    return n_files, n_mels


def train_val_idx(n_idx: int, validation_size: float = 0.1, seed: int | None = None) -> tuple[list[int], list[int]]:
    idx = list(range(n_idx))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n_idx * validation_size)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


