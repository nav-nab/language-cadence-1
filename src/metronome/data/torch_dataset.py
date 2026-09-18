"""
torch_dataset.py
=================

Thin PyTorch `Dataset` wrapper around the .npy window files that
preprocess.py writes. Kept separate from preprocess.py so that all the
actual data-wrangling logic (extraction, normalisation, vocab, splitting,
windowing) has zero dependency on PyTorch and can be built/tested without
it installed -- only this file, and later the model/training code, need
`torch` importable.

NOTE for the PyTorch-learning goal of this project: this is a good
minimal example of the standard PyTorch data-loading pattern --
implement a Dataset with __len__ and __getitem__, then hand it to a
DataLoader (in the training script, not here) to get automatic batching,
shuffling, and (optionally) multi-process loading for free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CharWindowDataset(Dataset):
    """
    Wraps one .npy file of shape (num_windows, window_length + 1) --
    produced by preprocess.documents_to_windows -- as a PyTorch Dataset
    of (input, target) pairs for next-character prediction.

    Each row of the underlying array is a window of window_length + 1
    character ids. We split it as:
        input  = window[:-1]   (the first window_length characters)
        target = window[1:]    (the same characters, shifted left by one)
    so that target[i] is exactly "the character that comes after
    input[i]" -- i.e. target[i] == input[i + 1] (except at the very last
    position, where target[-1] is the one extra character we kept around
    for exactly this purpose). This is the standard "teacher forcing"
    setup for training a language model on next-token prediction: at
    every position in the sequence, the model gets one loss term for
    predicting the next character from everything up to and including
    the current one.
    """

    def __init__(self, npy_path: str | Path):
        self.npy_path = Path(npy_path)
        # mmap_mode="r" avoids loading the whole array into RAM up front --
        # useful once these files get large; numpy pages data in from disk
        # as it's indexed, which suits random-access Dataset reads well.
        self.windows: np.ndarray = np.load(self.npy_path, mmap_mode="r")
        assert self.windows.ndim == 2, f"expected 2D array, got shape {self.windows.shape}"

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # np.asarray(...) copies the (read-only, memory-mapped) row into a
        # normal writable array before handing it to torch -- torch.from_numpy
        # would otherwise complain about (or silently share) a read-only buffer.
        window = np.asarray(self.windows[idx], dtype=np.int64)
        x = torch.from_numpy(window[:-1])  # shape: (window_length,)
        y = torch.from_numpy(window[1:])   # shape: (window_length,)
        return x, y
