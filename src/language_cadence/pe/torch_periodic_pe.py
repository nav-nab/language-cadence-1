"""
torch_periodic_pe.py
======================

PyTorch nn.Module wrapping the same period-P positional encoding as
periodic_pe.py (the NumPy reference), for use inside the actual GPT
model (Stage 3).

NOTE for PyTorch learning: this is a good example of a "parameter-free"
nn.Module -- it has no learnable weights (nn.Parameter). Instead, we
precompute the Π matrix once (in __init__, from D/H/P/N_max, all fixed
hyperparameters for a given run) and store it as a *buffer*
(self.register_buffer) rather than a plain attribute. A buffer is
PyTorch's mechanism for "this tensor is part of the module's state and
should move with it (e.g. .to(device), .cuda()) and be saved/loaded with
state_dict(), but it should NOT be updated by the optimizer during
training" -- exactly what we want for a fixed, non-learned encoding.

Design choice carried over from the NumPy version: H (= D/2) must stay
FIXED across the whole P-sweep for a given language (per the project
notes), so it's a required constructor argument, not inferred from
anything that could silently change between runs.
"""

from __future__ import annotations

import torch
from torch import nn


class PeriodicPositionalEncoding(nn.Module):
    """
    Precomputes Π (shape (D, N_max)) at construction time for a fixed
    (D, P, N_max), then serves Π[:, :N] -- transposed to (N, D), which is
    the more natural orientation for adding directly onto a (batch,
    seq_len, D) activation tensor, PyTorch's usual "batch first" layout
    -- via forward().

    Precomputing Π once at construction (rather than recomputing sin/cos
    on every forward pass) is both faster and guarantees every call sees
    numerically IDENTICAL positional encodings, which matters here since
    the whole experiment depends on Π being a fixed, well-defined
    function of position -- not something that could drift call to call
    due to floating-point non-determinism in repeated trig evaluations.
    """

    def __init__(self, D: int, P: int, max_seq_len: int):
        super().__init__()
        if D % 2 != 0:
            raise ValueError(f"D must be even (D = 2H row-pairs), got {D}")
        self.D = D
        self.H = D // 2
        self.P = P
        self.max_seq_len = max_seq_len

        pe = self._build_pe(D=D, N=max_seq_len, P=P)  # shape (D, max_seq_len)
        # Store as (max_seq_len, D) -- i.e. already transposed to the
        # (seq_len, D) orientation forward() will add onto activations in.
        pe = pe.transpose(0, 1).contiguous()  # shape (max_seq_len, D)
        # persistent=True (the default) means this buffer is included in
        # state_dict() -- fine here since Pi is cheap to store and this
        # keeps a saved checkpoint fully self-describing (you can recover
        # the exact positional encoding it was trained with even without
        # re-deriving it from D/P/max_seq_len).
        self.register_buffer("pe", pe)

    @staticmethod
    def _build_pe(D: int, N: int, P: int) -> torch.Tensor:
        """
        Torch equivalent of periodic_pe.periodic_pe -- deliberately
        mirrors that function's structure line-for-line (folded harmonic
        orders -> outer product angle -> sin/cos row-pairs) so the two
        implementations are easy to compare by eye, and are cross-checked
        for exact numerical agreement in tests/test_torch_periodic_pe.py.
        """
        H = D // 2
        j = torch.arange(1, H + 1, dtype=torch.float64)
        m = j % P
        k = torch.minimum(m, P - m)  # folded harmonic orders, shape (H,)

        n = torch.arange(N, dtype=torch.float64)  # shape (N,)
        angle = torch.outer(k, n) * (2.0 * torch.pi / P)  # shape (H, N)

        pe = torch.empty((D, N), dtype=torch.float64)
        pe[0::2, :] = torch.sin(angle)
        pe[1::2, :] = torch.cos(angle)
        return pe.to(torch.float32)  # match the model's usual working dtype

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns the (seq_len, D) slice of the precomputed Π, ready to be
        added onto an activation tensor of shape (batch, seq_len, D) via
        ordinary broadcasting (PyTorch broadcasts (seq_len, D) against
        (batch, seq_len, D) automatically -- no manual .unsqueeze/.expand
        needed).

        Takes seq_len rather than the activation tensor itself: this
        module has nothing to compute from the data, only from the
        requested length, which keeps it trivial to unit test in
        isolation (see tests/test_torch_periodic_pe.py) without needing
        to construct a fake activation tensor at all.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested seq_len={seq_len} exceeds max_seq_len={self.max_seq_len} "
                f"this module was constructed with."
            )
        return self.pe[:seq_len, :]
