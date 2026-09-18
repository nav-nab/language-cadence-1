"""
mlp.py
======

The feedforward ("MLP") sub-layer used inside each transformer block.

Where attention mixes information ACROSS positions (each position can
look at every earlier position), the feedforward block processes EACH
position independently and identically -- the same two-layer network is
applied at every position, with no mixing between positions at all. The
usual intuition: attention gathers relevant context from elsewhere in
the sequence, and the feedforward block then does further, position-local
computation on whatever attention just gathered.

Standard GPT convention: expand D -> ffn_mult*D (a wider hidden layer,
typically 4x), apply a nonlinearity, then project back down to D. The
expansion gives the network more room to compute in before compressing
back to the model's working dimension.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class FeedForward(nn.Module):
    """
    Two-layer MLP: Linear(D -> ffn_mult*D) -> GELU -> Linear(ffn_mult*D -> D) -> Dropout.

    GELU (Gaussian Error Linear Unit) is the standard nonlinearity used
    in GPT-family models (as opposed to plain ReLU) -- a smoother
    activation function that tends to train slightly better in practice
    for transformers. Input/output shape: (batch, seq_len, D) -> (batch, seq_len, D).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.ffn_mult * config.D
        self.net = nn.Sequential(
            nn.Linear(config.D, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.D),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
