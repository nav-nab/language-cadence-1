"""
attention.py
=============

Multi-head causal self-attention -- the mechanism that lets each
position in the sequence look back at (and combine information from)
earlier positions, weighted by learned relevance ("attention").

"Causal" means position i is only allowed to attend to positions
0..i (itself and everything before it), never anything after -- this is
what makes next-token prediction training valid at all: if position i
could see position i+1, predicting "what comes next" at position i would
be trivially solved by cheating, not by learning language structure.

PyTorch-learning notes:
- We compute Q, K, V (queries, keys, values) via a SINGLE combined linear
  layer (nn.Linear(D, 3*D)) rather than three separate ones -- purely an
  efficiency convention (one matrix multiply instead of three), the
  result is split apart afterward with a plain tensor slice.
- The causal mask is built once and stored as a registered buffer (see
  torch_periodic_pe.py for the same pattern) since it depends only on
  max_seq_len, not on any input data -- no need to rebuild it every
  forward call.
- torch.nn.functional.scaled_dot_product_attention is PyTorch's built-in,
  fused implementation of the "Attention is All You Need" attention
  formula: softmax(Q K^T / sqrt(head_dim)) V. We use it directly (with
  is_causal=True) instead of writing out the matrix multiplies by hand,
  both because it's what you'd actually use in real PyTorch code, and
  because it's simply faster (it avoids materialising the full N x N
  attention matrix in a naive way) -- but the plain-matmul formula is
  exactly what it computes, spelled out in the docstring below for
  anyone following along who wants to see the underlying math.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Standard multi-head causal self-attention.

    Input/output shape: (batch, seq_len, D) -> (batch, seq_len, D) --
    attention mixes information ACROSS the seq_len dimension but never
    changes the shape of the tensor, which is what lets us stack many of
    these (inside TransformerBlock) with plain residual addition.

    The underlying formula, per head, for a sequence of queries Q, keys
    K, and values V (each of shape (seq_len, head_dim)):

        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(head_dim) + mask ) @ V

    where `mask` sets the (i, j) entry to -infinity whenever j > i (a
    future position), so after the softmax those positions get exactly
    zero weight -- this is what "causal" means in code, not just in
    concept. n_heads independent copies of this run in parallel on
    different learned projections of the input, then their outputs are
    concatenated back together and passed through one more linear layer
    (this lets different heads specialise in different kinds of
    relationships between positions, e.g. one head might learn to attend
    to the immediately preceding character, another to whatever position
    is P characters back).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.D = config.D
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(config.D, 3 * config.D)
        self.out_proj = nn.Linear(config.D, config.D)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, D = x.shape
        assert D == self.D, f"expected last dim {self.D}, got {D}"

        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*D)
        q, k, v = qkv.split(self.D, dim=-1)  # each (batch, seq_len, D)

        # Reshape D -> (n_heads, head_dim) and move the head dimension
        # next to batch, so scaled_dot_product_attention treats
        # (batch, n_heads) as independent "batch" dimensions it attends
        # over separately. This reshape/transpose dance -- split the
        # feature dim into heads, then move heads next to batch -- is a
        # very common PyTorch pattern worth recognising elsewhere.
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (batch, seq_len, D) -> (batch, seq_len, n_heads, head_dim)
            #                      -> (batch, n_heads, seq_len, head_dim)
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # is_causal=True applies exactly the future-masking described in
        # the class docstring internally -- no need to build our own mask
        # tensor here.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )  # (batch, n_heads, seq_len, head_dim)

        # Undo split_heads: move heads back next to the feature dim, then
        # merge (n_heads, head_dim) back into a single D-sized dimension.
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, D)

        return self.resid_dropout(self.out_proj(attn_out))
