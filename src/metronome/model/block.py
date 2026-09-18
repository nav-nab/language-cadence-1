"""
block.py
========

One transformer block: self-attention followed by a feedforward layer,
each wrapped in a residual connection with layer normalisation.

We use "pre-norm" (LayerNorm BEFORE the sub-layer, not after), which is
the standard modern choice over the original Transformer paper's
"post-norm": x = x + Sublayer(LayerNorm(x)), rather than
x = LayerNorm(x + Sublayer(x)). Pre-norm keeps a clean, unnormalised
residual "stream" running through the whole depth of the network (the
same design idea used in ResNets), which empirically makes deep
transformers noticeably easier to train -- post-norm models are more
prone to unstable gradients, especially as depth increases.

Each of the two sub-layers (attention, feedforward) gets its own
LayerNorm and its own residual connection:

    x = x + Attention(LayerNorm1(x))
    x = x + FeedForward(LayerNorm2(x))

The residual "+" is doing a lot of work here: it means each sub-layer
only has to learn a *correction* to add onto the running representation,
rather than having to reconstruct the whole thing from scratch -- and,
just as importantly for training deep stacks, it gives gradients a
direct, unimpeded path (the identity path through the "+") all the way
back through every block during backpropagation.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .mlp import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.D)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.D)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
