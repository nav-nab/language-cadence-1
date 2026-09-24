"""
gpt.py
======

The full small GPT model: token embedding -> Pi positional encoding
(injected additively or via concat, per config) -> a stack of
TransformerBlocks -> final LayerNorm -> linear head projecting back to
vocab-sized logits.

PyTorch-learning note on the overall shape convention used throughout
this model: every intermediate tensor is (batch, seq_len, D) --
"batch first". Only right at the very start (turning integer token ids
into vectors) and the very end (turning D-dim vectors into per-vocab
logits) does the feature dimension change size.
"""

from __future__ import annotations

import torch
from torch import nn

from ..pe.torch_periodic_pe import PeriodicPositionalEncoding
from .block import TransformerBlock
from .config import ModelConfig
from .pe_injection import PEInjection


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # nn.Embedding is a lookup table: given an integer token id, it
        # returns that row's learned D-dimensional vector. Shape:
        # (batch, seq_len) integer ids -> (batch, seq_len, D) floats.
        self.token_embedding = nn.Embedding(config.vocab_size, config.D)

        self.pe = PeriodicPositionalEncoding(
            D=config.D, P=config.P, max_seq_len=config.max_seq_len
        )
        self.pe_injection = PEInjection(config)

        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_ln = nn.LayerNorm(config.D)

        # Projects the final D-dim representation at each position to a
        # score ("logit") for every symbol in the vocabulary -- softmax
        # over this (done outside the model, by the loss function) gives
        # the predicted next-character probability distribution.
        self.lm_head = nn.Linear(config.D, config.vocab_size, bias=False)

        # Weight tying: reuse the token embedding matrix as the output
        # projection (transposed), a well-established GPT convention.
        # Intuition: "the vector that represents token X as an input"
        # and "the direction that should score highly when the model
        # wants to predict token X as output" are related things to
        # learn, so sharing the parameter is both a parameter-count
        # saving (removes a D x vocab_size matrix that would otherwise
        # be learned from scratch) and, empirically, usually a mild
        # regulariser that doesn't hurt quality.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Standard GPT-style initialisation: small random normal weights,
        # zero biases, unit-scale LayerNorm. Reasonable defaults chosen
        # to keep early-training activations/gradients well-scaled; not
        # something this project needs to tune, so it's kept simple and
        # out of ModelConfig.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def num_parameters(self, exclude_embedding_duplication: bool = True) -> int:
        """
        Total trainable parameter count. Since lm_head's weight is tied
        to (the same tensor object as) token_embedding's weight, naively
        summing every parameter's numel() would double-count that one
        shared matrix; exclude_embedding_duplication=True (the default)
        corrects for that so this reports the true number of distinct
        learned values, matching how people usually report model size.
        """
        seen = set()
        total = 0
        for p in self.parameters():
            if exclude_embedding_duplication:
                if id(p) in seen:
                    continue
                seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) integer tensor of vocab ids.
        Returns logits of shape (batch, seq_len, vocab_size) -- one
        vocab-sized score vector per position, per batch element,
        representing "predicted distribution over the NEXT character,
        given everything up to and including this position" (recall
        CausalSelfAttention guarantees position i never sees position
        i+1 or later, so this is a well-formed next-token prediction at
        every position simultaneously, not just at the last one -- this
        is what lets one forward pass yield a loss term for every
        position in the window, per torch_dataset.py's docstring).
        """
        batch, seq_len = token_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds config.max_seq_len={self.config.max_seq_len}"
            )

        tok_emb = self.token_embedding(token_ids)  # (batch, seq_len, D)
        pe = self.pe(seq_len=seq_len)  # (seq_len, D)
        x = self.pe_injection(tok_emb, pe)  # (batch, seq_len, D)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits
