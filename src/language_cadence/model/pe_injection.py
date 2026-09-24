"""
pe_injection.py
=================

Wraps the two ways of combining the token embeddings with the Pi
positional encoding, switched via ModelConfig.pe_injection. This is one
of the two things Stage 3 was explicitly asked to make config-driven
rather than hardcoded -- see config.py's docstring and the project
notes' "small issues to resolve at implementation time" section.

Both modes take a token-embedding tensor of shape (batch, seq_len, D)
and a Pi tensor of shape (seq_len, D) (Pi has no batch dimension since
it's the same for every sequence in the batch -- see
torch_periodic_pe.py) and return a combined tensor of shape
(batch, seq_len, D), so the rest of the model (TransformerBlock etc.)
never needs to know or care which injection mode is active.

- "additive": x = token_emb + pe_amplitude * Pi. This is Prince's /
  GPT's default convention. His justification for just adding the two
  relies on D >> N so Pi occupies a near-orthogonal subspace from the
  token embeddings and the model can "read off" position and content
  separately -- an assumption the project notes flag as possibly NOT
  holding cleanly at this laptop scale, where D and N can be comparable.
  pe_amplitude is a free scalar controlling Pi's relative strength
  against the token embeddings; it must be fixed and held constant
  across the whole P-sweep (varying it would confound amplitude with
  periodicity, which is exactly the kind of experimental error the
  project notes call out).
- "concat": x = Linear(concat(token_emb, Pi)). Rather than asking the
  model to disentangle a sum, the token and positional information are
  kept in separate feature slots (doubling the feature dimension to 2*D)
  and a single learned linear layer decides how to mix them down to D.
  This sidesteps the near-orthogonality assumption entirely -- at the
  cost of extra learnable parameters (the projection layer) and a
  concatenation from which the model must actively *learn* to extract
  positional structure, rather than getting it "for free" as an additive
  offset it could in principle ignore across other dimensions.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class PEInjection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.method = config.pe_injection
        self.pe_amplitude = config.pe_amplitude
        if self.method == "concat":
            self.proj = nn.Linear(2 * config.D, config.D)
        else:
            self.proj = None  # additive mode has no extra parameters

    def forward(self, token_emb: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """
        token_emb: (batch, seq_len, D)
        pe:        (seq_len, D)  -- broadcasts across the batch dimension
        returns:   (batch, seq_len, D)
        """
        if self.method == "additive":
            return token_emb + self.pe_amplitude * pe  # pe broadcasts: (seq_len,D) + (batch,seq_len,D)
        else:  # "concat"
            batch, seq_len, _ = token_emb.shape
            pe_expanded = pe.unsqueeze(0).expand(batch, seq_len, -1)  # (batch, seq_len, D)
            combined = torch.cat([token_emb, pe_expanded], dim=-1)  # (batch, seq_len, 2*D)
            return self.proj(combined)
