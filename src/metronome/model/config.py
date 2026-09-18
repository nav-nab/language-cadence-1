"""
config.py
=========

A single dataclass holding every architectural and positional-encoding
hyperparameter the model needs. The point of centralising everything
here, rather than passing loose arguments around or hardcoding values
inside GPT/TransformerBlock/etc., is the project's Stage 3 requirement:
the injection method (additive vs concat) and the relative amplitude of
Pi vs the token embeddings are explicitly flagged as UNRESOLVED design
decisions in the project notes -- so the code must make them trivially
easy to change and compare, not bury them as implicit defaults deep
inside some layer's __init__.

D and H are related by H = D // 2 (see periodic_pe.py) -- H is derived,
not stored separately, so it's impossible for the two to accidentally
drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PEInjectionMethod = Literal["additive", "concat"]


@dataclass
class ModelConfig:
    # --- Data-dependent (set from the vocab/window sizes built in Stage 1) ---
    vocab_size: int
    max_seq_len: int  # must be >= the window_length used when building the dataset

    # --- Core architecture (fixed across the whole P-sweep, per the project notes:
    #     only P should vary between sweep runs for a fixed language) ---
    D: int = 64          # embedding / model dimension. Must be even (D = 2H).
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4    # feedforward hidden size = ffn_mult * D (standard GPT convention)
    dropout: float = 0.1

    # --- Positional encoding (Pi) ---
    P: int = 45           # the period being tested -- THIS is what the Stage 4 sweep varies
    pe_injection: PEInjectionMethod = "additive"
    # Only used when pe_injection == "additive": x = token_emb + pe_amplitude * Pi.
    # Flagged as an open decision in the project notes -- must be fixed and held
    # constant across the whole P-sweep once chosen, to avoid confounding
    # amplitude with periodicity. Default of 1.0 is a placeholder, not a
    # considered choice -- see README's "Stage 3 decisions" section.
    pe_amplitude: float = 1.0

    def __post_init__(self):
        if self.D % 2 != 0:
            raise ValueError(f"D must be even (D = 2H row-pairs for Pi), got {self.D}")
        if self.D % self.n_heads != 0:
            raise ValueError(f"D ({self.D}) must be divisible by n_heads ({self.n_heads})")
        if self.pe_injection not in ("additive", "concat"):
            raise ValueError(f"pe_injection must be 'additive' or 'concat', got {self.pe_injection!r}")

    @property
    def H(self) -> int:
        """Number of Pi row-pairs. Derived from D, never set independently."""
        return self.D // 2

    @property
    def head_dim(self) -> int:
        return self.D // self.n_heads
