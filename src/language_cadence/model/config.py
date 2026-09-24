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
    # Settled via a calibration pilot (see README's "Stage 3 decisions"
    # section and the project write-up notes for the full comparison and
    # the reasoning): 7 short runs compared additive at amplitude in
    # {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0} against concat. Additive at
    # amplitude=0.1 reached the numerically lowest eval loss, but only
    # narrowly ahead of concat (2.243 vs 2.264 -- a margin well within
    # what a single-seed, short, one-language pilot could produce as
    # noise). Chose CONCAT despite not being the literal top score,
    # because it has a real structural advantage the numbers alone don't
    # capture: no amplitude hyperparameter to get wrong. Amplitude turned
    # out to have a narrow, saturating sweet spot (see pe_amplitude's
    # docstring below) governed by an architecture detail unrelated to
    # the actual cadence hypothesis -- concat sidesteps that fragility
    # entirely.
    pe_injection: PEInjectionMethod = "concat"
    # Only used when pe_injection == "additive": x = token_emb + pe_amplitude * Pi.
    # Not used by the chosen "concat" default above -- kept configurable
    # (and documented) since additive remains available for later
    # comparison/ablation. Must stay fixed across the whole P-sweep if
    # ever switched back to additive, to avoid confounding amplitude with
    # periodicity: past a certain amplitude, every transformer block's
    # leading LayerNorm makes further increases invisible anyway (it's
    # scale-invariant, LayerNorm(c*v) == LayerNorm(v)), so amplitude only
    # matters in the small-to-moderate range where it's actually
    # competing with the token embedding's own magnitude. 0.1 was the
    # pilot's best-performing value in that range.
    pe_amplitude: float = 0.1

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
