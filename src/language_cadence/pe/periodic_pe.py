"""
periodic_pe.py
===============

The Π positional encoding matrix, exactly as specified in the project
design notes -- pure NumPy reference implementation.

Why a NumPy version at all, if we're going to need a PyTorch version for
the model (see torch_periodic_pe.py)? Two reasons: (1) it's the easiest
thing to unit test rigorously and to use for the pure-encoding-geometry
sanity checks (the distinguishability-vs-P heatmap) that don't involve
training a model at all, and (2) having an independent, simple reference
implementation is a good way to catch a bug in the PyTorch version later
(the two should always produce numerically identical output for the same
inputs -- see tests/test_torch_periodic_pe.py).

Notation (matching Prince's "Understanding Deep Learning", used
throughout the project notes): X is the D x N data matrix (D = embedding
dimension, N = sequence length). Π is the D x N positional encoding
matrix, added to X. We use H = D / 2 "row-pairs": row-pair j (for
j = 1..H) occupies rows (2j-2, 2j-1) of Π (0-indexed) and gets a sin/cos
pair oscillating at one particular harmonic.

The period-P encoding, with harmonic folding
----------------------------------------------
A naive periodic encoding would give row-pair j the harmonic order
k = j directly, i.e. oscillation frequency 2*pi*j/P. But for j > P/2,
this alias es back onto a *lower* harmonic that's already been used by
an earlier row-pair (sin/cos with frequency 2*pi*k/P is indistinguishable
from frequency 2*pi*(P-k)/P as a function of integer n, since folding a
frequency past the Nyquist-like point P/2 just reflects it). Row-pairs
whose harmonic order aliases onto one already used are wasted -- they
duplicate information instead of adding new distinguishing power.

So instead we FOLD the row-pair index j into an effective harmonic order:

    k_j = min(j mod P, P - (j mod P))

which maps j = 1, 2, ..., onto the sequence of *distinct, non-aliasing*
harmonic orders 1, 2, ..., floor(P/2), then wraps back down again rather
than continuing to alias. This is what makes Π exactly periodic in n
with period P (Π(n) == Π(n+P) for every n -- verified in the unit tests)
while still using every row-pair's embedding capacity productively,
rather than the geometric-frequency-spread encoding from the original
"Attention is All You Need" paper (which never exactly repeats, and
wasn't designed with a fixed target period in mind at all).

Row-pair j's two rows are then:

    Π[2j-2, n] = sin(n * 2*pi*k_j / P)
    Π[2j-1, n] = cos(n * 2*pi*k_j / P)
"""

from __future__ import annotations

import numpy as np


def harmonic_orders(H: int, P: int) -> np.ndarray:
    """
    Compute the folded harmonic order k_j for row-pairs j = 1..H, given
    period P. Returns an array of shape (H,), k_orders[j-1] == k_j.

    Folding formula: k_j = min(j mod P, P - (j mod P)).

    Worked example (P=5): j mod P cycles 1,2,3,4,0,1,2,3,4,0,...
    folding min(m, 5-m) turns that into 1,2,2,1,0,1,2,2,1,0,...
    -- i.e. distinct harmonics 0,1,2 (the maximum distinguishable
    harmonic for period 5 is floor(5/2)=2), each appearing twice before
    the pattern repeats, rather than continuing to climb to meaningless
    aliased frequencies.
    """
    if P <= 0:
        raise ValueError(f"P must be a positive integer, got {P}")
    j = np.arange(1, H + 1)
    m = j % P
    return np.minimum(m, P - m)


def periodic_pe(D: int, N: int, P: int) -> np.ndarray:
    """
    Build the full D x N periodic positional encoding matrix Π, for
    embedding dimension D (must be even -- D = 2H row-pairs) and
    sequence length N, at period P.

    Returns a float64 array of shape (D, N).
    """
    if D % 2 != 0:
        raise ValueError(f"D must be even (D = 2H row-pairs), got {D}")
    H = D // 2

    k = harmonic_orders(H, P)  # shape (H,)
    n = np.arange(N)  # shape (N,)

    # Outer product: angle[j, n] = n * 2*pi*k_j / P, shape (H, N).
    angle = np.outer(k, n) * (2.0 * np.pi / P)

    pe = np.empty((D, N), dtype=np.float64)
    pe[0::2, :] = np.sin(angle)  # rows 0, 2, 4, ... -> sin, one per row-pair
    pe[1::2, :] = np.cos(angle)  # rows 1, 3, 5, ... -> cos, one per row-pair
    return pe
