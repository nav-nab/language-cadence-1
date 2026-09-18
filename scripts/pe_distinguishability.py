#!/usr/bin/env python3
"""
pe_distinguishability.py
==========================

Reproduces the pure positional-encoding-geometry analysis from the
design phase (before any model training): for a fixed H (number of Π
row-pairs, i.e. fixed D = 2H), how "distinguishable" are neighbouring
position columns of Π, as a function of the period P?

This is a sanity check against Navid's earlier analysis, NOT a claim
about trained-model behaviour -- see the project write-up notes
(language-cadence-writeup.md) for why that distinction matters: the
geometric peak here is a property of the encoding alone, derived before
any training, and a hypothesis for where the empirical P* (from actual
test loss, Stage 4/5) might land -- not a guarantee.

Distinguishability metric
--------------------------
For a given P, take Π at that period (shape D x N for some N >= P), and
measure how different adjacent columns are: distinguishability(P) =
mean over n of ||Π(n) - Π(n+1)||_2 (mean L2 distance between
neighbouring position columns, averaged over one full period so the
starting phase doesn't matter). This is a direct, interpretable
"how much does moving one position actually change the encoding"
measure -- too small a period wastes row-pairs on aliased/redundant
harmonics (many effectively-duplicate directions -> small
column-to-column movement in some directions, but not this aggregate
metric directly -- see the note in the plot function about a second,
complementary metric); too large a period spaces adjacent columns too
closely together in absolute angle terms (slow-wavelength "packing"
decay) -- both hurt distinguishability, producing an interior optimum.

Outputs
-------
- figures/pe_distinguishability_vs_P.png -- line plot, distinguishability
  vs P, for the H used in the project design analysis (H=32, i.e. D=64).
- figures/pe_distinguishability_heatmap.png -- heatmap over (H, P), to
  see how the peak location shifts with H (should track ~2H).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # headless -- no display in this environment
import matplotlib.pyplot as plt
import numpy as np

from metronome.pe.periodic_pe import periodic_pe  # noqa: E402


def distinguishability(D: int, P: int) -> float:
    """
    Mean L2 distance between neighbouring position columns of Pi(D, N=P, P),
    averaged over exactly one period (n=0..P-1, wrapping n=P-1 -> n=0),
    for period P and embedding dim D.
    """
    pe = periodic_pe(D=D, N=P, P=P)  # shape (D, P) -- exactly one period
    # Columns 0..P-2 compared to the next column, PLUS the wrap-around
    # pair (P-1 compared to 0, since Pi is periodic with period P).
    diffs = pe[:, 1:] - pe[:, :-1]  # shape (D, P-1)
    wrap_diff = pe[:, 0] - pe[:, -1]  # shape (D,)
    all_diffs = np.concatenate([diffs, wrap_diff[:, None]], axis=1)  # (D, P)
    return float(np.linalg.norm(all_diffs, axis=0).mean())


def sweep_p(D: int, p_values: np.ndarray) -> np.ndarray:
    return np.array([distinguishability(D, int(p)) for p in p_values])


def plot_distinguishability_vs_p(H: int, p_values: np.ndarray, out_path: Path) -> int:
    """
    Line plot of distinguishability vs P for a single fixed H, with the
    P that achieves the maximum marked. Returns the argmax P for the
    caller to report.
    """
    D = 2 * H
    scores = sweep_p(D, p_values)
    p_star = int(p_values[np.argmax(scores)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_values, scores, color="#3b6ba5", linewidth=1.8)
    ax.axvline(p_star, color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"peak at P={p_star}")
    ax.axvline(2 * H, color="#7f8c8d", linestyle=":", linewidth=1.2,
               label=f"2H={2*H} (design-notes prediction)")
    ax.set_xlabel("Period P")
    ax.set_ylabel("Mean neighbouring-column L2 distance")
    ax.set_title(f"Pi distinguishability vs P (H={H}, D={D})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return p_star


def plot_heatmap(h_values: np.ndarray, p_values: np.ndarray, out_path: Path) -> None:
    """
    Heatmap of distinguishability over the (H, P) grid, with the
    per-H argmax-P traced as a line, to check whether the peak tracks
    the ~2H prediction across a range of H, not just at H=32.
    """
    grid = np.zeros((len(h_values), len(p_values)))
    peak_p_per_h = np.zeros(len(h_values))
    for i, h in enumerate(h_values):
        scores = sweep_p(2 * int(h), p_values)
        grid[i, :] = scores
        peak_p_per_h[i] = p_values[np.argmax(scores)]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap="viridis",
        extent=[p_values.min(), p_values.max(), h_values.min(), h_values.max()],
    )
    ax.plot(peak_p_per_h, h_values, color="white", linewidth=1.5, label="argmax P per H")
    ax.plot(2 * h_values, h_values, color="red", linestyle="--", linewidth=1.2, label="P = 2H")
    ax.set_xlabel("Period P")
    ax.set_ylabel("H (= D/2)")
    ax.set_title("Pi distinguishability over (H, P)")
    ax.legend(loc="upper left", facecolor="white", framealpha=0.85)
    fig.colorbar(im, ax=ax, label="Mean neighbouring-column L2 distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Single-H analysis at H=32 (D=64), matching the design-phase
    # analysis referenced in the project notes (peak reported near P≈45).
    p_values_fine = np.arange(2, 201)
    p_star = plot_distinguishability_vs_p(
        H=32, p_values=p_values_fine, out_path=figures_dir / "pe_distinguishability_vs_P.png"
    )
    print(f"H=32 (D=64): distinguishability-maximising P = {p_star} "
          f"(design notes reported peak near P≈45)")

    # Heatmap over a range of H, to see whether ~2H tracks the peak
    # more generally, not just at the one H value already checked.
    h_values = np.arange(4, 65, 4)
    p_values_coarse = np.arange(2, 141)
    plot_heatmap(h_values, p_values_coarse, figures_dir / "pe_distinguishability_heatmap.png")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
