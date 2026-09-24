#!/usr/bin/env python3
"""
plot_distinguishability_vs_loss.py
=====================================

A follow-up to Stage 5, prompted by a specific observation Navid made
while reviewing the results: Pi was deliberately DESIGNED (Stage 2) so
that an overly long period P makes neighbouring positions harder to
tell apart -- "distinguishability" (mean L2 distance between adjacent
position columns of Pi) peaks around P=45 for this project's D=64, then
falls off sharply for larger P. If that geometric property mattered for
trained-model performance, you'd expect test loss to get WORSE, not
better, as P grows well past 45 -- distinguishability and test loss
should move in OPPOSITE directions (more distinguishable -> lower
loss).

This script puts the two curves on one figure, over the exact same P
grid the real sweep used, to make that comparison directly rather than
across two separate plots. It doesn't recompute anything Stage 2/5
didn't already compute -- it just reuses pe_distinguishability.py's
`distinguishability()` (the exact Stage 2 metric, unchanged) and
analyze_sweep.py's `load_sweep_by_language()` (the exact Stage 5
per-language loss data), side by side.

What the numbers actually show (see the printed correlation and the
figure): distinguishability and test loss are POSITIVELY correlated
across the swept grid -- both fall together as P grows past 45 -- the
OPPOSITE sign from what "distinguishability drives predictive
performance" would predict. Framed plainly: P=384 is the single WORST
point on the whole grid by the distinguishability metric (~5x below
the P=45 peak), and simultaneously the BEST point by test loss, for
every language. The geometric trade-off Pi was built around had no
protective effect against long periods in the actual trained models.

Output
------
- figures/distinguishability_vs_loss.png -- dual-axis plot: Stage 2's
  distinguishability (left axis) and Stage 5's per-language + mean test
  loss (right axis), same P grid (log-x), P=45 marked for reference.
- Prints the Pearson correlation between distinguishability and mean
  test loss across the grid, plus the P=45 vs P=384 comparison on both
  metrics, for direct quoting in the write-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # scripts/ isn't a package -- see test_analyze_sweep.py for the same pattern

import matplotlib

matplotlib.use("Agg")  # headless -- no display in this environment
import matplotlib.pyplot as plt
import numpy as np

from analyze_sweep import LANGUAGE_COLORS, LANGUAGE_NAMES, load_sweep_by_language  # noqa: E402
from pe_distinguishability import distinguishability  # noqa: E402
from train_sweep import FIXED_HYPERPARAMETERS  # noqa: E402


def mean_loss_per_p(by_language: dict[str, dict[int, float]], p_grid: list[int]) -> list[float]:
    """
    Cross-language mean test loss at each P in p_grid -- the bold black
    curve in the figure. Pulled out as its own small function purely so
    it has something for tests/test_plot_distinguishability_vs_loss.py
    to call directly against synthetic data, rather than only being
    checkable by eyeballing the real figure.
    """
    return [float(np.mean([by_language[lang][p] for lang in by_language])) for p in p_grid]


def main() -> None:
    results_path = REPO_ROOT / "results" / "sweep_results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"No sweep results found at {results_path} -- run scripts/train_sweep.py first.")

    by_language = load_sweep_by_language(results_path)
    p_grid = sorted(next(iter(by_language.values())).keys())

    # D=64 throughout the sweep (Pi's own row-pair count H=D/2 is what
    # the distinguishability metric depends on) -- read from
    # train_sweep.py's FIXED_HYPERPARAMETERS rather than a second
    # hardcoded "64" here, so this can't silently drift out of sync with
    # what the sweep actually used.
    D = FIXED_HYPERPARAMETERS["D"]
    distinguishability_scores = [distinguishability(D=D, P=p) for p in p_grid]

    mean_loss = mean_loss_per_p(by_language, p_grid)

    # --- Quantify the "opposite direction" claim directly, for the write-up ---
    corr = float(np.corrcoef(distinguishability_scores, mean_loss)[0, 1])
    p45_idx, p384_idx = p_grid.index(45), p_grid.index(384)
    print(f"Pearson correlation, distinguishability vs mean test loss, across the P grid: r = {corr:+.3f}")
    print(
        "(If distinguishability drove predictive performance, this should be strongly NEGATIVE -- "
        "higher distinguishability -> lower loss. A positive value means they move together instead: "
        "the worse-distinguishability points tend to be the lower-loss points.)"
    )
    print(
        f"\nP=45 (geometric peak):  distinguishability={distinguishability_scores[p45_idx]:.3f}  "
        f"mean test loss={mean_loss[p45_idx]:.4f}"
    )
    print(
        f"P=384 (worst distinguishability, best loss):  "
        f"distinguishability={distinguishability_scores[p384_idx]:.3f}  mean test loss={mean_loss[p384_idx]:.4f}"
    )

    # --- Figure ---
    fig, ax_left = plt.subplots(figsize=(9, 6))
    ax_right = ax_left.twinx()

    line_d, = ax_left.plot(
        p_grid, distinguishability_scores, color="#3b6ba5", linewidth=2.2, marker="o",
        label="Pi distinguishability (Stage 2 metric)",
    )
    ax_left.set_ylabel("Pi distinguishability (mean L2 dist.)\nhigher = more distinguishable",
                        color="#3b6ba5")
    ax_left.tick_params(axis="y", labelcolor="#3b6ba5")

    for language in sorted(by_language):
        losses = [by_language[language][p] for p in p_grid]
        ax_right.plot(p_grid, losses, color=LANGUAGE_COLORS.get(language), alpha=0.35, linewidth=1.0, zorder=1)
    line_loss, = ax_right.plot(
        p_grid, mean_loss, color="#333333", linewidth=2.2, linestyle="--", marker="s",
        label="Mean test loss across languages (Stage 5)", zorder=4,
    )
    ax_right.set_ylabel("Test loss (nats/char)\nlower = better (axis inverted); faint = per-language",
                         color="#333333")
    ax_right.tick_params(axis="y", labelcolor="#333333")
    # Loss is "better when lower" while distinguishability is "better when
    # higher" -- inverting the right axis means "up" reads as "better" on
    # BOTH axes, so the two curves moving in visibly opposite directions
    # for P > 45 is the actual "distinguishability predicts the wrong
    # thing" story, not an artifact of one axis being upside down relative
    # to the other's meaning of "good".
    ax_right.invert_yaxis()

    ax_left.axvline(45, color="gray", linestyle=":", linewidth=1, zorder=0)
    ax_left.text(45, ax_left.get_ylim()[1], "  P=45 (geometric peak)", rotation=90,
                 va="top", ha="left", fontsize=8, color="gray")

    ax_left.set_xscale("log")
    ax_left.set_xticks(p_grid)
    ax_left.set_xticklabels([str(p) for p in p_grid], rotation=45)
    ax_left.set_xlabel("Period P (log scale)")
    ax_left.set_title("Pi's own geometric distinguishability vs. trained-model test loss")
    ax_left.grid(True, which="both", alpha=0.25)
    ax_left.legend(handles=[line_d, line_loss], loc="center left")

    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "distinguishability_vs_loss.png"
    # bbox_inches="tight" (on top of fig.tight_layout() above) -- with two
    # y-axes, the left axis's long rotated label can still get clipped by
    # the figure edge even after tight_layout(); bbox_inches="tight" on
    # the actual save crops/expands the saved image to the real rendered
    # extent instead of trusting the pre-computed layout, which is the
    # part that was clipping "distinguishab[ility]" before this.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
