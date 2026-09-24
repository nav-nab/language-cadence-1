#!/usr/bin/env python3
"""
analyze_sweep.py
==================

Stage 5: the actual test of the cadence hypothesis. Reads the real
sweep results (results/sweep_results.jsonl, produced by
scripts/train_sweep.py) and asks the two questions the whole project
was built to answer:

1. For each language, does test loss vs P have an INTERIOR minimum
   (loss goes down, then back up, somewhere strictly between the
   smallest and largest P tested) -- evidence for a genuine per-language
   optimal period? Or is it MONOTONIC across the tested range (loss
   just keeps improving as P grows) -- the "Ceiling Confound" from the
   design notes: P behaving like an enlarged effective context window
   rather than revealing an intrinsic cadence.
2. Do the empirical P* values (if interior) land anywhere near the
   Stage 2 geometric-distinguishability peak (P=45 for this project's
   H=32), or somewhere unrelated to it?

Classification rule (agreed with Navid before writing this): a language
is labelled "interior minimum" if its argmin over the tested P grid is
STRICTLY inside the range (not the smallest or largest P tested);
otherwise "monotonic / boundary". This is deliberately the simplest
possible rule, appropriate given this sweep has only 1 seed per point --
a fancier statistical test would imply a confidence the data doesn't
support. A smoothed curve (quadratic fit in log10(P)) is overlaid on the
plot purely as a visual aid for judging borderline cases; it does NOT
feed into the classification itself.

As a secondary, more exploratory check, this script also scans for any
LOCAL minima strictly inside the range (points below both neighbours)
even when the GLOBAL argmin is at the boundary -- a language could show
the dominant Ceiling Confound pattern overall while still having a
smaller secondary dip near, say, the Stage 2 geometric peak. Flagged
separately and described cautiously: with 1 seed, a small local dip is
exactly the kind of thing run-to-run noise can produce, so this is
reported as "worth a second seed to check", not as a finding on its own.

Outputs
-------
- figures/sweep_test_loss_vs_P.png -- the headline plot: test loss vs P
  (log-x), one line per language, smoothed overlay per language, Stage 2's
  P=45 geometric peak marked as a reference line.
- results/sweep_summary.json -- per-language argmin P, min loss, loss at
  P=45, classification, and any local minima found -- the numbers behind
  the plot, in a form Stage 5's write-up / project memory can quote directly.
- Prints the same summary to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # headless -- no display in this environment
import matplotlib.pyplot as plt
import numpy as np

GEOMETRIC_PEAK_P = 45  # Stage 2's distinguishability-maximising P, for H=32 (D=64)
LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German", "it": "Italian", "es": "Spanish"}
# A fixed, colourblind-distinguishable palette (not just matplotlib's default
# cycle) so the same language always gets the same colour across figures.
LANGUAGE_COLORS = {
    "en": "#1b9e77", "fr": "#d95f02", "de": "#7570b3", "it": "#e7298a", "es": "#66a61e",
}


def load_sweep_by_language(results_path: Path) -> dict[str, dict[int, float]]:
    """
    Reads the sweep's JSONL results file and returns
    {language: {P: final_eval_loss_full}}. Only the fields this analysis
    actually needs are pulled out -- the full per-run records (eval
    curves, hyperparameters, timing) stay in the results file itself for
    anyone who wants to dig further.

    If more than one seed's worth of results exists for a given
    (language, P) (e.g. after a later `train_sweep.py --seeds 0,1` run),
    this averages them -- a natural, harmless generalisation for when a
    second seed gets added, even though today's sweep has exactly one.
    """
    from collections import defaultdict

    sums: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sums[record["language"]][record["P"]].append(record["final_eval_loss_full"])

    return {
        language: {P: float(np.mean(vals)) for P, vals in p_to_losses.items()}
        for language, p_to_losses in sums.items()
    }


def classify_curve(p_grid: list[int], losses: list[float]) -> dict:
    """
    Applies the agreed classification rule to one language's (P, loss)
    curve. p_grid must be sorted ascending and losses aligned to it.

    Returns a dict with: argmin_p / argmin_loss (the global minimum),
    is_interior_minimum (the primary classification), and
    local_minima -- a list of P values that are strict local minima
    (loss[i] < loss[i-1] and loss[i] < loss[i+1]) among the INTERIOR
    points only (a boundary point can't be a "local minimum" in the
    usual sense, since it only has one neighbour) -- the secondary,
    more exploratory check described in the module docstring.
    """
    n = len(p_grid)
    assert n == len(losses) and n >= 3, "need at least 3 points to talk about interior/boundary at all"

    argmin_idx = int(np.argmin(losses))
    is_interior_minimum = 0 < argmin_idx < n - 1

    local_minima = [
        p_grid[i]
        for i in range(1, n - 1)
        if losses[i] < losses[i - 1] and losses[i] < losses[i + 1]
    ]

    return {
        "argmin_p": p_grid[argmin_idx],
        "argmin_loss": losses[argmin_idx],
        "is_interior_minimum": is_interior_minimum,
        "local_minima": local_minima,
    }


def fit_smoothed_curve(p_grid: list[int], losses: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    A degree-2 polynomial fit of loss against log10(P), evaluated on a
    dense grid for plotting -- purely a visual smoothing aid (see module
    docstring: this never feeds into classify_curve's labels). log10(P)
    rather than raw P because the P grid itself spans two orders of
    magnitude (3 to 384) and is far denser at the small end -- fitting
    in log-P space avoids the large-P points dominating the fit just by
    being numerically bigger.
    """
    log_p = np.log10(p_grid)
    coeffs = np.polyfit(log_p, losses, deg=2)
    dense_log_p = np.linspace(log_p.min(), log_p.max(), 200)
    dense_loss = np.polyval(coeffs, dense_log_p)
    return 10 ** dense_log_p, dense_loss


def make_plot(by_language: dict[str, dict[int, float]], p_grid: list[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for language in sorted(by_language):
        losses = [by_language[language][p] for p in p_grid]
        color = LANGUAGE_COLORS.get(language, None)
        label = LANGUAGE_NAMES.get(language, language)

        ax.plot(p_grid, losses, marker="o", linestyle="none", color=color, label=label, zorder=3)
        dense_p, dense_loss = fit_smoothed_curve(p_grid, losses)
        ax.plot(dense_p, dense_loss, linestyle="-", color=color, alpha=0.5, linewidth=1.5, zorder=2)

    ax.axvline(GEOMETRIC_PEAK_P, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.text(
        GEOMETRIC_PEAK_P, ax.get_ylim()[1], f"  Stage 2 geometric peak (P={GEOMETRIC_PEAK_P})",
        rotation=90, va="top", ha="left", fontsize=8, color="gray",
    )

    ax.set_xscale("log")
    ax.set_xticks(p_grid)
    ax.set_xticklabels([str(p) for p in p_grid], rotation=45)
    ax.set_xlabel("Period P (log scale)")
    ax.set_ylabel("Test loss (full test set, nats/char)")
    ax.set_title("Test loss vs positional-encoding period P, by language")
    ax.legend(title="Language")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    results_path = REPO_ROOT / "results" / "sweep_results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"No sweep results found at {results_path} -- run scripts/train_sweep.py first.")

    by_language = load_sweep_by_language(results_path)
    p_grid = sorted(next(iter(by_language.values())).keys())
    for language, p_to_loss in by_language.items():
        if sorted(p_to_loss.keys()) != p_grid:
            raise SystemExit(f"language={language} has a different P grid than the others -- sweep looks incomplete.")

    summary = {}
    print(f"{'lang':<5} {'argmin P':<9} {'min loss':<9} {'loss@P=45':<10} {'classification':<28} {'local minima'}")
    for language in sorted(by_language):
        losses = [by_language[language][p] for p in p_grid]
        result = classify_curve(p_grid, losses)
        result["loss_at_geometric_peak"] = by_language[language].get(GEOMETRIC_PEAK_P)
        summary[language] = result

        label = "interior minimum" if result["is_interior_minimum"] else "monotonic / boundary"
        print(
            f"{language:<5} {result['argmin_p']:<9} {result['argmin_loss']:<9.4f} "
            f"{result['loss_at_geometric_peak']:<10.4f} {label:<28} {result['local_minima']}"
        )

    n_interior = sum(1 for r in summary.values() if r["is_interior_minimum"])
    print(
        f"\n{n_interior}/{len(summary)} languages show an interior minimum within the tested P range "
        f"(P in [{p_grid[0]}, {p_grid[-1]}])."
    )
    if n_interior == 0:
        print(
            "All languages' test loss is monotonic across the tested range, decreasing all the way to "
            f"the largest P tested (P={p_grid[-1]} = max_seq_len). This is exactly the Ceiling Confound "
            "pattern flagged in the design notes: P is behaving like it enlarges the model's effective "
            "context/positional range, not like it's revealing an intrinsic per-language optimal period. "
            "See any local_minima reported above for a weaker, single-seed-only secondary signal worth "
            "checking with more seeds before reading anything into it."
        )

    summary_path = REPO_ROOT / "results" / "sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"p_grid": p_grid, "geometric_peak_p": GEOMETRIC_PEAK_P, "languages": summary}, f, indent=2)
    print(f"\nWrote {summary_path}")

    plot_path = REPO_ROOT / "figures" / "sweep_test_loss_vs_P.png"
    make_plot(by_language, p_grid, plot_path)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
