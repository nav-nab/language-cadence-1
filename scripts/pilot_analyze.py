#!/usr/bin/env python3
"""
pilot_analyze.py
==================

Reads results/pilot_results.jsonl (written by pilot_run.py) and produces:
- figures/pilot_eval_loss_curves.png: eval loss vs training step, one
  line per (injection, amplitude) combination tested.
- figures/pilot_final_comparison.png: bar chart of best eval loss reached
  by each combination, sorted, for an at-a-glance ranking.
- A printed summary table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    results_path = REPO_ROOT / "results" / "pilot_results.jsonl"
    runs = [json.loads(line) for line in open(results_path, encoding="utf-8")]

    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # --- eval loss curves ---
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    additive_runs = [r for r in runs if r["pe_injection"] == "additive"]
    additive_runs.sort(key=lambda r: r["pe_amplitude"])
    for i, r in enumerate(additive_runs):
        color = cmap(i / max(1, len(additive_runs) - 1))
        steps = [p["step"] for p in r["eval_points"]]
        losses = [p["eval_loss"] for p in r["eval_points"]]
        ax.plot(steps, losses, color=color, label=f"additive, amp={r['pe_amplitude']}")
    for r in runs:
        if r["pe_injection"] == "concat":
            steps = [p["step"] for p in r["eval_points"]]
            losses = [p["eval_loss"] for p in r["eval_points"]]
            ax.plot(steps, losses, color="red", linewidth=2.2, linestyle="--", label="concat")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval loss (held-out English windows)")
    ax.set_title("Pilot: eval loss vs step, by Pi injection method / amplitude\n(seq_len=128, D=64, 2 layers, P=45, single seed)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "pilot_eval_loss_curves.png", dpi=150)
    plt.close(fig)

    # --- final comparison bar chart ---
    labeled = [(r["run_label"], r["best_eval_loss"]) for r in runs]
    labeled.sort(key=lambda t: t[1])
    labels = [t[0] for t in labeled]
    values = [t[1] for t in labeled]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if lbl == labeled[0][0] else "#3b6ba5" for lbl in labels]
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()  # best (smallest) at top
    ax.set_xlabel("Best eval loss reached during the pilot run")
    ax.set_title("Pilot calibration: ranked comparison")
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "pilot_final_comparison.png", dpi=150)
    plt.close(fig)

    # --- printed table ---
    print(f"{'run':<18} {'final_eval':>11} {'best_eval':>10} {'elapsed_s':>10}")
    for r in sorted(runs, key=lambda r: r["best_eval_loss"]):
        print(f"{r['run_label']:<18} {r['final_eval_loss']:>11.4f} {r['best_eval_loss']:>10.4f} {r['elapsed_seconds']:>10.1f}")

    print(f"\nWrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
