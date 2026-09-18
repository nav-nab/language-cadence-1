#!/usr/bin/env python3
"""
train_sweep.py
===============

Stage 4: the CLI harness that runs the REAL cadence-hypothesis sweep --
training one GPT per (language, P, seed) combination and recording its
test loss -- across the full grid agreed with the user. Unlike
pilot_run.py (Stage 3), which ran a handful of short, deliberately rough
comparisons to settle the pe_injection/pe_amplitude question, every run
this script makes goes directly into the Stage 5 analysis.

Design agreed with the user before writing this file
------------------------------------------------------
- P grid (14 values): 3, 4, 6, 8, 16, 24, 32, 45, 64, 96, 128, 192, 256,
  384. Spans well below the Stage 2 geometric-distinguishability peak
  (P=45) with a few very small periods, a reasonably even spread above
  it, and P=384=max_seq_len exactly (Pi's period equal to the whole
  context window -- the key "Ceiling Confound" test point from the
  design notes: periodicity becomes structurally unobservable within a
  single window once P >= N).
- Languages: en, fr, de, it, es (all five processed in Stage 1).
- ASSUMPTION (the user asked me to pick a reasonable value and say so):
  seeds=1, steps=3000, batch_size=16 -- see the long comment further
  down headed "Why these numbers" for the full reasoning, which
  includes a real benchmark run in this environment, not just a guess.
- Resumability: append-only results file (default
  results/sweep_results.jsonl) plus skip-if-done -- before starting any
  (language, P, seed) run, the harness checks whether a result for that
  exact combination AND the same fixed hyperparameters already exists
  in the file, and skips it if so. This means the whole sweep can be
  interrupted (Ctrl+C, laptop sleep, closing the terminal) and resumed
  by simply re-running the same command -- exactly the pattern the user
  chose. No mid-run checkpointing: each individual run is short enough
  (well under the file-system-level risk of losing hours of progress)
  that redoing one interrupted run from step 0 is an acceptable cost.

Why "same fixed hyperparameters" matters for skip-if-done: it's not
just (language, P, seed) that has to match a prior result -- if you
later change, say, steps or n_layers and re-run, those new results must
NOT be silently skipped as if they were the old configuration's results
in disguise. _config_id() below hashes every OTHER fixed hyperparameter
(everything except language/P/seed, which are the three axes actually
being swept) into a short fingerprint that's stored with is compared
against every result, so a hyperparameter change is automatically
treated as "not done yet" rather than requiring you to remember to
clear out the results file yourself.

Why these numbers (steps / seeds / batch size)
-----------------------------------------------
The three P-grid/seed/step questions were put to the user directly; the
seed count and step budget question went unanswered, with instructions
to make a reasonable assumption and explain it. Before picking numbers,
I benchmarked one real training step at the sweep's actual architecture
(D=64, 4 layers, 4 heads, seq_len=384 -- the full ModelConfig defaults,
NOT the pilot's shrunk-down 2-layer/seq_len=128 config) on data/processed
in this sandboxed environment: ~1.2 seconds/step on this environment's
4 CPU cores. At the originally-recommended 6000 steps x 2 seeds, the
full 14-P x 5-language grid would take on the order of 280+ hours of
continuous compute -- clearly not what "a few sittings" was meant to
cover, so that combination was not a reasonable assumption to carry
forward silently.

Instead: steps=3000, batch_size=16 means each run sees
3000 * 16 = 48,000 windows of 384 characters = ~18.4M characters --
almost exactly ONE PASS over a language's ~17-18M-character training
set (see data/processed/manifest.json), which is both a clean,
easy-to-justify budget ("one epoch") and, per the benchmark above,
roughly 3000 * 1.2s ~= 60 minutes per run on hardware like this
sandbox's. seeds=1 (rather than 2) keeps the full grid at
14 * 5 * 1 = 70 runs (~70 hours on this sandbox's hardware -- likely
less on a real Mac with more CPU cores; --dry-run and the per-step
timing this script prints will tell you your actual rate within
minutes of starting). Because the results file is append-only and
skip-if-done keys on seed too, adding a second seed later (to check
run-to-run noise) is just `--seeds 0,1` in a later invocation -- it
will skip everything seed=0 already covers and only run the new seed=1
combinations, so nothing about this assumption is a one-way door.

Usage
-----
    # See the plan without running anything:
    python3 scripts/train_sweep.py --dry-run

    # Run the whole grid (resumable -- Ctrl+C any time, re-run to continue):
    python3 scripts/train_sweep.py

    # Run just one language, to spread the work across sittings:
    python3 scripts/train_sweep.py --languages en

    # Add a second seed later, after the first pass is done:
    python3 scripts/train_sweep.py --seeds 0,1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from metronome.model.config import ModelConfig  # noqa: E402
from metronome.train.loop import run_training  # noqa: E402

DEFAULT_LANGUAGES = ["en", "fr", "de", "it", "es"]
DEFAULT_P_GRID = [3, 4, 6, 8, 16, 24, 32, 45, 64, 96, 128, 192, 256, 384]
DEFAULT_SEEDS = [0]

# Fixed across the whole sweep (per the project notes: only P, language,
# and seed vary between runs) -- these match ModelConfig's own defaults,
# repeated explicitly here (rather than relying on ModelConfig's
# defaults silently) so that a future change to ModelConfig's defaults
# can't silently change an in-progress or already-partly-run sweep out
# from under you.
FIXED_HYPERPARAMETERS = dict(
    D=64,
    n_layers=4,
    n_heads=4,
    ffn_mult=4,
    dropout=0.1,
    pe_injection="concat",
    pe_amplitude=0.1,
    steps=3000,
    batch_size=16,
    lr=3e-4,
    weight_decay=0.01,
    eval_every=500,
    eval_subset_batches=32,
)


def _config_id(fixed: dict) -> str:
    """
    A short, stable fingerprint of every fixed (non-swept) hyperparameter,
    stored with each result and checked by skip-if-done. Sorting keys
    before hashing makes this independent of dict ordering; truncating
    to 12 hex characters keeps results files readable while still being
    astronomically unlikely to collide by accident for the handful of
    distinct configurations this project will ever actually run.
    """
    blob = json.dumps(fixed, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _load_language_data(language: str):
    data_dir = REPO_ROOT / "data" / "processed" / language
    train_windows = np.load(data_dir / "train_windows.npy", mmap_mode="r")
    test_windows = np.load(data_dir / "test_windows.npy", mmap_mode="r")
    return train_windows, test_windows


def _load_existing_keys(results_path: Path) -> set[tuple]:
    """
    Reads every already-recorded result and returns the set of
    (language, P, seed, config_id) keys it covers -- run_sweep() skips
    any (language, P, seed, config_id) combination already in this set.
    A missing or empty results file just means nothing is done yet.
    """
    keys: set[tuple] = set()
    if not results_path.exists():
        return keys
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            keys.add((record["language"], record["P"], record["seed"], record["config_id"]))
    return keys


def run_sweep(
    languages: list[str],
    p_grid: list[int],
    seeds: list[int],
    results_path: Path,
    dry_run: bool = False,
    hyperparameters: dict | None = None,
) -> None:
    # `hyperparameters` defaults to the agreed FIXED_HYPERPARAMETERS, but
    # can be overridden (via the --steps/--batch-size/etc CLI flags below)
    # for a fast sanity-check run before committing to the real sweep --
    # this is exactly how this script's own end-to-end behaviour was
    # verified in the sandboxed environment before being handed off.
    # Overriding these does NOT contaminate the real sweep's results:
    # _config_id() fingerprints whatever hyperparameters were actually
    # used, so a smoke-test run's results simply live under a different
    # config_id and are never mistaken for (or skipped in favour of) the
    # real sweep's.
    active_hyperparameters = dict(FIXED_HYPERPARAMETERS)
    if hyperparameters:
        active_hyperparameters.update(hyperparameters)

    config_id = _config_id(active_hyperparameters)
    existing = _load_existing_keys(results_path)

    plan = [
        (language, P, seed)
        for language in languages
        for P in p_grid
        for seed in seeds
    ]
    todo = [
        (language, P, seed)
        for (language, P, seed) in plan
        if (language, P, seed, config_id) not in existing
    ]

    print(
        f"config_id={config_id}  plan={len(plan)} runs total, "
        f"{len(plan) - len(todo)} already done, {len(todo)} remaining."
    )
    if dry_run:
        print("--dry-run: not training anything. Remaining runs:")
        for language, P, seed in todo:
            print(f"  language={language} P={P} seed={seed}")
        return

    vocab = json.load(open(REPO_ROOT / "data" / "processed" / "vocab.json"))
    vocab_size = len(vocab["id_to_char"])
    manifest = json.load(open(REPO_ROOT / "data" / "processed" / "manifest.json"))
    max_seq_len = manifest["window_length"]

    hp = active_hyperparameters  # short alias, used throughout the loop below

    # Cache each language's arrays across runs within this process --
    # avoids re-loading (memory-mapping) the same .npy files once per
    # (P, seed) combination when several runs for the same language are
    # done back to back, which is the common case since `plan` iterates
    # P and seed inside the language loop.
    data_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    results_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_t0 = time.time()

    for i, (language, P, seed) in enumerate(todo, start=1):
        if language not in data_cache:
            data_cache[language] = _load_language_data(language)
        train_windows, test_windows = data_cache[language]

        cfg = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            D=hp["D"],
            n_layers=hp["n_layers"],
            n_heads=hp["n_heads"],
            ffn_mult=hp["ffn_mult"],
            dropout=hp["dropout"],
            P=P,
            pe_injection=hp["pe_injection"],
            pe_amplitude=hp["pe_amplitude"],
        )

        run_label = f"[{i}/{len(todo)}] language={language} P={P} seed={seed}"
        print(f"{run_label}  starting...")
        run_t0 = time.time()

        result = run_training(
            cfg,
            train_windows,
            test_windows,
            steps=hp["steps"],
            batch_size=hp["batch_size"],
            seed=seed,
            lr=hp["lr"],
            weight_decay=hp["weight_decay"],
            eval_every=hp["eval_every"],
            eval_subset_batches=hp["eval_subset_batches"],
            log_prefix=f"{run_label}  ",
            log_every=0,  # keep the harness's own per-run line as the only output; set >0 for verbose per-step logging
        )
        result["language"] = language
        result["config_id"] = config_id

        # Append immediately after each run finishes (not batched up and
        # written at the end) -- this is what makes the harness safe to
        # interrupt at any point: whatever has already been appended is
        # never lost, and skip-if-done means re-running just continues.
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        run_elapsed = time.time() - run_t0
        sweep_elapsed = time.time() - sweep_t0
        mean_run_time = sweep_elapsed / i
        remaining = len(todo) - i
        eta_hours = mean_run_time * remaining / 3600
        print(
            f"{run_label}  DONE in {run_elapsed:.0f}s  "
            f"final_eval_loss_full={result['final_eval_loss_full']:.4f}  "
            f"({remaining} runs left, ETA ~{eta_hours:.1f}h at current pace)"
        )


def _parse_int_list(s: str) -> list[int]:
    return [int(part.strip()) for part in s.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                         default=DEFAULT_LANGUAGES, help="comma-separated language codes")
    parser.add_argument("--p-grid", type=_parse_int_list, default=DEFAULT_P_GRID,
                         help="comma-separated list of P values")
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_SEEDS,
                         help="comma-separated list of seeds")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "sweep_results.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="print the run plan and exit without training")
    # Optional overrides of the fixed hyperparameters -- mainly for a
    # fast sanity-check run (e.g. --steps 20) before committing to the
    # real, full-length sweep. Left unset, every one of these falls back
    # to FIXED_HYPERPARAMETERS above, i.e. the agreed real-sweep values.
    parser.add_argument("--steps", type=int, default=None, help="override: optimiser steps per run")
    parser.add_argument("--batch-size", type=int, default=None, help="override: batch size")
    parser.add_argument("--eval-every", type=int, default=None, help="override: steps between subset evals")
    parser.add_argument("--eval-subset-batches", type=int, default=None, help="override: batches per subset eval")
    args = parser.parse_args()

    overrides = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.eval_every is not None:
        overrides["eval_every"] = args.eval_every
    if args.eval_subset_batches is not None:
        overrides["eval_subset_batches"] = args.eval_subset_batches

    run_sweep(
        languages=args.languages,
        p_grid=args.p_grid,
        seeds=args.seeds,
        results_path=args.results,
        dry_run=args.dry_run,
        hyperparameters=overrides or None,
    )


if __name__ == "__main__":
    main()
