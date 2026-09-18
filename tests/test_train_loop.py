"""
test_train_loop.py
====================

Stage 4 tests. Two things are exercised here, deliberately kept
separate:

1. train.loop.run_training() -- the actual training/eval logic --
   against tiny SYNTHETIC windows (random token ids, a handful of
   steps, a tiny model). This never touches the real Europarl-derived
   data on disk, so these tests run in a couple of seconds and would
   pass even in a fresh checkout that hasn't run scripts/build_dataset.py
   yet -- matching how tests/test_model.py already tests the GPT model
   itself without any real data.

2. scripts/train_sweep.py's bookkeeping helpers (_config_id,
   _load_existing_keys) -- the skip-if-done resumability logic -- using
   temporary files, again with no dependency on real data or on
   actually training anything.

Together these are the "verify the harness with a tiny/fast run" step
agreed with the user before handing the real sweep off to them to run
on their own machine: correctness of the mechanics is checked here;
the real sweep's numbers are what scripts/train_sweep.py itself
produces when the user runs it for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from metronome.model.config import ModelConfig  # noqa: E402
from metronome.train.loop import run_training  # noqa: E402

import train_sweep  # noqa: E402  (scripts/ is not a package -- imported directly, added to sys.path above)


# --- Tiny synthetic data shared by several tests below -----------------

VOCAB_SIZE = 8
MAX_SEQ_LEN = 16   # small "context length" -- must be >= the P values used below
WINDOW_LEN = MAX_SEQ_LEN + 1  # matches the real data's (num_windows, window_length + 1) shape


def _make_synthetic_windows(num_windows: int, rng_seed: int) -> np.ndarray:
    """
    A stand-in for a real .npy window file: uniform-random token ids in
    [0, VOCAB_SIZE), same shape/dtype convention (uint8) as the real
    Stage 1 output. The model doesn't need the data to be linguistically
    real to sanity-check that training/eval mechanically works.
    """
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, VOCAB_SIZE, size=(num_windows, WINDOW_LEN), dtype=np.uint8)


def _tiny_config(P: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        D=8,          # tiny -- H = D/2 = 4, plenty for P up to MAX_SEQ_LEN in these tests
        n_layers=1,
        n_heads=2,
        ffn_mult=2,
        dropout=0.0,  # deterministic (no dropout noise) -- makes the reproducibility test exact
        P=P,
        pe_injection="concat",
        pe_amplitude=0.1,
    )


# --- run_training() ------------------------------------------------------


def test_run_training_returns_well_formed_result():
    train_windows = _make_synthetic_windows(40, rng_seed=1)
    test_windows = _make_synthetic_windows(12, rng_seed=2)
    cfg = _tiny_config(P=5)

    result = run_training(
        cfg, train_windows, test_windows,
        steps=6, batch_size=4, seed=0,
        eval_every=2, eval_subset_batches=2,
    )

    # Structural checks -- these are the fields scripts/train_sweep.py
    # and the eventual Stage 5 analysis rely on being present.
    for key in ("seed", "P", "vocab_size", "hyperparameters", "param_count",
                "elapsed_seconds", "eval_points", "final_train_loss",
                "final_eval_loss_subset", "final_eval_loss_full"):
        assert key in result, f"missing key: {key}"

    assert result["P"] == 5
    assert result["seed"] == 0
    assert result["param_count"] > 0
    # eval_every=2 over 6 steps -> eval points at steps 2, 4, 6
    assert [p["step"] for p in result["eval_points"]] == [2, 4, 6]
    # cross-entropy over a VOCAB_SIZE=8 vocab starts around ln(8) ~= 2.08
    # for an untrained model and should be a finite, sane float throughout.
    assert np.isfinite(result["final_eval_loss_full"])
    assert 0.0 < result["final_eval_loss_full"] < 20.0


def test_run_training_is_reproducible_given_same_seed():
    train_windows = _make_synthetic_windows(40, rng_seed=1)
    test_windows = _make_synthetic_windows(12, rng_seed=2)
    cfg = _tiny_config(P=5)

    kwargs = dict(steps=6, batch_size=4, seed=42, eval_every=3, eval_subset_batches=2)
    result_a = run_training(cfg, train_windows, test_windows, **kwargs)
    result_b = run_training(cfg, train_windows, test_windows, **kwargs)

    # Same seed controls both weight init and data shuffling order (see
    # loop.py's docstring), so two runs with identical arguments must
    # produce numerically identical loss trajectories.
    assert result_a["eval_points"] == result_b["eval_points"]
    assert result_a["final_eval_loss_full"] == result_b["final_eval_loss_full"]


def test_run_training_different_seeds_diverge():
    train_windows = _make_synthetic_windows(40, rng_seed=1)
    test_windows = _make_synthetic_windows(12, rng_seed=2)
    cfg = _tiny_config(P=5)

    kwargs = dict(steps=6, batch_size=4, eval_every=6, eval_subset_batches=2)
    result_a = run_training(cfg, train_windows, test_windows, seed=1, **kwargs)
    result_b = run_training(cfg, train_windows, test_windows, seed=2, **kwargs)

    # Different initial weights + different data order should (with
    # overwhelming probability, for two arbitrary seeds) not land on the
    # exact same eval loss -- this is really a check that `seed` is
    # actually wired through to something that matters, not a check of
    # any particular value.
    assert result_a["final_eval_loss_full"] != result_b["final_eval_loss_full"]


def test_run_training_handles_steps_beyond_one_epoch():
    # 8 train windows at batch_size=4 -> 2 batches/epoch. Requesting 10
    # steps forces the loop past one epoch boundary, exercising
    # _infinite_batches()'s re-shuffle-on-exhaustion path rather than
    # raising StopIteration.
    train_windows = _make_synthetic_windows(8, rng_seed=3)
    test_windows = _make_synthetic_windows(8, rng_seed=4)
    cfg = _tiny_config(P=3)

    result = run_training(
        cfg, train_windows, test_windows,
        steps=10, batch_size=4, seed=0, eval_every=10, eval_subset_batches=2,
    )
    assert result["eval_points"][-1]["step"] == 10


# --- scripts/train_sweep.py bookkeeping ------------------------------------


def test_config_id_is_stable_and_order_independent():
    a = train_sweep._config_id({"x": 1, "y": 2})
    b = train_sweep._config_id({"y": 2, "x": 1})  # same content, different key order
    assert a == b


def test_config_id_changes_when_hyperparameters_change():
    a = train_sweep._config_id({"steps": 3000})
    b = train_sweep._config_id({"steps": 3001})
    assert a != b


def test_load_existing_keys_and_skip_logic(tmp_path):
    results_path = tmp_path / "sweep_results.jsonl"
    config_id = train_sweep._config_id(train_sweep.FIXED_HYPERPARAMETERS)

    # Simulate two already-completed runs.
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"language": "en", "P": 45, "seed": 0, "config_id": config_id,
                             "final_eval_loss_full": 2.1}) + "\n")
        f.write(json.dumps({"language": "fr", "P": 45, "seed": 0, "config_id": config_id,
                             "final_eval_loss_full": 2.3}) + "\n")

    existing = train_sweep._load_existing_keys(results_path)
    assert ("en", 45, 0, config_id) in existing
    assert ("fr", 45, 0, config_id) in existing
    assert ("de", 45, 0, config_id) not in existing
    # A different config_id (e.g. after a hyperparameter change) must
    # NOT be treated as already done, even for the same language/P/seed.
    assert ("en", 45, 0, "some_other_config_id") not in existing


def test_load_existing_keys_missing_file_returns_empty(tmp_path):
    assert train_sweep._load_existing_keys(tmp_path / "does_not_exist.jsonl") == set()


def test_run_sweep_dry_run_reports_plan_without_training(tmp_path, capsys):
    results_path = tmp_path / "sweep_results.jsonl"
    train_sweep.run_sweep(
        languages=["en", "fr"], p_grid=[8, 16], seeds=[0],
        results_path=results_path, dry_run=True,
    )
    captured = capsys.readouterr()
    assert "4 runs total" in captured.out
    assert "4 remaining" in captured.out
    # dry-run must not create/write the results file.
    assert not results_path.exists()
