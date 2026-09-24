"""
test_analyze_sweep.py
=======================

Stage 5 tests. Only the pure logic in scripts/analyze_sweep.py is
tested here -- classify_curve() (the interior-minimum-vs-monotonic
rule) and load_sweep_by_language()'s multi-seed averaging -- using
small synthetic (P, loss) arrays, not the real sweep results. The
plotting function (make_plot) isn't unit tested (it's a thin,
visually-inspected wrapper around matplotlib -- see figures/ for the
actual output to eyeball), consistent with how pe_distinguishability.py
(Stage 2) was verified: by reading the produced figure, not asserting
pixel content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_sweep  # noqa: E402  (scripts/ is not a package -- imported directly, added to sys.path above)


def test_monotonic_decreasing_is_not_interior_minimum():
    # Exactly the real sweep's shape: loss keeps improving all the way
    # to the largest P -- the Ceiling Confound pattern.
    p_grid = [3, 8, 32, 128, 384]
    losses = [2.3, 2.2, 2.1, 2.0, 1.8]
    result = analyze_sweep.classify_curve(p_grid, losses)
    assert result["argmin_p"] == 384
    assert result["is_interior_minimum"] is False
    assert result["local_minima"] == []


def test_true_interior_minimum_is_detected():
    p_grid = [3, 8, 32, 128, 384]
    losses = [2.3, 2.1, 1.8, 2.0, 2.2]  # dips at P=32, rises again
    result = analyze_sweep.classify_curve(p_grid, losses)
    assert result["argmin_p"] == 32
    assert result["is_interior_minimum"] is True
    assert result["local_minima"] == [32]


def test_minimum_at_smallest_p_is_boundary_not_interior():
    p_grid = [3, 8, 32, 128, 384]
    losses = [1.8, 2.0, 2.1, 2.2, 2.3]  # monotonic INCREASING -- argmin at the left edge
    result = analyze_sweep.classify_curve(p_grid, losses)
    assert result["argmin_p"] == 3
    assert result["is_interior_minimum"] is False


def test_local_minimum_detected_even_when_global_argmin_is_at_boundary():
    # This is the real sweep's more interesting secondary pattern: a
    # dip near one interior point even though the curve overall still
    # bottoms out at the boundary.
    p_grid = [3, 8, 32, 45, 64, 384]
    losses = [2.3, 2.2, 2.15, 2.18, 2.10, 1.8]  # small local bump AT P=45 (worse than both neighbours)
    result = analyze_sweep.classify_curve(p_grid, losses)
    assert result["is_interior_minimum"] is False  # global min still at the boundary (P=384)
    assert result["argmin_p"] == 384
    # P=32 is a local minimum (below both P=8 and P=45's neighbours)
    assert 32 in result["local_minima"]


def test_rejects_too_few_points():
    import pytest
    with pytest.raises(AssertionError):
        analyze_sweep.classify_curve([3, 8], [2.0, 1.9])


def test_load_sweep_by_language_averages_multiple_seeds(tmp_path):
    results_path = tmp_path / "sweep_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"language": "en", "P": 45, "seed": 0, "final_eval_loss_full": 2.0}) + "\n")
        f.write(json.dumps({"language": "en", "P": 45, "seed": 1, "final_eval_loss_full": 2.2}) + "\n")
        f.write(json.dumps({"language": "en", "P": 64, "seed": 0, "final_eval_loss_full": 1.9}) + "\n")

    by_language = analyze_sweep.load_sweep_by_language(results_path)
    assert by_language["en"][45] == 2.1  # mean of 2.0 and 2.2
    assert by_language["en"][64] == 1.9  # single seed, unchanged


def test_load_sweep_by_language_single_seed_passthrough(tmp_path):
    results_path = tmp_path / "sweep_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"language": "fr", "P": 8, "seed": 0, "final_eval_loss_full": 2.25}) + "\n")

    by_language = analyze_sweep.load_sweep_by_language(results_path)
    assert by_language == {"fr": {8: 2.25}}
