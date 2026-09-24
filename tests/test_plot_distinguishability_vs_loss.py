"""
test_plot_distinguishability_vs_loss.py
=========================================

Tests for the small pure-logic pieces of
scripts/plot_distinguishability_vs_loss.py -- mean_loss_per_p()'s
cross-language averaging, and a golden-value cross-check that this
script's imported `distinguishability()` is genuinely the same Stage 2
metric (not a reimplementation that could silently drift), using the
real numbers computed while building this figure (D=64, at P=45 and
P=384 -- see the module docstring / Stage 5 write-up for what these
values mean).

The figure itself (make via `main()`) isn't unit tested here, same
rationale as analyze_sweep.py and pe_distinguishability.py: it's a
thin, visually-inspected matplotlib wrapper -- see figures/ for the
actual output to eyeball.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_distinguishability_vs_loss as pdvl  # noqa: E402


def test_mean_loss_per_p_averages_across_languages():
    by_language = {
        "en": {45: 2.0, 384: 1.0},
        "fr": {45: 3.0, 384: 2.0},
        "de": {45: 4.0, 384: 3.0},
    }
    result = pdvl.mean_loss_per_p(by_language, [45, 384])
    assert result == [3.0, 2.0]  # mean(2,3,4)=3.0, mean(1,2,3)=2.0


def test_mean_loss_per_p_single_language_passthrough():
    by_language = {"en": {8: 2.25, 16: 2.10}}
    assert pdvl.mean_loss_per_p(by_language, [8, 16]) == [2.25, 2.10]


def test_distinguishability_matches_known_sweep_values():
    # Golden values from the real sweep's D=64 grid (computed and
    # cross-checked when this figure was first built) -- guards against
    # this script's import of pe_distinguishability.distinguishability()
    # silently picking up a different function, or that function's
    # formula changing under it without anyone noticing here.
    assert abs(pdvl.distinguishability(D=64, P=45) - 8.895) < 0.01
    assert abs(pdvl.distinguishability(D=64, P=384) - 1.738) < 0.01
    # The headline claim: distinguishability at the geometric peak (P=45)
    # is well above distinguishability at P=384 -- roughly 5x, per the
    # write-up.
    assert pdvl.distinguishability(D=64, P=45) > 5 * pdvl.distinguishability(D=64, P=384)
