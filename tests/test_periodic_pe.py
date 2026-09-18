"""
Unit tests for the NumPy reference Π implementation (src/metronome/pe/periodic_pe.py).

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from metronome.pe.periodic_pe import harmonic_orders, periodic_pe  # noqa: E402


class TestHarmonicOrders(unittest.TestCase):
    def test_folding_example_from_docstring(self):
        # P=5: j mod P cycles 1,2,3,4,0,1,2,3,4,0 for j=1..10
        # folded min(m, 5-m): 1,2,2,1,0,1,2,2,1,0
        k = harmonic_orders(H=10, P=5)
        expected = [1, 2, 2, 1, 0, 1, 2, 2, 1, 0]
        self.assertEqual(list(k), expected)

    def test_max_folded_harmonic_is_floor_p_over_2(self):
        P = 11
        k = harmonic_orders(H=200, P=P)
        self.assertEqual(k.max(), P // 2)

    def test_rejects_nonpositive_period(self):
        with self.assertRaises(ValueError):
            harmonic_orders(H=5, P=0)


class TestPeriodicPE(unittest.TestCase):
    def test_shape(self):
        pe = periodic_pe(D=64, N=100, P=17)
        self.assertEqual(pe.shape, (64, 100))

    def test_rejects_odd_d(self):
        with self.assertRaises(ValueError):
            periodic_pe(D=63, N=10, P=5)

    def test_exact_period_property(self):
        # The defining property: Pi(n) == Pi(n + P) for every column n,
        # exactly (not just approximately close) -- this is what makes
        # this encoding different from Vaswani's geometric-frequency
        # original, which never exactly repeats.
        D, P = 64, 13
        N = 5 * P + 7  # several periods plus a partial one
        pe = periodic_pe(D=D, N=N, P=P)
        for n in range(N - P):
            np.testing.assert_allclose(
                pe[:, n], pe[:, n + P], atol=1e-12,
                err_msg=f"Pi(n={n}) != Pi(n+P={n+P})",
            )

    def test_not_periodic_at_a_smaller_period(self):
        # Sanity check on the test itself: confirm the matrix does NOT
        # (in general) also repeat at some smaller period, i.e. P is the
        # *fundamental* period, not just *a* period. Guards against a
        # degenerate implementation (e.g. an all-zero matrix) trivially
        # satisfying the period test above for the wrong reason.
        D, P = 64, 13
        N = 3 * P
        pe = periodic_pe(D=D, N=N, P=P)
        for smaller_p in range(1, P):
            if P % smaller_p == 0:
                continue  # only need to rule out divisors of P
            mismatch = not np.allclose(pe[:, 0], pe[:, smaller_p], atol=1e-12)
            self.assertTrue(mismatch, f"Unexpectedly periodic at p={smaller_p} < P={P}")

    def test_row_pairs_are_sin_cos_of_same_angle(self):
        D, N, P = 8, 20, 7
        pe = periodic_pe(D=D, N=N, P=P)
        H = D // 2
        k = harmonic_orders(H, P)
        n = np.arange(N)
        for j in range(1, H + 1):
            angle = n * 2 * np.pi * k[j - 1] / P
            np.testing.assert_allclose(pe[2 * j - 2, :], np.sin(angle), atol=1e-12)
            np.testing.assert_allclose(pe[2 * j - 1, :], np.cos(angle), atol=1e-12)

    def test_values_bounded(self):
        # sin/cos outputs, so every entry must lie in [-1, 1].
        pe = periodic_pe(D=32, N=200, P=31)
        self.assertTrue(np.all(pe >= -1.0 - 1e-12))
        self.assertTrue(np.all(pe <= 1.0 + 1e-12))


if __name__ == "__main__":
    unittest.main()
