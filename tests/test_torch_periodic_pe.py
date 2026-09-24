"""
Unit tests for the PyTorch Π module (src/language_cadence/pe/torch_periodic_pe.py).
Requires torch to be installed -- see README.md for environment notes.

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from language_cadence.pe.periodic_pe import periodic_pe  # noqa: E402
from language_cadence.pe.torch_periodic_pe import PeriodicPositionalEncoding  # noqa: E402


class TestTorchPeriodicPE(unittest.TestCase):
    def test_agrees_exactly_with_numpy_reference(self):
        # The most important test in this file: the torch module and the
        # numpy reference implementation must produce numerically
        # identical output for the same (D, N, P). If they ever diverge,
        # that's a real bug -- these two are supposed to be the same
        # math, just in two different tensor libraries.
        D, N, P = 64, 200, 23
        np_pe = periodic_pe(D=D, N=N, P=P)  # shape (D, N)

        module = PeriodicPositionalEncoding(D=D, P=P, max_seq_len=N)
        torch_pe = module(seq_len=N)  # shape (N, D)

        # np_pe is (D, N); torch_pe is (N, D) -- transpose to compare.
        np.testing.assert_allclose(
            torch_pe.detach().numpy(), np_pe.T, atol=1e-5,
            err_msg="torch Pi disagrees with numpy reference Pi",
        )

    def test_output_shape(self):
        module = PeriodicPositionalEncoding(D=32, P=11, max_seq_len=100)
        out = module(seq_len=50)
        self.assertEqual(out.shape, (50, 32))

    def test_exact_period_property_in_torch(self):
        D, P = 64, 13
        N = 5 * P + 7
        module = PeriodicPositionalEncoding(D=D, P=P, max_seq_len=N)
        pe = module(seq_len=N)  # (N, D)
        for n in range(N - P):
            torch.testing.assert_close(pe[n], pe[n + P], atol=1e-5, rtol=0)

    def test_rejects_seq_len_beyond_max(self):
        module = PeriodicPositionalEncoding(D=16, P=5, max_seq_len=10)
        with self.assertRaises(ValueError):
            module(seq_len=11)

    def test_broadcasts_onto_batched_activations(self):
        # The actual intended usage in the model: adding Pi onto a
        # (batch, seq_len, D) tensor via ordinary broadcasting.
        D, P, N = 32, 9, 40
        module = PeriodicPositionalEncoding(D=D, P=P, max_seq_len=N)
        pe = module(seq_len=N)  # (N, D)

        batch_size = 4
        x = torch.zeros(batch_size, N, D)
        result = x + pe  # relies on broadcasting (N, D) -> (batch, N, D)
        self.assertEqual(result.shape, (batch_size, N, D))
        # Every item in the batch should have received the identical
        # positional encoding (since x was all zeros here).
        for b in range(batch_size):
            torch.testing.assert_close(result[b], pe)

    def test_buffer_is_not_a_trainable_parameter(self):
        # Pi has no learnable weights -- confirms it was registered as a
        # buffer (module.buffers()), not a Parameter (module.parameters()),
        # so the optimizer will never try to update it during training.
        module = PeriodicPositionalEncoding(D=16, P=5, max_seq_len=20)
        self.assertEqual(list(module.parameters()), [])
        buffer_names = [name for name, _ in module.named_buffers()]
        self.assertIn("pe", buffer_names)

    def test_rejects_odd_d(self):
        with self.assertRaises(ValueError):
            PeriodicPositionalEncoding(D=15, P=5, max_seq_len=10)

    def test_moves_with_module_to(self):
        # Buffers should follow .to(device)/.to(dtype) calls like any
        # other module state -- sanity check with a dtype change (device
        # change would need a GPU, not available in this environment).
        module = PeriodicPositionalEncoding(D=16, P=5, max_seq_len=20)
        module_double = module.to(torch.float64)
        self.assertEqual(module_double.pe.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
