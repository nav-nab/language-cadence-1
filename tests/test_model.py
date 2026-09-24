"""
Unit tests for the Stage 3 GPT model (src/language_cadence/model/).
Requires torch -- see docs/DESIGN_LOG.md for environment notes.

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from language_cadence.model.config import ModelConfig  # noqa: E402
from language_cadence.model.gpt import GPT  # noqa: E402


def tiny_config(**overrides) -> ModelConfig:
    """A deliberately tiny config for fast tests -- correctness, not scale."""
    defaults = dict(
        vocab_size=20, max_seq_len=32, D=16, n_layers=2, n_heads=2,
        ffn_mult=4, dropout=0.0, P=7, pe_injection="additive", pe_amplitude=1.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestModelConfig(unittest.TestCase):
    def test_h_derived_from_d(self):
        cfg = tiny_config(D=64)
        self.assertEqual(cfg.H, 32)

    def test_rejects_odd_d(self):
        with self.assertRaises(ValueError):
            tiny_config(D=15)

    def test_rejects_d_not_divisible_by_heads(self):
        with self.assertRaises(ValueError):
            tiny_config(D=16, n_heads=3)

    def test_rejects_bad_injection_method(self):
        with self.assertRaises(ValueError):
            tiny_config(pe_injection="banana")


class TestGPTShapes(unittest.TestCase):
    def test_output_shape_additive(self):
        cfg = tiny_config(pe_injection="additive")
        model = GPT(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (3, 10))
        logits = model(token_ids)
        self.assertEqual(logits.shape, (3, 10, cfg.vocab_size))

    def test_output_shape_concat(self):
        cfg = tiny_config(pe_injection="concat")
        model = GPT(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (3, 10))
        logits = model(token_ids)
        self.assertEqual(logits.shape, (3, 10, cfg.vocab_size))

    def test_rejects_seq_len_beyond_max(self):
        cfg = tiny_config(max_seq_len=10)
        model = GPT(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (1, 11))
        with self.assertRaises(ValueError):
            model(token_ids)


class TestCausality(unittest.TestCase):
    """
    The most important correctness property of the whole model: changing
    a FUTURE token must not change the prediction at an EARLIER position.
    We don't just trust that is_causal=True was passed correctly --
    we verify the actual numerical behaviour.
    """

    def test_future_tokens_do_not_affect_earlier_logits(self):
        torch.manual_seed(0)
        cfg = tiny_config(dropout=0.0)  # dropout off -> deterministic comparison
        model = GPT(cfg)
        model.eval()  # also disables dropout, belt-and-braces with dropout=0.0 above

        seq_len = 12
        token_ids = torch.randint(0, cfg.vocab_size, (2, seq_len))

        with torch.no_grad():
            logits_original = model(token_ids)

            # Change only the LAST token and re-run.
            token_ids_modified = token_ids.clone()
            token_ids_modified[:, -1] = (token_ids_modified[:, -1] + 1) % cfg.vocab_size
            logits_modified = model(token_ids_modified)

        # Every position EXCEPT the last must be byte-for-byte identical,
        # since none of them could have "seen" the changed last token.
        torch.testing.assert_close(
            logits_original[:, :-1, :], logits_modified[:, :-1, :],
            msg="changing the last token changed an earlier position's logits -- causal mask is leaking",
        )
        # The last position's logits SHOULD differ (sanity check that the
        # test itself is meaningful -- if this failed, the model might
        # just be ignoring input entirely).
        self.assertFalse(
            torch.allclose(logits_original[:, -1, :], logits_modified[:, -1, :]),
            "last-position logits identical after changing the last token -- suspicious",
        )


class TestGradientFlow(unittest.TestCase):
    def _check_grad_flow(self, pe_injection: str):
        cfg = tiny_config(pe_injection=pe_injection)
        model = GPT(cfg)
        token_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        targets = torch.randint(0, cfg.vocab_size, (2, 8))

        logits = model(token_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
        )
        loss.backward()

        # Every learnable parameter should have received a gradient --
        # a None or all-zero grad on some parameter would indicate a
        # disconnected/dead part of the computation graph.
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} received no gradient")
            self.assertTrue(
                torch.any(param.grad != 0), f"{name} received an all-zero gradient"
            )

    def test_gradient_flow_additive(self):
        self._check_grad_flow("additive")

    def test_gradient_flow_concat(self):
        self._check_grad_flow("concat")


class TestParameterCount(unittest.TestCase):
    def test_weight_tying_shares_the_same_tensor(self):
        cfg = tiny_config(D=64, vocab_size=100, n_layers=2, n_heads=4)
        model = GPT(cfg)
        # token_embedding and lm_head must be the literal same tensor object
        # (not just equal in value) -- this is what "weight tying" means.
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

        # NOTE: PyTorch's .parameters() (and named_parameters(), by
        # default) already de-duplicates shared tensors under the hood --
        # a tied weight is NOT double-counted by a plain
        # `sum(p.numel() for p in model.parameters())`. So
        # num_parameters()'s exclude_embedding_duplication logic is
        # actually redundant with that built-in behaviour; we just check
        # here that both ways of counting agree, rather than expecting
        # them to differ.
        naive_total = sum(p.numel() for p in model.parameters())
        corrected_total = model.num_parameters(exclude_embedding_duplication=True)
        self.assertEqual(naive_total, corrected_total)

    def test_laptop_scale_param_count(self):
        # Sanity range for a realistic (non-tiny-test) laptop-scale
        # configuration within the project notes' D=64-128, 2-4 layer
        # guidance -- broad bounds, not a precise target: the notes'
        # "a few million parameters" guidance and "D 64-128 / 2-4 layers"
        # guidance are in mild tension for a small vocab (64 symbols), so
        # this just checks the model lands at a sane small-model scale
        # (neither degenerately tiny nor accidentally huge), not an exact
        # parameter budget.
        cfg = ModelConfig(
            vocab_size=64, max_seq_len=384, D=128, n_layers=4, n_heads=4, ffn_mult=4,
        )
        model = GPT(cfg)
        n_params = model.num_parameters()
        self.assertGreater(n_params, 100_000)
        self.assertLess(n_params, 10_000_000)


class TestPEConfigIsRespected(unittest.TestCase):
    def test_changing_p_changes_output(self):
        # Two models identical except for P should produce different
        # outputs -- confirms P is actually wired through to Pi and used,
        # not silently ignored somewhere.
        torch.manual_seed(0)
        cfg_a = tiny_config(P=5)
        model_a = GPT(cfg_a)
        torch.manual_seed(0)
        cfg_b = tiny_config(P=11)
        model_b = GPT(cfg_b)

        # Same seed -> same initial weights, so any output difference is
        # attributable to P (via Pi), not random initialisation.
        token_ids = torch.randint(0, cfg_a.vocab_size, (1, 16))
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            out_a = model_a(token_ids)
            out_b = model_b(token_ids)
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_changing_amplitude_changes_output(self):
        torch.manual_seed(0)
        cfg_a = tiny_config(pe_injection="additive", pe_amplitude=1.0)
        model_a = GPT(cfg_a)
        torch.manual_seed(0)
        cfg_b = tiny_config(pe_injection="additive", pe_amplitude=5.0)
        model_b = GPT(cfg_b)

        token_ids = torch.randint(0, cfg_a.vocab_size, (1, 16))
        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            out_a = model_a(token_ids)
            out_b = model_b(token_ids)
        self.assertFalse(torch.allclose(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
