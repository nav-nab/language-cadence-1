#!/usr/bin/env python3
"""
pilot_run.py
=============

Stage 3/4 calibration pilot: a handful of SHORT training runs to pick a
defensible, fixed (pe_injection, pe_amplitude) combination for the real
Stage 4 sweep -- NOT the real experiment itself. Deliberately small
(seq_len=128, D=64, 2 layers, 2000 steps) purely for speed; conclusions
here are about which combination trains well at all in this rough
setting, not final numbers.

Design, to keep the comparison fair:
- Single language (English), single P (45, the geometric-distinguishability
  optimum from Stage 2) -- holds everything except injection/amplitude fixed.
- Same model seed and same data-order seed across all runs, so any
  difference in the loss curves is attributable to the injection method /
  amplitude, not to lucky initialisation or a lucky batch order.
- Real train/test split from Stage 1 (not the overfit-a-single-batch
  sanity check from Stage 3) -- eval loss is measured on genuinely held
  out windows, so this reflects generalisation, which is what we
  actually care about picking a good injection setup for.
- Each run appends one JSON line to results/pilot_results.jsonl, so runs
  can be done one at a time across several tool calls (each run only
  takes ~1-2 minutes, but doing all of them together would not fit in a
  single command's time limit) without losing earlier results.

Usage:
    python3 scripts/pilot_run.py --injection additive --amplitude 1.0
    python3 scripts/pilot_run.py --injection concat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from metronome.model.config import ModelConfig  # noqa: E402
from metronome.model.gpt import GPT  # noqa: E402

SEQ_LEN = 128
D = 64
N_LAYERS = 2
N_HEADS = 4
FFN_MULT = 4
DROPOUT = 0.1
BATCH_SIZE = 16
STEPS = 2000
EVAL_EVERY = 100
EVAL_BATCHES = 8  # averaged together for each eval point, to reduce noise
LR = 3e-4
WEIGHT_DECAY = 0.01
P = 45  # fixed at the Stage 2 geometric-distinguishability optimum for H=32... but
         # note D=64 here too (H=32), matching that analysis directly.
DATA_SEED = 1234    # controls which windows get sampled, same across all pilot runs
MODEL_SEED = 5678   # controls initial weights, same across all pilot runs


def sample_batch(windows: np.ndarray, rng: np.random.Generator, batch_size: int):
    idx = rng.integers(0, windows.shape[0], size=batch_size)
    rows = np.asarray(windows[idx], dtype=np.int64)[:, : SEQ_LEN + 1]
    x = torch.from_numpy(rows[:, :-1])
    y = torch.from_numpy(rows[:, 1:])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--injection", choices=["additive", "concat"], required=True)
    parser.add_argument("--amplitude", type=float, default=1.0, help="only used for additive")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "pilot_results.jsonl")
    args = parser.parse_args()

    vocab = json.load(open(REPO_ROOT / "data" / "processed" / "vocab.json"))
    vocab_size = len(vocab["id_to_char"])

    train_windows = np.load(REPO_ROOT / "data" / "processed" / "en" / "train_windows.npy", mmap_mode="r")
    test_windows = np.load(REPO_ROOT / "data" / "processed" / "en" / "test_windows.npy", mmap_mode="r")

    train_rng = np.random.default_rng(DATA_SEED)
    eval_rng = np.random.default_rng(DATA_SEED + 1)  # separate stream, still reproducible

    torch.manual_seed(MODEL_SEED)
    cfg = ModelConfig(
        vocab_size=vocab_size, max_seq_len=SEQ_LEN, D=D, n_layers=N_LAYERS, n_heads=N_HEADS,
        ffn_mult=FFN_MULT, dropout=DROPOUT, P=P,
        pe_injection=args.injection, pe_amplitude=args.amplitude,
    )
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    run_label = f"{args.injection}" + (f"_amp{args.amplitude}" if args.injection == "additive" else "")
    print(f"[{run_label}] params={model.num_parameters()} starting {STEPS} steps...")

    eval_points = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        x, y = sample_batch(train_windows, train_rng, BATCH_SIZE)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    ex, ey = sample_batch(test_windows, eval_rng, BATCH_SIZE)
                    elogits = model(ex)
                    eloss = torch.nn.functional.cross_entropy(elogits.reshape(-1, vocab_size), ey.reshape(-1))
                    eval_losses.append(eloss.item())
            mean_eval_loss = float(np.mean(eval_losses))
            eval_points.append({"step": step, "train_loss": loss.item(), "eval_loss": mean_eval_loss})
            print(f"[{run_label}] step {step}/{STEPS}  train_loss={loss.item():.4f}  eval_loss={mean_eval_loss:.4f}")

    elapsed = time.time() - t0
    result = {
        "run_label": run_label,
        "pe_injection": args.injection,
        "pe_amplitude": args.amplitude if args.injection == "additive" else None,
        "config": {"D": D, "n_layers": N_LAYERS, "n_heads": N_HEADS, "seq_len": SEQ_LEN,
                    "P": P, "steps": STEPS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY},
        "param_count": model.num_parameters(),
        "elapsed_seconds": elapsed,
        "eval_points": eval_points,
        "final_train_loss": eval_points[-1]["train_loss"],
        "final_eval_loss": eval_points[-1]["eval_loss"],
        "best_eval_loss": min(p["eval_loss"] for p in eval_points),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    print(f"[{run_label}] DONE in {elapsed:.1f}s. final_eval_loss={result['final_eval_loss']:.4f} "
          f"best_eval_loss={result['best_eval_loss']:.4f}. Appended to {args.out}")


if __name__ == "__main__":
    main()
