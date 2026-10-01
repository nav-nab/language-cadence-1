"""
train_checkpoint.py
====================

Trains a GPT and saves it to disk as a checkpoint the inference GUI
(scripts/inference/generate_gui.py) can load. The project's Stage 4/5
sweep deliberately never saved model weights -- only loss numbers, in
an append-only results file (see docs/DESIGN_LOG.md's Stage 4
decisions) -- so a checkpoint has to be produced separately, here.

Defaults to the sweep's own best-performing configuration for English
(see results/sweep_summary.json: P=384, i.e. the Ceiling Confound
boundary, achieved the lowest test loss of any (language, P) pair) so
the demo model is at least representative of the project's real
headline finding, not an arbitrary unrelated run.

RESUMABLE BY DESIGN: this script is meant to be run repeatedly in
short chunks (each invocation trains for --steps steps and exits),
rather than once for the full step budget in one long-running process
-- useful when training is driven from an environment with a wall-clock
limit per command. Pass --resume-from <checkpoint> to continue from a
previously-saved checkpoint's weights, optimizer state, and cumulative
step count; omit it to start fresh. Each run overwrites --out with the
new cumulative state, so pointing --resume-from and --out at the same
path turns repeated invocations into "keep training this one
checkpoint further".

Only a cheap, fixed-size-subset eval is run at the end of each chunk
(for a progress readout) -- a full-test-set eval is comparatively
expensive (the whole point of Stage 4/5's design was to only pay for
that once per real sweep run, see loop.py's run_training) and isn't
needed just to produce an inference checkpoint. Pass --full-eval to
run one anyway, e.g. on the final chunk.

Usage:
    PYTHONPATH=src python3 scripts/inference/train_checkpoint.py --steps 150 \
        --out checkpoints/en_P384.pt
    PYTHONPATH=src python3 scripts/inference/train_checkpoint.py --steps 150 \
        --resume-from checkpoints/en_P384.pt --out checkpoints/en_P384.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from language_cadence.data.preprocess import Vocab
from language_cadence.data.torch_dataset import CharWindowDataset
from language_cadence.model.config import ModelConfig
from language_cadence.model.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def _eval_subset(model: GPT, loader: DataLoader, vocab_size: int, max_batches: int) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(x)
        loss_sum = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum")
        total_loss += loss_sum.item()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="en", choices=["en", "fr", "de", "it", "es"])
    parser.add_argument("--P", type=int, default=384, help="positional-encoding period (default: 384, the sweep's best-performing value)")
    parser.add_argument("--steps", type=int, default=150, help="steps to train THIS invocation (a chunk, not the total budget)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=20, help="batches used for the cheap end-of-chunk progress eval")
    parser.add_argument("--full-eval", action="store_true", help="also run a full-test-set eval at the end of this chunk")
    parser.add_argument("--resume-from", default=None, help="checkpoint to resume weights/optimizer/step-count from")
    parser.add_argument("--out", default=None, help="checkpoint output path (default: checkpoints/<language>_P<P>.pt)")
    args = parser.parse_args()

    vocab = Vocab.from_json(json.loads((ROOT / "data/processed/vocab.json").read_text()))
    data_dir = ROOT / "data/processed" / args.language
    train_windows = np.load(data_dir / "train_windows.npy")
    test_windows = np.load(data_dir / "test_windows.npy")
    max_seq_len = train_windows.shape[1] - 1  # windows are (window_length + 1) long, see torch_dataset.py

    out_path = Path(args.out) if args.out else ROOT / "checkpoints" / f"{args.language}_P{args.P}.pt"

    cfg = ModelConfig(
        vocab_size=len(vocab), max_seq_len=max_seq_len, P=args.P,
        D=64, n_layers=4, n_heads=4, ffn_mult=4, dropout=0.1,
        pe_injection="concat", pe_amplitude=0.1,
    )

    torch.manual_seed(args.seed)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps_so_far = 0

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        steps_so_far = ckpt.get("steps", 0)
        print(f"Resumed from {args.resume_from} at step {steps_so_far}.", flush=True)

    train_dataset = CharWindowDataset(train_windows)
    test_dataset = CharWindowDataset(test_windows)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(
        f"Training language={args.language} P={cfg.P} chunk_steps={args.steps} "
        f"(cumulative so far: {steps_so_far}) batch_size={args.batch_size} "
        f"on {train_windows.shape[0]} train windows...",
        flush=True,
    )

    t0 = time.time()
    it = iter(train_loader)
    model.train()
    last_loss = None
    for i in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if args.log_every and i % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[train_checkpoint] chunk step {i}/{args.steps} "
                f"(cumulative {steps_so_far + i})  train_loss={last_loss:.4f}  "
                f"elapsed={elapsed:.0f}s  {i / elapsed:.2f} steps/s",
                flush=True,
            )

    eval_loss_subset = _eval_subset(model, eval_loader, cfg.vocab_size, max_batches=args.eval_batches)
    eval_loss_full = None
    if args.full_eval:
        eval_loss_full = _eval_subset(model, eval_loader, cfg.vocab_size, max_batches=10**9)

    steps_so_far += args.steps
    print(
        f"Chunk done in {time.time() - t0:.0f}s. cumulative_steps={steps_so_far} "
        f"train_loss={last_loss:.4f} eval_loss_subset={eval_loss_subset:.4f}"
        + (f" eval_loss_full={eval_loss_full:.4f}" if eval_loss_full is not None else ""),
        flush=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "vocab": vocab.to_json(),
            "language": args.language,
            "steps": steps_so_far,
            "eval_loss_subset": eval_loss_subset,
            "eval_loss_full": eval_loss_full,
        },
        out_path,
    )
    print(f"Checkpoint saved to {out_path} (cumulative_steps={steps_so_far})", flush=True)


if __name__ == "__main__":
    main()
