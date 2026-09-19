"""
loop.py
=======

Stage 4: the reusable training-and-evaluation loop for ONE (language, P,
seed) run of the real cadence-hypothesis sweep. Deliberately separated
from scripts/train_sweep.py (the CLI harness that drives many calls to
this function and manages the results file) so the actual training
logic can be unit-tested in isolation -- see tests/test_train_loop.py,
which calls run_training() directly against tiny synthetic windows,
with no dependency on the real Europarl-derived .npy files on disk.

PyTorch-learning notes for this stage, building on what pilot_run.py
(Stage 3) already introduced:

- pilot_run.py sampled batches by drawing random row indices with
  replacement on every step -- a fine shortcut for a fast, rough
  comparison, but not a real "epoch" (some windows could be seen many
  times before others are seen once). This loop instead uses the
  idiomatic PyTorch Dataset/DataLoader pattern that torch_dataset.py's
  docstring already pointed towards but nothing had used yet:
  CharWindowDataset + torch.utils.data.DataLoader(shuffle=True).
  DataLoader handles batching and shuffling for us, and shuffle=True
  means each full pass over the dataset sees every window exactly once,
  in a freshly randomised order.
- Because the requested step budget (`steps`) and "one epoch" (however
  many batches DataLoader produces before the dataset is exhausted)
  don't generally divide evenly, _infinite_batches() below wraps the
  DataLoader so iteration can continue past one epoch into a second,
  reshuffled one. See that function's own docstring for a specific
  itertools.cycle() gotcha this deliberately avoids.
- Training and evaluation both reuse the exact same GPT.forward() /
  cross-entropy setup as pilot_run.py -- see that script if you want a
  side-by-side comparison of the "quick pilot" vs "real sweep" training
  code.
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.torch_dataset import CharWindowDataset
from ..model.config import ModelConfig
from ..model.gpt import GPT


def _infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Yields (x, y) batches from `loader` forever, by re-entering the
    `for batch in loader` loop every time one pass finishes rather than
    iterating a single DataLoader iterator to exhaustion once.

    This distinction matters specifically because `loader` is built
    with shuffle=True: what actually triggers a fresh reshuffle is
    calling iter(loader) (which `for batch in loader` does implicitly
    every time the loop is (re-)entered) -- so structuring the infinite
    stream this way gives a genuinely new random order on every epoch.

    A tempting one-line alternative, itertools.cycle(loader), does NOT
    do this: cycle() consumes the underlying iterator exactly once,
    caches every item it yielded, and then replays that same cached
    sequence forever -- silently freezing the "random" order to
    whatever the first epoch happened to be. Worth knowing as a general
    PyTorch/itertools gotcha, not just a detail of this project.
    """
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()  # disables gradient tracking for everything inside -- pure inference, faster and uses less memory
def _evaluate(model: GPT, loader: DataLoader, vocab_size: int, max_batches: int | None) -> float:
    """
    Runs `model` over (up to `max_batches` batches of) `loader` and
    returns the mean cross-entropy loss, averaged per-token (i.e. over
    every (position, batch-element) pair seen, not just per-batch) so
    that batches of slightly different size (the last batch of an
    epoch is often smaller) don't get over- or under-weighted.

    model.eval() switches off dropout (and would switch off
    train/eval-mode-dependent behaviour in other layers, e.g.
    BatchNorm, though this model has none) -- always required before
    evaluating, and the caller is responsible for switching back to
    model.train() afterwards if more training follows.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(x)  # (batch, seq_len, vocab_size)
        # reduction="sum" (rather than the default "mean") so we can
        # correctly weight by the actual number of tokens in this batch
        # ourselves below, instead of accidentally averaging-of-averages
        # across batches of unequal size.
        loss_sum = F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
        )
        total_loss += loss_sum.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


def run_training(
    cfg: ModelConfig,
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    eval_every: int = 500,
    eval_subset_batches: int = 32,
    log_prefix: str = "",
    log_every: int = 0,
) -> dict:
    """
    Trains one GPT (built fresh from `cfg`) on `train_windows` for
    exactly `steps` optimiser steps, periodically evaluating on
    `test_windows`, and returns a plain-dict summary suitable for
    JSON-serialising straight into the sweep's results file.

    Arguments
    ---------
    cfg:
        A fully-specified ModelConfig -- crucially including the P
        value and language-specific vocab_size this particular run is
        for. Every other architectural field (D, n_layers, n_heads,
        ffn_mult, dropout, pe_injection, pe_amplitude, max_seq_len)
        should be held fixed across the whole sweep; it's the sweep
        harness's job (not this function's) to enforce that, by
        constructing every cfg from the same base settings and only
        varying P (and vocab_size/max_seq_len, which are language- and
        dataset-fixed, not swept).
    train_windows, test_windows:
        Arrays of shape (num_windows, cfg.max_seq_len + 1), exactly the
        format preprocess.documents_to_windows produces and
        build_dataset.py saves to disk -- see CharWindowDataset's
        docstring for how a row splits into (input, target).
    steps:
        Total optimiser steps (i.e. total batches trained on, NOT
        epochs -- may span more than one epoch if steps * batch_size
        exceeds train_windows.shape[0]).
    seed:
        Controls BOTH the model's random initialisation (via
        torch.manual_seed, set before constructing the model) and the
        DataLoader's shuffling order (via a torch.Generator passed to
        DataLoader) -- i.e. this one seed fixes everything random about
        this run, so re-running with the same seed reproduces it
        exactly, and different seeds for the same (language, P) give
        the run-to-run noise estimate the sweep needs.
    eval_every / eval_subset_batches:
        Every `eval_every` steps (and always at the final step), the
        model is evaluated on `eval_subset_batches` batches of test
        data -- a cheap, fixed-size subset, purely to track the loss
        curve's shape as training progresses (useful for spotting
        divergence, instability, or a run that hasn't converged yet).
        The FINAL evaluation (after all `steps` are done) additionally
        runs over the entire test set (every window, once), since that
        full-test-set number -- not any subset one -- is the number
        that belongs in the actual cadence-vs-P analysis.

    Returns
    -------
    A dict with: the run's identifying fields (seed, P, vocab_size),
    architecture/training hyperparameters actually used, parameter
    count, wall-clock time, the periodic eval_points list, and the
    final full-test-set loss under the key "final_eval_loss_full" (the
    number Stage 5's analysis should read) alongside
    "final_eval_loss_subset" (the last subset-based point, kept only
    for continuity with the eval_points curve).
    """
    vocab_size = cfg.vocab_size

    # --- Seeding: torch.manual_seed here controls the RANDOM WEIGHT
    # INITIALISATION that happens inside GPT.__init__ (via
    # GPT._init_weights, applied a few lines below) -- it must be set
    # BEFORE the model is constructed to have any effect on it. ---
    torch.manual_seed(seed)
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- A SEPARATE torch.Generator controls the DataLoader's shuffling
    # order. Using the same integer seed to build it (rather than
    # sharing global RNG state with the line above) keeps "which windows
    # appear in which order" reproducible independently of exactly how
    # many random calls model construction happened to make -- i.e.
    # changing the model architecture slightly won't silently change the
    # data order too. ---
    data_generator = torch.Generator().manual_seed(seed)

    train_dataset = CharWindowDataset(train_windows)
    test_dataset = CharWindowDataset(test_windows)

    # shuffle=True + the generator above: a fresh random order every
    # epoch, reproducibly. drop_last=False keeps every window (the final,
    # possibly-smaller batch of an epoch is still used) -- fine here
    # since _evaluate/run_training both weight losses by actual token
    # count rather than assuming a fixed batch size throughout.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=data_generator
    )
    # No shuffling needed for eval -- order doesn't matter when we're
    # just averaging a loss over everything (or a fixed-size subset).
    eval_subset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_full_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    batches = _infinite_batches(train_loader)

    eval_points: list[dict] = []
    t0 = time.time()
    train_loss_running = None  # simple running value for the log line only

    for step in range(1, steps + 1):
        model.train()
        x, y = next(batches)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_running = loss.item()

        if log_every and step % log_every == 0:
            # elapsed/rate/ETA are computed fresh from THIS run's own
            # step 1..step timing (not the sweep-level ETA that
            # train_sweep.py prints between runs) -- this is the number
            # that answers "is THIS run about to finish soon", which
            # matters most while you're staring at a silent terminal.
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            remaining_steps = steps - step
            run_eta_min = (remaining_steps / steps_per_sec / 60.0) if steps_per_sec > 0 else float("nan")
            print(
                f"{log_prefix}step {step}/{steps} ({100.0 * step / steps:.0f}%)  "
                f"train_loss={train_loss_running:.4f}  "
                f"elapsed={elapsed:.0f}s  {steps_per_sec:.2f} steps/s  "
                f"this-run ETA ~{run_eta_min:.1f} min",
                flush=True,  # flush=True: print output is meant to be watched live (e.g. via
                             # `tail -f` on a piped log file), not just seen once the run ends --
                             # without this, Python may buffer many lines before they actually
                             # appear, especially when stdout isn't a real terminal (piped to
                             # `tee`, redirected to a file, etc).
            )

        if step % eval_every == 0 or step == steps:
            eval_loss = _evaluate(model, eval_subset_loader, vocab_size, max_batches=eval_subset_batches)
            eval_points.append({"step": step, "train_loss": train_loss_running, "eval_loss_subset": eval_loss})
            if log_every:
                print(f"{log_prefix}step {step}/{steps}  eval_loss_subset={eval_loss:.4f}", flush=True)

    # Final, full-test-set evaluation -- the number that actually goes
    # into the P-vs-loss analysis. Kept separate from the eval_points
    # curve above (which only ever sees the cheaper fixed-size subset)
    # so the curve and the "official" final number aren't accidentally
    # conflated.
    final_eval_loss_full = _evaluate(model, eval_full_loader, vocab_size, max_batches=None)

    elapsed = time.time() - t0

    return {
        "seed": seed,
        "P": cfg.P,
        "vocab_size": vocab_size,
        "hyperparameters": {
            "D": cfg.D,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "ffn_mult": cfg.ffn_mult,
            "dropout": cfg.dropout,
            "max_seq_len": cfg.max_seq_len,
            "pe_injection": cfg.pe_injection,
            "pe_amplitude": cfg.pe_amplitude,
            "steps": steps,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        "param_count": model.num_parameters(),
        "elapsed_seconds": elapsed,
        "eval_points": eval_points,
        "final_train_loss": train_loss_running,
        "final_eval_loss_subset": eval_points[-1]["eval_loss_subset"] if eval_points else None,
        "final_eval_loss_full": final_eval_loss_full,
    }
