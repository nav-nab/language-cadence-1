"""
train/
======

Stage 4: the training-and-evaluation machinery for the real cadence
sweep, split into two layers:

- loop.py:        run_training() -- trains ONE (language, P, seed)
                   model to completion and returns its result as a
                   dict. Pure function of its arguments: no file I/O
                   for the sweep's results log, no CLI, no dedup logic.
                   This is what tests/test_train_loop.py exercises
                   directly, with tiny synthetic data, so the training
                   logic itself is verified independently of the real
                   ~18M-character-per-language dataset and of the sweep
                   harness's bookkeeping.
- scripts/train_sweep.py (not in this package -- see the repo's
  scripts/ directory): the CLI harness that loads real per-language
  data, iterates over the full (language, P, seed) grid, calls
  run_training() for each combination not already present in the
  results file, and appends each result as one JSON line -- this is
  the append-only, skip-if-done resumability design agreed with the
  user, matching the pattern scripts/pilot_run.py already established
  in Stage 3.
"""
