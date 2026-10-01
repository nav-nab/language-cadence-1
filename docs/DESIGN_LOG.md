# language-cadence

Small laptop-scale ML research project testing the "cadence" hypothesis:
does the period of a *periodic* positional encoding, in an otherwise fixed
GPT architecture, affect next-token test loss differently across
languages? If each language has an intrinsic optimal period P*, that's
evidence for a per-language structural "cadence".

> **This is a design log, not the project overview.** For the project summary and
> the paper, see the [root README](../README.md); the full write-up is at
> [`writeup/writeup.pdf`](../writeup/writeup.pdf).

Full experimental design lives in project memory (`language-cadence-research.md`,
`language-cadence-writeup.md`) -- this document records implementation status
and decisions made while turning that design into code.

## Status

- [x] Stage 1: data pipeline
- [x] Stage 2: positional encoding module (Π)
- [x] Stage 3: model
- [x] Stage 4: training/eval + sweep harness (built and verified here; **real sweep not yet run** -- see Stage 4 decisions)
- [x] Stage 5: analysis (real sweep analyzed -- **headline result: Ceiling Confound dominates, see Stage 5 decisions**)

## Layout

```
data/
  raw/          Europarl .tgz archives go here (not checked into git -- see raw/README.md)
  processed/    output of scripts/build_dataset.py: vocab.json, manifest.json,
                and per-language {train,test}_windows.npy
src/language_cadence/
  data/
    extract_europarl.py   parses raw Europarl text into Documents (pseudo-docs of 500 sentences --
                           the real release has no <CHAPTER> tags, see module docstring)
    preprocess.py          NFC normalise, lowercase, build shared vocab, split, window
    torch_dataset.py        PyTorch Dataset wrapper around the saved .npy window files
  pe/
    periodic_pe.py          Pi positional encoding, NumPy reference implementation
    torch_periodic_pe.py    Pi as a PyTorch nn.Module (buffer, not a learned parameter)
  model/
    config.py       ModelConfig dataclass -- ALL architecture + Pi-injection hyperparameters,
                     including the two Stage 3 open decisions (pe_injection, pe_amplitude)
    attention.py     causal multi-head self-attention
    mlp.py           feedforward sub-layer
    block.py         pre-norm transformer block (attention + feedforward, each with residual + LN)
    pe_injection.py  switches between additive (x + amplitude*Pi) and concat (Linear(concat(x,Pi)))
    gpt.py           full model: embedding -> Pi injection -> blocks -> final LN -> tied lm_head
  train/
    loop.py          Stage 4: run_training() -- trains + evaluates ONE (language, P, seed) run,
                      given already-loaded window arrays; unit-tested with tiny synthetic data
scripts/
  build_dataset.py         Stage 1 CLI: raw archives -> processed windows
  pe_distinguishability.py Stage 2: reproduces the distinguishability-vs-P analysis, writes figures/
  pilot_run.py             Stage 3 calibration pilot: one short training run per (injection, amplitude)
                            combination, appends its result to results/pilot_results.jsonl
  pilot_analyze.py         reads results/pilot_results.jsonl, writes the pilot comparison figures
  train_sweep.py           Stage 4 CLI: runs the real (language, P, seed) grid via train.loop.run_training,
                            resumable (append-only results file + skip-if-done) -- see Stage 4 decisions.
                            **This is the script Navid runs on his own Mac for the real sweep.**
  analyze_sweep.py         Stage 5: reads results/sweep_results.jsonl, classifies each language's
                            test-loss-vs-P curve (interior minimum vs monotonic/boundary), writes
                            results/sweep_summary.json and figures/sweep_test_loss_vs_P.png
  plot_distinguishability_vs_loss.py  Stage 5 follow-up: puts Stage 2's Pi-distinguishability curve
                            and Stage 5's test-loss curve on one dual-axis figure over the same P grid --
                            see Stage 5 decisions for what it shows (they're POSITIVELY correlated,
                            the opposite of what "distinguishability drives performance" predicts)
tests/
  test_preprocess.py       Stage 1 unit tests (stdlib unittest, no torch/pytest required)
  test_periodic_pe.py      Stage 2 unit tests, NumPy reference (incl. exact Pi(n)==Pi(n+P) check)
  test_torch_periodic_pe.py Stage 2 unit tests, PyTorch module (incl. cross-check vs NumPy reference)
  test_model.py            Stage 3 unit tests: shapes, real (not just present) causal masking,
                            gradient flow under both injection modes, parameter-count sanity
  test_train_loop.py       Stage 4 unit tests (pytest, not stdlib unittest -- uses tmp_path/capsys
                            fixtures): run_training() against tiny synthetic data (well-formed result,
                            reproducibility given a seed, seeds actually diverge, steps spanning more
                            than one epoch), plus train_sweep.py's config-fingerprinting and
                            skip-if-done resumability logic
  test_analyze_sweep.py    Stage 5 unit tests: the interior-minimum-vs-monotonic classification rule
                            (incl. the real sweep's actual shape: monotonic decreasing to the boundary),
                            local-minima detection, and multi-seed averaging in load_sweep_by_language()
  test_plot_distinguishability_vs_loss.py  Stage 5 follow-up tests: cross-language loss averaging,
                            and a golden-value cross-check on the imported distinguishability() values
                            at P=45/P=384 so this figure can't silently drift from Stage 2's real metric
results/
  pilot_results.jsonl      one JSON line per pilot_run.py run: config, eval-loss curve, best/final loss
  sweep_results.jsonl      the real Stage 4 sweep, 70 lines (14 P values x 5 languages x 1 seed): one
                            JSON line per run -- config_id, eval-loss curve, final_eval_loss_full
  sweep_summary.json       Stage 5 output: per-language argmin P, min loss, loss at P=45, classification
                            (interior minimum vs monotonic/boundary), any local minima found
figures/
  pe_distinguishability_vs_P.png    line plot at H=32 (matches design-phase P≈45 finding)
  pe_distinguishability_heatmap.png distinguishability over (H, P), argmax-P vs H compared to P=2H
  pilot_eval_loss_curves.png        eval loss vs step, one line per pilot (injection, amplitude) combo
  pilot_final_comparison.png        ranked bar chart of best eval loss per combo
  sweep_test_loss_vs_P.png          Stage 5 headline figure: test loss vs P (log-x), one line per
                                     language, smoothed overlay, Stage 2's P=45 marked for reference
  distinguishability_vs_loss.png    Stage 5 follow-up: Pi distinguishability (Stage 2) and mean test
                                     loss (Stage 5) on one dual-axis figure, same P grid -- see below
```

Run the pipeline once the raw archives are in place:

```
python3 scripts/build_dataset.py
```

Run tests any time (pytest picks up all 51 tests; plain `unittest discover`
works too but silently skips `test_train_loop.py`, which needs pytest's
`tmp_path`/`capsys` fixtures):

```
python3 -m pytest tests/ -v
```

Run the real Stage 4 sweep (resumable -- safe to interrupt and re-run):

```
python3 scripts/train_sweep.py
```

## Stage 1 decisions

These were either explicit open questions in the design notes, or small
implementation choices needed to turn the notes into runnable code.
Flagged (**agreed with Navid**) ones were confirmed in conversation
before implementing; the rest are small enough that they're just
documented here, but are easy to revisit if they turn out to matter.

- **Vocabulary scope (agreed):** one character vocabulary shared across
  all five languages (not per-language). Keeps the embedding table --
  and therefore D -- identical across languages by construction, which
  matters since D/H must stay fixed across the whole sweep per language
  and we want languages to be comparable.
- **Rare symbols (agreed):** characters below a frequency threshold
  (default `min_freq=1e-4`, i.e. must appear in at least 1 per 10,000
  characters) are folded into a single `<UNK>` symbol rather than each
  getting their own vocab slot. Keeps the vocab close to the ~80-120
  target from the notes.
- **Window length / overlap / data volume (agreed):** N=384 (within the
  256-512 guide band from the notes), non-overlapping windows (avoids
  near-duplicate training examples from overlapping slices), targeting
  roughly 10M characters/language. Actual achieved volume, per
  `data/processed/manifest.json` from the real downloaded archives:
  ~17.6-18.0M train characters and ~1.8-2.0M test characters per
  language (comfortably past the 10M target), ~208-239 train documents
  and ~23-26 test documents per language, and 45.6K-46.7K train windows
  / 4.8K-5.1K test windows per language -- consistent across all five
  languages, as intended by extracting a fixed raw-byte budget per
  language rather than a fixed document count.
- **Document granularity:** the real Europarl v7 release turned out to
  have NO document/chapter markup at all (confirmed directly against the
  downloaded archives -- an earlier assumption about `<CHAPTER>` tags was
  wrong, see `extract_europarl.py`'s module docstring for the full
  story). So "documents", for the purposes of the document-level
  train/test split, are fixed-size chunks of 500 consecutive sentences
  each, in original order -- not real debate/session boundaries, but
  still enough to keep topically/stylistically correlated sentences on
  one side of the split rather than scattered randomly across it. The
  code still recognises real `<CHAPTER>` tags as document boundaries if
  a file happens to have them (e.g. a differently-formatted release),
  but that's the exception path now, not the expected one.
- **English source file:** Europarl v7 ships as language-PAIR archives,
  always paired with English (`de-en.tgz`, `fr-en.tgz`, etc.), so there
  are 4 candidate English text files. Rather than merge/dedupe across
  all 4 (which have near-duplicate but not identical sentences due to
  translation-pipeline differences), the English corpus is taken from
  exactly one pair (`fr-en.tgz`, arbitrary choice) -- see
  `ENGLISH_SOURCE_PAIR` in `extract_europarl.py`. Doesn't affect the
  other 4 languages.
- **Window construction:** each window is `window_length + 1` characters
  (385 for N=384), so it can be split into `(input, target)` =
  `(window[:-1], window[1:])` for next-token prediction without any
  padding. Any leftover tail of a document shorter than a full window is
  dropped rather than padded -- keeps every training example "real" at
  the cost of a small amount of discarded data per document. Windows
  never cross a document boundary (tested in `test_preprocess.py`).
- **Train/test split fraction:** default 10% of documents held out per
  language (`--test-frac 0.1`), fixed seed for reproducibility. Easy to
  change via CLI flag.

## Stage 2 decisions

- **Distinguishability metric:** mean L2 distance between neighbouring
  position columns of Pi, averaged over one full period (including the
  wrap-around pair, since Pi is exactly periodic). A direct, easy-to-
  interpret "how much does moving one position actually change the
  encoding" measure -- not the only reasonable choice, but the one
  matching the informal description of the original design-phase
  analysis this stage was reproducing.
- **Verified, not just re-derived:** the reproduction independently
  confirmed the design-phase finding (P=45 at H=32) exactly, and
  additionally swept H itself (not just P), which showed the
  distinguishability-maximising P scales roughly linearly with H across
  the whole range tested, but consistently runs below the naive P=2H
  line -- closer to ~1.4H. Documented here since it refines (mildly
  contradicts the precise multiplier of) the "peak near P≈2H" framing in
  the original design notes, though the qualitative story (bigger H
  wants bigger P) holds.
- **This is encoding geometry only, not trained-model behaviour** -- a
  hypothesis for where empirical P* might land in Stage 4/5, not a
  prediction guaranteed to hold. Flagged explicitly because the project
  write-up notes call out this exact distinction as something to keep
  separate, not blur together.

## Stage 3 decisions

- **Pi injection method and amplitude are config, not hardcoded** (per
  the project notes' explicit requirement): `ModelConfig.pe_injection`
  ("additive" or "concat") and `ModelConfig.pe_amplitude`. **Settled**
  via a calibration pilot (`scripts/pilot_run.py` /
  `scripts/pilot_analyze.py`, results in `results/pilot_results.jsonl`,
  figures in `figures/pilot_*.png`): 8 short training runs: 7 additive amplitudes plus one concat (English only,
  P=45 fixed, seq_len=128, D=64, 2 layers, 2000 steps, one seed held
  constant across runs) comparing additive at amplitude in
  {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0} against concat. Note the pilot used a smaller model than the sweep (2 layers, seq_len=128, 2000 steps). Additive at
  amplitude=0.1 reached the numerically lowest eval loss (2.243), but
  only narrowly ahead of concat (2.264) -- a margin well within what a
  single-seed, short, one-language pilot could produce as noise. **Chose
  `pe_injection="concat"`** despite not being the literal top score,
  because of a structural advantage the numbers alone don't capture: no
  amplitude hyperparameter to get wrong. Amplitude turned out to have a
  narrow, saturating sweet spot -- amplitude=2.0 and amplitude=4.0
  produced eval curves agreeing to within about 0.002, because every transformer
  block's leading LayerNorm is scale-invariant
  (`LayerNorm(c*v) == LayerNorm(v)`), so past a certain point the token
  embedding becomes negligible next to amplitude*Pi and further scaling
  is invisible -- an architecture detail unrelated to the actual cadence
  hypothesis, which concat sidesteps entirely. (`pe_amplitude` stays
  configurable at `0.1`, the pilot's best in-range value, for anyone
  switching back to additive later for comparison.) This pilot is
  explicitly low-fidelity (one seed, one language, short context/run) --
  good enough to fix reasonable Stage 4 defaults, not a substitute for
  what Stage 4/5 will actually show; per the project notes, whichever
  combination is used must stay fixed across the whole P-sweep to avoid
  confounding amplitude/injection-method with periodicity.
- **Pre-norm transformer blocks** (LayerNorm before each sub-layer, not
  after) -- the modern standard over the original Transformer paper's
  post-norm, chosen for training stability at negligible cost/complexity;
  not something the project notes discuss, so not flagged as an open
  decision requiring sign-off.
- **Weight tying** between the token embedding and the output (`lm_head`)
  projection -- standard GPT convention, reduces parameter count with
  usually-neutral-to-positive effect on quality.
- **Concat injection projects 2*D -> D via one linear layer** after
  concatenating token embeddings and Pi (rather than, say, reducing the
  token embedding's own dimension to leave room for Pi) -- keeps Pi's own
  D/H exactly as specified in Stage 2 regardless of which injection mode
  is active, at the cost of one extra learned projection layer only used
  in concat mode.

## Stage 4 decisions

- **Split between reusable loop and CLI harness (agreed pattern from
  Stage 3):** `src/language_cadence/train/loop.py`'s `run_training()` trains
  and evaluates exactly ONE `(language, P, seed)` run, given
  already-loaded window arrays -- no file I/O, no results-file
  bookkeeping -- so it can be unit-tested against tiny synthetic data
  (`tests/test_train_loop.py`) independently of the real ~18M-character
  Europarl-derived dataset. `scripts/train_sweep.py` is the thin CLI
  layer on top: loads real per-language data, iterates the full grid,
  and manages the results file.
- **Idiomatic Dataset/DataLoader training loop, not the pilot's
  sample-with-replacement shortcut:** `pilot_run.py` (Stage 3) drew
  random batches by index, with replacement, on every step -- fine for
  a fast rough comparison, but not a real epoch. `run_training()`
  instead uses `CharWindowDataset` (extended to accept an in-memory
  array as well as a file path) + `torch.utils.data.DataLoader(shuffle=True)`,
  so each pass over a language's data is a genuine, freshly-reshuffled
  epoch. See `loop.py`'s docstring for a specific `itertools.cycle()`
  pitfall this deliberately avoids when extending training past one
  epoch.
- **Eval strategy:** periodic evals during training (every `eval_every`
  steps) run over a small, fixed-size subset of the test set -- cheap,
  purely for tracking the loss curve's shape as training progresses.
  The FINAL eval, after all training steps, runs over the *entire* test
  set exactly once -- that full-test-set number
  (`final_eval_loss_full`) is what Stage 5's analysis will actually
  read, kept clearly distinct in the result dict from the cheaper
  subset-based curve.
- **Resumability (agreed): append-only results file + skip-if-done, no
  mid-run checkpointing.** Same pattern as `pilot_run.py`: each finished
  run appends one JSON line to `results/sweep_results.jsonl`
  immediately. Before starting any `(language, P, seed)` combination,
  the harness checks whether it's already present in that file *and*
  was run under the same fixed hyperparameters (via a short hash --
  `_config_id()` -- of everything else: D, n_layers, steps, batch size,
  etc.), so a later hyperparameter change is never silently mistaken for
  "already done." This means the whole sweep can be interrupted (Ctrl+C,
  laptop sleep, closing the terminal) at any point and continued by
  simply re-running the same command.
- **P grid (agreed): 14 values -- 3, 4, 6, 8, 16, 24, 32, 45, 64, 96,
  128, 192, 256, 384.** A few very small periods (down to P=3) plus a
  spread above and below the Stage 2 geometric-distinguishability peak
  (P=45), up to and including P=384=`max_seq_len` exactly -- the "Pi's
  period equals the whole context window" point that makes periodicity
  structurally unobservable within a single window, directly testing the
  design notes' "Ceiling Confound" question.
- **Seeds and step budget -- ASSUMPTION, not confirmed with Navid** (he
  asked me to pick a reasonable value and explain it, rather than
  confirm it up front): **1 seed per `(language, P)`, 3000 steps,
  batch_size=16.** The originally-discussed 2 seeds / 6000 steps turned
  out not to be reasonable once actually benchmarked: one training step
  at the sweep's real architecture (D=64, 4 layers, seq_len=384 -- NOT
  the pilot's shrunk-down seq_len=128/2-layer config) measured ~1.2
  seconds/step on this sandboxed environment's 4 CPU cores, which would
  put the full 14 P values x 5 languages x 2 seeds x 6000 steps grid at
  280+ hours of continuous compute -- not "a few sittings" by any
  reading. 3000 steps x batch_size=16 = 48,000 windows of 384 characters
  seen per run = ~18.4M characters -- almost exactly ONE PASS over a
  language's ~17-18M-character training set (see
  `data/processed/manifest.json`), a clean way to justify the number,
  and roughly 60 minutes/run at the benchmarked rate (likely faster on
  Navid's actual Mac). 1 seed keeps the full grid at
  14 x 5 x 1 = 70 runs (~70 hours at this environment's rate). Because
  results are append-only and skip-if-done keys on seed too, this is
  not a one-way door: running `train_sweep.py --seeds 0,1` later adds a
  second seed for noise-checking without redoing anything already done.
  **Navid: if you'd rather use a different step/seed budget once you've
  seen the real per-step speed on your Mac (`--dry-run` plus a short
  `--steps N` trial run will tell you quickly), everything past this
  point is easy to change before committing to the full grid.**
- **Verified in this environment, not run for real here:** per the
  compute-constraint discussion before this stage, this sandboxed
  environment cannot practically run the real sweep (~70+ hours, no
  persistent background execution across tool calls, ~170s per call).
  What WAS verified here: all 9 new unit tests pass
  (`tests/test_train_loop.py`); a real, tiny end-to-end run against
  actual English Stage 1 data (15 steps, real architecture) produced a
  sensible decreasing loss curve; and the CLI harness itself
  (`train_sweep.py`) was run twice against a tiny 2-run grid with real
  data -- confirming the second run correctly skipped both (already
  done) and that changing a hyperparameter (`--steps`) correctly
  produced a different `config_id` and was NOT skipped. **The real
  sweep -- the full 70-run grid -- has not been run; that's the next
  step, on Navid's own Mac.**

## Stage 5 decisions: results and the Ceiling Confound

The real sweep (`results/sweep_results.jsonl`, 70 runs: 14 P values x 5
languages x 1 seed) completed cleanly -- one consistent `config_id`
across all 70 runs, no missing or duplicate `(language, P, seed)`
combinations, no NaN/non-finite losses, all `final_eval_loss_full`
values in a sane range (1.74-2.26, well below the untrained-model
baseline of ln(64)~4.16). `scripts/analyze_sweep.py` (`python3
scripts/analyze_sweep.py`) reads it, classifies each language's curve,
and writes `results/sweep_summary.json` and
`figures/sweep_test_loss_vs_P.png`.

**Headline result: the Ceiling Confound dominates, for every language.**
All 5 languages' test loss trends downward across the tested range (not strictly
monotone: see the local dips below) -- the global minimum is at P=384 (=`max_seq_len`, the
largest P tested) for English, French, German, Italian, AND Spanish,
with no exceptions. Per the classification rule agreed before running
this (interior minimum = argmin strictly inside the tested range, not at
either boundary): **0/5 languages show an interior minimum.** This is
exactly the failure mode Issue 1 in the design notes was worried about:
P is behaving like it enlarges the model's effective positional/context
range (bigger P -> Pi repeats less within one window -> more
distinguishable positions available to the model), not like it's
revealing an intrinsic per-language optimal period. At this project's
scale (N=384 fixed, one seed), this result gives no evidence for the
cadence hypothesis as originally framed -- the P=45 geometric peak from
Stage 2 is NOT where any language's test loss bottoms out; loss keeps
improving well past it for every language.

**Secondary, much weaker signal (flagged, not relied on):** even though
every language's GLOBAL minimum is at the boundary, a few languages show
small LOCAL dips at interior P values -- e.g. German at P=32,
English at P=32, Spanish at P=24 (full list in
`sweep_summary.json`'s `local_minima` field per language). These don't
cluster consistently near P=45 across languages, and with only one seed
per point, a small single-point dip is exactly what run-to-run noise
would produce -- **not read as evidence of a real effect** without at
least a second seed to check whether the same dips reproduce. Recorded
here (and in project memory for the write-up) as "worth a second seed",
not as a finding.

**What this means for the project, going forward (not yet acted on --
for discussion with Navid):** the current setup can't distinguish "P
gives the model a genuinely better positional encoding at this specific
period" from "P just gives the model more usable context, up to the
N=384 ceiling" -- because P=384 was included as the direct Ceiling
Confound test point, and it had the lowest loss every time (margin over P=256 ranges from 0.021 for Italian to 0.176 for English). A cleaner
test would need P's tested range to stay well below N (e.g. re-running
with a much larger max_seq_len so that even P=384 is a small fraction of
context, or restricting the P grid to well under N and checking whether
an interior optimum appears once the boundary effect is no longer
directly reachable). That's a design decision for a possible Stage 6,
not implemented here.

**Follow-up figure (prompted by Navid's own observation while reviewing
the results):** With H fixed, an
overly long P makes neighbouring positions harder to distinguish (a geometric consequence of the Stage 2 construction, measured after the fact, not a penalty built into the code) --
distinguishability peaks at P=45 then falls sharply for larger P. If
that geometric property mattered for trained performance, distinguishability
and test loss should move in OPPOSITE directions (more distinguishable
-> lower loss). `scripts/plot_distinguishability_vs_loss.py` puts both
curves on one figure (`figures/distinguishability_vs_loss.png`) to check
directly. They don't move in opposite directions -- they're **positively
correlated** (Pearson r = +0.930 across the swept grid): both curves
fall together as P grows past 45. Concretely, P=384 is the single WORST
point on the entire grid by the distinguishability metric (1.74, vs the
P=45 peak of 8.89 -- roughly a 5x drop) and simultaneously the BEST
point by test loss, for every language. The geometric trade-off Pi was
explicitly built around -- the reason not to just always pick the
largest possible P -- had no protective effect in the actual trained
models; whatever made large P better for prediction, it wasn't
adjacent-position distinguishability.

## Testing

61 unit tests total (`python3 -m pytest tests/ -v` -- see the note above
on why pytest, not plain `unittest`, is needed for full coverage), all
passing as of Stage 5:

- **`test_preprocess.py`** (Stage 1): chapter-boundary parsing (incl. the
  no-`<CHAPTER>`-tags path, which prints a note rather than a warning --
  it's the expected case for the real Europarl release, not an anomaly),
  NFC normalisation (including that combining-character sequences
  collapse to the same id as their precomposed form), lowercasing, vocab
  building (rare-char -> UNK folding, shared-across-languages behaviour),
  document-level split (no document in both splits, reproducible given a
  seed), and windowing (correct shape, windows never cross a document
  boundary).
- **`test_periodic_pe.py`** (Stage 2, NumPy): the folded-harmonic
  formula against a worked example, the exact `Pi(n) == Pi(n+P)` property,
  that P is the *fundamental* period (not just *a* period), and output
  shape/bounds.
- **`test_torch_periodic_pe.py`** (Stage 2, PyTorch): numerical agreement
  with the NumPy reference (to float32 precision), the same exact-period
  property in torch, correct broadcasting onto batched activations, and
  that Pi is registered as a non-trainable buffer, not a Parameter.
- **`test_model.py`** (Stage 3): output shapes under both injection
  modes; that causal masking is real (changing only the *last* token in
  a sequence leaves every earlier position's logits byte-for-byte
  identical -- verified numerically, not just "a mask was passed");
  gradient flow through every parameter under both injection modes; that
  weight tying shares one tensor, not two equal ones; a parameter-count
  sanity range; and that both `P` and `pe_amplitude` actually change the
  model's output (i.e. they're genuinely wired through, not silently
  ignored).
- **`test_train_loop.py`** (Stage 4): `run_training()` against tiny
  synthetic (random-token) windows -- a well-formed result dict with all
  the fields the sweep harness/Stage 5 analysis depend on; exact
  reproducibility given the same seed (weight init AND data shuffling
  order both derive from it); that different seeds actually diverge
  (i.e. `seed` is wired through to something that matters); and correct
  behaviour when the step budget spans more than one epoch. Plus
  `train_sweep.py`'s bookkeeping: `_config_id()` is stable/order-independent
  and changes when a hyperparameter changes, and `_load_existing_keys()` /
  the dry-run skip logic correctly identify already-done vs remaining
  runs.
- **`test_analyze_sweep.py`** (Stage 5): the interior-minimum-vs-monotonic
  classification rule against synthetic curves (monotonic decreasing --
  the real sweep's actual shape --, a true interior dip, monotonic
  increasing, and a local dip that coexists with a boundary global
  minimum), and that `load_sweep_by_language()` correctly averages
  multiple seeds per `(language, P)` while passing single-seed data
  through unchanged.
- **`test_plot_distinguishability_vs_loss.py`** (Stage 5 follow-up):
  cross-language mean-loss averaging, and a golden-value check that the
  imported `distinguishability()` still matches the real values this
  figure was built from at P=45/P=384 -- so a future change to Stage 2's
  metric (or an accidental reimplementation drifting from it) would be
  caught here rather than silently changing the figure's story.

Beyond unit tests, several things were checked against real (not
synthetic) data, not just automated fixtures:
`scripts/build_dataset.py` run against the actual downloaded Europarl
archives (see `data/processed/manifest.json` for the real per-language
counts, summarised in the Stage 1 decisions section above); an ad hoc,
uncommitted overfit-a-single-batch script (loaded a real `en` window
batch via `CharWindowDataset`, trained a real `GPT` on it repeatedly
with plain Adam, no weight decay) confirming loss drops from the
random-guessing baseline (~ln(vocab_size) = 4.16) to ~0.01 -- i.e. the
full data -> model -> loss -> backward -> step chain is correctly
wired, not just that shapes line up; and `scripts/pilot_run.py`, which
is effectively a proper, committed, more thorough successor to that ad
hoc check -- real held-out train/eval loss curves (not just overfitting
one batch) across 7 different model configurations, all on real English
data (see "Stage 3 decisions" above and `results/pilot_results.jsonl`
for the results).

The Stage 1 pipeline was additionally smoke-tested end-to-end against a
small synthetic multi-language corpus (fake `<CHAPTER>`-tagged text) to
confirm `build_dataset.py` runs cleanly from raw archive to saved
`.npy` files -- that test data was generated, checked, and deleted, not
committed.

## Post-Stage-5 addition: an inference GUI (since removed)

**Note:** the `scripts/inference/` files described below were deleted in the "Tweaks" commit and are no longer in the repo. The section is kept as history. `checkpoints/en_P384.pt` remains.

Added after the write-up was finalized, purely for interactive
exploration -- not part of the Stage 1-5 research pipeline and not
discussed in the paper. `scripts/inference/train_checkpoint.py` fills
the gap the sweep deliberately left (Stage 4's append-only
results-file design never saved model weights, only loss numbers) by
training a GPT and saving a checkpoint; `scripts/inference/generate_gui.py`
is a tkinter desktop app that loads a checkpoint and streams
generated text, one character at a time, from an optional prompt,
until stopped. See `scripts/inference/README.md` for usage.

`checkpoints/en_P384.pt` (committed) uses the sweep's own
best-performing configuration -- English, P=384 -- trained for 1200
steps in short, resumable chunks (`train_checkpoint.py --resume-from`)
rather than the real sweep's 3000-step budget per run. At 1200 steps
it produces word-fragment-level text, not fluent English; this is a
proof of concept for the inference path, not a claim about generation
quality.
