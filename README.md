# metronome

Small laptop-scale ML research project testing the "cadence" hypothesis:
does the period of a *periodic* positional encoding, in an otherwise fixed
GPT architecture, affect next-token test loss differently across
languages? If each language has an intrinsic optimal period P*, that's
evidence for a per-language structural "cadence".

Full experimental design lives in project memory (`language-cadence-research.md`,
`language-cadence-writeup.md`) -- this README documents implementation status
and decisions made while turning that design into code.

## Status

- [x] Stage 1: data pipeline
- [x] Stage 2: positional encoding module (Π)
- [ ] Stage 3: model
- [ ] Stage 4: training/eval + sweep harness
- [ ] Stage 5: analysis

## Environment notes

- **Europarl data:** downloaded and in `data/raw/` (done).
- **PyTorch:** installed (`torch 2.14.0+cpu`, CPU-only build) in the
  sandboxed Linux VM Claude uses to run commands against this folder,
  via `pip install --user --index-url https://download.pytorch.org/whl/cpu
  torch`. This required adding `pypi.org`, `files.pythonhosted.org`,
  `download.pytorch.org`, and `download-r2.pytorch.org` to this
  environment's network allowlist -- by default that sandbox can't reach
  the internet at all. Verified working: forward pass, backward pass /
  autograd, and `torch_dataset.py`'s `CharWindowDataset` all run
  correctly. Note this is a *separate* Python environment from the one
  VS Code uses on your actual machine (which has its own working PyTorch
  install via Anaconda, `/opt/anaconda3/bin/python`) -- the sandbox VM's
  install is what lets Claude run and test PyTorch code directly, not a
  replacement for your own environment.

## Layout

```
data/
  raw/          Europarl .tgz archives go here (not checked into git -- see raw/README.md)
  processed/    output of scripts/build_dataset.py: vocab.json, manifest.json,
                and per-language {train,test}_windows.npy
src/metronome/
  data/
    extract_europarl.py   parses raw Europarl text into Documents (pseudo-docs of 500 sentences --
                           the real release has no <CHAPTER> tags, see module docstring)
    preprocess.py          NFC normalise, lowercase, build shared vocab, split, window
    torch_dataset.py        PyTorch Dataset wrapper around the saved .npy window files
  pe/
    periodic_pe.py          Pi positional encoding, NumPy reference implementation
    torch_periodic_pe.py    Pi as a PyTorch nn.Module (buffer, not a learned parameter)
scripts/
  build_dataset.py         Stage 1 CLI: raw archives -> processed windows
  pe_distinguishability.py Stage 2: reproduces the distinguishability-vs-P analysis, writes figures/
tests/
  test_preprocess.py       Stage 1 unit tests (stdlib unittest, no torch/pytest required)
  test_periodic_pe.py      Stage 2 unit tests, NumPy reference (incl. exact Pi(n)==Pi(n+P) check)
  test_torch_periodic_pe.py Stage 2 unit tests, PyTorch module (incl. cross-check vs NumPy reference)
figures/
  pe_distinguishability_vs_P.png    line plot at H=32 (matches design-phase P≈45 finding)
  pe_distinguishability_heatmap.png distinguishability over (H, P), argmax-P vs H compared to P=2H
```

Run the pipeline once the raw archives are in place:

```
python3 scripts/build_dataset.py
```

Run tests any time:

```
python3 -m unittest discover -s tests -v
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
  roughly 10M characters/language. Note the actual character count
  depends on what's in the real Europarl archives once downloaded --
  the manifest.json written by build_dataset.py reports the true count
  per language per run so this can be checked/adjusted.
- **Document granularity:** one Europarl `<CHAPTER>` (one parliamentary
  debate/session) = one document, for the purposes of the document-level
  train/test split. This is the natural document unit Europarl's raw
  release provides. If a raw file unexpectedly has no `<CHAPTER>` tags,
  the code falls back to fixed-size pseudo-documents and prints a loud
  warning -- worth double-checking against that warning once real files
  are downloaded, in case the release format differs from what's
  assumed here.
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

## Testing

`tests/test_preprocess.py` covers: chapter-boundary parsing (incl. the
no-`<CHAPTER>`-tags fallback and its warning), NFC normalisation
(including that combining-character sequences collapse to the same id as
their precomposed form), lowercasing, vocab building (rare-char -> UNK
folding, shared-across-languages behaviour), document-level split
(no document in both splits, reproducible given a seed), and windowing
(correct shape, and that windows never cross a document boundary).

The full pipeline was additionally smoke-tested end-to-end against a
small synthetic multi-language corpus (fake `<CHAPTER>`-tagged text) to
confirm `build_dataset.py` runs cleanly from raw archive to saved
`.npy` files -- that test data was generated, checked, and deleted, not
committed.
