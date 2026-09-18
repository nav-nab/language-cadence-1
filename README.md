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
- [x] Stage 3: model
- [ ] Stage 4: training/eval + sweep harness
- [ ] Stage 5: analysis

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
  model/
    config.py       ModelConfig dataclass -- ALL architecture + Pi-injection hyperparameters,
                     including the two Stage 3 open decisions (pe_injection, pe_amplitude)
    attention.py     causal multi-head self-attention
    mlp.py           feedforward sub-layer
    block.py         pre-norm transformer block (attention + feedforward, each with residual + LN)
    pe_injection.py  switches between additive (x + amplitude*Pi) and concat (Linear(concat(x,Pi)))
    gpt.py           full model: embedding -> Pi injection -> blocks -> final LN -> tied lm_head
scripts/
  build_dataset.py         Stage 1 CLI: raw archives -> processed windows
  pe_distinguishability.py Stage 2: reproduces the distinguishability-vs-P analysis, writes figures/
tests/
  test_preprocess.py       Stage 1 unit tests (stdlib unittest, no torch/pytest required)
  test_periodic_pe.py      Stage 2 unit tests, NumPy reference (incl. exact Pi(n)==Pi(n+P) check)
  test_torch_periodic_pe.py Stage 2 unit tests, PyTorch module (incl. cross-check vs NumPy reference)
  test_model.py            Stage 3 unit tests: shapes, real (not just present) causal masking,
                            gradient flow under both injection modes, parameter-count sanity
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
  ("additive" or "concat") and `ModelConfig.pe_amplitude`. Neither has
  been experimentally chosen yet -- current defaults (`"additive"`,
  amplitude `1.0`) are placeholders for getting the model running end to
  end, not a considered decision. Per the project notes, whichever
  combination is used must be held fixed across the whole P-sweep in
  Stage 4 to avoid confounding amplitude/injection-method with
  periodicity -- worth a explicit decision (and check-in) before Stage 4
  locks one in for the real sweep.
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

## Testing

42 unit tests total (`python3 -m unittest discover -s tests -v`), all
passing as of Stage 3:

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

Beyond unit tests, two things were checked against real (not synthetic)
data, not just automated: `scripts/build_dataset.py` run against the
actual downloaded Europarl archives (see `data/processed/manifest.json`
for the real per-language counts), and an ad hoc end-to-end script
(loaded a real `en` window batch via `CharWindowDataset`, trained a real
`GPT` on it repeatedly with plain Adam, no weight decay) confirming loss
drops from the random-guessing baseline (~ln(vocab_size) = 4.16) to
~0.01 -- i.e. the full data -> model -> loss -> backward -> step chain
is correctly wired, not just that shapes line up. That script wasn't
saved as a committed file (it was a one-off sanity check, not a
reusable tool) -- worth turning into a proper `scripts/` entry if it'd
be useful to re-run later, e.g. after future changes to the data
pipeline or model code.

The Stage 1 pipeline was additionally smoke-tested end-to-end against a
small synthetic multi-language corpus (fake `<CHAPTER>`-tagged text) to
confirm `build_dataset.py` runs cleanly from raw archive to saved
`.npy` files -- that test data was generated, checked, and deleted, not
committed.
