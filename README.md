# Language Cadence: A Null Result

Small laptop-scale ML research project testing whether tuning the period of
a *periodic* positional encoding, per language, improves GPT pretraining
loss. Read the full write-up: **[`writeup/writeup.pdf`](writeup/writeup.pdf)**.

> Positional encodings in transformer language models are usually
> language-agnostic: the original geometric encoding of Vaswani et al.
> (2017) uses a fixed spread of frequencies regardless of what is being
> modeled. This work asks a narrower question: if a strictly *periodic*
> positional encoding is used instead, does tuning its period per language
> improve next-token pretraining loss in an otherwise-fixed, from-scratch
> GPT? The idea has some loose motivation from linguistics — languages are
> sometimes claimed to differ in rhythmic timing — but the experiment is
> evaluated purely on pretraining loss, independent of whether that
> linguistic distinction holds up. We sweep the period across 14 values and
> five languages (English, French, German, Italian, Spanish; 70 runs total)
> and find no benefit from per-language tuning: for every language, test
> loss decreases monotonically as the period grows, all the way to the
> model's context length. This is a boundary effect rather than a genuine
> positional optimum — a long period simply gives the model more usable
> context within a fixed window, rather than encoding any exploitable
> periodic structure. Supporting this reading, a geometric property of the
> encoding explicitly designed to penalize very long periods turns out to
> be *positively* correlated with test loss across the sweep (Pearson
> r = +0.930), the opposite of what that design predicts. We report this
> as a null result and identify the character-level tokenizer used here as
> a likely reason the underlying idea was not given a fair test.

![Test loss vs. period, by language](figures/sweep_test_loss_vs_P.png)

## Contents

- **[`writeup/writeup.pdf`](writeup/writeup.pdf)** — the paper (start here).
- **[`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md)** — implementation log: how the
  idea was turned into code, stage by stage, and every design decision made
  along the way.
- **`src/language_cadence/`** — data pipeline, positional encoding module,
  model, and training loop.
- **`scripts/`** — CLI entry points for each stage (data build, sweep,
  analysis).
- **`data/`, `results/`, `figures/`** — processed data, raw sweep results,
  and generated figures.
- **`tests/`** — unit tests (`python3 -m pytest tests/ -v`).

## Reproducing

```
python3 scripts/build_dataset.py     # Stage 1: raw archives -> processed windows
python3 -m pytest tests/ -v          # run all unit tests
python3 scripts/train_sweep.py       # Stage 4: the real (language, P, seed) sweep
python3 scripts/analyze_sweep.py     # Stage 5: summary + headline figure
```

See [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) for full details on the
data, architecture, and every implementation decision.
