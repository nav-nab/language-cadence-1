# Inference

Everything here is for playing with a trained model interactively, separate
from the Stage 1-5 research pipeline (which never saved model weights --
only loss numbers -- see [`docs/DESIGN_LOG.md`](../../docs/DESIGN_LOG.md)).

## Checkpoints

`train_checkpoint.py` trains a GPT and saves its weights to `checkpoints/`.
It's resumable by design, so it can be run in short chunks:

```
PYTHONPATH=src python3 scripts/inference/train_checkpoint.py \
    --steps 150 --out checkpoints/en_P384.pt
PYTHONPATH=src python3 scripts/inference/train_checkpoint.py \
    --steps 150 --resume-from checkpoints/en_P384.pt --out checkpoints/en_P384.pt
# ...repeat to train further
```

`checkpoints/en_P384.pt` (committed to the repo) was trained this way for
1200 steps on English at P=384 -- the sweep's own best-performing
configuration (see `results/sweep_summary.json`), but well short of the
3000-step budget the real sweep used per run. Treat its output as a proof
of concept, not representative generation quality: at 1200 steps it's
still producing word-fragment-level text rather than fluent English. Train
further with the commands above (each chunk is a few minutes) for better
quality.

## GUI

`generate_gui.py` is a small tkinter desktop app: type an optional prompt,
hit Generate, and the model streams out text one character at a time until
you hit Stop.

```
python3 scripts/inference/generate_gui.py
python3 scripts/inference/generate_gui.py --checkpoint checkpoints/en_P384.pt
```

tkinter ships with the standard python.org macOS/Windows installers; if
your Python is missing it (common on minimal Linux installs), install your
platform's `python3-tk` equivalent.

Generation always samples (temperature + top-k controls in the GUI), not
greedy decoding -- greedy decoding on a small character-level model tends
to loop. The model was trained on fixed-length windows (`max_seq_len`
characters), so "infinite" generation past that length works by always
feeding the most recent `max_seq_len` characters back in as context, not
by growing the context indefinitely.
