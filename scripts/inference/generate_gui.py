"""
generate_gui.py
================

A small desktop GUI for running inference on a trained Language Cadence
checkpoint (see train_checkpoint.py): type an optional starting prompt,
hit Generate, and the model keeps producing text, one character at a
time, until you hit Stop. Built on tkinter (Python's standard-library
GUI toolkit) so there's nothing extra to install.

Generation runs on a background thread so the window stays responsive
(you can hit Stop immediately, scroll, resize, etc. while text is still
streaming in) and only ever talks back to the GUI thread through a
thread-safe queue, which the main loop drains on a timer -- the
standard pattern for "long-running work + live-updating tkinter UI".

tkinter is only imported inside App.__init__, not at module level, so
load_model()/generate_stream() below can still be imported and used
headlessly (tests, a notebook, a CLI script) on a Python install that
happens to lack tkinter -- common on minimal Linux setups even though
it ships with the standard macOS/Windows installers.

Usage:
    python3 scripts/inference/generate_gui.py
    python3 scripts/inference/generate_gui.py --checkpoint checkpoints/en_P384.pt
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from language_cadence.data.preprocess import Vocab  # noqa: E402
from language_cadence.model.config import ModelConfig  # noqa: E402
from language_cadence.model.gpt import GPT  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "en_P384.pt"


def load_model(checkpoint_path: Path) -> tuple[GPT, Vocab, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    vocab = Vocab.from_json(ckpt["vocab"])
    return model, vocab, ckpt


@torch.no_grad()
def generate_stream(model: GPT, vocab: Vocab, prompt: str, temperature: float, top_k: int, stop_event: threading.Event):
    """
    Yields one generated character at a time, forever (until
    stop_event is set). Sampling, not greedy argmax: temperature
    rescales the logits before softmax (lower = more conservative/
    repetitive, higher = more random) and top_k restricts sampling to
    the k highest-probability characters at each step, a standard way
    to avoid an occasional very-low-probability character derailing
    the whole continuation.

    The model has a fixed max_seq_len (it was trained on fixed-length
    windows -- see ModelConfig/GPT.forward). "Infinite" generation
    past that length works by always feeding only the most recent
    max_seq_len characters as context (a sliding window), exactly the
    same context size the model was trained with, rather than trying
    to feed it an ever-growing sequence it was never trained to handle.
    """
    max_seq_len = model.config.max_seq_len
    ids = vocab.encode(prompt) if prompt else [vocab.char_to_id.get(" ", 0)]

    while not stop_event.is_set():
        context = ids[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long)
        logits = model(x)[0, -1]  # (vocab_size,) -- next-char distribution after the last context position
        logits = logits / max(temperature, 1e-6)

        if top_k and top_k < logits.shape[-1]:
            top_values, top_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_values, dim=-1)
            next_id = top_indices[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        ids.append(next_id)
        yield vocab.id_to_char[next_id]


class GenerationWorker:
    """Runs generate_stream on a background thread, pushing each new
    character into a queue the GUI thread polls -- keeps the tkinter
    main loop (which must stay on the main thread) free to repaint and
    respond to the Stop button immediately, instead of blocking inside
    one long model.forward()-calling Python loop."""

    def __init__(self, model: GPT, vocab: Vocab):
        self.model = model
        self.vocab = vocab
        self.char_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, prompt: str, temperature: float, top_k: int) -> None:
        self.stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(prompt, temperature, top_k), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self, prompt: str, temperature: float, top_k: int) -> None:
        try:
            for ch in generate_stream(self.model, self.vocab, prompt, temperature, top_k, self.stop_event):
                self.char_queue.put(ch)
        except Exception as exc:  # surfaces any runtime error in the GUI instead of a silent dead thread
            self.char_queue.put(f"\n[generation error: {exc}]")
            self.stop_event.set()


class App:
    POLL_MS = 30  # how often the GUI drains the character queue -- frequent enough to feel live

    # Explicit colors throughout, rather than relying on ttk's theme
    # defaults: ttk's native themes (e.g. "aqua" on macOS) render some
    # widgets -- buttons especially -- with the OS's own light/dark-mode
    # colors and largely ignore foreground/background options passed to
    # the widget, which is how a dark-mode system can end up rendering
    # white text on a white background. Switching to the "clam" theme
    # (a cross-platform, Tcl-only theme, not a native one) makes ttk
    # actually respect the colors set below, on any platform.
    BG = "#f5f5f5"
    FG = "#111111"
    MUTED_FG = "#555555"
    TEXT_BG = "#ffffff"
    TEXT_FG = "#111111"
    BUTTON_BG = "#e4e4e4"

    def __init__(self, root, checkpoint_path: Path):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        root.title("Language Cadence -- Inference")
        root.geometry("720x520")
        root.configure(bg=self.BG)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED_FG)
        style.configure("TButton", background=self.BUTTON_BG, foreground=self.FG)
        style.map(
            "TButton",
            background=[("disabled", self.BG), ("active", "#d0d0d0")],
            foreground=[("disabled", self.MUTED_FG)],
        )
        style.configure("TSpinbox", fieldbackground=self.TEXT_BG, foreground=self.TEXT_FG)

        self.model, self.vocab, ckpt = load_model(checkpoint_path)
        self.worker = GenerationWorker(self.model, self.vocab)

        info = (
            f"checkpoint: {checkpoint_path.name}   "
            f"language: {ckpt.get('language', '?')}   "
            f"P: {ckpt['config']['P']}   "
            f"steps trained: {ckpt.get('steps', '?')}"
        )
        ttk.Label(root, text=info, style="Muted.TLabel").pack(anchor="w", padx=10, pady=(8, 0))

        prompt_frame = ttk.Frame(root)
        prompt_frame.pack(fill="x", padx=10, pady=8)
        ttk.Label(prompt_frame, text="Prompt (optional):").pack(anchor="w")
        self.prompt_entry = tk.Text(
            prompt_frame, height=3, wrap="word",
            bg=self.TEXT_BG, fg=self.TEXT_FG, insertbackground=self.TEXT_FG,
        )
        self.prompt_entry.pack(fill="x", pady=(2, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=10, pady=4)

        ttk.Label(controls, text="Temperature:").grid(row=0, column=0, sticky="w")
        self.temperature_var = tk.DoubleVar(value=0.8)
        ttk.Scale(controls, from_=0.1, to=2.0, variable=self.temperature_var, orient="horizontal", length=160).grid(row=0, column=1, padx=(4, 16))

        ttk.Label(controls, text="Top-k:").grid(row=0, column=2, sticky="w")
        self.top_k_var = tk.IntVar(value=10)
        ttk.Spinbox(controls, from_=1, to=64, textvariable=self.top_k_var, width=5).grid(row=0, column=3, padx=(4, 16))

        self.generate_button = ttk.Button(controls, text="Generate", command=self.on_generate)
        self.generate_button.grid(row=0, column=4, padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop", command=self.on_stop, state="disabled")
        self.stop_button.grid(row=0, column=5, padx=(0, 8))
        self.clear_button = ttk.Button(controls, text="Clear output", command=self.on_clear)
        self.clear_button.grid(row=0, column=6)

        output_frame = ttk.Frame(root)
        output_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.output_text = tk.Text(
            output_frame, wrap="word", state="disabled",
            bg=self.TEXT_BG, fg=self.TEXT_FG, insertbackground=self.TEXT_FG,
        )
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.root.after(self.POLL_MS, self._poll_queue)

    def on_generate(self) -> None:
        prompt = self.prompt_entry.get("1.0", "end").strip()
        self._append_output(prompt)
        self.worker.start(prompt, self.temperature_var.get(), self.top_k_var.get())
        self.generate_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def on_stop(self) -> None:
        self.worker.stop()
        self.generate_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def on_clear(self) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")

    def _append_output(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def _poll_queue(self) -> None:
        drained = []
        try:
            while True:
                drained.append(self.worker.char_queue.get_nowait())
        except queue.Empty:
            pass
        if drained:
            self._append_output("".join(drained))
        self.root.after(self.POLL_MS, self._poll_queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="path to a checkpoint saved by train_checkpoint.py")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train one first, e.g.:\n"
            f"  PYTHONPATH=src python3 scripts/inference/train_checkpoint.py --out {checkpoint_path}"
        )

    import tkinter as tk

    root = tk.Tk()
    App(root, checkpoint_path)
    root.mainloop()


if __name__ == "__main__":
    main()
