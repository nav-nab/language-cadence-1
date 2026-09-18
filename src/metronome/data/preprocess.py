"""
preprocess.py
=============

Turns the Documents produced by extract_europarl.py into the fixed-length
integer windows the model will actually train on. Four steps, in order:

  1. Text normalisation (NFC unicode normalisation + lowercasing), applied
     identically to every language so orthographic differences in *how*
     text is encoded don't leak into the "cadence" signal we're trying to
     measure.
  2. Vocabulary construction: a single character-level vocabulary shared
     across all five languages (decision: shared, not per-language --
     keeps the embedding table, and therefore D, identical across
     languages by construction). Rare characters (below a frequency
     threshold) are mapped to a single <UNK> symbol so the vocab stays
     close to the ~80-120 symbol target in the project notes instead of
     growing unboundedly with typos/stray unicode/rare punctuation.
  3. Document-level train/test split: whole documents (Europarl chapters)
     are assigned to train or test, never split partway -- this is what
     prevents leakage between train and test.
  4. Windowing: each document's character stream is cut into
     non-overlapping fixed-length windows of `window_length + 1`
     characters. The "+1" is deliberate: a window of length N+1 gives you
     N next-token-prediction pairs (input = window[:-1], target =
     window[1:]) without needing any separate padding logic. Any leftover
     tail of a document shorter than window_length + 1 is dropped rather
     than padded, which keeps every window "real" (no padding token that
     the model has to learn to ignore) at the cost of throwing away a
     small amount of data per document.

Everything in this file is pure Python + numpy -- deliberately no PyTorch
dependency, so the pipeline can be built and unit-tested even before
PyTorch is installed. The PyTorch `Dataset` wrapper around the .npy files
this module writes lives in torch_dataset.py.
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .extract_europarl import Document

# Special vocabulary symbols. Index 0 is reserved for <UNK> so that if a
# bug ever produces an all-zero array, it reads as "all unknown" (loud)
# rather than silently decoding as some particular common letter.
UNK_TOKEN = "<UNK>"
UNK_ID = 0


def normalise_text(text: str) -> str:
    """NFC-normalise and lowercase a string. Applied to every sentence,
    every language, identically -- this is the 'fixed preprocessing'
    methodology step from the project notes."""
    return unicodedata.normalize("NFC", text).lower()


@dataclass
class Vocab:
    """A character<->integer-id mapping, shared across all languages."""

    char_to_id: dict[str, int]
    id_to_char: list[str]

    def __len__(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(ch, UNK_ID) for ch in text]

    def to_json(self) -> dict:
        return {"id_to_char": self.id_to_char}

    @classmethod
    def from_json(cls, data: dict) -> "Vocab":
        id_to_char = data["id_to_char"]
        char_to_id = {ch: i for i, ch in enumerate(id_to_char)}
        return cls(char_to_id=char_to_id, id_to_char=id_to_char)


def build_vocab(
    documents_by_lang: dict[str, list[Document]],
    min_freq: float = 1e-4,
) -> Vocab:
    """
    Build a single shared vocabulary from (already-normalised) documents
    across all languages.

    min_freq: characters that make up less than this *fraction* of all
    characters seen (across all languages combined) are treated as rare
    and folded into <UNK> instead of getting their own vocab slot. E.g.
    min_freq=1e-4 means a character needs to appear in at least 1 out of
    every 10,000 characters, on average, to earn its own id.
    """
    counts: Counter[str] = Counter()
    total_chars = 0
    for documents in documents_by_lang.values():
        for doc in documents:
            for sentence in doc.sentences:
                counts.update(sentence)
                total_chars += len(sentence)

    threshold_count = min_freq * total_chars
    kept_chars = sorted(ch for ch, n in counts.items() if n >= threshold_count)

    id_to_char = [UNK_TOKEN] + kept_chars
    char_to_id = {ch: i for i, ch in enumerate(id_to_char)}

    dropped = sorted(ch for ch, n in counts.items() if n < threshold_count)
    if dropped:
        print(
            f"[build_vocab] kept {len(kept_chars)} chars, folded "
            f"{len(dropped)} rare chars into <UNK> "
            f"(threshold: {threshold_count:.1f} occurrences out of {total_chars})."
        )

    return Vocab(char_to_id=char_to_id, id_to_char=id_to_char)


@dataclass
class SplitDocuments:
    """Documents for one language, already divided into train/test."""

    train: list[Document]
    test: list[Document]


def split_documents(
    documents: list[Document],
    test_frac: float = 0.1,
    seed: int = 0,
) -> SplitDocuments:
    """
    Randomly assign whole documents to train/test (document-level split,
    per the project notes -- never split a single document across the
    boundary, since that's exactly the leakage the notes want to avoid).

    We shuffle with a fixed seed so the split is reproducible, and we
    shuffle a *copy* of the id order so this function has no side effects
    on the input list.
    """
    rng = random.Random(seed)
    order = list(range(len(documents)))
    rng.shuffle(order)

    n_test = max(1, round(len(documents) * test_frac)) if documents else 0
    test_idx = set(order[:n_test])

    train = [doc for i, doc in enumerate(documents) if i not in test_idx]
    test = [doc for i, doc in enumerate(documents) if i in test_idx]
    return SplitDocuments(train=train, test=test)


def documents_to_windows(
    documents: list[Document],
    vocab: Vocab,
    window_length: int,
) -> np.ndarray:
    """
    Concatenate each document's (normalised) sentences with newline
    separators, encode to vocab ids, and chunk into non-overlapping
    windows of length `window_length + 1`.

    Chunking is done *per document*, not by concatenating all documents
    together first -- this matters because we never want a window to
    straddle two unrelated documents (that would hand the model a fake
    "sentence" made of the tail of one debate glued to the head of
    another, which is not a real example of the language's structure).
    Any leftover characters at the end of a document that don't fill a
    full window are simply dropped.

    Returns an integer array of shape (num_windows, window_length + 1)
    and dtype uint8 (safe as long as the vocab has <= 256 symbols, which
    it will -- we're targeting ~80-120).
    """
    chunk_len = window_length + 1
    assert len(vocab) <= 256, "vocab grew past 256 symbols -- widen the dtype below"

    all_windows: list[np.ndarray] = []
    for doc in documents:
        text = "\n".join(normalise_text(s) for s in doc.sentences)
        ids = np.asarray(vocab.encode(text), dtype=np.uint8)
        n_full_chunks = len(ids) // chunk_len
        if n_full_chunks == 0:
            continue
        usable = ids[: n_full_chunks * chunk_len]
        all_windows.append(usable.reshape(n_full_chunks, chunk_len))

    if not all_windows:
        return np.zeros((0, chunk_len), dtype=np.uint8)
    return np.concatenate(all_windows, axis=0)


@dataclass
class LanguageDatasetStats:
    lang: str
    n_train_docs: int
    n_test_docs: int
    n_train_windows: int
    n_test_windows: int
    n_train_chars: int
    n_test_chars: int


def build_language_dataset(
    lang: str,
    documents: list[Document],
    vocab: Vocab,
    window_length: int,
    test_frac: float,
    seed: int,
    out_dir: Path,
) -> LanguageDatasetStats:
    """
    Full per-language pipeline: split documents, window each split, save
    both splits as .npy files under out_dir/{lang}/, and return summary
    stats for a human-readable report.
    """
    split = split_documents(documents, test_frac=test_frac, seed=seed)
    train_windows = documents_to_windows(split.train, vocab, window_length)
    test_windows = documents_to_windows(split.test, vocab, window_length)

    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    np.save(lang_dir / "train_windows.npy", train_windows)
    np.save(lang_dir / "test_windows.npy", test_windows)

    return LanguageDatasetStats(
        lang=lang,
        n_train_docs=len(split.train),
        n_test_docs=len(split.test),
        n_train_windows=int(train_windows.shape[0]),
        n_test_windows=int(test_windows.shape[0]),
        n_train_chars=sum(d.char_length() for d in split.train),
        n_test_chars=sum(d.char_length() for d in split.test),
    )
