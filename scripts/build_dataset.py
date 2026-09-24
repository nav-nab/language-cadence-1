#!/usr/bin/env python3
"""
build_dataset.py
=================

Orchestrates Stage 1 end to end:

    raw Europarl .tgz archives (data/raw/)
        -> extract_europarl.load_language_documents   (per language)
        -> preprocess.normalise_text                   (NFC + lowercase)
        -> preprocess.build_vocab                       (shared, across languages)
        -> preprocess.build_language_dataset            (split, window, save)
        -> data/processed/{lang}/{train,test}_windows.npy + vocab.json + manifest.json

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --languages en de fr --window 384

Run it from anywhere -- paths are resolved relative to this file, not
the current working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src/` importable without requiring the package to be pip-installed.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from language_cadence.data.extract_europarl import Document, load_language_documents  # noqa: E402
from language_cadence.data.preprocess import (  # noqa: E402
    LanguageDatasetStats,
    build_language_dataset,
    build_vocab,
    normalise_text,
)

DEFAULT_LANGUAGES = ["en", "fr", "de", "it", "es"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    p.add_argument("--window", type=int, default=384, help="window_length N (excludes the +1 target char)")
    p.add_argument("--min-freq", type=float, default=1e-4, help="vocab rare-char cutoff, as a fraction of all chars")
    p.add_argument("--test-frac", type=float, default=0.1, help="fraction of documents held out per language")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-raw-bytes-per-lang",
        type=int,
        default=20_000_000,
        help=(
            "cap on raw bytes read from each language's Europarl file before any "
            "processing -- the full files are far larger than this laptop-scale "
            "project needs (target ~10M characters/language after preprocessing); "
            "reading a bounded prefix instead of the whole file is a large speedup. "
            "Pass 0 to read the whole file."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading and normalising documents for: {args.languages}")
    documents_by_lang: dict[str, list[Document]] = {}
    max_bytes = None if args.max_raw_bytes_per_lang == 0 else args.max_raw_bytes_per_lang
    for lang in args.languages:
        try:
            docs = load_language_documents(args.raw_dir, lang, max_bytes=max_bytes)
        except FileNotFoundError as e:
            print(f"\n[!] Skipping '{lang}': {e}\n")
            continue
        # Normalise sentences in place before anything else touches them,
        # so vocab-building and windowing both see already-normalised text.
        for doc in docs:
            doc.sentences = [normalise_text(s) for s in doc.sentences]
        documents_by_lang[lang] = docs
        n_chars = sum(d.char_length() for d in docs)
        print(f"  {lang}: {len(docs)} documents, {n_chars:,} characters")

    if not documents_by_lang:
        print(
            "\nNo languages loaded -- nothing to do. Check that the Europarl "
            f".tgz archives are in {args.raw_dir} (see data/raw/README.md)."
        )
        return

    print("\nBuilding shared vocabulary...")
    vocab = build_vocab(documents_by_lang, min_freq=args.min_freq)
    print(f"  vocab size (incl. <UNK>): {len(vocab)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.to_json(), f, ensure_ascii=False, indent=2)

    print("\nSplitting, windowing, and saving per language...")
    all_stats: list[LanguageDatasetStats] = []
    for lang, docs in documents_by_lang.items():
        stats = build_language_dataset(
            lang=lang,
            documents=docs,
            vocab=vocab,
            window_length=args.window,
            test_frac=args.test_frac,
            seed=args.seed,
            out_dir=args.out_dir,
        )
        all_stats.append(stats)
        print(
            f"  {lang}: {stats.n_train_docs} train docs / {stats.n_test_docs} test docs -> "
            f"{stats.n_train_windows:,} train windows / {stats.n_test_windows:,} test windows"
        )

    manifest = {
        "window_length": args.window,
        "min_freq": args.min_freq,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "vocab_size": len(vocab),
        "languages": {
            s.lang: {
                "n_train_docs": s.n_train_docs,
                "n_test_docs": s.n_test_docs,
                "n_train_windows": s.n_train_windows,
                "n_test_windows": s.n_test_windows,
                "n_train_chars": s.n_train_chars,
                "n_test_chars": s.n_test_chars,
            }
            for s in all_stats
        },
    }
    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Wrote vocab.json, manifest.json, and per-language windows to {args.out_dir}")


if __name__ == "__main__":
    main()
