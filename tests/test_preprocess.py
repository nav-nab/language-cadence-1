"""
Unit tests for the Stage 1 data pipeline. Pure stdlib `unittest` (no
pytest dependency required) so these run with just `python3 -m unittest`
even before any extra packages are installed.

Run with:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from language_cadence.data.extract_europarl import Document, parse_documents  # noqa: E402
from language_cadence.data.preprocess import (  # noqa: E402
    UNK_ID,
    build_vocab,
    documents_to_windows,
    normalise_text,
    split_documents,
)


class TestParseDocuments(unittest.TestCase):
    def test_splits_on_chapter_tags(self):
        raw = (
            "<CHAPTER ID=1>\n"
            "First sentence of chapter one.\n"
            "<SPEAKER ID=2 NAME=\"Someone\">\n"
            "Second sentence of chapter one.\n"
            "<CHAPTER ID=2>\n"
            "Only sentence of chapter two.\n"
        )
        docs = parse_documents(raw, source_name="test")
        self.assertEqual(len(docs), 2)
        self.assertEqual(
            docs[0].sentences,
            ["First sentence of chapter one.", "Second sentence of chapter one."],
        )
        self.assertEqual(docs[1].sentences, ["Only sentence of chapter two."])

    def test_falls_back_when_no_chapter_tags(self):
        # 3 sentences, well under the fallback chunk size, so we expect
        # exactly one pseudo-document containing all of them. This is the
        # NORMAL path for the real Europarl v7 release, which has no
        # <CHAPTER> tags at all -- verified against the actual downloaded
        # archives -- so it just prints a note, not a warning.
        raw = "Sentence one.\nSentence two.\nSentence three.\n"
        docs = parse_documents(raw, source_name="test")
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(docs[0].sentences), 3)

    def test_blank_lines_ignored(self):
        raw = "<CHAPTER ID=1>\nHello.\n\n   \nWorld.\n"
        docs = parse_documents(raw, source_name="test")
        self.assertEqual(docs[0].sentences, ["Hello.", "World."])


class TestNormalise(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalise_text("Bonjour Le Monde"), "bonjour le monde")

    def test_nfc_normalises_combining_characters(self):
        # "e" + combining acute accent (2 codepoints) should NFC-normalise
        # to the single precomposed "é" codepoint. This matters because
        # otherwise the same visible character could get two different
        # vocab ids depending on which unicode form happened to be used
        # in the source text.
        decomposed = "é"  # 'e' + COMBINING ACUTE ACCENT
        precomposed = "é"  # 'é'
        self.assertEqual(normalise_text(decomposed), precomposed)
        self.assertEqual(len(normalise_text(decomposed)), 1)


class TestVocab(unittest.TestCase):
    def test_rare_chars_folded_to_unk(self):
        # 'a' is common (999 occurrences), 'z' appears exactly once out of
        # 1000 characters total. min_freq=2e-3 -> threshold = 2.0
        # occurrences, so 'z' (count 1, below threshold) is dropped while
        # 'a' (count 999, above threshold) is kept.
        docs = [Document(doc_id="d1", sentences=["a" * 999 + "z"])]
        vocab = build_vocab({"en": docs}, min_freq=2e-3)
        self.assertIn("a", vocab.char_to_id)
        self.assertNotIn("z", vocab.char_to_id)
        encoded = vocab.encode("az")
        self.assertEqual(encoded[1], UNK_ID)

    def test_vocab_shared_across_languages(self):
        docs_en = [Document(doc_id="d1", sentences=["hello" * 100])]
        docs_fr = [Document(doc_id="d2", sentences=["ça va" * 100])]
        vocab = build_vocab({"en": docs_en, "fr": docs_fr}, min_freq=1e-3)
        # A character that's rare in English but common in French should
        # still get its own id, because the vocab is built from the
        # combined character counts across all languages.
        self.assertIn("ç", vocab.char_to_id)


class TestSplitDocuments(unittest.TestCase):
    def test_no_document_appears_in_both_splits(self):
        docs = [Document(doc_id=f"d{i}", sentences=[f"sentence {i}"]) for i in range(20)]
        split = split_documents(docs, test_frac=0.25, seed=0)
        train_ids = {d.doc_id for d in split.train}
        test_ids = {d.doc_id for d in split.test}
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(len(train_ids) + len(test_ids), 20)
        self.assertEqual(len(test_ids), 5)

    def test_reproducible_with_same_seed(self):
        docs = [Document(doc_id=f"d{i}", sentences=[f"s{i}"]) for i in range(50)]
        split_a = split_documents(docs, test_frac=0.2, seed=42)
        split_b = split_documents(docs, test_frac=0.2, seed=42)
        self.assertEqual([d.doc_id for d in split_a.test], [d.doc_id for d in split_b.test])


class TestWindowing(unittest.TestCase):
    def test_windows_are_nonoverlapping_and_correct_length(self):
        # vocab covering just 'a' and 'b'
        docs = [Document(doc_id="d1", sentences=["a" * 50])]
        vocab = build_vocab({"en": docs}, min_freq=0.0)
        windows = documents_to_windows(docs, vocab, window_length=10)
        # sentence "a"*50 + no extra newline (single sentence) = 50 chars
        # chunk_len = 11 -> floor(50/11) = 4 full windows, 6 chars dropped
        self.assertEqual(windows.shape, (4, 11))

    def test_windows_never_cross_document_boundary(self):
        # Two short documents that would combine to exactly one window's
        # worth of characters if (incorrectly) concatenated together --
        # but each is too short on its own to produce any window, so we
        # expect ZERO windows, proving they aren't being glued together.
        short_docs = [
            Document(doc_id="d1", sentences=["a" * 5]),
            Document(doc_id="d2", sentences=["b" * 5]),
        ]
        vocab = build_vocab({"en": short_docs}, min_freq=0.0)
        windows = documents_to_windows(short_docs, vocab, window_length=10)  # chunk_len=11
        self.assertEqual(windows.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
