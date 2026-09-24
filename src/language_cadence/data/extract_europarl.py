"""
extract_europarl.py
====================

Turns the raw Europarl v7 archives (as downloaded from statmt.org) into
plain Python data structures we can preprocess: for each target language,
a list of "documents", where each document is itself a list of sentence
strings.

Why "documents" matter
-----------------------
The project notes are explicit that the train/test split must happen at
the *document* level, not the sentence level. If we split at the sentence
level, near-duplicate or highly-correlated sentences from the same debate
could end up on both sides of the split, which would let the model
partially "memorise" test sentences via train sentences from the same
document (leakage) rather than genuinely generalising. Europarl gives us
a natural document unit: each parliamentary debate/session is wrapped in
a `<CHAPTER ID=...>` tag in the raw release. We treat one <CHAPTER> as one
document.

Europarl file format (verified against the real statmt.org release)
-----------------------------------------------------------------------
Each archive (e.g. de-en.tgz) is a .tar.gz containing two plain-text
files, one per language in the pair (e.g. europarl-v7.de-en.de and
europarl-v7.de-en.en). Each file is UTF-8, ONE SENTENCE PER LINE, with
NO markup and no explicit document/chapter boundaries -- checked
directly against the downloaded archives. (An earlier version of this
docstring assumed inline <CHAPTER>/<SPEAKER> SGML tags, based on
half-remembered details of other Europarl-adjacent releases; that
assumption was wrong for this release and has been corrected here.)

Since there is no real document boundary information in the file at
all, "documents" for the purposes of the leakage-safe train/test split
(see preprocess.split_documents) are constructed as fixed-size chunks
of FALLBACK_PSEUDO_DOC_SENTENCES consecutive sentences, in their
original order. This isn't a true parliamentary session/debate boundary,
but it still serves the actual goal: sentences that are topically or
stylistically correlated (adjacent lines from the same speech) stay
together on one side of the split, rather than being scattered randomly
across train and test.

The code below still also recognises `<CHAPTER ...>` tags as document
boundaries IF a file happens to contain them (e.g. if you point it at a
differently-formatted Europarl-family release some day) -- in that case
real chapter boundaries are used instead of the fixed-size fallback, and
no warning is printed.
"""

from __future__ import annotations

import io
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# How many consecutive sentences make up one pseudo-document in the
# fallback path (used only if a file has no <CHAPTER> tags at all).
FALLBACK_PSEUDO_DOC_SENTENCES = 500

# Europarl v7 ships as language-pair archives, always paired with English.
# To cover en/fr/de/it/es we need these four archives; English text is
# taken from exactly one of them (see `EUROPARL_PAIRS` below) rather than
# merged across all four, to avoid dealing with near-duplicate English
# sentences that appear (with small translation-pipeline differences)
# across multiple pair files. This is a deliberate simplification, noted
# in docs/DESIGN_LOG.md, and doesn't affect the four non-English
# languages at all.
EUROPARL_PAIRS = {
    "de": "de-en",
    "fr": "fr-en",
    "it": "it-en",
    "es": "es-en",
}
ENGLISH_SOURCE_PAIR = "fr-en"  # which pair's English side we use for "en"


@dataclass
class Document:
    """One Europarl <CHAPTER>, as a list of sentence strings (in order)."""

    doc_id: str
    sentences: list[str]

    def char_length(self) -> int:
        # +1 per sentence for the newline we'll join them with later.
        return sum(len(s) + 1 for s in self.sentences)


def _member_text_for_language(
    tgz_path: Path, lang_code: str, max_bytes: int | None = None
) -> str:
    """
    Open a Europarl .tgz archive and return the decoded text of the
    member file for the given language (e.g. lang_code="de" finds the
    member ending in ".de").

    max_bytes: if given, read at most this many raw bytes from the
    (streaming-decompressed) member instead of the whole file, then trim
    back to the last complete line so we never return a truncated
    sentence. The real Europarl language files are far larger than this
    laptop-scale project needs (tens of millions of characters), so
    capping how much we even decompress/read is a meaningful speedup, not
    just a memory optimisation -- gzip decompression + tarfile only do
    work for the bytes actually read off the stream.
    """
    with tarfile.open(tgz_path, mode="r:gz") as tar:
        candidates = [
            m for m in tar.getmembers() if m.isfile() and m.name.endswith(f".{lang_code}")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No member ending in '.{lang_code}' found inside {tgz_path}. "
                f"Archive contains: {[m.name for m in tar.getmembers()][:10]}..."
            )
        if len(candidates) > 1:
            # Shouldn't happen for a standard Europarl pair archive, but
            # fail loudly rather than silently picking the wrong one.
            raise RuntimeError(
                f"Multiple members ending in '.{lang_code}' found in {tgz_path}: "
                f"{[m.name for m in candidates]}"
            )
        member = candidates[0]
        fileobj = tar.extractfile(member)
        if max_bytes is None:
            raw_bytes = fileobj.read()
        else:
            raw_bytes = fileobj.read(max_bytes)
            last_newline = raw_bytes.rfind(b"\n")
            if last_newline != -1:
                raw_bytes = raw_bytes[: last_newline + 1]
    return raw_bytes.decode("utf-8", errors="replace")


def parse_documents(raw_text: str, source_name: str = "<unknown>") -> list[Document]:
    """
    Parse the raw Europarl text of a single language file into a list of
    Document objects, splitting on <CHAPTER ...> tags. Pure function of
    the text, so it's easy to unit test with a small in-memory fixture.
    """
    documents: list[Document] = []
    current_sentences: list[str] = []
    current_id = None
    saw_any_chapter_tag = False

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("<CHAPTER"):
            saw_any_chapter_tag = True
            # Flush the previous document (if it had any content).
            if current_sentences:
                documents.append(Document(doc_id=current_id, sentences=current_sentences))
            # e.g. '<CHAPTER ID=1>' -> we just use a running index as the
            # id; the exact numbering scheme doesn't matter, only that
            # every document gets a distinct id for the split logic.
            current_id = f"{source_name}:chapter{len(documents) + 1}"
            current_sentences = []
        elif line.startswith("<"):
            # Other markup (<SPEAKER ...>, <P>, </P>, etc.) -- drop the
            # tag line itself, keep accumulating into the current document.
            continue
        else:
            current_sentences.append(line)

    if current_sentences:
        documents.append(Document(doc_id=current_id, sentences=current_sentences))

    if not saw_any_chapter_tag:
        # Expected for the standard Europarl v7 release (see module
        # docstring) -- not an error, just worth a quiet note of which
        # document-construction strategy is in effect for this file.
        print(
            f"[parse_documents] {source_name}: no <CHAPTER> tags (expected for "
            f"the standard Europarl v7 release) -- using pseudo-documents of "
            f"{FALLBACK_PSEUDO_DOC_SENTENCES} consecutive sentences each as the "
            f"document unit for the train/test split."
        )
        # Re-flatten and re-chunk into fixed-size pseudo-documents.
        all_sentences = [s for doc in documents for s in doc.sentences] or current_sentences
        documents = [
            Document(
                doc_id=f"{source_name}:pseudo{i // FALLBACK_PSEUDO_DOC_SENTENCES}",
                sentences=all_sentences[i : i + FALLBACK_PSEUDO_DOC_SENTENCES],
            )
            for i in range(0, len(all_sentences), FALLBACK_PSEUDO_DOC_SENTENCES)
        ]

    return documents


def load_language_documents(
    raw_dir: Path, lang_code: str, max_bytes: int | None = None
) -> list[Document]:
    """
    Load all Document objects for one of our five target languages
    (en/fr/de/it/es) from the archives in `raw_dir`.

    max_bytes: see `_member_text_for_language` -- caps how much raw text
    is read/decompressed from the archive at all, rather than loading
    the full multi-tens-of-millions-of-character file and discarding
    most of it later. Pass None to read the whole file.
    """
    if lang_code == "en":
        pair = ENGLISH_SOURCE_PAIR
    elif lang_code in EUROPARL_PAIRS:
        pair = EUROPARL_PAIRS[lang_code]
    else:
        raise ValueError(f"Unsupported language code: {lang_code!r}")

    tgz_path = raw_dir / f"{pair}.tgz"
    if not tgz_path.exists():
        raise FileNotFoundError(
            f"Expected archive not found: {tgz_path}. "
            f"Download it from https://www.statmt.org/europarl/v7/ and place it there."
        )

    raw_text = _member_text_for_language(tgz_path, lang_code, max_bytes=max_bytes)
    return parse_documents(raw_text, source_name=f"{pair}.{lang_code}")


def iter_all_languages(
    raw_dir: Path, lang_codes: list[str], max_bytes: int | None = None
) -> Iterator[tuple[str, list[Document]]]:
    """Convenience generator: yields (lang_code, documents) for each requested language."""
    for lang_code in lang_codes:
        yield lang_code, load_language_documents(raw_dir, lang_code, max_bytes=max_bytes)
