# Session transcript: auditing the write-up against the codebase

**Date range:** 2026-10-01 to 2026-10-02. The session started on 2026-10-01. The session clock rolled over to 2026-10-02 before the final export request. Per-message timestamps were not available to the assistant, so no finer range is given.

**Session type:** An interactive Claude (Cowork, desktop app) session linked to the author's computer. Its purpose was to check the *Language Cadence: A Null Result* write-up for claims unsupported by the codebase and results, then fix them.

**Session ID:** https://claude.ai/code/session_01HfsjLU936fEKJ5fsZG9enG

**Completeness note:** This is **not** a complete verbatim transcript.
- **User prompts:** all eight are reproduced verbatim, including the final export request.
- **Assistant responses:** every assistant response is a condensed paraphrase written at export time. None of the assistant text is verbatim.
  - Raw tool output (file dumps, command output, search results) is summarized or omitted.
  - Short status messages are paraphrased.
- **Compaction:** the conversation had no auto-compaction as far as the assistant can tell. The condensed sections come from the full conversation context, not from an internal summary.
- **Omissions:** system reminders and tool schemas are omitted.

---

## 1. User

> Look inside the following path:
>
> [local path]/language-cadence-1
>
> Here you will find an ML research project. There is a codebase and an accompany write-up.
> Your task is to check that the information contained in the write-up is consistent with the codebase, i.e. that the write-up does not contain hallucinations. Ask me before making any changes.

## 1. Assistant (condensed)

**Access.** The session's connected folder was `language-cadence-2`, not `language-cadence-1`. The assistant requested access to `language-cadence-1`, and access was granted.

**What was read.**
- `writeup/writeup.tex`
- `README.md`
- `docs/DESIGN_LOG.md`
- The data manifest, vocabulary and results files
- The model, positional-encoding, preprocessing, training-loop and sweep scripts

**Recomputation.** The assistant recomputed from the saved results and code:
- the per-language losses
- the distinguishability scores
- the Pearson correlation
- the local minima
- the UNK fraction in the processed windows

It also checked three citations on the web.

**A progress message was sent to the user midway:** "Still working: I've read the write-up and most of the code. Next I'm checking the numbers against the raw results, then the preprocessing and training loop." A second one followed: "Nearly done: I've checked the numbers and the citations. I'm now writing up the findings. I haven't changed anything."

**What matched.**
- All 10 table values (P = 384 and P = 45, five languages).
- 70 runs, 14 P values, 1 seed, 3000 steps, batch 16, D = 64, 4 layers, 4 heads, concatenation injection, N = 384.
- Distinguishability peak at P = 45 (score 8.895) and score 1.738 at P = 384. The peak scales at about 1.4H.
- Pearson r = +0.930 between distinguishability and mean test loss. Mean loss is 2.123 at P = 45 and 1.808 at P = 384.
- The list of interior local minima per language.
- The 64-symbol vocabulary breakdown, the 500-sentence pseudo-documents, the 10% document-level split, and the test-set sizes.
- The worked P = 8 matrix.
- Citations: Ghosh et al. (2024) exists and is summarized accurately (22 languages, 5 tasks). Stojanović (2013) exists and is summarized accurately (21 languages, transcribed written text, rhythm metrics track phonotactic complexity).

**What did not match** (reported to the user):

1. "Test loss decreases monotonically for every language" is false for all five languages. For English, the loss is 2.128 at P = 32, 2.182 at P = 45 and 2.155 at P = 64. The write-up's own paragraph on local dips contradicts the claim.
2. "P = 384 wins by a wide margin every time" is overstated. The margin over P = 256 is 0.021 for Italian and 0.047 for German, about the size of the dips attributed to noise. Italian's margin of P = 45 over P = 64 is only 0.0006.
3. The calibration pilot was 8 runs (7 additive amplitudes plus concatenation), not 7. It used a smaller model (2 layers, context 128, 2000 steps) than the sweep, which the write-up did not say. Amplitudes 2 and 4 agree to within about 0.002, not identically to three decimals.
4. The newline that joins sentences is not in the vocabulary, so every sentence boundary is encoded as `<UNK>`. UNK is about 0.6–0.7% of characters. Letters such as ç and ì are also folded into UNK.
5. Undisclosed dataset details:
   - only the first 20 MB of raw text per language is read;
   - English is taken from the French–English file only;
   - windows are 385 characters (384 inputs plus the target).
6. The write-up (and README and design log) say the encoding was "designed to penalize long periods." No code does this. Distinguishability is a measurement made after the fact, a geometric consequence of H = 32 with harmonic folding.
7. Related work: the formal-language claim (period-2 PE solves (aa)* but fails on (aaaa)*) has no citation. For Pellegrino et al. (2011), the "near-identical languages" wording could not be verified because the paper could not be opened. The assistant recalled seven languages, including Japanese and Mandarin.
8. README links `scripts/inference/`, which the latest commit ("Tweaks") deleted. `scripts/analysis/` and `scripts/pilot/` are empty directories.

**Smaller points.** The LayerNorm explanation for amplitude saturation is only approximate. `config.py` and the design log also say "7 pilot runs".

**Closing.** The assistant offered to fix items 1–3 in the write-up first, and asked whether to update the README and design log as well. **No changes were made at this stage.**

---

## 2. User

> Yes do all of that

## 2. Assistant (condensed)

The device tools briefly disconnected and then reconnected. The assistant retried and continued.

**Write-up (`writeup.tex`).**
- **Monotonic claim:** the abstract, results text, figure caption and conclusion now say the loss trends downward and reaches its minimum at P = 384 for every language, with small local dips.
- **P = 384 margin:** the results text now gives the margin over P = 256 (0.021 Italian to 0.176 English) and notes that the Italian and German margins are comparable to the noise-level dips. The 0.0006 Italian margin over P = 64 at P = 45 is stated.
- **Distinguishability:** now described as a geometric consequence of the construction that the code only measures after the fact. The "designed to penalize" wording was removed.
- **Pilot:** now described as 8 runs on a smaller model. The amplitude 2 vs 4 statement now says the curves agree to within about 0.002.
- **Tokenizer:** documents UNK handling of newlines and of some letters (e.g. ç, ì), with the 0.6–0.7% figure.
- **Dataset:** documents the 20 MB prefix, the English source file and the 385-character windows.
- **Related work:** the (aa)* sentence was marked "reference to be added". Pellegrino was softened to "a set of languages that includes" the five. The Ghosh et al. language count was left at "22–23" for the moment.

**Other files.**
- `README.md`: removed the dead `scripts/inference/` link.
- `docs/DESIGN_LOG.md`:
  - corrected the pilot count and noted the smaller pilot model;
  - fixed the monotonic and "wide margin" wording;
  - reworded the distinguishability passage;
  - marked the inference-GUI section as removed.
- `config.py`: comment now says 8 pilot runs.

**Verification.** The LaTeX source compiled and the PDF was rebuilt with no warnings. A `.git/index.lock` warning appeared but did not affect the edits.

**Left open for the user.**
- The (aa)* reference.
- The Pellegrino language list, still unverified.
- Stray LaTeX build files that the assistant could not delete.
- The edits were not committed.

---

## 3. User

> Look further for the (aa)* result

## 3. Assistant (condensed)

The assistant ran several web searches and read candidate papers. **No source states the claim.**

Related results that were verified:
- **Yang, Chiang & Angluin, "Masked Hard-Attention Transformers Recognize Exactly the Star-Free Languages" (arXiv 2310.13897).** Corollary 8 shows that masked hard-attention transformers with rational sinusoidal (periodic) position embeddings recognize exactly the regular languages in AC⁰. These are the languages definable with modular position predicates. This is a general theorem, not the specific (aa)* vs (aaaa)* example. The version read was labelled a preprint, so no venue was confirmed.
- **Chiang & Cholak (ACL 2022), "Overcoming a Theoretical Limitation of Self-Attention."** They use a period-2 cosine positional encoding as part of a parity construction. They do not compare periods or discuss (aaaa)*.
- arXiv 2506.06398 and 2605.07705 were checked and do not contain the result.

The assistant proposed replacing the sentence with the accurate Yang et al. result plus the Chiang & Cholak example, and asked whether the user remembered the source. **No edit was made.**

---

## 4. User

> No, I think the (aa)* must be a hallucination. Remove all references to it, and check that the resulting file still makes sense

## 4. Assistant (condensed)

A search of the whole repo found one remaining mention, the Related Work item. The assistant deleted it and rebuilt the PDF with no errors. The section now lists three precedents (Ghosh, Stojanović, Pellegrino), and the introductory sentence and the cross-reference from the Stojanović item still read correctly. No other mention of formal-language results exists in the repo. The assistant noted that the Ghosh item still said "22–23 languages" while the abstract checked said 22, and offered to change it.

---

## 5. User

> Change to 22.
>
> Is that everything?

## 5. Assistant (condensed)

Changed to "22 languages" in `writeup.tex` and the rebuilt PDF. The assistant listed what remained:
- stray LaTeX build files it could not delete;
- the Pellegrino wording, still a hedge because the paper could not be opened;
- unchecked items (the intro's claims about the phonetics literature on isochrony, and whether the figures in the PDF still match the data);
- nothing committed yet.

---

## 6. User

> Remove the build files: I'll give you the permission

## 6. Assistant (condensed)

The assistant requested delete permission for the repo folder, and it was granted. It checked which files git tracks (`references.bib`, `writeup.bbl`, `writeup.pdf`, `writeup.tex`) and deleted the untracked build files: `.aux`, `.log`, `.fdb_latexmk`, `.fls`, `.out`, `.blg` and `tmp1.tmp`. Two more empty files, `tmp2.tmp` and `tmp3.tmp`, had appeared during rebuilds and were also removed. `writeup/` now holds only the four tracked files. Deletion remained enabled for that folder for the rest of the session, and nothing else was deleted.

---

## 7. User

> Now commit locally

## 7. Assistant (condensed)

A stale, empty `.git/index.lock` was blocking git. It had been left by an earlier `git status` in this session, and no git process was running, so the assistant removed it. It then staged the five modified files and committed them locally as `aa9ba7b` ("Align write-up with code and results"). The working tree was clean afterwards. **Nothing was pushed.**

The commit message summarizes the write-up changes and lists the README, DESIGN_LOG and config.py fixes. It ends with the session's attribution lines.

---

## 8. User

> This session is wrapping up. Before we're done, export this entire conversation as a clean, readable Markdown transcript — it's going into a public GitHub repo's prompt_history/ folder, as part of disclosing how this project was built.
>
> Format:
>
> My prompts verbatim, exactly as I wrote them.
> Your own responses condensed wherever they're dominated by raw tool output, long file dumps, or exploratory back-and-forth — but preserve the actual substance: real decisions made, numbers found, reasoning followed, and anything explicitly left open or deferred.
> Start the file with a short header block: the date range this conversation actually covered (from the real first/last messages — don't guess or round), a one-line description of what kind of session this was and what it was for, a link or ID for this session if you have one (omit if you don't), and a completeness note — is this a complete verbatim transcript, or is any part of it (e.g. anything before an auto-compaction) a reconstruction from an internal summary rather than the original wording? If so, say exactly which part, and don't present a reconstructed section as if it were verbatim.
>
> If you have file/shell tools: save the result as a new file at the repo root, prompt_history/YYYY-MM-DD_<short-label>.md (pick the date from when this session actually started, and a short label describing what it did) — just write the file, do not git commit it, and do not push anything.
>
> If you don't have file tools (e.g. this is a plain chat): just give me the Markdown directly in your reply instead, and I'll save it myself.

## 8. Assistant (condensed)

The assistant wrote this file to `prompt_history/2026-10-01_writeup-vs-code-audit.md` at the repo root. It was not committed or pushed.

---

## Open items at the end of the session

- The Pellegrino et al. (2011) language list was never verified.
- The intro's claims about the phonetics literature on isochrony were not checked.
- It was not checked whether the figures in the PDF still match the data.
- Delete permission remained enabled for the repo folder until the session ended.
- The commit `aa9ba7b` and this transcript file are local only. The commit is unpushed, and this file is uncommitted.
