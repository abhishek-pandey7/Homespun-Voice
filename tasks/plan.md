# Implementation Plan: Homespun

## Overview

Homespun adapts open-weight speech recognition to **Awadhi**, a low-resource Indo-Aryan
dialect spoken across the Awadh region of Uttar Pradesh. General-purpose speech models
transcribe Awadhi by quietly translating it: local vocabulary becomes standard Hindi,
idiom is dropped, and the way people actually speak is flattened into newscaster
register. For dictation that scarcely matters. For recording how elders describe birth
customs, naming ceremonies and family tradition, it means the transcript is not what
was said.

A LoRA adapter is trained on `openai/whisper-small` over the **SpeeD-IA Awadhi corpus**,
measured against the stock model on the corpus authors' own held-out split, and then
used to drive a readable, narrated storybook of the life-cycle narratives the corpus
contains.

The project is two separable things: a reproducible ASR experiment, and a usable
artifact. Neither is allowed to prop up the other — a weak benchmark is reported as a
weak benchmark, and a polished reader does not stand in for a result.

## Corpus

**SpeeD-IA** (Speech Datasets for Indo-Aryan languages), Dr. Bhimrao Ambedkar University
and the Council for Strategic and Defence Research, with Karya Inc. and UnReaL-TecE LLP.
Published at the Speech for Social Good Workshop, Interspeech 2022.

- Transcriptions: `github.com/unrealtecellp/SpeeD-IA`
- Audio: Google Drive folder linked from that repository
- Licence: **CC BY-NC-SA 4.0** — attribution, non-commercial, share-alike

| Subset | train | test | total |
|---|---|---|---|
| lifecycle | 357 | 90 | 447 |
| translation | 1,713 | 429 | 2,142 |
| **total utterances** | **2,070** | **519** | **2,589** |

Audio arrives pre-segmented: one WAV plus one JSON per utterance, grouped by speaker.

### Why this corpus rather than scraped video

Three things it settles at once. The transcriptions already exist, which removes the
single largest risk in the original plan — hours of manual correction producing perhaps
25 minutes of labelled audio. The split is defined by the corpus authors, so results are
comparable to published work rather than to a split invented here. And the licence is
explicit and citable, so provenance is a fact in the manifest rather than an assumption.

### Licence obligations

CC BY-NC-SA 4.0 is not decorative. Three consequences the project must honour:

- **Attribution** — the Interspeech 2022 paper is cited in the README, the dataset card,
  the reader UI and any write-up.
- **Non-commercial** — the adapter and the artifact are not sold or used commercially.
- **Share-alike** — a model fine-tuned on this data, and a storybook derived from its
  transcripts, are both derivative works. Each carries CC BY-NC-SA 4.0. This is stated
  up front rather than discovered at publication.

## Architecture Decisions

- **Train locally.** RTX 4050 (6 GB). `whisper-small` is 244M params; a LoRA adapter on
  `q_proj`/`v_proj` in fp16 with batch 2-4 and gradient accumulation fits. No hosted
  notebook, no session timeouts, no dataset re-upload.
- **Python 3.12, pinned.** The 3.14 interpreter on `py` has no stable torch wheels.
- **No VAD stage.** The original plan chunked long recordings with Silero VAD. SpeeD-IA
  ships one file per utterance, already segmented by the collection app, so that stage
  is removed rather than kept as ceremony. Utterances exceeding Whisper's 30-second
  window are reported and excluded, not re-split.
- **The authors' split is used verbatim.** Re-splitting would break comparability and
  risk leaking a speaker across the boundary.
- **Metric plumbing before the model.** The evaluation harness is built and proven
  against the stock model before any adapter exists, so the numbers later come from code
  already known to work.
- **Baseline is measured on the real test set, once, and frozen.** It is the number the
  adapter must beat.
- **Verbatim quote preservation is a hard constraint on the story engine.** Gemma 2 may
  structure and connect; quoted Awadhi is copied exactly and checked programmatically. A
  generation step that silently normalises dialect would undo the entire point.
- **Static frontend, prebuilt audio.** No cold start, no keys in the deployed artifact.
- **Secrets never enter the repo.** Keys live in a gitignored `.env`.

## Dependency Graph

```
repo scaffold + pinned env  (T1, T2)  [done]
        |
        v
  corpus import (T3)
        |
        v
  corpus QA + manifest (T4)
        |
        +-------------------+
        v                   v
 baseline STT (T5)   eval harness (T6)
        +---------+---------+
                  v
         baseline frozen (T6)
                  v
         LoRA fine-tune (T7)
                  v
      benchmark baseline vs tuned (T8)
                  v
        story engine / Gemma 2 (T9)
                  v
       narration / ElevenLabs (T10)
                  v
         storybook reader (T11)
                  v
            deployment (T12)
```

## Task List

### Phase 0: Foundation
- [x] Task 1: Repository scaffold and ignore rules
- [x] Task 2: Pinned Python environment with GPU verification

### Phase 1: Corpus
- [x] Task 3: Import the SpeeD-IA Awadhi corpus
- [x] Task 4: Corpus QA, normalisation and manifest

### Checkpoint: Corpus

### Phase 2: Baseline and Measurement
- [x] Task 5: Baseline transcription with stock whisper-small
- [x] Task 6: WER/CER evaluation harness and frozen baseline

### Checkpoint: Baseline

### Phase 3: Adaptation
- [x] Task 7: LoRA fine-tune of whisper-small
- [x] Task 8: Baseline vs adapted benchmark report

### Checkpoint: Benchmark

### Phase 4: Story and Narration
- [x] Task 9: Gemma 2 story engine with verbatim quote preservation
- [x] Task 10: ElevenLabs narration

### Phase 5: Reader and Deployment
- [x] Task 11: Static storybook reader
- [ ] Task 12: Deployment to Render

### Checkpoint: Complete

Per-task acceptance criteria and verification steps are in `tasks/todo.md`.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Whisper's tokeniser has no Awadhi language id; forcing `hi` may cap achievable WER | **High** | Fix the language hint to `hi` for both baseline and adapted runs so the comparison stays fair; report it as a known ceiling rather than tuning it per-model. |
| Baseline WER on Awadhi may be so high (>80%) that the metric is noisy | High | Report CER alongside WER — it degrades more gracefully. Include a dialect-vocabulary recall count as a third, more interpretable measure. |
| `translation` and `lifecycle` subsets may differ in register and difficulty | Medium | Score them separately as well as pooled; never report one blended number. |
| Audio may vary in sample rate or channel count across speakers | Medium | T4 normalises everything to 16 kHz mono and reports any file that needed conversion. |
| Utterances longer than 30s silently truncate in Whisper | Medium | T4 flags and excludes them, with the count recorded in the dataset card. |
| 6 GB VRAM OOM during training | Medium | fp16, batch 2 with gradient accumulation, gradient checkpointing, encoder freeze if needed. |
| Google Drive throttles or partially completes the bulk download | Medium | `--continue` resumes; T4 verifies every transcript ID has a matching audio file and reports gaps rather than training on a silently short corpus. |
| Share-alike obligations overlooked at publication | Medium | Licence terms recorded in README, dataset card and reader UI during T4, not at the end. |
| ElevenLabs Devanagari pronunciation on Awadhi text | Medium | Audition one chapter before the full run; fall back to narrating a standardised variant while on-screen text stays verbatim. |
| Story engine normalises dialect quotes | Medium | Programmatic verbatim check; failures block the write. |

## Open Questions

1. **Gemma 2 size and host:** `gemma-2-2b-it` locally, or `gemma-2-9b-it` via a hosted
   endpoint? Affects T9 only; deferrable to Phase 4.
2. **Story source subset:** the `lifecycle` narratives (birth customs, naming ceremonies,
   tradition) are the natural storybook material. `translation` utterances are likely
   prompted sentences rather than narrative. Confirm after T4 inspection.
3. ~~Fate of the YouTube material in `data/raw/`.~~ **Resolved: deleted.** The corpus is
   now single-source and cleanly licensed. `scripts/ingest.py` is retained for importing
   directly recorded audio should any be added later; no current task depends on it.

## Definition of Done (project-wide)

- Code runs end to end from a clean checkout by following `README.md`.
- Scripts are re-runnable and idempotent; re-running does not corrupt prior output.
- No secrets, audio, or model weights committed.
- Every data-producing stage writes a manifest recording what it produced and from what.
- Reported metrics are reproducible from a committed script plus the corpus split.
- Attribution and licence terms are present wherever the data or its derivatives appear.
