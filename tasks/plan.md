# Implementation Plan: Vocalia

## Overview

Vocalia adapts open-weight speech recognition to a specific North Indian regional dialect so that spoken personal stories are transcribed as actually spoken, rather than normalised into standard Hindi. A parameter-efficient (LoRA) adapter is trained on a small, hand-corrected dialect corpus on top of `openai/whisper-small`, measured against the stock model on a held-out set, and then used to drive a readable storybook with synthesised narration and a small web reader.

The project is deliberately two things at once: a measurable ASR experiment (does the adapter actually help, and by how much) and a usable artifact (a story someone can read and listen to). The plan keeps those separable, so that a weak benchmark result does not invalidate the artifact, and a polished artifact cannot paper over a weak benchmark.

## Architecture Decisions

- **Train locally, not on hosted notebooks.** The machine has an RTX 4050 (6 GB VRAM). `whisper-small` is 244M params; a LoRA adapter on `q_proj`/`v_proj` in fp16 with batch size 2-4 and gradient accumulation fits inside 6 GB. This removes hosted-notebook session timeouts and dataset re-upload friction. A notebook is kept as a fallback path, not the primary one.
- **Python 3.12, pinned.** Python 3.14 is installed as `py`, but torch has no stable wheels for it. All tooling targets the 3.12 interpreter explicitly.
- **Walking skeleton before bulk labelling.** The first vertical slice pushes roughly 3 minutes of audio through every stage (ingest, VAD, draft STT, metric) before significant human transcription effort is spent. Labelling is the most expensive and least reversible input; the pipeline must be proven to consume it correctly first.
- **Single dialect target: rural UP/Bihar Hindi.** Decided. Awadhi and Bhojpuri are out of scope; splitting ~25 minutes across three varieties would leave too little signal per variety to move the metric.
- **Two speakers, multiple sessions each.** The corpus comes from two speakers in their seventies (one man, one woman) from rural UP. Recording several shorter sessions per speaker rather than one long one per speaker is a deliberate choice: it allows a held-out split where each test session shares a speaker with training, so the benchmark isolates dialect adaptation instead of speaker and room change.
- **Metric plumbing before the model.** The evaluation harness is built and exercised on a tiny hand-corrected sample early, so that the WER/CER numbers reported later come from code already known to work.
- **Split by recording, never by chunk.** Chunks from one recording share speaker, microphone and room. A random chunk-level split would leak and inflate the result.
- **Verbatim quote preservation is a hard constraint on the story engine.** Gemma 2 may structure, order and connect, but dialect utterances it is asked to preserve must be copied exactly. A generation step that silently "corrects" dialect would undo the entire point of the fine-tune.
- **Static frontend with prebuilt assets.** The storybook reader ships as static HTML/JS with audio generated ahead of time, so the hosted surface has no cold-start behaviour and no API keys in the deployed artifact.
- **Secrets never enter the repo.** API keys live in a local `.env` that is gitignored, and in the host's environment settings.

## Dependency Graph

```
repo scaffold + pinned env  (T1, T2)
        |
        v
  audio ingest (T3)
        |
        v
  VAD chunking (T4)
        |
        +-------------------+
        v                   v
 draft STT (T5)      eval harness (T6)
        |                   |
        +---------+---------+
                  v
         correction tooling (T7)
                  v
       gold dataset + split (T8)
                  v
         LoRA fine-tune (T9)
                  v
      benchmark baseline vs tuned (T10)
                  v
        story engine / Gemma 2 (T11)
                  v
       narration / ElevenLabs (T12)
                  v
         storybook frontend (T13)
                  v
            deployment (T14)
```

T6 depends only on T4 plus a few manually corrected chunks, so it can be built in parallel with T5.

## Task List

### Phase 0: Foundation
- [x] Task 1: Repository scaffold and ignore rules
- [ ] Task 2: Pinned Python environment with GPU verification

### Checkpoint: Foundation

### Phase 1: Walking Skeleton (~3 minutes of audio)
- [ ] Task 3: Audio ingest to 16 kHz mono WAV with a source manifest
- [ ] Task 4: Silero VAD chunking to 5-25s segments
- [ ] Task 5: Baseline transcription with stock whisper-small
- [ ] Task 6: WER/CER evaluation harness

### Checkpoint: Skeleton

### Phase 2: Gold-Standard Dataset
- [ ] Task 7: Transcript correction tool
- [ ] Task 8: Corrected corpus with leakage-safe train/test split

### Checkpoint: Dataset Frozen

### Phase 3: Adaptation and Measurement
- [ ] Task 9: LoRA fine-tune of whisper-small
- [ ] Task 10: Baseline vs adapted benchmark report

### Checkpoint: Benchmark

### Phase 4: Story and Narration
- [ ] Task 11: Gemma 2 story engine with verbatim quote preservation
- [ ] Task 12: ElevenLabs narration for finalised chapters

### Phase 5: Reader and Deployment
- [ ] Task 13: Static storybook reader
- [ ] Task 14: Deployment to Render

### Checkpoint: Complete

Full task detail, acceptance criteria and verification steps are in `tasks/todo.md`.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ~25 min of labelled audio is very small for ASR adaptation; WER may barely move or regress | **High** | Treat the measurement as the deliverable, not a guaranteed win. Report CER alongside WER (more sensitive at this scale) plus a dialect-vocabulary recall count. Freeze the test set before training. Publish the honest number either way. |
| Overfitting to a handful of recordings | High | Low LR (1e-4), `q_proj`/`v_proj` only, early stopping on held-out loss, small adapter rank (r=8-16). |
| Held-out recording differs in speaker or room, so the split measures domain shift rather than dialect gain | High | Prefer multiple recordings per speaker so at least one test recording shares a speaker with train. Report per-recording WER, never a single pooled number. |
| Manual correction of 25 min is the schedule bottleneck (realistically 6-10x audio duration) | High | Build T7 tooling before labelling; support Devanagari input; allow partial progress and resume. Scope down to 15 min if time runs short, and record that in the dataset card. |
| 6 GB VRAM OOM during training | Medium | fp16, batch 2 with gradient accumulation, gradient checkpointing, freeze the encoder if needed. Notebook fallback retained. |
| ElevenLabs Devanagari pronunciation quality on dialect text | Medium | Audition voices early in T12. Fall back to narrating a lightly standardised variant while keeping dialect text on screen verbatim. |
| `gemma-2-9b-it` too heavy for 6 GB local inference | Medium | Use a hosted endpoint for 9b, or run `gemma-2-2b-it` locally. Decide in T11. |
| Story engine silently normalises dialect quotes | Medium | Programmatic check: every quoted span in the output must appear verbatim in the source transcript, otherwise the step fails. |
| Source audio licensing and consent | Medium | Record a consenting speaker directly where possible. Otherwise keep a provenance manifest with URL, channel and licence per source, and do not redistribute raw audio in the repo. |

## Open Questions

**Resolved:**

- *Dialect target:* rural UP/Bihar Hindi.
- *Speakers:* two, approximately 70 years old, one man and one woman, from rural UP. Recorded directly with consent rather than sourced from public video, which also settles provenance and licensing.

**Still open:**

1. **Session structure for recording.** The plan assumes 6 sessions of roughly 5-8 minutes, 3 per speaker, so that train holds 2 sessions per speaker and test holds 1 per speaker. Fewer, longer sessions would weaken the split. Confirm this is practical with the speakers before T3.
2. **Gemma 2 size and host:** `gemma-2-2b-it` locally, or `gemma-2-9b-it` via a hosted endpoint? Affects T11 only; can be deferred to Phase 4.
3. **How much audio can realistically be hand-corrected?** This sets the ceiling on everything downstream. Expect 6-10x the audio duration in effort, so 25 minutes is a 3-4 hour sitting. An honest number now is worth more than an optimistic one.

**Recording notes (elderly speakers, affects T3 quality):** quiet room with no fan, TV or background conversation; phone or mic within about 30 cm; one speaker at a time with no overlapping speech; prompt with open questions about memory and let them talk uninterrupted. Record at the highest quality the device offers and downsample later — resampling down is lossless in effect, upsampling recovers nothing.

## Definition of Done (project-wide)

Every task clears this bar before it counts as done:

- Code runs end to end from a clean checkout by following `README.md`.
- Scripts are re-runnable and idempotent; re-running does not corrupt prior output.
- No secrets, raw audio, or model weights committed.
- Every data-producing stage writes a manifest recording what it produced and from what.
- Reported metrics are reproducible from a committed script plus the frozen test set.
