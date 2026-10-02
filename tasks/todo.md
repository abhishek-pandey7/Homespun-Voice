# Vocalia — Task List

Plan document: `tasks/plan.md`

---

## Phase 0: Foundation

## Task 1: Repository scaffold and ignore rules

**Description:** Create the directory layout, README skeleton, and ignore rules so that no later task has to guess where things belong. The ignore rules matter most: audio, model weights and secrets must be excluded before any of them exist, not after they are accidentally staged.

**Acceptance criteria:**
- [x] Directory layout exists: `src/vocalia/`, `scripts/`, `data/{raw,chunks,transcripts,gold,story}/`, `models/`, `reports/`, `configs/`, `prompts/`, `web/`
- [x] `.gitignore` excludes `data/`, `models/`, `*.wav`, `*.mp3`, `.env`, `__pycache__/`, `.venv/`
- [x] `data/` subdirectories are tracked as empty via `.gitkeep` so the layout survives a clean clone
- [x] `README.md` states what the project is, the single target dialect, and a run order placeholder
- [x] `.env.example` lists required variable names with empty values
- [x] `.gitattributes` normalises line endings to LF and marks audio/weights binary

**Verification:**
- [x] `git status --porcelain` shows no `.wav`, `.env`, or weight files as stageable after dropping dummy files into `data/raw/`, `data/chunks/`, `models/` and `web/assets/audio/`
- [x] `git check-ignore` confirms ignored paths ignored *and* manifest/`.gitkeep` paths still trackable
- [x] Forbidden-term scan over tracked files returns no matches

**Note:** `data/**` alone silently killed every negation beneath it — git does not descend into an ignored directory, so `!data/raw/manifest.jsonl` was dead until `!data/**/` was added to re-include directories. Verified empirically rather than assumed.

**Dependencies:** None

**Files likely touched:** `.gitignore`, `README.md`, `.env.example`, `data/**/.gitkeep`

**Estimated scope:** S

---

## Task 2: Pinned Python environment with GPU verification

**Description:** Pin dependencies against Python 3.12 and prove torch sees the GPU before any training code is written. A silent fall back to CPU would make T9 appear to work while taking 50x longer.

**Acceptance criteria:**
- [ ] `requirements.txt` pins torch (CUDA build), transformers, peft, datasets, accelerate, jiwer, silero-vad, soundfile, librosa, yt-dlp
- [ ] A `scripts/check_env.py` prints Python version, torch version, `torch.cuda.is_available()`, and the GPU name and total VRAM
- [ ] `yt-dlp` is installed and resolves on PATH
- [ ] README documents the exact interpreter used (3.12, not the 3.14 on `py`)

**Verification:**
- [ ] `python scripts/check_env.py` reports `cuda: True` and names the RTX 4050
- [ ] `yt-dlp --version` succeeds
- [ ] Manual check: a one-line torch matmul on `cuda` completes without error

**Dependencies:** Task 1

**Files likely touched:** `requirements.txt`, `scripts/check_env.py`, `README.md`

**Estimated scope:** S

---

### Checkpoint: Foundation
- [ ] Clean clone plus documented setup steps produces a working GPU-enabled environment
- [ ] No data or secrets are committable
- [ ] Target dialect decided and written into the README (Open Question 2 resolved)

---

## Phase 1: Walking Skeleton (~3 minutes of audio)

## Task 3: Audio ingest to 16 kHz mono WAV with a source manifest

**Description:** Fetch or import source audio and normalise it to 16 kHz mono WAV, recording provenance for every source. Provenance is part of the deliverable, not bookkeeping: the dataset card and the licensing position both depend on it.

**Acceptance criteria:**
- [ ] `scripts/ingest.py` accepts either a local file or a URL and writes `data/raw/<source_id>.wav` at 16 kHz mono
- [ ] `data/raw/manifest.jsonl` gains one row per source: `source_id`, origin (URL or "direct recording"), `speaker_id`, duration, licence or consent note, ingest timestamp
- [ ] `speaker_id` distinguishes the two speakers (e.g. `spk_m70`, `spk_f70`) so the T8 split can be speaker-aware
- [ ] Target intake: 6 sessions of roughly 5-8 minutes, 3 per speaker
- [ ] Re-running on an existing `source_id` is a no-op unless `--force` is passed
- [ ] Works on roughly 3 minutes of audio for the skeleton pass

**Verification:**
- [ ] `ffprobe` on an output file confirms 16000 Hz, 1 channel
- [ ] Manifest row count equals the number of files in `data/raw/`
- [ ] Manual check: audio plays back intelligibly, not sped up or truncated

**Dependencies:** Task 2

**Files likely touched:** `scripts/ingest.py`, `src/vocalia/manifest.py`, `data/raw/manifest.jsonl`

**Estimated scope:** M

---

## Task 4: Silero VAD chunking to 5-25s segments

**Description:** Split each source into speech-only chunks of 5-25 seconds using Silero VAD, so that Whisper never hits its 30-second context limit mid-utterance and silence is not paid for in labelling time.

**Acceptance criteria:**
- [ ] `scripts/chunk.py` reads `data/raw/manifest.jsonl` and writes chunks to `data/chunks/<source_id>/<nnn>.wav`
- [ ] Every chunk is between 5 and 25 seconds; chunks that cannot be split within bounds are logged and skipped rather than emitted oversized
- [ ] `data/chunks/manifest.jsonl` records `chunk_id`, `source_id`, start, end, duration
- [ ] Chunk boundaries do not cut mid-word on spot-checked samples

**Verification:**
- [ ] A script assertion confirms no chunk duration falls outside 5-25s
- [ ] Total chunk duration is within 20% of source speech duration (the gap being removed silence)
- [ ] Manual check: listen to 5 random chunks; each starts and ends at a plausible pause

**Dependencies:** Task 3

**Files likely touched:** `scripts/chunk.py`, `src/vocalia/vad.py`, `data/chunks/manifest.jsonl`

**Estimated scope:** M

---

## Task 5: Baseline transcription with stock whisper-small

**Description:** Run unmodified `openai/whisper-small` over every chunk to produce draft transcripts. These serve two purposes: the starting point for human correction, and the baseline the adapter is later measured against.

**Acceptance criteria:**
- [ ] `scripts/transcribe.py --model baseline` writes `data/transcripts/baseline.jsonl` with `chunk_id` and `text`
- [ ] Runs on GPU; a CPU fall back warns loudly rather than proceeding silently
- [ ] Output is deterministic for a fixed chunk set (greedy decoding, fixed language hint)
- [ ] Resumable: an interrupted run continues rather than restarting from zero

**Verification:**
- [ ] Row count in `baseline.jsonl` equals chunk count in `data/chunks/manifest.jsonl`
- [ ] Re-running produces byte-identical output
- [ ] Manual check: transcripts are Devanagari Hindi and visibly imperfect on dialect terms — this is the expected failure the project exists to address

**Dependencies:** Task 4

**Files likely touched:** `scripts/transcribe.py`, `src/vocalia/asr.py`, `data/transcripts/baseline.jsonl`

**Estimated scope:** M

---

## Task 6: WER/CER evaluation harness

**Description:** Build the metric path with jiwer and exercise it on a handful of hand-corrected chunks, before the full labelling effort. Reporting is per-recording as well as pooled, because a single pooled number can hide a split that measures the wrong thing.

**Acceptance criteria:**
- [ ] `scripts/evaluate.py` takes a reference file and one or more hypothesis files and emits WER and CER
- [ ] Output includes a per-`source_id` breakdown alongside the pooled figure
- [ ] Text normalisation before scoring is explicit, documented, and applied identically to both sides (whitespace, punctuation, Devanagari digit handling)
- [ ] Results write to `reports/eval_<timestamp>.json` and a readable Markdown table
- [ ] Verified against a tiny set of 5-10 manually corrected chunks

**Verification:**
- [ ] Scoring a reference against itself yields WER 0.0 and CER 0.0
- [ ] A deliberately corrupted hypothesis (one word substituted in a known chunk) produces the hand-calculated WER
- [ ] Manual check: the Markdown table is legible and labels which model produced each column

**Dependencies:** Task 4 (plus a few corrected chunks); parallel with Task 5

**Files likely touched:** `scripts/evaluate.py`, `src/vocalia/metrics.py`, `reports/`

**Estimated scope:** M

---

### Checkpoint: Skeleton
- [ ] Ingest, chunk, transcribe and evaluate all run end to end on ~3 minutes of audio
- [ ] Self-scoring sanity checks pass (identity = 0.0, known corruption = expected value)
- [ ] Baseline failure modes on dialect terms are visible and noted — these become the side-by-side examples later
- [ ] Review with human before committing to bulk labelling

---

## Phase 2: Gold-Standard Dataset

## Task 7: Transcript correction tool

**Description:** A minimal local tool for correcting draft transcripts chunk by chunk: play audio, edit text, save, advance. Labelling is the project's bottleneck, so friction here multiplies across every minute of audio. Correctness of the saved output matters more than polish.

**Acceptance criteria:**
- [ ] Presents one chunk at a time with audio playback and the baseline draft pre-filled as the starting text
- [ ] Accepts Devanagari input and saves to `data/gold/corrected.jsonl` without mangling encoding
- [ ] Progress persists; reopening resumes at the first uncorrected chunk
- [ ] Shows a running count of corrected chunks and total corrected audio duration
- [ ] A chunk can be flagged as unusable (crosstalk, music, unintelligible) and excluded rather than force-corrected

**Verification:**
- [ ] Correct 3 chunks, close the tool, reopen: prior edits are intact and position resumes correctly
- [ ] Saved file round-trips through `json.loads` with Devanagari preserved exactly
- [ ] Manual check: a full correct-and-advance cycle takes only a few keystrokes

**Dependencies:** Task 5

**Files likely touched:** `scripts/correct.py` or `web/correct/`, `src/vocalia/gold.py`, `data/gold/corrected.jsonl`

**Estimated scope:** M

---

## Task 8: Corrected corpus with leakage-safe train/test split

**Description:** Complete the correction pass over the target duration, then split 80/20 grouped by `source_id` so no recording appears on both sides, while keeping both speakers present on both sides. The split is written to disk and frozen before any training occurs.

**Acceptance criteria:**
- [ ] At least 15 minutes of corrected audio, targeting 25 minutes (record whichever is achieved)
- [ ] `data/gold/{train,test}.jsonl` written, split by `source_id` with no overlap
- [ ] A programmatic assertion proves the `source_id` sets are disjoint
- [ ] Both `speaker_id` values appear in train and in test — one held-out session per speaker — so the benchmark measures dialect adaptation rather than speaker shift
- [ ] `data/gold/DATASET_CARD.md` records per-source duration, speaker labels, dialect, correction conventions used, and known limitations
- [ ] Test split holds roughly 20% of duration and at least 2 distinct recordings

**Verification:**
- [ ] Assertion script confirms zero `source_id` overlap and reports duration per split
- [ ] Spot-check 5 corrected transcripts against audio for accuracy
- [ ] Manual check: dataset card states the honest corrected duration, not the target

**Dependencies:** Task 7

**Files likely touched:** `scripts/split_dataset.py`, `data/gold/{train,test}.jsonl`, `data/gold/DATASET_CARD.md`

**Estimated scope:** M

---

### Checkpoint: Dataset Frozen
- [ ] Split is committed and will not be touched again; any later change invalidates the benchmark
- [ ] Leakage assertion passes
- [ ] Baseline re-scored against the frozen test set, giving the number the adapter must beat
- [ ] Review with human — this is the last point where scope can be cut cheaply

---

## Phase 3: Adaptation and Measurement

## Task 9: LoRA fine-tune of whisper-small

**Description:** Train a LoRA adapter on `openai/whisper-small` over the frozen training split, sized to fit 6 GB of VRAM, with early stopping on held-out loss to limit overfitting on a small corpus.

**Acceptance criteria:**
- [ ] `scripts/train_lora.py` fine-tunes with PEFT targeting `["q_proj", "v_proj"]`, lr 1e-4, rank 8-16
- [ ] Trains within 6 GB using fp16, batch size 2-4 and gradient accumulation; peak VRAM is logged
- [ ] Adapter weights save to `models/vocalia-lora/` (gitignored)
- [ ] Training and validation loss per epoch are logged to `reports/train_log.jsonl`
- [ ] Hyperparameters are read from a committed config file, not hardcoded at the call site

**Verification:**
- [ ] Training completes without OOM; peak VRAM stays under 6 GB
- [ ] Validation loss decreases for at least the first epochs; a flat or rising curve from step one is reported rather than hidden
- [ ] Adapter loads cleanly onto the base model in a fresh process
- [ ] Manual check: adapter directory is a few MB, not hundreds — confirming LoRA rather than a full fine-tune

**Dependencies:** Task 8

**Files likely touched:** `scripts/train_lora.py`, `src/vocalia/train.py`, `configs/lora.yaml`, `reports/train_log.jsonl`

**Estimated scope:** M

---

## Task 10: Baseline vs adapted benchmark report

**Description:** Transcribe the frozen test split with both the stock model and the adapted model, score both through the T6 harness, and publish the comparison including per-recording figures and qualitative examples.

**Acceptance criteria:**
- [ ] `scripts/transcribe.py --model lora` produces `data/transcripts/lora.jsonl` over the test split
- [ ] `reports/benchmark.md` contains a WER and CER table, baseline vs adapted, pooled and per-`source_id`
- [ ] At least 5 side-by-side examples where the models differ, with the gold reference shown
- [ ] A dialect-vocabulary recall count: how many target dialect terms each model got right
- [ ] If the adapter does not improve on a metric, the report says so plainly with a hypothesis as to why

**Verification:**
- [ ] Both models scored against the identical frozen test set via the same script invocation
- [ ] Re-running the report reproduces the same numbers
- [ ] Manual check: at least one example clearly shows preserved dialect vocabulary where the baseline normalised it

**Dependencies:** Task 9

**Files likely touched:** `scripts/transcribe.py`, `scripts/evaluate.py`, `reports/benchmark.md`

**Estimated scope:** S

---

### Checkpoint: Benchmark
- [ ] Honest, reproducible numbers exist for baseline vs adapted
- [ ] Per-recording breakdown reviewed for domain-shift confounds
- [ ] Decision point: the artifact proceeds on the adapted transcripts regardless of result, with the result stated accurately
- [ ] Review with human

---

## Phase 4: Story and Narration

## Task 11: Gemma 2 story engine with verbatim quote preservation

**Description:** Turn corrected transcripts into structured storybook chapters using Gemma 2, with a programmatic guarantee that quoted dialect spans are reproduced exactly as transcribed.

**Acceptance criteria:**
- [ ] `scripts/build_story.py` reads transcripts and writes `data/story/chapters.json` with title, ordered chapters, and per-chapter quoted spans
- [ ] Model choice (`gemma-2-2b-it` local vs `gemma-2-9b-it` hosted) is configurable; the resolved choice is recorded in the output
- [ ] A validator asserts every quoted span appears verbatim in the source transcript; failures block the write
- [ ] Narrative connective text is clearly separated from quoted speech in the output schema
- [ ] Prompt template is a committed file, not an inline string

**Verification:**
- [ ] Validator passes on a real run; an injected mutated quote makes it fail as intended
- [ ] Manual check: chapters read coherently and quoted dialect is untouched, including non-standard spellings
- [ ] Re-running with a fixed seed and the same input produces stable chapter structure

**Dependencies:** Task 10

**Files likely touched:** `scripts/build_story.py`, `src/vocalia/story.py`, `prompts/story.txt`, `data/story/chapters.json`

**Estimated scope:** M

---

## Task 12: ElevenLabs narration for finalised chapters

**Description:** Synthesise narration audio per chapter via ElevenLabs, written to static files for the reader to serve. API key comes from the environment and never from the repo.

**Acceptance criteria:**
- [ ] `scripts/narrate.py` reads `chapters.json` and writes `web/assets/audio/chapter_<n>.mp3`
- [ ] API key is read from the environment; absence produces a clear error, never a committed fallback
- [ ] Already-synthesised chapters are skipped unless `--force`, so credits are not spent twice
- [ ] `data/story/narration_manifest.json` records voice id, model, and character count per chapter
- [ ] Hindi/Devanagari pronunciation is auditioned on one chapter before the full run

**Verification:**
- [ ] Generated audio plays and matches chapter text
- [ ] Re-running without `--force` makes zero API calls (verified via the manifest timestamps)
- [ ] `git status` shows no key material and no unexpected large files staged
- [ ] Manual check: pronunciation is acceptable; if not, the documented fallback is applied

**Dependencies:** Task 11

**Files likely touched:** `scripts/narrate.py`, `src/vocalia/tts.py`, `web/assets/audio/`, `data/story/narration_manifest.json`

**Estimated scope:** M

---

## Phase 5: Reader and Deployment

## Task 13: Static storybook reader

**Description:** A responsive static page presenting the story chapters with narration playback beside the text, plus the before/after transcription comparison that shows what the adapter changed.

**Acceptance criteria:**
- [ ] `web/index.html` renders chapters from `chapters.json` with per-chapter audio controls
- [ ] A comparison section shows baseline vs adapted transcripts for selected utterances
- [ ] Readable on a phone viewport with no horizontal scroll
- [ ] Devanagari renders correctly with an explicit font stack and sensible line height
- [ ] No API keys or network calls to paid services from the client

**Verification:**
- [ ] Served locally, every chapter's audio plays and text matches
- [ ] Checked at 375px and 1440px widths
- [ ] Browser console is free of errors
- [ ] Manual check: Devanagari is not clipped or substituted by a fallback font

**Dependencies:** Task 12

**Files likely touched:** `web/index.html`, `web/app.js`, `web/styles.css`

**Estimated scope:** M

---

## Task 14: Deployment to Render

**Description:** Publish the reader as a Render static site with a reproducible build, and document the deployment so it can be repeated from scratch.

**Acceptance criteria:**
- [ ] Render static site serves `web/` at a public URL
- [ ] Deployment config is committed (`render.yaml` or documented dashboard settings)
- [ ] Audio assets load over HTTPS without mixed-content warnings
- [ ] README documents the deploy steps and the live URL
- [ ] No secrets in committed config

**Verification:**
- [ ] Public URL loads on a device that has never seen the project
- [ ] Audio plays from the deployed site, not just locally
- [ ] A fresh deploy from a clean clone succeeds
- [ ] Manual check: page is reachable on mobile data

**Dependencies:** Task 13

**Files likely touched:** `render.yaml`, `README.md`

**Estimated scope:** S

---

### Checkpoint: Complete
- [ ] All task acceptance criteria met
- [ ] Benchmark numbers reproducible from the committed scripts and frozen test set
- [ ] Dataset card and provenance manifest complete and accurate
- [ ] No secrets, raw audio, or weights in git history
- [ ] Live URL works from an unrelated device
- [ ] Ready for review
