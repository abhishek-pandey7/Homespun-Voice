# Vocalia — Task List

Plan document: `tasks/plan.md`
Corpus: SpeeD-IA Awadhi (CC BY-NC-SA 4.0), Interspeech 2022

---

## Phase 0: Foundation — complete

## Task 1: Repository scaffold and ignore rules — DONE

- [x] Directory layout, `.gitkeep` placeholders, `README.md`, `.env.example`
- [x] `.gitignore` excludes audio, weights, secrets; manifests stay tracked
- [x] `.gitattributes` normalises line endings, marks audio binary
- [x] Verified with `git check-ignore`, both directions

**Note:** `data/**` alone disabled every negation beneath it — git does not descend into
an ignored directory. `!data/**/` restores the manifest and `.gitkeep` exceptions.

---

## Task 2: Pinned Python environment with GPU verification — DONE

- [x] `requirements.txt` pinned against Python 3.12, CUDA index documented inline
- [x] `scripts/check_env.py` fails loudly on a CPU-only torch build
- [x] Verified: python 3.12.10, torch 2.5.1+cu121, RTX 4050 (6.0 GB), fp16 matmul on device

**Note:** yt-dlp 2026.08.19 returned HTTP 403 on media streams while metadata resolved;
upgrading to 2026.09.27 fixed extraction. No cookies or auth needed.

---

## Phase 1: Corpus

## Task 3: Import the SpeeD-IA Awadhi corpus - DONE

**Description:** Fetch the audio from the authors' Google Drive folder and the
transcriptions from the GitHub repository, then join them into a single local corpus
tree. The join is on utterance ID, which appears both as the WAV filename and as the
`ID` column in the TSV transcription files.

**Acceptance criteria:**
- [x] Audio downloaded to `data/speedia/` preserving the subset and speaker structure
- [x] Transcriptions for `lifecycle` and `translation`, train and test, fetched to `data/speedia/transcripts/`
- [x] `scripts/import_speedia.py` joins audio to transcript on utterance ID
- [x] Every joined record carries: `utt_id`, `audio_path`, `text`, `subset`, `split`, `speaker_dir`
- [x] IDs present in a transcript but missing audio are reported and counted, not silently dropped
- [x] IDs present as audio but missing a transcript are reported and counted
- [x] Re-running does not re-download existing files

**Verification:**
- [x] Joined record count reconciles against 2,070 train + 519 test from the TSVs
- [x] A random sample of 5 records: audio plays and matches its transcript
- [x] Missing-file counts printed explicitly, including zero

**Dependencies:** Task 2

**Files likely touched:** `scripts/import_speedia.py`, `src/vocalia/speedia.py`, `data/speedia/`

**Estimated scope:** M

---

## Task 4: Corpus QA, normalisation and manifest - DONE

**Description:** Normalise every audio file to 16 kHz mono, measure the corpus, exclude
what Whisper cannot consume, and write the dataset card. This is the gate: nothing
downstream may run on an unmeasured corpus.

**Acceptance criteria:**
- [x] All audio normalised to 16 kHz mono PCM; any file requiring conversion is counted
- [x] `data/speedia/manifest.jsonl` holds one row per usable utterance with duration
- [x] Utterances longer than 30s flagged, excluded, counted (Whisper truncates silently)
- [x] Zero-length, silent, or unreadable files excluded and counted
- [x] Totals reported: utterances, total duration, per split, per subset, per speaker
- [x] `data/speedia/DATASET_CARD.md` records source, citation, CC BY-NC-SA 4.0 terms, the
      counts above, exclusions with reasons, and known limitations
- [x] README and dataset card both carry the Interspeech 2022 citation

**Verification:**
- [x] `ffprobe` on 10 random files confirms 16000 Hz / 1 channel
- [x] Sum of per-split durations equals the reported total
- [x] Excluded count plus usable count equals the joined record count from T3
- [x] Manual check: read the dataset card as a stranger — is provenance unambiguous?

**Dependencies:** Task 3

**Files likely touched:** `scripts/corpus_qa.py`, `src/vocalia/audio.py`, `data/speedia/manifest.jsonl`, `data/speedia/DATASET_CARD.md`

**Estimated scope:** M

---

### Checkpoint: Corpus - PASSED

**Measured corpus (from `scripts/corpus_qa.py`):**

| split | subset | utts | duration | mean |
|---|---|---:|---:|---:|
| test | lifecycle | 80 | 15m35s | 11.7s |
| test | translation | 429 | 22m32s | 3.2s |
| train | lifecycle | 316 | 1h04m25s | 12.2s |
| train | translation | 1713 | 1h31m49s | 3.2s |
| **test** | **all** | **509** | **38m07s** | 4.5s |
| **train** | **all** | **2029** | **2h36m15s** | 4.6s |

Total usable: **2,538 utterances / 3h14m22s**, 18 speakers. Train/test is 80/20 by
duration. 46 excluded (45 longer than Whisper's 30s window, 1 under 0.2s).
Duration p50 3.2s, p90 8.4s, p99 25.3s.

- [x] Corpus measured; estimates replaced with real numbers
- [x] Licence and citation present in README and dataset card
- [x] YouTube material deleted; corpus is single-source
- [x] Review with human before spending GPU time

---

## Phase 2: Baseline and Measurement

## Task 5: Baseline transcription with stock whisper-small - DONE

**Description:** Transcribe the full test split with unmodified `openai/whisper-small`.
This is the number the adapter must beat, and the evidence for how the stock model fails
on Awadhi.

**Acceptance criteria:**
- [x] `scripts/transcribe.py --model baseline --split test` writes `data/transcripts/baseline_test.jsonl`
- [x] Language hint fixed to `hi`, recorded in the output, identical for every later run
- [x] Greedy decoding, fixed seed — byte-identical output across runs
- [x] Runs on GPU; a CPU fallback warns loudly rather than proceeding quietly
- [x] Resumable after interruption
- [x] Throughput and total wall time logged

**Verification:**
- [x] Output row count equals test-split utterance count
- [x] Re-running produces byte-identical output
- [x] Manual check: transcripts are Devanagari and visibly mis-handle Awadhi vocabulary

**Dependencies:** Task 4

**Files likely touched:** `scripts/transcribe.py`, `src/vocalia/asr.py`, `data/transcripts/baseline_test.jsonl`

**Estimated scope:** M

---

## Task 6: WER/CER evaluation harness and frozen baseline - DONE

**Description:** Score hypotheses against references with jiwer, broken out by subset and
split, and freeze the baseline. Normalisation is explicit and applied identically to both
sides, since silent normalisation differences are the usual way ASR benchmarks become
meaningless.

**Acceptance criteria:**
- [x] `scripts/evaluate.py` takes reference and one or more hypothesis files, emits WER and CER
- [x] Breakdown by subset (`lifecycle`, `translation`) as well as pooled
- [x] Normalisation documented and shared by both sides: whitespace, punctuation, Devanagari digits
- [x] Results to `reports/eval_<timestamp>.json` plus a readable Markdown table
- [x] Baseline frozen to `reports/baseline.json`, never overwritten by later runs

**Verification:**
- [x] Scoring a reference against itself yields WER 0.0 and CER 0.0
- [x] A hypothesis with one known substituted word yields the hand-calculated WER
- [x] An empty hypothesis yields WER 1.0
- [x] Manual check: the table states which model produced each column

**Dependencies:** Task 5

**Files likely touched:** `scripts/evaluate.py`, `src/vocalia/metrics.py`, `reports/baseline.json`

**Estimated scope:** M

---

### Checkpoint: Baseline - PASSED

**Frozen baseline** (`reports/baseline.json`), stock whisper-small, 509 test utterances:

| subset | utts | WER | CER |
|---|---:|---:|---:|
| lifecycle | 80 | 0.9762 | 0.6181 |
| translation | 429 | 1.0592 | 0.6476 |
| **all** | **509** | **1.0210** | **0.6335** |

Dialect-marker recall: **36/709 = 5.1%**.

WER exceeds 1.0 because the model inserts as well as substitutes - it hallucinates
on longer clips. The metric is therefore saturated and CER is the more informative
figure at this baseline. Throughput 4.5x realtime, 463s for 34.6 min of audio.

- [x] Baseline WER and CER known, frozen, reproducible
- [x] Harness proven by self-scoring and known-corruption tests
- [x] Qualitative failure examples collected
- [x] Review with human

---

## Phase 3: Adaptation

## Task 7: LoRA fine-tune of whisper-small - DONE

**Acceptance criteria:**
- [x] `scripts/train_lora.py` uses PEFT targeting `["q_proj", "v_proj"]`, lr 1e-4, rank 8-16
- [x] Fits in 6 GB using fp16, batch 2-4, gradient accumulation; peak VRAM logged
- [x] A validation slice held out of train — the test split is never seen during training
- [x] Adapter saves to `models/vocalia-lora/` (gitignored)
- [x] Per-epoch train and validation loss logged to `reports/train_log.jsonl`
- [x] Hyperparameters read from `configs/lora.yaml`, not hardcoded

**Verification:**
- [x] Completes without OOM; peak VRAM under 6 GB
- [x] Validation loss curve recorded; a flat or rising curve is reported, not hidden
- [x] Adapter loads onto the base model in a fresh process
- [x] Adapter directory is a few MB — confirming LoRA, not a full fine-tune
- [x] Assertion: no test-split utterance ID appears in training

**Dependencies:** Task 6

**Files likely touched:** `scripts/train_lora.py`, `src/vocalia/train.py`, `configs/lora.yaml`

**Estimated scope:** M

---

## Task 8: Baseline vs adapted benchmark report - DONE

**Acceptance criteria:**
- [x] `scripts/transcribe.py --model lora --split test` produces `data/transcripts/lora_test.jsonl`
- [x] `reports/benchmark.md` holds WER and CER, baseline vs adapted, pooled and per subset
- [x] At least 5 side-by-side examples where the models differ, with the reference shown
- [x] A dialect-vocabulary recall count for each model
- [x] Absolute and relative change stated for each metric
- [x] If the adapter fails to improve a metric, the report says so with a hypothesis

**Verification:**
- [x] Both models scored on the identical frozen test set through one script
- [x] Re-running reproduces the numbers
- [x] Manual check: at least one example shows preserved Awadhi vocabulary where the
      baseline substituted standard Hindi

**Dependencies:** Task 7

**Estimated scope:** S

---

### Checkpoint: Benchmark - PASSED

| metric | baseline | adapted | change |
|---|---:|---:|---:|
| WER | 1.0210 | 0.6967 | -31.8% |
| CER | 0.6335 | 0.3960 | -37.5% |
| dialect markers | 36/709 (5.1%) | 288/709 (40.6%) | 8x |

Per subset: translation WER -46.7% / CER -59.5%; lifecycle WER -12.8% / CER -12.3%.
Training: 360 steps, 25.6 min, peak VRAM 3.72 GB, val loss 1.1242 -> 0.8888.
Adapter 6.8 MB. Decoding 7.6x faster (67s vs 463s) because the baseline generated
runaway tokens the adapted model terminates.

**Open limitation:** 35 of 509 utterances (6.9%) still regress by more than 0.05
CER, with degenerate repetition loops. Reduced from baseline but not eliminated.
A `repetition_penalty` / `no_repeat_ngram_size` ablation would have to be applied
to BOTH models to keep the comparison fair.

- [x] Honest, reproducible comparison exists
- [x] Per-subset breakdown reviewed
- [x] Result stated accurately, including the regression tail
- [ ] Review with human

---

## Phase 4: Story and Narration

## Task 9: Gemma 2 story engine with verbatim quote preservation

**Acceptance criteria:**
- [ ] `scripts/build_story.py` writes `data/story/chapters.json` — title, ordered chapters, quoted spans
- [ ] Model choice configurable; resolved choice recorded in the output
- [ ] Validator asserts every quoted span appears verbatim in its source; failure blocks the write
- [ ] Narrative connective text separated from quoted speech in the schema
- [ ] Prompt template committed to `prompts/story.txt`
- [ ] Output carries source attribution and CC BY-NC-SA 4.0

**Verification:**
- [ ] Validator passes on a real run; an injected mutated quote makes it fail
- [ ] Manual check: chapters read coherently, Awadhi quotes untouched including spelling
- [ ] Re-running with a fixed seed yields stable structure

**Dependencies:** Task 8

**Estimated scope:** M

---

## Task 10: ElevenLabs narration

**Acceptance criteria:**
- [ ] `scripts/narrate.py` writes `web/assets/audio/chapter_<n>.mp3`
- [ ] Missing API key produces a clear error, never a committed fallback
- [ ] Existing chapters skipped unless `--force`, so credits are not spent twice
- [ ] `data/story/narration_manifest.json` records voice id, model, character count
- [ ] One chapter auditioned for Devanagari pronunciation before the full run

**Verification:**
- [ ] Audio plays and matches chapter text
- [ ] Re-running without `--force` makes zero API calls
- [ ] `git status` shows no key material and no large files staged

**Dependencies:** Task 9

**Estimated scope:** M

---

## Phase 5: Reader and Deployment

## Task 11: Static storybook reader

**Acceptance criteria:**
- [ ] `web/index.html` renders chapters with per-chapter audio controls
- [ ] Comparison section shows baseline vs adapted transcripts for selected utterances
- [ ] Readable at phone width, no horizontal scroll
- [ ] Devanagari renders with an explicit font stack and adequate line height
- [ ] Corpus attribution, citation and licence visible in the UI
- [ ] No API keys and no paid-service calls from the client

**Verification:**
- [ ] Served locally, every chapter plays and text matches
- [ ] Checked at 375px and 1440px
- [ ] Console free of errors

**Dependencies:** Task 10

**Estimated scope:** M

---

## Task 12: Deployment to Render

**Acceptance criteria:**
- [ ] Render static site serves `web/` at a public URL
- [ ] Deployment config committed
- [ ] Audio loads over HTTPS without mixed-content warnings
- [ ] README documents deploy steps and the live URL
- [ ] No secrets in committed config

**Verification:**
- [ ] Public URL loads on a device that has never seen the project
- [ ] Audio plays from the deployed site
- [ ] Fresh deploy from a clean clone succeeds

**Dependencies:** Task 11

**Estimated scope:** S

---

### Checkpoint: Complete
- [ ] All acceptance criteria met
- [ ] Benchmark reproducible from committed scripts and the corpus split
- [ ] Dataset card, citation and licence accurate everywhere they appear
- [ ] No secrets, audio, or weights in git history
- [ ] Live URL works from an unrelated device
