# Vocalia

Speech recognition that keeps a dialect intact.

Standard speech-to-text models transcribe **rural UP/Bihar Hindi** by quietly
translating it. Local vocabulary becomes standard Hindi, colloquial idioms get
dropped, and the cadence of how someone actually speaks is flattened into
newscaster register. For ordinary dictation that hardly matters. For recording the
spoken memories of two people in their seventies, it means the transcript is not
what they said.

Vocalia trains a small LoRA adapter on top of `openai/whisper-small` using a
hand-corrected corpus of that dialect, measures whether it actually helps against
the stock model, and then turns the corrected transcripts into a readable, narrated
storybook.

## Target

| | |
|---|---|
| **Dialect** | Rural UP/Bihar Hindi |
| **Speakers** | Two, approximately 70 years old — one man, one woman, rural Uttar Pradesh |
| **Source audio** | Recorded directly, with consent |
| **Base model** | `openai/whisper-small` (244M) |
| **Adaptation** | LoRA on `q_proj` / `v_proj`, rank 8-16, lr 1e-4 |
| **Metrics** | WER and CER via `jiwer`, reported per recording as well as pooled |

## What this is honest about

The corpus is small — tens of minutes, not tens of hours. That is far below what
ASR adaptation usually needs, so the adapter may deliver only a modest gain, or
none. The benchmark is therefore the deliverable rather than a formality: the test
split is frozen before training, both models are scored by the same committed
script, results are broken out per recording, and the number is published whichever
way it falls. A storybook built on an adapter that did not help is still a
storybook; a benchmark nobody can reproduce is worth nothing.

## Layout

```
src/vocalia/      library code
scripts/          runnable pipeline stages
configs/          training and generation hyperparameters
prompts/          prompt templates (committed, not inlined)
data/             audio, transcripts, gold corpus  (gitignored except manifests)
models/           LoRA adapter output              (gitignored)
reports/          benchmarks and training logs
web/              static storybook reader
tasks/            implementation plan and task list
```

Audio, weights and secrets are gitignored. The manifests that record *where each
recording came from* are tracked, so provenance survives without redistributing
the recordings themselves.

## Setup

Requires **Python 3.12**. A 3.14 interpreter may also be present on this machine as
`py` — do not use it, as torch has no stable wheels for it.

```bash
python --version          # expect 3.12.x
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); use .venv\Scripts\activate in PowerShell
pip install -r requirements.txt
python scripts/check_env.py     # must report cuda: True
```

Copy `.env.example` to `.env` and fill in keys as the later stages need them.

## Pipeline

Run order. Each stage writes a manifest and is safe to re-run.

```bash
python scripts/ingest.py      # recordings -> 16 kHz mono WAV + provenance manifest
python scripts/chunk.py       # Silero VAD -> 5-25s segments
python scripts/transcribe.py --model baseline
python scripts/correct.py     # human-in-the-loop correction
python scripts/split_dataset.py
python scripts/train_lora.py
python scripts/transcribe.py --model lora
python scripts/evaluate.py    # WER / CER, baseline vs adapted
python scripts/build_story.py
python scripts/narrate.py
```

## Status

Scaffolding. See `tasks/todo.md` for the task list and `tasks/plan.md` for
architecture decisions, risks and open questions.

## Licence and consent

Source recordings were made with the speakers' consent and are not published in
this repository. Any transcript excerpt used for illustration appears with their
permission.
