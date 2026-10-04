# Homespun

Speech recognition that keeps a dialect intact.

**Awadhi** is spoken by roughly 3.85 million people across the Awadh region of Uttar
Pradesh, and by around half a million more in Nepal. General-purpose speech models
transcribe it by quietly translating it: local vocabulary becomes standard Hindi,
idiom is dropped, and the way people actually speak is flattened into newscaster
register. For dictation that scarcely matters. For recording how elders describe
birth customs, naming ceremonies and family tradition, it means the transcript is
not what was said.

Homespun trains a LoRA adapter on `openai/whisper-small` over a real Awadhi speech
corpus, measures whether it helps against the stock model on the corpus authors'
own held-out split, and turns the corrected transcripts into a readable, narrated
storybook.

## Corpus

**SpeeD-IA** — Speech Datasets for Indo-Aryan Languages. Collected by Dr. Bhimrao
Ambedkar University and the Council for Strategic and Defence Research, with
Karya Inc. and UnReaL-TecE LLP, and published at the Speech for Social Good
Workshop, Interspeech 2022.

- Transcriptions: <https://github.com/unrealtecellp/SpeeD-IA>
- Audio: Google Drive folder linked from that repository
- Licence: **CC BY-NC-SA 4.0**

| Subset | train | test | total |
|---|---|---|---|
| lifecycle | 357 | 90 | 447 |
| translation | 1,713 | 429 | 2,142 |
| **total** | **2,070** | **519** | **2,589** |

Speakers were asked questions about birth and naming customs and answered in
Awadhi; the TSV files transcribe those spoken answers. One WAV and one JSON per
utterance, grouped by speaker.

### Citation

```bibtex
@inproceedings{interspeech2022,
    author    = {Kumar, Ritesh and Singh, Siddharth and Ratan, Shyam and Raj, Mohit
                 and Sinha, Sonal and Lahiri, Bornini and Seshadri, Vivek
                 and Bali, Kalika and Ojha, Atul Kr.},
    title     = {Annotated Speech Corpus for Low Resource Indian Languages:
                 Awadhi, Bhojpuri, Braj and Magahi},
    booktitle = {Proceedings of Speech for Social Good Workshop, Interspeech 2022},
    year      = {2022}
}
```

### Licence obligations

CC BY-NC-SA 4.0 is not decorative. The adapter trained on this data and any
storybook derived from its transcripts are both derivative works:

- **Attribution** — the citation above appears in this README, the dataset card,
  the reader UI and any write-up.
- **Non-commercial** — neither the adapter nor the artifact is sold or used
  commercially.
- **Share-alike** — derivatives carry CC BY-NC-SA 4.0.

## Three things this corpus will do to the results

Recorded here because each one shapes how the benchmark should be read, and none
of them is visible from the file listing.

1. **The audio is 8 kHz, not 16 kHz.** Whisper expects 16 kHz, so every clip is
   upsampled — but nothing above 4 kHz was ever captured. Fricatives and sibilants
   live up there. This caps how good any model can get on this data, baseline and
   adapted alike.
2. **The split is by utterance, not by speaker.** The same speaker appears in both
   train and test. Results therefore describe adaptation *with speaker overlap*,
   and say nothing about generalising to an unseen voice.
3. **The JSON sidecar is the question, not the answer.** Its `data` field holds the
   prompt the speaker was asked. The spoken response is transcribed only in the
   TSV. Reference text comes from the TSVs alone.

## Honest about scale

Whisper has no Awadhi language id, so the language hint is fixed to `hi` for every
run, baseline and adapted, to keep the comparison fair. Baseline WER on a dialect
the model has never been tuned for may be high enough to make the metric noisy, so
CER is reported alongside it, plus a dialect-vocabulary recall count. The test
split is frozen before training, both models are scored by the same committed
script, and the number is published whichever way it falls.

## Layout

```
src/homespun/      library code
scripts/          runnable pipeline stages
configs/          training hyperparameters
prompts/          prompt templates (committed, not inlined)
data/             corpus, transcripts, derived audio  (gitignored except manifests)
models/           LoRA adapter output                 (gitignored)
reports/          benchmarks and training logs
web/              static storybook reader
tasks/            implementation plan and task list
```

Audio and weights are gitignored. Manifests and the dataset card are tracked, so
provenance is version-controlled without redistributing the corpus.

## Setup

Requires **Python 3.12**. A 3.14 interpreter may also be present as `py` — do not
use it; torch has no stable wheels for it.

```bash
python -m venv .venv
source .venv/Scripts/activate          # Git Bash; .venv\Scripts\activate in PowerShell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/check_env.py            # must report cuda: True
```

Copy `.env.example` to `.env` and fill in keys as the later stages need them.

## Pipeline

```bash
python scripts/import_speedia.py   # join audio to transcriptions on utterance id
python scripts/corpus_qa.py        # normalise to 16 kHz mono, measure, write dataset card
python scripts/transcribe.py --model baseline --split test
python scripts/evaluate.py         # freeze the baseline
python scripts/train_lora.py
python scripts/transcribe.py --model lora --split test
python scripts/evaluate.py         # baseline vs adapted
python scripts/build_story.py
python scripts/narrate.py
```

## Status

Phase 1, corpus import. See `tasks/todo.md` for the task list and `tasks/plan.md`
for architecture decisions, risks and open questions.
