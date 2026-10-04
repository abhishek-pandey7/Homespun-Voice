# Homespun

**A speech model retrained to write Awadhi down the way it is actually spoken.**

[Adapter](https://huggingface.co/abhshkp/homespun-awadhi-lora) ·
[Browser build](https://huggingface.co/abhshkp/homespun-awadhi-web) ·
[Benchmark](reports/benchmark.md)

## Why I built this

I grew up in Mumbai. Hindi is everywhere here. It is on the news, in school, in
every shop, and it is the Hindi the whole country has agreed on.

My grandparents did not grow up here. They spent their entire lives in a village
in Uttar Pradesh, and they speak **Awadhi**. It sits close enough to Hindi that
you feel you ought to follow it, and far enough that you often do not. My
siblings and I catch about half of what they say. We nod through the rest, and
then one of us asks our parents afterwards.

So I did the obvious thing and pointed a transcription app at a recording. It
handed back standard Hindi. Not a transcription, a translation, and a lossy one:

| what was said | what came back | both mean |
|---|---|---|
| `हो थय` *(ho thay)* | `होता है` *(hota hai)* | "it happens" |
| `अहय` *(ahay)* | nothing, it vanished | "is" |
| `पहिले` *(pahile)* | `पहले` *(pahle)* | "before" |

Every row loses the same thing. The meaning survives and the voice does not.
`अहय` is not a mistake for `है`; it is how you say "is" where my grandparents
are from, and a model trained on newsreaders has simply never met it.

On anything longer than a sentence it gave up altogether and repeated a single
syllable until it ran out of room.

That is not one bad app. Awadhi has roughly four million speakers and almost no
presence in the data these models learn from, so every model treats it as broken
Hindi rather than as a language of its own. The part that makes my grandparents
sound like my grandparents is precisely the part that gets corrected away.

I wanted something my siblings and I could use. That is the whole brief.

## What it does

A LoRA adapter on `openai/whisper-small`, trained on three hours of Awadhi
speech, measured against the unmodified model on a held-out split the corpus
authors defined. Both models decode through the same script with the same
settings, so the adapter is the only thing that differs.

| | before | after |
|---|---:|---:|
| Words wrong | 0.9323 | **0.5853** |
| Characters wrong | 0.6200 | **0.3357** |
| Awadhi marker words kept | 5.1% | **45.3%** |

The last row is the one I care about. Those are the everyday Awadhi words a
general model quietly deletes: `अहय` *(ahay, "is")*, `थय` *(thay, the past
marker)*, `होत` *(hot, "happens")*, `जौन` *(jaun, "which")*. Function words,
the kind you cannot speak a sentence without. The adapter went from keeping one
in twenty to keeping almost half.

The adapter is **8.7 MB**. It trained in **52 minutes** on a laptop GPU.

## A word error rate above 1.0 is not a typo

The stock model scores worse than 100 percent word error, which sounds
impossible until you watch it work. It does not merely pick wrong words. On
longer clips it stops transcribing and emits one syllable over and over until it
reaches the token limit, so it produces more errors than the reference has
words.

```
said      ई लम्बा पेड अहय ।            "this is a tall tree"
before    इज़ंबा पेर आख आख आख आख आख आख आख आख आख आख आख आख आख ...
after     ई लम्बा पेर अहय ।            "this is a tall tree"
```

It is also why the adapted model decodes about seven times faster. It stops when
the sentence does.

## What the numbers do not say

Four things, written here rather than buried, because anyone reading the results
properly will find them anyway.

**The recordings are 8 kHz.** Nothing above 4 kHz was ever captured, and that is
where much of the energy separating fricatives lives. The ceiling applies to
both models equally, so the comparison holds while the absolute figures stay
worse than they would be on clean audio.

**The same speakers appear in training and testing.** The corpus authors split
by utterance, not by speaker, so these results describe adaptation to familiar
voices. They say nothing about a voice the model has never heard, which is
awkwardly the case I care about most.

**Long answers improved least.** Short prompted sentences gained roughly four
times as much as the long spontaneous narratives, which had a fifth of the
training data. Doubling the epochs narrowed that gap without closing it, which
points at how much data exists rather than how long it trained.

**Twenty of 509 recordings still get worse.** The repetition collapse is mostly
gone, not entirely. It was 35 before the longer run. The demo page shows one of
those failures on purpose, because a reader who finds a hidden failure stops
trusting everything else on the page.

## The site

Four static pages, no build step.

| | |
|---|---|
| **Listen** | The testimony. Each quotation plays in the speaker's own voice, with an English translation underneath wherever the translator was confident enough to give one. |
| **Try it** | Record or upload audio and the model runs inside your browser. Nothing is uploaded. |
| **Evidence** | The benchmark, broken out by subset. |
| **Method** | How it was trained, and what that cost. |

The browser demo runs an int8 ONNX build through Transformers.js. It is a
measurably worse model than the one in the table above, 0.2943 character error
against 0.2553, because quantising it down to 279 MB costs accuracy. That trade
is stated on the Method page rather than hidden, and it still halves the stock
model's error.

Audio never leaving the machine is not a technical convenience. A tool for
keeping family recordings should not require handing them to someone else's
server in order to prove it works.

## The corpus

**SpeeD-IA**, collected by Dr. Bhimrao Ambedkar University and the Council for
Strategic and Defence Research with Karya Inc. and UnReaL-TecE LLP, published at
Interspeech 2022. Licensed CC BY-NC-SA 4.0.

2,538 usable utterances, 3 hours 14 minutes, 18 speakers, answering questions
about birth, marriage and mourning customs in their own words. Three properties
shaped everything downstream, and two of them are traps:

- The audio is 8 kHz and has to be upsampled before Whisper will take it.
- The JSON sidecar shipped beside each clip holds the **question the speaker was
  asked**, not their answer. Pairing clips with it would have trained the model
  against entirely the wrong words.
- 840 clips carry no transcription at all and are useless for supervised
  training.

I did not record these speakers. The motivation is mine; the voices belong to
people who gave their time to a research corpus, and `data/speedia/DATASET_CARD.md`
says exactly what came from where.

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

## Running it

Requires **Python 3.12**. Not 3.14, which has no stable torch wheels.

```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash; .venv\Scripts\activate on PowerShell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/check_env.py       # must report cuda: True
```

Then, in order:

```bash
python scripts/import_speedia.py                      # join audio to transcriptions
python scripts/corpus_qa.py                           # resample, measure, write the dataset card
python scripts/transcribe.py --model baseline --split test --beams 5
python scripts/evaluate.py data/transcripts/baseline_test.jsonl --freeze-baseline
python scripts/train_lora.py --config configs/lora-v2.yaml
python scripts/transcribe.py --model lora --split test --beams 5
python scripts/report_benchmark.py                    # writes reports/benchmark.md
python scripts/build_story.py && python scripts/translate_story.py
python scripts/narrate.py && python scripts/export_web.py
```

Every stage writes a manifest, is safe to re-run, and counts what it excluded
rather than quietly dropping it.

## Five things that went wrong, in case they save you an afternoon

**`data/**` in `.gitignore` silently killed every negation beneath it.** Git does
not descend into an ignored directory, so `!data/raw/manifest.jsonl` was dead
until `!data/**/` went back in.

**PEFT plus reentrant gradient checkpointing severs the autograd graph.** With a
frozen base no input requires grad, so backward finds nothing to do. It needs
`use_reentrant: False`.

**Batch 8 without gradient checkpointing allocated 9.65 GB on a 6 GB card and
did not crash.** Windows spills to system RAM instead of raising OOM, so it
reported success while running at half the speed of the setting it replaced.

**Quantising a merged ONNX decoder does nothing at all.** It is built around an
`If` node and `quantize_dynamic` will not descend into subgraphs, so a 739 MB
file came out at 739 MB. Quantise the two graphs while they are still flat, then
merge.

**Blocking repeated n-grams made transcription worse than greedy decoding.** It
is the obvious fix for a repetition loop, and Awadhi genuinely repeats, so the
block deletes real words to prevent a failure affecting 4 percent of clips.

## Licence

Code is MIT. The adapter, and anything else derived from the corpus, carries
CC BY-NC-SA 4.0 in line with the corpus itself.
