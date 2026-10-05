*This is a submission for the [Hacktoberfest Weekend Challenge: Build for a Friend](https://dev.to/challenges/hacktoberfest-weekend-2026-10-01)*

## What I Built

**Homespun** is a speech model retrained to write **Awadhi** down the way it is actually spoken, instead of quietly correcting it into standard Hindi.

I grew up in Mumbai. Hindi is everywhere here, and it is the Hindi the whole country has agreed on. My grandparents did not grow up here. They spent their entire lives in a village in Uttar Pradesh, and they speak Awadhi. It sits close enough to Hindi that you feel you ought to follow it, and far enough that you often do not. My siblings and I catch about half of what they say. We nod through the rest and ask our parents afterwards.

So I did the obvious thing and pointed a transcription app at a recording. It handed back standard Hindi. Not a transcription, a translation, and a lossy one:

| what was said | what came back | both mean |
|---|---|---|
| `हो थय` *(ho thay)* | `होता है` *(hota hai)* | "it happens" |
| `अहय` *(ahay)* | nothing, it vanished | "is" |
| `पहिले` *(pahile)* | `पहले` *(pahle)* | "before" |

Every row loses the same thing. The meaning survives and the voice does not. `अहय` is not a mistake for `है`; it is how you say "is" where my grandparents are from, and a model trained on newsreaders has simply never met it.

Awadhi has roughly four million speakers and almost no presence in the data these models learn from, so every one of them treats it as broken Hindi rather than as a language. The friends I built this for are my siblings. We are the ones who will want these recordings in twenty years, and we are the ones who cannot fully follow them today.

## Demo

**Live site: https://homespun-awadhi.onrender.com**

Four pages:

- **Listen** - the testimony. Each quotation plays in the speaker's own voice, with an English translation underneath.
- **Try it** - record or upload audio and the retrained model runs **inside your browser**. Nothing is uploaded.
- **Evidence** - the full benchmark, broken out by subset.
- **Method** - how it was trained, and what that cost.

### The headline

Both models read the same held-out recordings through the same script with the same settings. The only difference between them is an 8.7 MB adapter.

| | before | after |
|---|---:|---:|
| Words wrong | 0.9323 | **0.5853** |
| Characters wrong | 0.6200 | **0.3357** |
| Awadhi marker words kept | 5.4% | **48.8%** |

That last row is the one I care about. Those are the everyday words a general model deletes: `अहय` *(is)*, `थय` *(past marker)*, `होत` *(happens)*, `जौन` *(which)*. Function words you cannot speak a sentence without. It went from keeping one in twenty to keeping almost half.

### A word error rate above 1.0 is not a typo

The stock model scores worse than 100 percent word error, which sounds impossible until you watch it work. It does not merely pick wrong words. On longer clips it stops transcribing and emits one syllable over and over until it hits the token limit, so it produces more errors than the reference has words.

```
said      ई लम्बा पेड अहय ।            "this is a tall tree"
before    इज़ंबा पेर आख आख आख आख आख आख आख आख आख आख आख ...
after     ई लम्बा पेर अहय ।            "this is a tall tree"
```

It is also why the adapted model decodes about seven times faster. It stops when the sentence does.

## Code

{% embed https://github.com/abhishek-pandey7/Homespun-Voice %}

Published artefacts:

- Adapter: [abhshkp/homespun-awadhi-lora](https://huggingface.co/abhshkp/homespun-awadhi-lora), 8.7 MB
- Browser build: [abhshkp/homespun-awadhi-web](https://huggingface.co/abhshkp/homespun-awadhi-web), int8 ONNX, 279 MB

## How I Built It

Everything in the pipeline is open weights or open source. Nothing in the chain required an API key to produce the result.

**Speech recognition: `openai/whisper-small` with a LoRA adapter (PEFT).** 244M parameter base, rank 32 on the attention projections, trained on 3 hours 14 minutes of Awadhi. The adapter is 8.7 MB and trained in 52 minutes on one **RTX 4050 laptop GPU with 6 GB**, peaking at 3.7 GB.

**Corpus: SpeeD-IA** (Kumar et al., Interspeech 2022), collected by Dr. Bhimrao Ambedkar University and the Council for Strategic and Defence Research with Karya Inc. and UnReaL-TecE LLP. CC BY-NC-SA 4.0. 2,538 usable utterances, 18 speakers, answering questions about birth, marriage and mourning customs in their own words.

**To be exact about provenance: these are not recordings of my grandparents.** The motivation is mine; the voices belong to people who gave their time to a research corpus, and the dataset card in the repo says precisely what came from where. Recording my own family is the obvious next step and the right one.

**Storybook and translation: `gemma-2-2b-it`, quantised to 4-bit NF4** and run locally. At bf16 it needs about 5.2 GB for weights alone and OOMs mid-generation on a 6 GB card; NF4 brings it to roughly 1.5 GB.

Gemma does **not** touch the Awadhi. It returns chapter titles, English narration and a list of utterance ids; the quotations are looked up from the corpus and inserted afterwards. That is deliberate. My first design asked it to copy each quote and validated afterwards, and it kept truncating them or appending a full stop. Now the model has no channel through which to alter a word, so fidelity is structural rather than checked.

**Narration: ElevenLabs** (`eleven_multilingual_v2`) for the English chapter introductions. The Awadhi is never sent to a text to speech service. The reader plays the speakers' own recordings, because synthesising their words would impose Hindi phonology on Awadhi and flatten exactly what the project exists to preserve.

**In-browser inference: Transformers.js** against an int8 ONNX build, WebGPU with a WASM fallback. Your audio never leaves your machine. That is not a technical convenience: a tool for keeping family recordings should not require handing them to someone else's server to prove it works.

**Hosting: Render**, static site, no build step.

**Evaluation: jiwer**, corpus level rather than an average of per-utterance rates, with normalisation applied identically to both sides. The harness verifies itself against known answers before it reports anything: a reference scored against itself must give zero, an empty hypothesis must give one, and a single substituted word must match the hand calculation.

## Why Does Open Innovation Matter?

**This project is only possible because the weights were open.** Not "easier". Possible.

A closed speech API gives you one lever: the audio you send it. You cannot teach it a dialect. I did not need a better API, I needed a different model, and the only way to get one was to reach inside and change the weights. LoRA let me do that with 8.7 MB of trained parameters on a laptop, for the cost of 52 minutes of electricity.

**The economics only work because the model is open.** Four million Awadhi speakers is a rounding error to anyone selling a transcription API. Nobody is going to fund that work. But the marginal cost of adapting an open model to a long tail language is now one evening and a mid-range GPU, which means it does not need funding. It needs someone who cares, and there is at least one of those per language.

**And the failure was only visible because I could measure it.** A closed API returns standard Hindi and looks like it worked. You cannot see what it deleted. Running the stock model myself, over a corpus with human transcriptions, is what turned "this feels wrong" into "it keeps 5 percent of the dialect markers".

The same argument holds for the rest of the stack. Gemma runs on my own machine, so no transcript of my family's speech is sent anywhere to be turned into a storybook. Transformers.js means the demo runs in the visitor's browser, so the privacy claim on the site is a fact about the architecture rather than a promise in a policy.

## What the numbers do not say

Four things, written here rather than buried, because anyone reading the results properly will find them anyway.

**The recordings are 8 kHz.** Nothing above 4 kHz was ever captured, which is where much of the energy separating fricatives lives. The ceiling applies to both models equally, so the comparison holds while the absolute figures stay worse than they would be on clean audio.

**The same speakers appear in training and testing.** The corpus authors split by utterance, not by speaker. These results describe adaptation to familiar voices, and say nothing about a voice the model has never heard, which is awkwardly the case I care about most.

**Long answers improved least.** Short prompted sentences gained roughly four times as much as the long spontaneous narratives, which had a fifth of the training data. Doubling the epochs narrowed that gap without closing it, which points at how much data exists rather than how long it trained.

**Twenty of 509 recordings still get worse.** The repetition collapse is mostly gone, not entirely. It was 35 before the longer run. The demo page shows one of those failures on purpose, because a reader who finds a hidden failure stops trusting everything else on the page.

## Five things that went wrong

In case they save someone an afternoon.

**`data/**` in `.gitignore` silently killed every negation beneath it.** Git does not descend into an ignored directory, so `!data/raw/manifest.jsonl` was dead until `!data/**/` went back in. The same trap later 404'd four audio clips in production while they worked perfectly in local testing.

**PEFT plus reentrant gradient checkpointing severs the autograd graph.** With a frozen base, no input requires grad, so backward finds nothing to do. It needs `use_reentrant: False`.

**Batch 8 without gradient checkpointing allocated 9.65 GB on a 6 GB card and did not crash.** Windows spills to system RAM instead of raising OOM, so it reported success while running at half the speed of the setting it replaced.

**Quantising a merged ONNX decoder does nothing at all.** It is built around an `If` node and `quantize_dynamic` will not descend into subgraphs, so a 739 MB file came out at 739 MB. Quantise the two graphs while they are still flat, then merge.

**Blocking repeated n-grams made transcription worse than greedy decoding.** It is the obvious fix for a repetition loop, and it lost twice: Awadhi genuinely repeats, so the block deletes real words to prevent a failure affecting 4 percent of clips. Beam search removes the loops and is more accurate than both.

## Prize Categories

- **Best Use of Gemma** - `gemma-2-2b-it` at 4-bit, running locally, assembling the storybook and translating every quotation, architecturally prevented from touching the Awadhi itself
- **Best Use of ElevenLabs** - `eleven_multilingual_v2` narrating the English chapter introductions, deliberately not the dialect
- **Best Use of Render** - hosting the four page static reader, including the in-browser model demo

## Licence

Code is MIT. The adapter and anything derived from the corpus carry CC BY-NC-SA 4.0, following the corpus.
