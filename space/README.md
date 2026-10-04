---
title: Homespun
emoji: 🧵
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
---

# Homespun

A speech model retrained to transcribe **Awadhi** without translating it into
standard Hindi.

Record or upload a clip and it runs through stock `openai/whisper-small` and
through the same model with a LoRA adapter trained on the SpeeD-IA Awadhi
corpus. Decoding settings are identical for both, so the adapter is the only
difference between the two columns.

On the held-out test split: word error 1.02 to 0.62, character error 0.63 to
0.36, Awadhi marker words kept 5 percent to 45 percent.

Corpus: SpeeD-IA (Kumar et al., Interspeech 2022), CC BY-NC-SA 4.0.
The adapter is a derivative work and carries the same licence.
