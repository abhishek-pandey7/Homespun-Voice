"""Homespun - hear what each model makes of Awadhi speech.

Runs two models over the same recording: stock openai/whisper-small, and the
same model with a LoRA adapter trained on the SpeeD-IA Awadhi corpus. Both use
identical decoding settings, so the only difference between the two columns is
the adapter.

Runs on CPU. The Space has no GPU, and whisper-small is small enough that a
short clip returns in a few seconds.
"""

from __future__ import annotations

import time

import gradio as gr
import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-small"
ADAPTER = "abhshkp/homespun-awadhi-lora"
LANGUAGE = "hi"          # Whisper has no Awadhi id; pinned for both models
TARGET_SR = 16_000
MAX_SECONDS = 30         # Whisper's window; longer input is silently truncated

print("loading processor and base model")
processor = WhisperProcessor.from_pretrained(BASE)
stock = WhisperForConditionalGeneration.from_pretrained(BASE).eval()

print("loading adapter")
_adapted = WhisperForConditionalGeneration.from_pretrained(BASE)
adapted = PeftModel.from_pretrained(_adapted, ADAPTER).merge_and_unload().eval()

FORCED = processor.get_decoder_prompt_ids(language=LANGUAGE, task="transcribe")


def to_mono_16k(sr: int, data: np.ndarray) -> np.ndarray:
    """Gradio hands back whatever the browser recorded. Whisper wants one
    channel at 16 kHz, so normalise here rather than trusting the input."""
    x = data.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.max(np.abs(x)) or 1.0
    if peak > 1.0:
        x = x / peak
    if sr != TARGET_SR:
        try:
            import soxr

            x = soxr.resample(x, sr, TARGET_SR, quality="HQ")
        except ImportError:
            n = int(round(len(x) * TARGET_SR / sr))
            x = np.interp(
                np.linspace(0, len(x) - 1, n), np.arange(len(x)), x
            ).astype(np.float32)
    return x


@torch.inference_mode()
def run(model, audio: np.ndarray) -> str:
    feats = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features
    ids = model.generate(
        feats,
        forced_decoder_ids=FORCED,
        num_beams=1,          # greedy, so results are reproducible
        do_sample=False,
        max_new_tokens=200,
    )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe(recorded, uploaded):
    source = recorded or uploaded
    if source is None:
        return "", "", "Record something or upload a file first."

    sr, data = source
    audio = to_mono_16k(sr, data)
    seconds = len(audio) / TARGET_SR

    if seconds < 0.3:
        return "", "", "That clip is too short to transcribe."
    truncated = seconds > MAX_SECONDS
    if truncated:
        audio = audio[: MAX_SECONDS * TARGET_SR]

    t0 = time.time()
    a = run(stock, audio)
    b = run(adapted, audio)
    elapsed = time.time() - t0

    note = f"{seconds:.1f}s of audio, both models, {elapsed:.1f}s on CPU."
    if truncated:
        note += f" Only the first {MAX_SECONDS}s was transcribed."
    if a.strip() == b.strip():
        note += " Both models agree on this one."
    return a, b, note


INTRO = """
# Homespun

**Awadhi** is spoken by around four million people in Uttar Pradesh. General
speech models transcribe it by translating it into standard Hindi, dropping the
local vocabulary, and on longer clips they stop transcribing and repeat one
syllable until they run out of room.

Record yourself or upload a clip. The same audio goes through stock
`whisper-small` and through the same model with an 8.7 MB adapter trained on
Awadhi. Identical decoding settings; the adapter is the only difference.

Awadhi or Hindi speech works best. The model was trained on 8 kHz telephone
quality recordings of people describing birth, marriage and mourning customs.
"""

NOTE = """
Trained on **SpeeD-IA Awadhi** (Kumar et al., Interspeech 2022), CC BY-NC-SA 4.0.
The adapter is a derivative work under the same licence.
On the held-out test split: word error 1.02 to 0.62, character error 0.63 to
0.36, Awadhi marker words kept 5% to 45%.
"""

with gr.Blocks(title="Homespun", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="numpy", label="Record")
        upload = gr.Audio(sources=["upload"], type="numpy", label="Or upload wav / mp3")

    go = gr.Button("Transcribe with both models", variant="primary")

    with gr.Row():
        out_stock = gr.Textbox(label="Stock whisper-small", lines=4, show_copy_button=True)
        out_tuned = gr.Textbox(label="Retrained on Awadhi", lines=4, show_copy_button=True)
    status = gr.Markdown()

    go.click(transcribe, inputs=[mic, upload], outputs=[out_stock, out_tuned, status])
    gr.Markdown(NOTE)

if __name__ == "__main__":
    demo.launch()
