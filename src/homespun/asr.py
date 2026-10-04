"""Whisper inference, with or without a LoRA adapter.

The same code path serves the baseline and the adapted model, and the decoding
configuration is fixed rather than passed in. That is deliberate: if the two runs
could differ in language hint, beam count or temperature, any WER difference
between them would be unattributable. The only thing allowed to vary is the
adapter.

Whisper has no Awadhi language id, so the hint is pinned to Hindi (`hi`) for every
run. This caps achievable accuracy, equally for both models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"
SAMPLE_RATE = 16_000


@dataclass
class DecodeConfig:
    """Decoding settings, recorded into every output file.

    Whatever these are, both models get the same ones. The point of the
    benchmark is the adapter, so decoding must not be a second variable.
    """

    language: str = LANGUAGE
    task: str = TASK
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 200
    # 0 disables. Whisper's characteristic failure on out-of-distribution audio
    # is to lock into one syllable and emit it until the token limit; blocking
    # repeated n-grams attacks that directly rather than hoping the model
    # outgrows it.
    no_repeat_ngram_size: int = 0

    def to_dict(self) -> dict:
        return {
            "base_model": BASE_MODEL,
            "language": self.language,
            "task": self.task,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }


class Transcriber:
    def __init__(
        self,
        adapter_path: str | Path | None = None,
        device: str | None = None,
        config: DecodeConfig | None = None,
    ) -> None:
        self.config = config or DecodeConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("  [warn] CUDA unavailable - running on CPU, which will be ~50x slower")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(BASE_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=dtype
        )

        self.adapter_path = str(adapter_path) if adapter_path else None
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()  # fold LoRA in; no runtime overhead

        self.model = model.to(self.device).eval()
        self.forced_ids = self.processor.get_decoder_prompt_ids(
            language=self.config.language, task=self.config.task
        )

    def _load(self, path: str | Path) -> np.ndarray:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{path} is {sr} Hz; expected {SAMPLE_RATE}. Run corpus_qa.py.")
        return x

    @torch.inference_mode()
    def transcribe_batch(self, paths: list[str | Path]) -> list[str]:
        audio = [self._load(p) for p in paths]
        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        features = inputs.input_features.to(self.device, self.model.dtype)
        kwargs = {}
        if self.config.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        generated = self.model.generate(
            features,
            forced_decoder_ids=self.forced_ids,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            max_new_tokens=self.config.max_new_tokens,
            **kwargs,
        )
        return [
            t.strip()
            for t in self.processor.batch_decode(generated, skip_special_tokens=True)
        ]

    def describe(self) -> dict:
        return {
            **self.config.to_dict(),
            "adapter": self.adapter_path,
            "device": self.device,
            "dtype": str(self.model.dtype),
        }
