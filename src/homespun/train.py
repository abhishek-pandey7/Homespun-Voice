"""Dataset, collator and LoRA setup for fine-tuning whisper-small.

Features are extracted on the fly rather than precomputed. A single 30-second
log-mel spectrogram is 80 x 3000 floats, so caching the corpus would cost several
gigabytes on disk to save work that the GPU is not waiting on anyway.

The validation slice is carved out of the *training* split. The corpus test split
is never read here, by anything, for any reason -- early stopping against the test
set is the most common way a reported benchmark becomes a lie.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000


class SpeechDataset(Dataset):
    """Utterance rows -> (input_features, labels)."""

    def __init__(self, rows: list[dict], processor, language: str, task: str) -> None:
        self.rows = rows
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.set_prefix_tokens(language=language, task=task)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{row['audio_path']} is {sr} Hz; expected {SAMPLE_RATE}")

        features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_features[0]
        labels = self.tokenizer(row["text"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class SpeechCollator:
    """Pad a batch; mask padded label positions with -100 so loss ignores them."""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The tokenizer prepends BOS and the model adds it again when shifting
        # labels right, so strip the duplicate.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def split_train_validation(
    rows: list[dict], fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Carve a validation slice out of train, deterministically.

    Split at the utterance level, matching how the corpus authors split, and
    seeded so a re-run trains on exactly the same data.
    """
    rows = sorted(rows, key=lambda r: r["utt_id"])  # stable regardless of file order
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def build_model(cfg: dict):
    """Load whisper-small and wrap the attention projections in LoRA adapters."""
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])

    # Generation-time constraints must be cleared for training, or the forced
    # prefix fights the labels being learned.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    tr = cfg["training"]
    if tr.get("spec_augment"):
        # Masking spans of time and frequency during training only. For a small
        # single-domain corpus this is the cheapest regularisation available,
        # and it costs nothing at inference.
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = tr.get("mask_time_prob", 0.05)
        model.config.mask_feature_prob = tr.get("mask_feature_prob", 0.05)

    if cfg["training"]["gradient_checkpointing"]:
        model.config.use_cache = False
        # Reentrant checkpointing only records a graph for inputs that require
        # grad. With a frozen base model none do, so backward finds no grad_fn
        # and raises. Making the embedding outputs require grad restores the
        # path; the trainer is also told to use the non-reentrant implementation.
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    return model


def trainable_summary(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return (f"trainable {trainable:,} / {total:,} params "
            f"({trainable / total:.3%})")


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3

MAX_LABEL_TOKENS = 448  # whisper max_target_positions


def filter_by_label_length(
    rows: list[dict], processor, language: str, task: str,
    max_tokens: int = MAX_LABEL_TOKENS,
) -> tuple[list[dict], list[dict]]:
    """Drop utterances whose transcript exceeds Whisper's decoder limit.

    Whisper cannot represent more than 448 target tokens, and passing a longer
    label raises rather than truncating. Truncating ourselves would be worse than
    dropping: the model would learn to stop mid-sentence. So these are excluded
    and counted, like every other exclusion in this pipeline.
    """
    tok = processor.tokenizer
    tok.set_prefix_tokens(language=language, task=task)
    kept, dropped = [], []
    for row in rows:
        n = len(tok(row["text"]).input_ids)
        (kept if n <= max_tokens else dropped).append({**row, "label_tokens": n})
    return kept, dropped
