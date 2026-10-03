"""Story assembly: model access, JSON extraction, and the verbatim validator.

The validator is the important part of this module. A language model asked to
arrange dialect testimony will, left alone, quietly tidy it -- converting होत to
होता है, अहय to है, मा to में. That tidying is precisely the harm this project
exists to measure, so it is checked mechanically rather than trusted: every quote
must appear character for character in its source utterance, and a single
mismatch fails the whole generation.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a model response that may be fenced or chatty."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def _canon(s: str) -> str:
    """NFC plus whitespace collapse.

    Deliberately narrow. Unicode normalisation and spacing are rendering
    concerns; anything beyond that -- punctuation, spelling, word forms -- is
    content, and letting it differ is exactly what the validator is for.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


@dataclass
class ValidationResult:
    ok: bool
    checked: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    def report(self) -> str:
        if self.ok:
            return f"all {self.checked} quotes verbatim"
        lines = [f"{len(self.failures)} of {self.checked} quotes are not verbatim:"]
        for f in self.failures[:5]:
            lines += [
                f"  utt_id {f['utt_id']}: {f['problem']}",
                f"    source: {f.get('source', '')[:110]}",
                f"    quote : {f.get('quote', '')[:110]}",
            ]
        return "\n".join(lines)


def validate_quotes(story: dict[str, Any], sources: dict[str, str]) -> ValidationResult:
    """Every quote must be a verbatim substring of its source utterance.

    Substring rather than equality: quoting part of a long answer is legitimate
    editing. Altering any character of it is not.
    """
    result = ValidationResult(ok=True)
    for chapter in story.get("chapters", []):
        for entry in chapter.get("entries", []):
            result.checked += 1
            utt_id = str(entry.get("utt_id", ""))
            quote = entry.get("quote", "")
            source = sources.get(utt_id)
            if source is None:
                result.failures.append({
                    "utt_id": utt_id, "quote": quote,
                    "problem": "utt_id not in the source set (invented or mistyped)",
                })
                continue
            if _canon(quote) not in _canon(source):
                result.failures.append({
                    "utt_id": utt_id, "quote": quote, "source": source,
                    "problem": "quote is not a verbatim substring of the source",
                })
    result.ok = not result.failures
    return result


def check_coverage(story: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    used: list[str] = []
    for chapter in story.get("chapters", []):
        for entry in chapter.get("entries", []):
            used.append(str(entry.get("utt_id", "")))
    duplicates = sorted({u for u in used if used.count(u) > 1})
    return {
        "used": len(set(used)),
        "available": len(sources),
        "duplicates": duplicates,
        "unused": sorted(set(sources) - set(used)),
    }


# --------------------------------------------------------------------- models ---

class StoryModel:
    """Base interface so the script does not care where Gemma runs."""

    name = "unset"

    def generate(self, prompt: str, max_new_tokens: int = 4096) -> str:
        raise NotImplementedError


class LocalGemma(StoryModel):
    """gemma-2-2b-it through transformers, on the local GPU."""

    def __init__(self, model_id: str = "google/gemma-2-2b-it") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"{model_id} (local)"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()

    def generate(self, prompt: str, max_new_tokens: int = 4096) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        return self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )


class HostedGemma(StoryModel):
    """Any OpenAI-compatible chat endpoint, e.g. a hosted gemma-2-9b-it."""

    def __init__(self, model_id: str = "gemma-2-9b-it") -> None:
        import requests

        self.requests = requests
        self.base = os.environ.get("GEMMA_API_BASE", "").rstrip("/")
        self.key = os.environ.get("GEMMA_API_KEY", "")
        if not self.base or not self.key:
            raise RuntimeError(
                "GEMMA_API_BASE and GEMMA_API_KEY must be set for the hosted model. "
                "Copy .env.example to .env and fill them in, or use --model local."
            )
        self.model_id = model_id
        self.name = f"{model_id} (hosted)"

    def generate(self, prompt: str, max_new_tokens: int = 4096) -> str:
        response = self.requests.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
