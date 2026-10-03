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


def validate_quotes(story: dict[str, Any], sources: dict[str, str],
                    min_quotes: int = 1) -> ValidationResult:
    """Every quote must be a verbatim substring of its source utterance.

    Substring rather than equality: quoting part of a long answer is legitimate
    editing. Altering any character of it is not.

    ``min_quotes`` closes a hole that a first version shipped with: a story
    containing no quotes at all passed, because every one of its zero quotes was
    verbatim. A check that accepts an empty result is worse than no check, since
    it reports success.
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
    if result.checked < min_quotes:
        result.failures.append({
            "utt_id": "-",
            "problem": f"only {result.checked} quotes, need at least {min_quotes}",
        })
    result.ok = not result.failures
    return result


def assemble(plan: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    """Turn the model's id-only plan into a story, inserting the text ourselves.

    The model never handles Awadhi text. It returns chapter titles, English
    narration and lists of utterance ids; the quotes are looked up here straight
    from the corpus. Verbatim fidelity stops being something to verify after the
    fact and becomes structurally impossible to break -- the model has no channel
    through which to alter a word.

    Ids the model invented are dropped and reported by the caller via coverage.
    """
    # Small models return the chapter list bare about as often as they wrap it
    # in the requested object, so accept either shape rather than crashing.
    if isinstance(plan, list):
        plan = {"chapters": plan}
    chapters = []
    seen: set[str] = set()
    for ch in plan.get("chapters", []):
        if not isinstance(ch, dict):
            continue
        entries = []
        for utt_id in ch.get("utt_ids", []):
            utt_id = str(utt_id)
            if utt_id in seen or utt_id not in sources:
                continue
            seen.add(utt_id)
            entries.append({"utt_id": utt_id, "quote": sources[utt_id], "bridge": ""})
        if entries:
            chapters.append({
                "title": ch.get("title", ""),
                "subtitle": ch.get("subtitle", ""),
                "intro": ch.get("intro", ""),
                "entries": entries,
            })
    return {
        "title": plan.get("title", "Vocalia"),
        "subtitle": plan.get("subtitle", ""),
        "chapters": chapters,
    }


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
    """gemma-2-2b-it through transformers, on the local GPU.

    Quantised to 4-bit by default. At bf16 the 2.6B parameters occupy about
    5.2 GB, which on a 6 GB card leaves nothing for the prompt and KV cache --
    it OOMs partway through generation. NF4 brings the weights to roughly
    1.5 GB and leaves room to actually generate.
    """

    def __init__(self, model_id: str = "google/gemma-2-2b-it",
                 load_4bit: bool = True) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        kwargs: dict = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if load_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float32

        self.name = f"{model_id} (local{', 4-bit nf4' if load_4bit else ''})"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

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
