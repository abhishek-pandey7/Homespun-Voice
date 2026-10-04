"""Translate each Awadhi quotation into English for the reader.

The Awadhi is the record. The translation is a reading aid sitting beneath it,
labelled as machine output, and it never replaces or edits the original: this
script only adds an `english` field, so the quotation itself cannot be touched
by a translation pass that misreads a word.

One quote per request. Gemma 2B at 4-bit degrades badly on long prompts, and a
single bad batch would otherwise corrupt fifteen translations at once.

    python scripts/translate_story.py
    python scripts/translate_story.py --force     # re-translate everything
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homespun.story import HostedGemma, LocalGemma  # noqa: E402

STORY = Path("data/story/chapters.json")

# No description of what the sentences are about. An earlier version opened with
# "It is spoken testimony about family customs" and the 2B model echoed that line
# back as the translation for a third of the quotes. Few-shot examples teach the
# shape of the task without handing the model a sentence to copy.
PROMPT = """Translate Hindi and Awadhi sentences into English. Output only the English sentence.

Hindi: लरिका कय नामकरन वोहकय महतारी-बाप करा थीं ।
English: The child's naming is done by his mother and father.

Hindi: बारात आये के बाद दुआरे कय पूजा हो थय ।
English: After the wedding procession arrives, the doorway worship is performed.

Hindi: {text}
English:"""

# Phrases that mean the model described the task instead of performing it.
ECHOES = (
    "testimony", "family customs", "this sentence", "the sentence",
    "translation of", "awadhi", "dialect of hindi", "spoken about",
)

# The few-shot answers themselves. A small model will copy an example verbatim
# when it cannot read the input, which looks like a plausible translation until
# you notice the same sentence appearing under two different quotations.
EXAMPLE_OUTPUTS = (
    "the child's naming is done by his mother and father",
    "after the wedding procession arrives, the doorway worship is performed",
)


def clean(raw: str) -> str:
    """Take the first real line and strip the wrappers small models add."""
    text = raw.strip()
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
    # Drop a leading label the model sometimes repeats back.
    text = re.sub(r"^(english|translation)\s*[:\-]\s*", "", text, flags=re.I)
    for line in text.splitlines():
        line = line.strip().strip('"').strip()
        if line and not line.lower().startswith(("note", "here is", "this ")):
            text = line
            break
    # Em and en dashes are stripped for the same reason as everywhere else.
    return text.replace("—", " - ").replace("–", "-").strip()


def plausible(english: str, awadhi: str, seen: set[str]) -> tuple[bool, str]:
    """Reject obvious failures rather than publishing them."""
    if len(english) < 10:
        return False, "too short"
    if re.search(r"[ऀ-ॿ]", english):
        return False, "still contains Devanagari"
    if len(english) > len(awadhi) * 4:
        return False, "implausibly long"
    low = english.lower()
    hit = next((e for e in ECHOES if e in low), None)
    if hit:
        return False, f"echoes the prompt ({hit!r})"
    if any(ex in low for ex in EXAMPLE_OUTPUTS):
        return False, "copied a few-shot example"
    # Two different sentences translating identically means the model is
    # producing a stock phrase rather than reading the input.
    key = re.sub(r"[^a-z ]", "", low).strip()
    if key in seen:
        return False, "duplicate of an earlier translation"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--story", type=Path, default=STORY)
    ap.add_argument("--model", choices=["local", "hosted"], default="local")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-translate entries that already have english")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    if not args.story.exists():
        sys.exit(f"error: {args.story} not found - run scripts/build_story.py first")
    story = json.loads(args.story.read_text(encoding="utf-8"))

    entries = [e for ch in story.get("chapters", []) for e in ch.get("entries", [])]
    todo = [e for e in entries if args.force or not e.get("english")]
    print(f"{len(entries)} quotations, {len(todo)} to translate")
    if not todo:
        print("[ok] nothing to do")
        return 0

    if args.model == "local":
        model = LocalGemma(args.model_id or "google/gemma-2-2b-it",
                           load_4bit=not args.no_4bit)
    else:
        model = HostedGemma(args.model_id or "gemma-2-9b-it")
    print(f"model: {model.name}\n")

    done = failed = 0
    seen: set[str] = set()
    for i, entry in enumerate(todo, 1):
        awadhi = entry.get("quote", "")
        english, why = "", "no attempt"
        for _ in range(args.retries + 1):
            candidate = clean(model.generate(PROMPT.format(text=awadhi),
                                             max_new_tokens=220))
            ok, why = plausible(candidate, awadhi, seen)
            if ok:
                english = candidate
                seen.add(re.sub(r"[^a-z ]", "", candidate.lower()).strip())
                break

        if english:
            entry["english"] = english
            done += 1
            print(f"  {i}/{len(todo)}  {english[:72]}")
        else:
            entry["english"] = ""
            failed += 1
            print(f"  {i}/{len(todo)}  SKIPPED ({why})")

    story["translation"] = {
        "model": model.name,
        "note": "English translations are machine generated. The Awadhi "
                "quotation is the record; the translation is a reading aid.",
    }
    args.story.write_text(json.dumps(story, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print(f"\n[ok] {args.story}")
    print(f"     translated {done}, skipped {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
