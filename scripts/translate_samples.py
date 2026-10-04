"""Translate the reference text behind every comparison shown on the site.

The exhibit and the worked examples were two blocks of Devanagari side by side.
A reader who does not know the script can see that they differ and cannot see
which one is right, which is most of the point lost.

Only the reference is translated: what the speaker actually said. The model
outputs are not, because translating a failure is meaningless - a line that
repeats one syllable forty times has no English, and inventing one would
flatter it. Those get a computed description instead.

Results cache to data/story/sample_translations.json keyed by the Awadhi text,
so re-running the export does not re-run the model.

    python scripts/translate_samples.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from translate_story import (  # noqa: E402
    translate_in_pieces,
    translate_once,
)
from homespun.story import HostedGemma, LocalGemma  # noqa: E402

CACHE = Path("data/story/sample_translations.json")


def collect_references() -> list[str]:
    """Every reference string the site renders in a comparison."""
    texts: list[str] = []
    for path, key in ((Path("web/data/metrics.json"), "samples"),
                      (Path("web/data/examples.json"), "examples")):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get(key, []):
            if row.get("reference"):
                texts.append(row["reference"])
    # Stable order, no duplicates.
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["local", "hosted"], default="local")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    refs = collect_references()
    if not refs:
        sys.exit("error: no references found - run scripts/export_web.py first")

    cache: dict[str, str] = {}
    if CACHE.exists() and not args.force:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))

    todo = [r for r in refs if r not in cache or not cache[r]]
    print(f"{len(refs)} references, {len(todo)} to translate")
    if not todo:
        print("[ok] all cached")
        return 0

    model = (LocalGemma("google/gemma-2-2b-it", load_4bit=not args.no_4bit)
             if args.model == "local" else HostedGemma("gemma-2-9b-it"))
    print(f"model: {model.name}\n")

    done = failed = 0
    for i, text in enumerate(todo, 1):
        english, why = translate_once(model, text, set(), args.retries)
        if not english:
            english, why = translate_in_pieces(model, text, args.retries)
        cache[text] = english
        if english:
            done += 1
            print(f"  {i}/{len(todo)}  {english[:66]}")
        else:
            failed += 1
            print(f"  {i}/{len(todo)}  SKIPPED ({why})")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] {CACHE}")
    print(f"     translated {done}, skipped {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
