"""Task 9 - assemble storybook chapters from Awadhi testimony with Gemma 2.

Quote text comes from the corpus's human transcriptions, not from the ASR output.
At the adapter's current error rate an ASR-sourced storybook would be substantially
wrong, which would contradict the project's own argument about preserving what was
actually said. The adapter's contribution is demonstrated in the benchmark, and the
reader shows reference, baseline and adapted side by side so nothing is hidden.

Generation is rejected unless every quote is verbatim.

    python scripts/build_story.py --model local
    python scripts/build_story.py --model hosted --utterances 40
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocalia.story import (  # noqa: E402
    HostedGemma,
    LocalGemma,
    check_coverage,
    extract_json,
    validate_quotes,
)

MANIFEST = Path("data/speedia/manifest.jsonl")
PROMPT = Path("prompts/story.txt")
OUT = Path("data/story/chapters.json")

ATTRIBUTION = {
    "corpus": "SpeeD-IA Awadhi",
    "citation": "Kumar et al., Annotated Speech Corpus for Low Resource Indian "
                "Languages: Awadhi, Bhojpuri, Braj and Magahi, "
                "Speech for Social Good Workshop, Interspeech 2022",
    "corpus_url": "https://github.com/unrealtecellp/SpeeD-IA",
    "licence": "CC BY-NC-SA 4.0",
    "note": "Quote text is the corpus's human transcription. This derivative work "
            "carries the same licence.",
}


def load_utterances(manifest: Path, subset: str, limit: int) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["subset"] == subset:
                    rows.append(r)
    # Longest first: substantial answers make better testimony than one-liners.
    rows.sort(key=lambda r: -len(r["text"]))
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["local", "hosted"], default="local")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--subset", default="lifecycle")
    ap.add_argument("--utterances", type=int, default=35)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--prompt", type=Path, default=PROMPT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--retries", type=int, default=2,
                    help="re-ask with the validator's complaint attached")
    args = ap.parse_args()

    rows = load_utterances(args.manifest, args.subset, args.utterances)
    if not rows:
        sys.exit(f"error: no '{args.subset}' rows in {args.manifest}")
    sources = {r["utt_id"]: r["text"] for r in rows}
    print(f"{len(rows)} '{args.subset}' utterances as source material")

    template = args.prompt.read_text(encoding="utf-8")
    listing = "\n".join(f'{r["utt_id"]}\t{r["text"]}' for r in rows)
    prompt = template + listing + "\n"

    if args.model == "local":
        model = LocalGemma(args.model_id or "google/gemma-2-2b-it")
    else:
        model = HostedGemma(args.model_id or "gemma-2-9b-it")
    print(f"model: {model.name}")

    story = None
    last = ""
    for attempt in range(1, args.retries + 2):
        print(f"\nattempt {attempt}...")
        ask = prompt if attempt == 1 else (
            prompt + "\n\nYour previous response was rejected:\n" + last +
            "\nCopy every quote character for character from the source above.\n"
        )
        raw = model.generate(ask)
        try:
            candidate = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            last = f"the response was not valid JSON ({exc})"
            print(f"  rejected: {last}")
            continue

        result = validate_quotes(candidate, sources)
        print(f"  {result.report()}")
        if result.ok:
            story = candidate
            break
        last = result.report()

    if story is None:
        print("\nfailed: no generation passed the verbatim check.")
        print("Nothing written - a storybook with rewritten dialect is worse than none.")
        return 1

    coverage = check_coverage(story, sources)
    print(f"\ncoverage: {coverage['used']}/{coverage['available']} utterances used")
    if coverage["duplicates"]:
        print(f"  warning: repeated utt_ids {coverage['duplicates'][:5]}")

    payload = {
        "generated_on": date.today().isoformat(),
        "model": model.name,
        "subset": args.subset,
        "quote_source": "corpus human transcription",
        "attribution": ATTRIBUTION,
        "coverage": {k: v for k, v in coverage.items() if k != "unused"},
        **story,
        "audio": {r["utt_id"]: Path(r["audio_path"]).name for r in rows},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n[ok] {args.out}")
    print(f"     {len(story.get('chapters', []))} chapters, "
          f"{coverage['used']} quotes, all verbatim")
    for ch in story.get("chapters", []):
        print(f"     - {ch.get('title', '?')} ({len(ch.get('entries', []))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
