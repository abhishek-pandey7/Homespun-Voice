"""Task 10 - synthesise English narration for each chapter via ElevenLabs.

Only the narration is synthesised. The Awadhi is never sent to a TTS service:
the reader plays the speakers' own recordings for every quote. Synthesising
their words would impose Hindi phonology on Awadhi and flatten exactly the
distinctions this project exists to preserve -- and a real voice is better than
a generated one regardless.

Credits are finite, so an already-rendered chapter is skipped unless --force, and
--audition renders only the first chapter so a voice can be judged before the
whole run is paid for.

    python scripts/narrate.py --audition
    python scripts/narrate.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

CHAPTERS = Path("data/story/chapters.json")
OUT_DIR = Path("web/assets/audio")
MANIFEST = Path("data/story/narration_manifest.json")
API = "https://api.elevenlabs.io/v1/text-to-speech"
MODEL = "eleven_multilingual_v2"

# Free accounts cannot call Voice Library voices through the API; the request
# comes back 402 paid_plan_required. This one is reachable without a paid plan,
# so it is used as a fallback rather than failing the run outright.
FALLBACK_VOICE = "pNInz6obpgDQGcFmaJgB"  # Adam


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader. Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _words(text: str) -> set[str]:
    return {w.strip(".,;:").lower() for w in text.split() if len(w) > 3}


def narration_for(chapter: dict, index: int) -> str:
    """The English text for one chapter. Quotes are deliberately excluded.

    The subtitle and intro are dropped when they restate each other or the
    title. A small planner tends to paraphrase itself - harmless on the page
    where the eye skips it, grating when read aloud twice in ten seconds.
    """
    title = (chapter.get("title") or "").strip()
    parts = [f"Chapter {index}. {title}"]
    said = _words(title)

    for key in ("subtitle", "intro"):
        text = (chapter.get(key) or "").strip()
        if not text:
            continue
        new = _words(text)
        # Mostly-recycled wording adds nothing a listener has not just heard.
        if new and len(new - said) / len(new) < 0.4:
            continue
        said |= new
        parts.append(text if text.endswith(".") else text + ".")

    return " ".join(parts)


def _post(text: str, voice_id: str, api_key: str):
    return requests.post(
        f"{API}/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": MODEL,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=120,
    )


def synthesise(text: str, voice_id: str, api_key: str, dest: Path) -> tuple[int, str]:
    response = _post(text, voice_id, api_key)

    if response.status_code == 402 and voice_id != FALLBACK_VOICE:
        print(f"       voice {voice_id} needs a paid plan; "
              f"falling back to {FALLBACK_VOICE}")
        voice_id = FALLBACK_VOICE
        response = _post(text, voice_id, api_key)

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs returned {response.status_code}: {response.text[:300]}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return len(response.content), voice_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chapters", type=Path, default=CHAPTERS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--audition", action="store_true",
                    help="render only chapter 1, to judge the voice first")
    ap.add_argument("--force", action="store_true",
                    help="re-render chapters that already exist (spends credits)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the text and character cost, call nothing")
    args = ap.parse_args()

    load_env()
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")

    if not args.dry_run:
        if not api_key:
            sys.exit("error: ELEVENLABS_API_KEY not set. Put it in .env.")
        if not voice_id:
            sys.exit("error: ELEVENLABS_VOICE_ID not set. Put it in .env.")

    if not args.chapters.exists():
        sys.exit(f"error: {args.chapters} not found - run scripts/build_story.py")
    story = json.loads(args.chapters.read_text(encoding="utf-8"))
    chapters = story.get("chapters", [])
    if not chapters:
        sys.exit("error: the story has no chapters")

    if args.audition:
        chapters = chapters[:1]
        print("audition: rendering chapter 1 only\n")

    records, total_chars, spent = [], 0, 0
    for i, chapter in enumerate(chapters, 1):
        text = narration_for(chapter, i)
        total_chars += len(text)
        dest = args.out / f"chapter_{i}.mp3"

        if args.dry_run:
            print(f"chapter {i}: {len(text)} chars\n  {text}\n")
            continue
        if dest.exists() and not args.force:
            print(f"[skip] {dest.name} exists ({dest.stat().st_size // 1024} KB)")
            records.append({"chapter": i, "file": dest.name, "status": "existing"})
            continue

        print(f"[tts ] chapter {i}: {len(text)} chars -> {dest.name}")
        size, used_voice = synthesise(text, voice_id, api_key, dest)
        spent += len(text)
        records.append({
            "chapter": i,
            "title": chapter.get("title", ""),
            "file": dest.name,
            "characters": len(text),
            "bytes": size,
            "voice_id": used_voice,
            "model": MODEL,
            "rendered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "rendered",
        })

    if args.dry_run:
        print(f"total {total_chars} characters across {len(chapters)} chapters "
              f"- nothing was sent")
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if MANIFEST.exists():
        existing = {r["chapter"]: r for r in
                    json.loads(MANIFEST.read_text(encoding="utf-8")).get("chapters", [])}
    for r in records:
        if r.get("status") == "rendered" or r["chapter"] not in existing:
            existing[r["chapter"]] = r
    MANIFEST.write_text(json.dumps({
        "model": MODEL,
        "voice_id": voice_id,
        "chapters": [existing[k] for k in sorted(existing)],
    }, indent=2), encoding="utf-8")

    print(f"\n[ok] {MANIFEST}")
    print(f"     {spent} characters sent this run")
    if args.audition:
        print(f"     listen to {args.out / 'chapter_1.mp3'}, then run without "
              f"--audition for the rest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
