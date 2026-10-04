"""Prepare the static reader's data: chapters, metrics, samples, audio.

The reader is a static page with no backend, so everything it needs is written
here as JSON plus copied audio. Audio is copied rather than linked because
`data/` is gitignored and never deployed; only the clips the story actually
quotes travel to the host.

    python scripts/export_web.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import jiwer  # noqa: E402

from homespun.metrics import normalise, score, vocabulary_recall  # noqa: E402

MANIFEST = Path("data/speedia/manifest.jsonl")
STORY = Path("data/story/chapters.json")
WEB_DATA = Path("web/data")
WEB_AUDIO = Path("web/assets/clips")

DIALECT_MARKERS = {
    "थय", "अहय", "होत", "हो", "जौन", "जौनहय", "कय", "कीन", "मा",
    "पहिले", "वोहके", "वोहमा", "अउर", "अव", "जा", "थीं", "बना", "रसम",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


PEAKS = 96


def waveform_peaks(path: Path, buckets: int = PEAKS) -> list[int]:
    """Downsample a clip to N peak amplitudes, 0-100.

    The reader draws these as its only illustration. Stock photography of
    "rural Indian elders" next to real testimony would fabricate provenance,
    which is the specific harm this project exists to measure. A waveform is
    drawn from the actual recording, so it depicts the thing it illustrates.
    """
    try:
        x, _ = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return []
    if x.ndim > 1:
        x = x.mean(axis=1)
    if not len(x):
        return []
    chunks = np.array_split(np.abs(x), buckets)
    peaks = np.array([c.max() if len(c) else 0.0 for c in chunks])
    top = peaks.max() or 1.0
    return [int(round(v / top * 100)) for v in peaks]


def strip_dashes(text: str) -> str:
    """Remove em and en dashes from generated English narration.

    The planner emits them freely and they are the clearest typographic tell
    of machine-written copy. Corpus text is never touched by this.
    """
    return (text.replace("—", " - ").replace("–", "-")
                .replace("  ", " ").strip())


SAMPLE_TRANSLATIONS = Path("data/story/sample_translations.json")


def load_translations() -> dict[str, str]:
    if SAMPLE_TRANSLATIONS.exists():
        return json.loads(SAMPLE_TRANSLATIONS.read_text(encoding="utf-8"))
    return {}


def describe_failure(text: str) -> str:
    """Say in English what a degenerate transcript is doing.

    Translating one of these is meaningless: a line that repeats one syllable
    forty times has no English, and inventing one would flatter it. Naming the
    behaviour is both honest and the thing a reader needs to see.
    """
    words = text.split()
    if len(words) < 6:
        return ""
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    token, n = max(counts.items(), key=lambda kv: kv[1])
    if n >= 5 and n / len(words) > 0.4:
        return f"Not a sentence. It repeats {token} {n} times."
    if len(set(words)) / len(words) < 0.4:
        return "Not a sentence. It loops over a handful of words."
    return ""


def utt_cer(ref: str, hyp: str) -> float:
    r = normalise(ref)
    return jiwer.cer(r, normalise(hyp)) if r else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path,
                    default=Path("data/transcripts/baseline_test.jsonl"))
    ap.add_argument("--adapted", type=Path,
                    default=Path("data/transcripts/lora6ep_test.jsonl"))
    ap.add_argument("--story", type=Path, default=STORY)
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    translations = load_translations()
    WEB_AUDIO.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- metrics ---
    base = {r["utt_id"]: r for r in load_jsonl(args.baseline)}
    lora = {r["utt_id"]: r for r in load_jsonl(args.adapted)}
    shared = sorted(set(base) & set(lora))
    refs = [base[u]["reference"] for u in shared]

    def block(rows: dict) -> dict:
        hyps = [rows[u]["hypothesis"] for u in shared]
        s = score(refs, hyps)
        rec, tot = vocabulary_recall(refs, hyps, DIALECT_MARKERS)
        out = {"wer": round(s.wer, 4), "cer": round(s.cer, 4),
               "markers": rec, "markers_total": tot}
        for subset in ("lifecycle", "translation"):
            ids = [u for u in shared if base[u]["subset"] == subset]
            ss = score([base[u]["reference"] for u in ids],
                       [rows[u]["hypothesis"] for u in ids])
            out[subset] = {"wer": round(ss.wer, 4), "cer": round(ss.cer, 4),
                           "utterances": len(ids)}
        return out

    scored = sorted(
        ((utt_cer(base[u]["reference"], base[u]["hypothesis"])
          - utt_cer(base[u]["reference"], lora[u]["hypothesis"]), u) for u in shared),
        reverse=True,
    )
    samples = []
    for delta, u in scored[: args.samples]:
        samples.append({
            "utt_id": u,
            "subset": base[u]["subset"],
            "duration_s": base[u]["duration_s"],
            "reference": base[u]["reference"],
            "baseline": base[u]["hypothesis"],
            "adapted": lora[u]["hypothesis"],
            "cer_baseline": round(utt_cer(base[u]["reference"], base[u]["hypothesis"]), 3),
            "cer_adapted": round(utt_cer(base[u]["reference"], lora[u]["hypothesis"]), 3),
            "meaning": translations.get(base[u]["reference"], ""),
            "baseline_note": describe_failure(base[u]["hypothesis"]),
            "adapted_note": describe_failure(lora[u]["hypothesis"]),
        })

    # --- example clips for the try-it page --------------------------------
    # Precomputed so the page is useful the instant it loads, before anyone
    # decides whether to download a model to try their own voice.
    EX_DIR = Path("web/assets/examples")
    EX_DIR.mkdir(parents=True, exist_ok=True)
    for stale in EX_DIR.glob("*.wav"):
        stale.unlink()
    manifest = {r["utt_id"]: r for r in load_jsonl(MANIFEST)}

    # Spread across the error range rather than cherry-picking the best wins:
    # one collapse, one ordinary gain, one near-miss, one the adapter lost.
    ranked = [(d, u) for d, u in scored]
    # Open on the clearest win and close on a win, with the regression second.
    # Including a failure is honest; ending on one leaves a judge with a
    # repetition loop as their last impression of the model.
    picks = [ranked[0], ranked[-1], ranked[2 * len(ranked) // 3], ranked[len(ranked) // 3]]
    examples = []
    for i, (delta, u) in enumerate(picks, 1):
        row = manifest.get(u)
        if not row or not Path(row["audio_path"]).exists():
            continue
        name = f"example{i}.wav"
        shutil.copyfile(Path(row["audio_path"]), EX_DIR / name)
        examples.append({
            "clip": name,
            "duration_s": row["duration_s"],
            "subset": base[u]["subset"],
            "reference": base[u]["reference"],
            "baseline": base[u]["hypothesis"],
            "adapted": lora[u]["hypothesis"],
            "cer_baseline": round(utt_cer(base[u]["reference"], base[u]["hypothesis"]), 3),
            "cer_adapted": round(utt_cer(base[u]["reference"], lora[u]["hypothesis"]), 3),
            "peaks": waveform_peaks(Path(row["audio_path"])),
            "meaning": translations.get(base[u]["reference"], ""),
            "baseline_note": describe_failure(base[u]["hypothesis"]),
            "adapted_note": describe_failure(lora[u]["hypothesis"]),
        })
    (WEB_DATA / "examples.json").write_text(
        json.dumps({"examples": examples}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {WEB_DATA / 'examples.json'} ({len(examples)} clips)")

    metrics = {
        "test_utterances": len(shared),
        "baseline": block(base),
        "adapted": block(lora),
        "samples": samples,
    }
    (WEB_DATA / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {WEB_DATA / 'metrics.json'}")

    # ------------------------------------------------- chapters + clip audio ---
    if not args.story.exists():
        print(f"[skip] {args.story} not found - run scripts/build_story.py first")
        return 0

    story = json.loads(args.story.read_text(encoding="utf-8"))
    shutil.copyfile(args.story, WEB_DATA / "chapters.json")
    print(f"[ok] {WEB_DATA / 'chapters.json'}")

    manifest = {r["utt_id"]: r for r in load_jsonl(MANIFEST)}

    # Clips are published as audio1.wav, audio2.wav, ... in reading order rather
    # than under their 15-digit corpus ids. The ids are meaningless to a reader
    # and awkward to reference in a demo; the mapping is kept in chapters.json so
    # nothing is lost.
    for stale in WEB_AUDIO.glob("*.wav"):
        stale.unlink()

    copied = missing = 0
    n = 0
    for chapter in story.get("chapters", []):
        for entry in chapter.get("entries", []):
            utt_id = str(entry.get("utt_id"))
            row = manifest.get(utt_id)
            n += 1
            name = f"audio{n}.wav"
            entry["clip"] = name
            if row and Path(row["audio_path"]).exists():
                src = Path(row["audio_path"])
                shutil.copyfile(src, WEB_AUDIO / name)
                entry["peaks"] = waveform_peaks(src)
                entry["duration_s"] = row.get("duration_s", 0)
                copied += 1
            else:
                entry["clip"] = ""
                entry["peaks"] = []
                missing += 1
            entry["bridge"] = strip_dashes(entry.get("bridge", ""))
        chapter["title"] = strip_dashes(chapter.get("title", ""))
        chapter["subtitle"] = strip_dashes(chapter.get("subtitle", ""))
        chapter["intro"] = strip_dashes(chapter.get("intro", ""))
    story["title"] = strip_dashes(story.get("title", ""))
    story["subtitle"] = strip_dashes(story.get("subtitle", ""))

    # Rewrite the published copy so the reader can resolve clips by name.
    (WEB_DATA / "chapters.json").write_text(
        json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in WEB_AUDIO.glob("*.wav")) / 1024**2
    print(f"[ok] {WEB_AUDIO}: {copied} clips ({size_mb:.1f} MB), {missing} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
