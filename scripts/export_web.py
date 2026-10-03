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

import jiwer  # noqa: E402

from vocalia.metrics import normalise, score, vocabulary_recall  # noqa: E402

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
        })

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
    wanted = [str(e.get("utt_id")) for ch in story.get("chapters", [])
              for e in ch.get("entries", [])]
    copied = missing = 0
    for utt_id in wanted:
        row = manifest.get(utt_id)
        if not row:
            missing += 1
            continue
        src = Path(row["audio_path"])
        if src.exists():
            shutil.copyfile(src, WEB_AUDIO / f"{utt_id}.wav")
            copied += 1
        else:
            missing += 1
    size_mb = sum(f.stat().st_size for f in WEB_AUDIO.glob("*.wav")) / 1024**2
    print(f"[ok] {WEB_AUDIO}: {copied} clips ({size_mb:.1f} MB), {missing} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
