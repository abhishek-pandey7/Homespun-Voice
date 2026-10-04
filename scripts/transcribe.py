"""Task 5 / Task 8 - transcribe a split with the stock model or the adapted one.

One script serves both so that decoding cannot drift between the two runs. The
output records the exact decode configuration, which is what makes a later
comparison defensible.

Resumable: an interrupted run continues from what is already on disk rather than
re-spending GPU time.

    python scripts/transcribe.py --model baseline --split test
    python scripts/transcribe.py --model lora --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MANIFEST = Path("data/speedia/manifest.jsonl")
OUT_DIR = Path("data/transcripts")
ADAPTER = Path("models/homespun-lora")


def load_manifest(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    # Longest first: a batch is padded to its longest member, so grouping similar
    # lengths together wastes less compute than manifest order would.
    return sorted(rows, key=lambda r: -r["duration_s"])


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done[row["utt_id"]] = row
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["baseline", "lora"], required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--adapter", type=Path, default=ADAPTER)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ignore existing output")
    ap.add_argument("--label", default=None,
                    help="name recorded in the output rows; defaults to --model. "
                         "Use it to distinguish two adapters in one comparison.")
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.split)
    if not rows:
        sys.exit(f"error: no '{args.split}' rows in {args.manifest}")
    if args.limit:
        rows = rows[: args.limit]

    out = args.out or OUT_DIR / f"{args.label or args.model}_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {} if args.force else load_done(out)
    todo = [r for r in rows if r["utt_id"] not in done]
    print(f"{args.label or args.model} / {args.split}: {len(rows)} utterances, "
          f"{len(done)} already done, {len(todo)} to go")
    if not todo:
        print("[ok] nothing to do")
        return 0

    from homespun.asr import Transcriber  # imported late so --help stays instant

    adapter = args.adapter if args.model == "lora" else None
    if adapter and not Path(adapter).exists():
        sys.exit(f"error: adapter not found at {adapter} - run train_lora.py first")

    print(f"loading model{' + adapter' if adapter else ''}...")
    t = Transcriber(adapter_path=adapter)
    cfg = t.describe()
    print(f"  {cfg['base_model']}  device={cfg['device']}  dtype={cfg['dtype']}  "
          f"lang={cfg['language']}  beams={cfg['num_beams']}")

    mode = "w" if args.force else "a"
    t0 = time.time()
    audio_s = 0.0
    written = 0

    with out.open(mode, encoding="utf-8", newline="\n") as fh:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            texts = t.transcribe_batch([r["audio_path"] for r in batch])
            for row, text in zip(batch, texts):
                fh.write(json.dumps({
                    "utt_id": row["utt_id"],
                    "hypothesis": text,
                    "reference": row["text"],
                    "split": row["split"],
                    "subset": row["subset"],
                    "speaker_id": row["speaker_id"],
                    "duration_s": row["duration_s"],
                    "model": args.label or args.model,
                }, ensure_ascii=False) + "\n")
                audio_s += row["duration_s"]
                written += 1
            fh.flush()
            elapsed = time.time() - t0
            print(f"  {written}/{len(todo)}  {elapsed:5.0f}s  "
                  f"{audio_s/elapsed:5.1f}x realtime", end="\r")

    elapsed = time.time() - t0
    print()
    print(f"[ok] {out}  ({written} new, {len(done) + written} total)")
    print(f"     {elapsed:.0f}s for {audio_s/60:.1f} min of audio "
          f"({audio_s/elapsed:.1f}x realtime)")

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        **cfg,
        "split": args.split,
        "utterances": len(done) + written,
        "audio_seconds": round(audio_s, 1),
        "wall_seconds": round(elapsed, 1),
    }, indent=2), encoding="utf-8")
    print(f"[ok] {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
