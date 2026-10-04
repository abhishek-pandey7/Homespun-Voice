"""Task 3 - join the SpeeD-IA Awadhi audio to its transcriptions.

Audio comes from the authors' Google Drive folder; transcriptions come from the
GitHub repository. Neither half is useful alone, and the join is on utterance id.

Safe to run while the download is still in progress: it reports coverage against
the full transcript set, so a partial corpus is visible as a partial corpus
rather than mistaken for a complete one.

    python scripts/import_speedia.py
    python scripts/import_speedia.py --audio-root data/speedia_tmp --strict
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homespun.speedia import (  # noqa: E402
    index_audio,
    join,
    read_transcripts,
    write_index,
)

TRANSCRIPT_DIR = Path("data/speedia/transcripts")
AUDIO_ROOT = Path("data/speedia_tmp")
INDEX_PATH = Path("data/speedia/index.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript-dir", type=Path, default=TRANSCRIPT_DIR)
    ap.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    ap.add_argument("--out", type=Path, default=INDEX_PATH)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any transcript lacks audio")
    args = ap.parse_args()

    if not args.transcript_dir.exists():
        sys.exit(f"error: {args.transcript_dir} not found - fetch the TSVs first")

    transcripts = read_transcripts(args.transcript_dir)
    audio = index_audio(args.audio_root)
    joined, missing_audio, missing_text = join(transcripts, audio)

    print(f"transcripts : {len(transcripts)}")
    print(f"audio files : {len(audio)}")
    print(f"joined      : {len(joined)}")
    print()

    expected = collections.Counter((s, ss) for _, s, ss in transcripts.values())
    got = collections.Counter((u.split, u.subset) for u in joined)
    print(f"{'split':6} {'subset':12} {'joined':>7} {'expected':>9} {'coverage':>9}")
    for key in sorted(expected):
        split, subset = key
        e, g = expected[key], got.get(key, 0)
        print(f"{split:6} {subset:12} {g:7} {e:9} {g / e:8.1%}")

    print()
    print(f"transcript without audio : {len(missing_audio)}")
    print(f"audio without transcript : {len(missing_text)}")
    if missing_text[:3]:
        print(f"  e.g. {', '.join(missing_text[:3])}")

    speakers = collections.defaultdict(set)
    for u in joined:
        speakers[u.speaker_id].add(u.split)
    both = [s for s, sp in speakers.items() if len(sp) > 1]
    print()
    print(f"speakers : {len(speakers)}")
    if both:
        print(f"  {len(both)} appear in BOTH train and test - the corpus split is")
        print("  by utterance, not by speaker. Results describe adaptation with")
        print("  speaker overlap and say nothing about an unseen speaker.")

    write_index(args.out, joined)
    print()
    print(f"[ok] wrote {args.out} ({len(joined)} records)")

    if args.strict and missing_audio:
        print(f"[strict] {len(missing_audio)} transcripts have no audio")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
