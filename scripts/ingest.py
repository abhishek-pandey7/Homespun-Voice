"""Stage 1 — bring audio into the corpus as 16 kHz mono WAV with provenance.

Accepts either a local recording or a URL. Everything is normalised to the
format Whisper expects (16 kHz, mono, PCM) and recorded in
`data/raw/manifest.jsonl` alongside where it came from.

Provenance is not bookkeeping here. The dataset card, the licensing position and
the train/test split all read from this manifest, and `speaker_id` in particular
is what lets the split keep both speakers on both sides.

Examples
--------
Local recording from a consenting speaker::

    python scripts/ingest.py --file recordings/nani_01.m4a \
        --source-id sess_01 --speaker-id spk_f70 --split train \
        --origin "direct recording" --consent "verbal, recorded 2026-10-03"

Pipeline smoke test from a URL, first three minutes only::

    python scripts/ingest.py --url "https://youtu.be/VIDEO_ID" \
        --source-id smoke_01 --speaker-id unknown --split none \
        --purpose smoke-test --duration 180
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homespun.manifest import has_id, upsert_row  # noqa: E402

RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.jsonl"
SAMPLE_RATE = 16_000


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        sys.exit(f"error: {binary} not found on PATH")
    return path


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        sys.exit(f"error: {what} failed\n  " + "\n  ".join(tail))
    return result


def probe_duration(path: Path) -> float:
    """Duration in seconds via ffprobe, which reads the container rather than guessing."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0.0
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return round(float(result.stdout.strip()), 2)
    except ValueError:
        return 0.0


def fetch_url(url: str, dest_stem: Path) -> tuple[Path, dict[str, str]]:
    """Download bestaudio to a temp file and capture the upstream metadata."""
    ytdlp = shutil.which("yt-dlp") or sys.executable
    base = [ytdlp] if shutil.which("yt-dlp") else [sys.executable, "-m", "yt_dlp"]

    meta_proc = _run(
        [*base, "--skip-download", "--no-warnings", "--print-json", url],
        "metadata lookup",
    )
    meta_raw = json.loads(meta_proc.stdout.splitlines()[0])
    meta = {
        "title": meta_raw.get("title", ""),
        "uploader": meta_raw.get("uploader", ""),
        "upload_date": meta_raw.get("upload_date", ""),
        "video_id": meta_raw.get("id", ""),
        "licence": meta_raw.get("license") or "unspecified (YouTube standard)",
        "source_duration_s": meta_raw.get("duration", 0),
    }

    tmp = dest_stem.with_suffix(".download.m4a")
    _run(
        [*base, "-f", "bestaudio", "--no-warnings", "-o", str(tmp), url],
        "audio download",
    )
    if not tmp.exists():
        matches = list(dest_stem.parent.glob(f"{dest_stem.name}.download.*"))
        if not matches:
            sys.exit("error: download reported success but produced no file")
        tmp = matches[0]
    return tmp, meta


def to_wav(src: Path, dest: Path, start: float | None, duration: float | None) -> None:
    """Transcode to 16 kHz mono PCM, optionally trimming a window."""
    ffmpeg = _require("ffmpeg")
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dest)]
    _run(cmd, "ffmpeg transcode")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="path to a local recording")
    src.add_argument("--url", help="URL to fetch audio from")

    ap.add_argument("--source-id", required=True,
                    help="stable id, e.g. sess_01 — becomes the filename")
    ap.add_argument("--speaker-id", required=True,
                    help="speaker label, e.g. spk_m70 / spk_f70 / unknown")
    ap.add_argument("--split", default="unassigned",
                    choices=["train", "test", "none", "unassigned"],
                    help="intended split; the authoritative split is written in Task 8")
    ap.add_argument("--origin", default="",
                    help="free text, e.g. 'direct recording'. Defaults to the URL.")
    ap.add_argument("--consent", default="",
                    help="consent note for directly recorded speakers")
    ap.add_argument("--purpose", default="corpus",
                    choices=["corpus", "smoke-test"],
                    help="smoke-test audio is excluded from the dataset in Task 8")
    ap.add_argument("--notes", default="")
    ap.add_argument("--start", type=float, default=None, help="trim start, seconds")
    ap.add_argument("--duration", type=float, default=None, help="trim length, seconds")
    ap.add_argument("--force", action="store_true",
                    help="re-ingest a source_id that already exists")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{args.source_id}.wav"

    if dest.exists() and not args.force:
        print(f"[skip] {dest} already exists - pass --force to replace")
        return 0
    if has_id(MANIFEST, "source_id", args.source_id) and not args.force:
        print(f"[skip] {args.source_id} already in manifest - pass --force to replace")
        return 0

    meta: dict[str, object] = {}
    tmp: Path | None = None

    if args.url:
        print(f"[fetch] {args.url}")
        tmp, meta = fetch_url(args.url, RAW_DIR / args.source_id)
        source_path = tmp
        origin = args.origin or args.url
    else:
        source_path = Path(args.file)
        if not source_path.exists():
            sys.exit(f"error: {source_path} does not exist")
        origin = args.origin or "direct recording"

    print(f"[transcode] -> {dest} ({SAMPLE_RATE} Hz mono)")
    to_wav(source_path, dest, args.start, args.duration)

    if tmp is not None and tmp.exists():
        tmp.unlink()

    duration = probe_duration(dest)
    row = {
        "source_id": args.source_id,
        "speaker_id": args.speaker_id,
        "split_hint": args.split,
        "purpose": args.purpose,
        "origin": origin,
        "consent": args.consent,
        "duration_s": duration,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "path": str(dest).replace("\\", "/"),
        "trim_start_s": args.start,
        "trim_duration_s": args.duration,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": args.notes,
        **({"upstream": meta} if meta else {}),
    }
    upsert_row(MANIFEST, row, key="source_id")

    mins, secs = divmod(duration, 60)
    print(f"[ok] {args.source_id}: {int(mins)}m{int(secs):02d}s  speaker={args.speaker_id}  split={args.split}")
    if args.purpose == "smoke-test":
        # ASCII only: the Windows console defaults to cp1252 and mangles em dashes.
        print("     marked smoke-test - excluded from the corpus at split time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
