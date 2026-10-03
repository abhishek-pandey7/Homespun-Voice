"""Task 4 - measure the corpus, normalise it to 16 kHz mono, write the dataset card.

This is the gate before any GPU time is spent. Nothing downstream may run on an
unmeasured corpus, because every later number is quoted against these totals.

Exclusions are counted and reported with reasons rather than silently dropped: a
corpus that is quietly short is the specific failure this guards against.

    python scripts/corpus_qa.py
    python scripts/corpus_qa.py --limit 50        # quick pass while iterating
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocalia.audio import MAX_SECONDS, TARGET_SR, inspect_and_convert  # noqa: E402
from vocalia.speedia import read_index  # noqa: E402

INDEX = Path("data/speedia/index.jsonl")
AUDIO_OUT = Path("data/speedia/audio16k")
MANIFEST = Path("data/speedia/manifest.jsonl")
CARD = Path("data/speedia/DATASET_CARD.md")

CITATION = """@inproceedings{interspeech2022,
    author    = {Kumar, Ritesh and Singh, Siddharth and Ratan, Shyam and Raj, Mohit
                 and Sinha, Sonal and Lahiri, Bornini and Seshadri, Vivek
                 and Bali, Kalika and Ojha, Atul Kr.},
    title     = {Annotated Speech Corpus for Low Resource Indian Languages:
                 Awadhi, Bhojpuri, Braj and Magahi},
    booktitle = {Proceedings of Speech for Social Good Workshop, Interspeech 2022},
    year      = {2022}
}"""


def hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--audio-out", type=Path, default=AUDIO_OUT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--card", type=Path, default=CARD)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N")
    ap.add_argument("--force", action="store_true",
                    help="re-convert clips whose 16 kHz copy already exists")
    args = ap.parse_args()

    records = list(read_index(args.index))
    if not records:
        sys.exit(f"error: {args.index} is empty - run scripts/import_speedia.py first")
    if args.limit:
        records = records[: args.limit]

    print(f"processing {len(records)} utterances\n")

    kept: list[dict] = []
    excluded: list[dict] = []
    src_rates: collections.Counter[int] = collections.Counter()
    resampled_n = 0

    for i, u in enumerate(records, 1):
        dest = args.audio_out / u.split / f"{u.utt_id}.wav"
        skip_convert = dest.exists() and not args.force
        stats = inspect_and_convert(u.audio_path, None if skip_convert else dest)
        src_rates[stats.sample_rate] += 1
        resampled_n += bool(stats.resampled and stats.ok)

        row = {
            "utt_id": u.utt_id,
            "text": u.text,
            "split": u.split,
            "subset": u.subset,
            "speaker_id": u.speaker_id,
            "audio_path": str(dest).replace("\\", "/"),
            "source_path": u.audio_path,
            "duration_s": stats.duration_s,
            "source_sample_rate": stats.sample_rate,
            "sample_rate": TARGET_SR,
            "rms": stats.rms,
        }
        if stats.ok:
            kept.append(row)
        else:
            excluded.append({**row, "reason": stats.reason})

        if i % 250 == 0 or i == len(records):
            print(f"  {i}/{len(records)}  kept={len(kept)} excluded={len(excluded)}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="\n") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- report ---
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    by_subset: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    by_speaker: dict[str, list[dict]] = collections.defaultdict(list)
    for r in kept:
        by_split[r["split"]].append(r)
        by_subset[(r["split"], r["subset"])].append(r)
        by_speaker[r["speaker_id"]].append(r)

    total_s = sum(r["duration_s"] for r in kept)
    durations = sorted(r["duration_s"] for r in kept)

    def pct(p: float) -> float:
        return durations[min(int(len(durations) * p), len(durations) - 1)] if durations else 0.0

    print(f"\nkept     : {len(kept)}")
    print(f"excluded : {len(excluded)}")
    print(f"total    : {hms(total_s)}")
    print(f"source sample rates: {dict(src_rates)}  (resampled {resampled_n})")

    print(f"\n{'split':6} {'subset':12} {'utts':>6} {'duration':>10} {'mean':>7}")
    for key in sorted(by_subset):
        rows = by_subset[key]
        d = sum(r["duration_s"] for r in rows)
        print(f"{key[0]:6} {key[1]:12} {len(rows):6} {hms(d):>10} {d/len(rows):6.1f}s")
    for split in sorted(by_split):
        rows = by_split[split]
        d = sum(r["duration_s"] for r in rows)
        print(f"{split:6} {'ALL':12} {len(rows):6} {hms(d):>10} {d/len(rows):6.1f}s")

    print(f"\nduration p50={pct(0.50):.1f}s p90={pct(0.90):.1f}s "
          f"p99={pct(0.99):.1f}s max={durations[-1] if durations else 0:.1f}s")
    print(f"speakers : {len(by_speaker)}")

    if excluded:
        reasons = collections.Counter(e["reason"].split("(")[0].strip() for e in excluded)
        print("\nexclusions:")
        for reason, n in reasons.most_common():
            print(f"  {n:5}  {reason}")

    # ------------------------------------------------------------ dataset card ---
    both = sum(1 for s, rows in by_speaker.items()
               if len({r["split"] for r in rows}) > 1)
    lines = [
        "# Dataset Card - SpeeD-IA Awadhi (as used by Vocalia)",
        "",
        f"Generated by `scripts/corpus_qa.py` on {date.today().isoformat()}.",
        "",
        "## Source",
        "",
        "SpeeD-IA - Speech Datasets for Indo-Aryan Languages. Collected by Dr. Bhimrao",
        "Ambedkar University and the Council for Strategic and Defence Research, with",
        "Karya Inc. and UnReaL-TecE LLP.",
        "",
        "- Transcriptions: <https://github.com/unrealtecellp/SpeeD-IA>",
        "- Audio: Google Drive folder linked from that repository",
        "- Licence: **CC BY-NC-SA 4.0**",
        "",
        "Speakers were asked questions about birth and naming customs and answered in",
        "Awadhi. The TSV files transcribe those spoken answers.",
        "",
        "## Citation",
        "",
        "```bibtex",
        CITATION,
        "```",
        "",
        "## Contents as used here",
        "",
        f"- Utterances: **{len(kept)}**",
        f"- Total audio: **{hms(total_s)}**",
        f"- Speakers: **{len(by_speaker)}**",
        f"- Source sample rate: {', '.join(f'{k} Hz x{v}' for k, v in src_rates.items())}",
        f"- Delivered to the model at: {TARGET_SR} Hz mono",
        "",
        "| split | subset | utterances | duration | mean |",
        "|---|---|---:|---:|---:|",
    ]
    for key in sorted(by_subset):
        rows = by_subset[key]
        d = sum(r["duration_s"] for r in rows)
        lines.append(f"| {key[0]} | {key[1]} | {len(rows)} | {hms(d)} | {d/len(rows):.1f}s |")
    for split in sorted(by_split):
        rows = by_split[split]
        d = sum(r["duration_s"] for r in rows)
        lines.append(f"| **{split}** | **all** | **{len(rows)}** | **{hms(d)}** | {d/len(rows):.1f}s |")

    lines += [
        "",
        "## Exclusions",
        "",
        f"{len(excluded)} of {len(records)} joined utterances were excluded:",
        "",
    ]
    if excluded:
        reasons = collections.Counter(e["reason"].split("(")[0].strip() for e in excluded)
        lines += ["| reason | count |", "|---|---:|"]
        lines += [f"| {r} | {n} |" for r, n in reasons.most_common()]
    else:
        lines.append("None.")

    lines += [
        "",
        "## Limitations",
        "",
        "Three properties of this corpus shape how any result on it should be read.",
        "",
        "1. **Source audio is 8 kHz.** Clips are upsampled to 16 kHz because that is",
        "   what Whisper consumes, but nothing above 4 kHz was ever recorded, and much",
        "   of the energy distinguishing fricatives and sibilants sits above it. This",
        "   caps achievable accuracy for the baseline and the adapted model equally,",
        "   so the comparison stays fair while the absolute numbers stay depressed.",
        "2. **The split is by utterance, not by speaker.**",
        f"   {both} of {len(by_speaker)} speakers appear in both train and test. Results",
        "   describe adaptation with speaker overlap and say nothing about",
        "   generalising to an unseen voice.",
        "3. **The JSON sidecars hold the question, not the answer.** Their `data` field",
        "   is the prompt the speaker was asked. Reference text is taken from the TSVs",
        "   only; pairing clips with the sidecar text would train against the wrong",
        "   words entirely.",
        "",
        "Additionally, 840 clips in the distributed audio have no transcription and are",
        "unusable for supervised training; 5 transcriptions have no matching audio.",
        "",
        "## Licence obligations",
        "",
        "CC BY-NC-SA 4.0 applies to derivatives. A model fine-tuned on this corpus and",
        "any text generated from its transcripts are both derivative works: they carry",
        "attribution, stay non-commercial, and are shared under the same licence.",
        "",
    ]
    args.card.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ok] {args.manifest}  ({len(kept)} rows)")
    print(f"[ok] {args.card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
