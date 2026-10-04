"""Task 6 - score transcription output, with the harness proving itself first.

`--selftest` runs three checks whose answers are known in advance: a reference
scored against itself must be 0.0, an empty hypothesis must be 1.0, and a single
substituted word in a known sentence must give the hand-calculated rate. If those
do not hold, nothing the harness reports afterwards means anything, so the
selftest runs by default before scoring.

    python scripts/evaluate.py --selftest
    python scripts/evaluate.py data/transcripts/baseline_test.jsonl
    python scripts/evaluate.py data/transcripts/baseline_test.jsonl \
                               data/transcripts/lora_test.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homespun.metrics import NORMALISATION, score, vocabulary_recall  # noqa: E402

REPORTS = Path("reports")

# Awadhi markers that standard-Hindi-trained models characteristically replace.
# Counted directly because WER weights them the same as any other token, while
# for this project they are the whole point.
DIALECT_MARKERS = {
    "थय", "अहय", "होत", "हो", "जौन", "जौनहय", "कय", "कीन", "मा",
    "पहिले", "वोहके", "वोहमा", "अउर", "अव", "जा", "थीं", "बना", "रसम",
}


def selftest() -> bool:
    ref = ["जनम अउर नामकरन कय रसम होत है", "बियाह के रसम मा बारात आवा थय"]
    ok = True

    s = score(ref, list(ref))
    print(f"  identity            WER={s.wer:.4f} CER={s.cer:.4f}  expect 0.0000")
    ok &= abs(s.wer) < 1e-9 and abs(s.cer) < 1e-9

    s = score(ref, ["", ""])
    print(f"  empty hypothesis    WER={s.wer:.4f}              expect 1.0000")
    ok &= abs(s.wer - 1.0) < 1e-9

    # One substitution across 14 reference words (7 + 7).
    n_words = sum(len(r.split()) for r in ref)
    assert n_words == 14, f"test fixture changed: {n_words} words"
    mutated = ["जनम अउर नामकरन कय रसम होत XXX", ref[1]]
    s = score(ref, mutated)
    expected = 1 / n_words
    print(f"  one substitution    WER={s.wer:.4f}              expect {expected:.4f}")
    ok &= abs(s.wer - expected) < 1e-9

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hypothesis_files", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true", help="run checks and exit")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--freeze-baseline", action="store_true",
                    help="also write reports/baseline.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.selftest or not args.skip_selftest:
        print("harness selftest:")
        if not selftest():
            sys.exit("error: selftest failed - scores would be meaningless")
        print()
    if args.selftest:
        return 0
    if not args.hypothesis_files:
        sys.exit("error: give at least one hypothesis file")

    REPORTS.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    for path in args.hypothesis_files:
        if not path.exists():
            sys.exit(f"error: {path} not found")
        rows = load(path)
        name = rows[0].get("model", path.stem) if rows else path.stem

        refs = [r["reference"] for r in rows]
        hyps = [r["hypothesis"] for r in rows]
        overall = score(refs, hyps)
        recalled, total = vocabulary_recall(refs, hyps, DIALECT_MARKERS)

        by_subset: dict[str, dict] = {}
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for r in rows:
            groups[r["subset"]].append(r)
        for subset, g in sorted(groups.items()):
            s = score([r["reference"] for r in g], [r["hypothesis"] for r in g])
            by_subset[subset] = s.to_dict()

        results[name] = {
            "file": str(path).replace("\\", "/"),
            "overall": overall.to_dict(),
            "by_subset": by_subset,
            "dialect_markers": {
                "recalled": recalled,
                "total": total,
                "recall": round(recalled / total, 4) if total else None,
            },
        }

    # ------------------------------------------------------------------ print ---
    print(f"normalisation: {NORMALISATION}\n")
    names = list(results)
    print(f"{'model':12} {'subset':12} {'utts':>6} {'WER':>8} {'CER':>8}")
    for name in names:
        r = results[name]
        for subset, s in r["by_subset"].items():
            print(f"{name:12} {subset:12} {s['utterances']:6} "
                  f"{s['wer']:8.4f} {s['cer']:8.4f}")
        o = r["overall"]
        print(f"{name:12} {'ALL':12} {o['utterances']:6} "
              f"{o['wer']:8.4f} {o['cer']:8.4f}")

    print(f"\n{'model':12} {'dialect markers recalled':>28}")
    for name in names:
        d = results[name]["dialect_markers"]
        pct = f"{d['recall']:.1%}" if d["recall"] is not None else "n/a"
        print(f"{name:12} {d['recalled']:>12}/{d['total']:<6} {pct:>8}")

    if len(names) == 2:
        a, b = names
        print(f"\ndelta ({b} vs {a}):")
        for metric in ("wer", "cer"):
            va = results[a]["overall"][metric]
            vb = results[b]["overall"][metric]
            rel = (vb - va) / va * 100 if va else 0.0
            arrow = "better" if vb < va else "worse" if vb > va else "same"
            print(f"  {metric.upper():4} {va:.4f} -> {vb:.4f}  "
                  f"({vb - va:+.4f}, {rel:+.1f}%)  {arrow}")

    # ------------------------------------------------------------------- write ---
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "normalisation": NORMALISATION,
        "dialect_markers": sorted(DIALECT_MARKERS),
        "results": results,
    }
    out = args.out or REPORTS / f"eval_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] {out}")

    if args.freeze_baseline:
        frozen = REPORTS / "baseline.json"
        if frozen.exists():
            print(f"[skip] {frozen} already exists - the baseline stays frozen")
        else:
            frozen.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
            print(f"[ok] {frozen} (frozen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
