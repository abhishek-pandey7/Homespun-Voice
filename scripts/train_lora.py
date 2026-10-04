"""Task 7 - fine-tune whisper-small on Awadhi with a LoRA adapter.

    python scripts/train_lora.py --smoke          # 50 steps, proves VRAM fits
    python scripts/train_lora.py                  # full run per configs/lora.yaml

The test split is never loaded here. Early stopping uses a validation slice taken
out of train, because stopping against the test set would make the reported
benchmark meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MANIFEST = Path("data/speedia/manifest.jsonl")
CONFIG = Path("configs/lora.yaml")
OUT = Path("models/homespun-lora")
LOG = Path("reports/train_log.jsonl")


def load_rows(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="50 steps on a small slice, to prove it fits in VRAM")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0,
                    help="override configs/lora.yaml epochs, for ablations")
    ap.add_argument("--tag", default="full",
                    help="label recorded in reports/train_log.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tr = cfg["training"]
    if args.epochs:
        tr["epochs"] = args.epochs

    import torch
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperProcessor,
    )

    from homespun.train import (
        MAX_LABEL_TOKENS,
        SpeechCollator,
        SpeechDataset,
        build_model,
        filter_by_label_length,
        peak_vram_gb,
        split_train_validation,
        trainable_summary,
    )

    if not torch.cuda.is_available():
        print("[warn] CUDA unavailable - this will be unusably slow on CPU")

    rows = load_rows(args.manifest, "train")
    if not rows:
        sys.exit(f"error: no train rows in {args.manifest}")

    train_rows, val_rows = split_train_validation(
        rows, tr["validation_fraction"], tr["seed"]
    )
    if args.smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:16]

    # The test split must never reach training. Assert it rather than trust it.
    test_ids = {r["utt_id"] for r in load_rows(args.manifest, "test")}
    used_ids = {r["utt_id"] for r in train_rows + val_rows}
    leaked = test_ids & used_ids
    assert not leaked, f"test ids leaked into training: {sorted(leaked)[:5]}"

    audio_min = sum(r["duration_s"] for r in train_rows) / 60
    print(f"train {len(train_rows)} utts ({audio_min:.1f} min)   "
          f"validation {len(val_rows)} utts")
    print(f"test ids held out: {len(test_ids)}  leak check: clean")

    processor = WhisperProcessor.from_pretrained(
        cfg["base_model"], language=cfg["language"], task=cfg["task"]
    )

    train_rows, dropped_tr = filter_by_label_length(
        train_rows, processor, cfg["language"], cfg["task"]
    )
    val_rows, dropped_val = filter_by_label_length(
        val_rows, processor, cfg["language"], cfg["task"]
    )
    n_dropped = len(dropped_tr) + len(dropped_val)
    if n_dropped:
        longest = max(r["label_tokens"] for r in dropped_tr + dropped_val)
        print(f"dropped {n_dropped} utterances over {MAX_LABEL_TOKENS} label "
              f"tokens (longest {longest}) - Whisper cannot represent them")

    model = build_model(cfg)
    print(trainable_summary(model))

    train_ds = SpeechDataset(train_rows, processor, cfg["language"], cfg["task"])
    val_ds = SpeechDataset(val_rows, processor, cfg["language"], cfg["task"])
    collator = SpeechCollator(processor=processor)

    max_steps = args.max_steps or (50 if args.smoke else -1)
    args.out.mkdir(parents=True, exist_ok=True)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=tr["per_device_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        num_train_epochs=tr["epochs"],
        max_steps=max_steps,
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        fp16=tr["fp16"] and torch.cuda.is_available(),
        gradient_checkpointing=tr["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=10 if args.smoke else tr["eval_steps"],
        save_strategy="steps",
        save_steps=50 if args.smoke else tr["save_steps"],
        save_total_limit=2,
        logging_steps=tr["logging_steps"],
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=tr["seed"],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,  # Windows: workers reimport and slow startup
    )

    callbacks = []
    if not args.smoke:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=tr["early_stopping_patience"]
        ))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    print(f"\n{'=' * 60}")
    print("SMOKE TEST - 50 steps" if args.smoke else "FULL RUN")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    peak = peak_vram_gb()
    print(f"\nwall time  : {elapsed / 60:.1f} min")
    print(f"peak VRAM  : {peak:.2f} GB")
    print(f"train loss : {result.training_loss:.4f}")

    metrics = trainer.evaluate()
    print(f"eval loss  : {metrics.get('eval_loss', float('nan')):.4f}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({
            "mode": "smoke" if args.smoke else args.tag,
            "train_utterances": len(train_rows),
            "validation_utterances": len(val_rows),
            "train_audio_minutes": round(audio_min, 1),
            "steps": result.global_step,
            "train_loss": round(result.training_loss, 4),
            "eval_loss": round(metrics.get("eval_loss", 0.0), 4),
            "wall_minutes": round(elapsed / 60, 2),
            "peak_vram_gb": round(peak, 2),
            "config": cfg,
            "log_history": trainer.state.log_history,
        }, ensure_ascii=False) + "\n")
    print(f"[ok] appended to {LOG}")

    if args.smoke:
        print("\nsmoke test complete - adapter NOT saved")
        headroom = 6.0 - peak
        print(f"headroom on a 6 GB card: {headroom:.2f} GB")
        if peak > 5.5:
            print("  tight. Reduce per_device_batch_size or train on a larger card.")
        else:
            print("  comfortable. The full run should fit.")
        return 0

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    size_mb = sum(f.stat().st_size for f in args.out.rglob("*")
                  if f.is_file() and "checkpoints" not in f.parts) / 1024**2
    print(f"[ok] adapter saved to {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
