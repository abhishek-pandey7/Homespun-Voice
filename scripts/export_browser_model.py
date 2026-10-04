"""Prepare the adapted Whisper for in-browser inference with Transformers.js.

The fp32 export is 1.76 GB, which no one is going to download to try a demo.
Dynamic int8 quantisation brings it to roughly a quarter of that, and Whisper
tolerates it well: the encoder is convolution and attention heavy, and weight
quantisation costs far less accuracy here than the 8 kHz source audio already
does.

Produces models/homespun-web/ in the layout Transformers.js expects:

    config.json, tokenizer.json, preprocessor_config.json, ...
    onnx/encoder_model_quantized.onnx
    onnx/decoder_model_merged_quantized.onnx
"""

from __future__ import annotations

import pathlib
import shutil
import sys

SRC = pathlib.Path("models/homespun-onnx")
OUT = pathlib.Path("models/homespun-web")


def mb(p: pathlib.Path) -> float:
    return p.stat().st_size / 1048576


def main() -> int:
    if not SRC.exists():
        sys.exit(f"error: {SRC} not found - run the optimum export first")

    onnx_dir = OUT / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # --- merge the two decoder graphs -------------------------------------
    merged = SRC / "decoder_model_merged.onnx"
    if not merged.exists():
        from optimum.onnx import merge_decoders

        # strict=False: the no-past graph emits encoder present-key-values that
        # the with-past graph does not, which is expected for Whisper and not a
        # reason to refuse the merge.
        merge_decoders(
            str(SRC / "decoder_model.onnx"),
            str(SRC / "decoder_with_past_model.onnx"),
            save_path=str(merged),
            strict=False,
        )
    print(f"merged decoder: {mb(merged):.0f} MB")

    # --- quantise ----------------------------------------------------------
    from onnxruntime.quantization import QuantType, quantize_dynamic

    jobs = [
        (SRC / "encoder_model.onnx", onnx_dir / "encoder_model_quantized.onnx"),
        (merged, onnx_dir / "decoder_model_merged_quantized.onnx"),
    ]
    for src, dst in jobs:
        if dst.exists():
            print(f"[skip] {dst.name} exists ({mb(dst):.0f} MB)")
            continue
        print(f"quantising {src.name} ({mb(src):.0f} MB) ...")
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False,
        )
        print(f"  -> {dst.name}  {mb(dst):.0f} MB")

    # --- companion files ---------------------------------------------------
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "preprocessor_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt",
                 "added_tokens.json", "normalizer.json"):
        src = SRC / name
        if src.exists():
            shutil.copyfile(src, OUT / name)

    total = sum(mb(f) for f in OUT.rglob("*") if f.is_file())
    print(f"\n[ok] {OUT}  {total:.0f} MB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
