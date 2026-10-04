"""Verify the environment can actually train before any training code runs.

A CPU fallback is the failure this guards against: torch installed from the
default index works fine, reports no error, and makes a LoRA fine-tune roughly
50x slower. That looks like "training is slow" rather than "torch is wrong",
which is an expensive thing to discover late.

Exit code is non-zero when something would block training, so this can gate a
pipeline run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

MIN_VRAM_GB = 5.0  # whisper-small LoRA in fp16 needs headroom inside 6 GB


def _ok(label: str, value: str) -> None:
    print(f"  [ok]   {label}: {value}")


def _warn(label: str, value: str) -> None:
    print(f"  [warn] {label}: {value}")


def _fail(label: str, value: str) -> None:
    print(f"  [FAIL] {label}: {value}")


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) == (3, 12):
        _ok("python", version)
        return True
    _fail("python", f"{version} — expected 3.12.x (torch has no stable 3.14 wheels)")
    return False


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        _fail("torch", "not installed — pip install -r requirements.txt")
        return False

    _ok("torch", torch.__version__)

    if "+cu" not in torch.__version__:
        _fail(
            "torch build",
            f"{torch.__version__} looks like the CPU wheel; reinstall with "
            "--extra-index-url https://download.pytorch.org/whl/cu121",
        )
        return False
    _ok("torch build", "CUDA wheel")

    if not torch.cuda.is_available():
        _fail("cuda", "False — training would silently run on CPU")
        return False
    _ok("cuda", "True")

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    _ok("gpu", f"{name} ({vram_gb:.1f} GB)")

    if vram_gb < MIN_VRAM_GB:
        _warn("vram", f"{vram_gb:.1f} GB is tight; expect to reduce batch size")

    # Prove a real kernel launches. cuda.is_available() can be True while the
    # driver/runtime pairing is broken.
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        (a @ a).sum().item()
        torch.cuda.synchronize()
        _ok("fp16 matmul", "executed on device")
    except Exception as exc:  # noqa: BLE001 — surface whatever the driver says
        _fail("fp16 matmul", f"{type(exc).__name__}: {exc}")
        return False

    return True


def check_binary(name: str, args: list[str]) -> bool:
    path = shutil.which(name)
    if path is None:
        _fail(name, "not found on PATH")
        return False
    try:
        out = subprocess.run(
            [path, *args], capture_output=True, text=True, timeout=30, check=True
        )
        _ok(name, out.stdout.strip().splitlines()[0][:60])
        return True
    except Exception as exc:  # noqa: BLE001
        _fail(name, f"present but not runnable: {exc}")
        return False


def check_imports() -> bool:
    required = ["transformers", "peft", "datasets", "jiwer", "soundfile", "librosa"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        _fail("packages", f"missing: {', '.join(missing)}")
        return False
    _ok("packages", f"{len(required)} core imports resolve")
    return True


def main() -> int:
    print("Homespun environment check\n")
    results = [
        check_python(),
        check_torch(),
        check_imports(),
        check_binary("ffmpeg", ["-version"]),
    ]

    # yt-dlp is only needed for the ingest stage, so a miss is not fatal.
    if shutil.which("yt-dlp") is None:
        _warn("yt-dlp", "not on PATH — ingest from URL unavailable")
    else:
        check_binary("yt-dlp", ["--version"])

    print()
    if all(results):
        print("Environment is ready for training.")
        return 0
    print("Environment is NOT ready — resolve the [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
