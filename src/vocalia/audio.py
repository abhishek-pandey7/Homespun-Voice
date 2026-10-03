"""Audio loading, resampling and quality checks.

Whisper consumes 16 kHz mono. SpeeD-IA ships 8 kHz mono, so every clip is
resampled on the way in. Worth being clear about what that does and does not
achieve: resampling makes the data *consumable*, it does not recover
information. Nothing above 4 kHz was captured, and that is where much of the
energy of fricatives and sibilants sits. The ceiling this imposes applies to the
baseline and the adapted model alike, so the comparison stays fair even though
the absolute numbers are depressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000
MAX_SECONDS = 30.0  # Whisper's receptive field; longer input is silently truncated
MIN_SECONDS = 0.20
SILENCE_RMS = 1e-4


@dataclass
class AudioStats:
    duration_s: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    resampled: bool
    ok: bool
    reason: str = ""


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample with soxr when available, falling back to linear interpolation.

    soxr is a dependency of librosa and gives a far better anti-aliasing filter;
    the fallback exists so a missing optional package degrades quality rather
    than stopping the pipeline.
    """
    if sr == TARGET_SR:
        return x
    try:
        import soxr

        return soxr.resample(x, sr, TARGET_SR, quality="HQ")
    except ImportError:
        ratio = TARGET_SR / sr
        n_out = int(round(len(x) * ratio))
        return np.interp(
            np.linspace(0.0, len(x) - 1, n_out, dtype=np.float64),
            np.arange(len(x), dtype=np.float64),
            x,
        ).astype(np.float32)


def inspect_and_convert(
    src: str | Path,
    dest: str | Path | None = None,
) -> AudioStats:
    """Read a clip, measure it, and optionally write a 16 kHz mono copy.

    Returns stats with ``ok=False`` and a reason for anything the training
    pipeline must not consume, rather than raising: the caller counts and
    reports exclusions, and a corpus that is quietly short is the failure this
    guards against.
    """
    src = Path(src)
    try:
        x, sr = sf.read(str(src), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - unreadable is a result, not a crash
        return AudioStats(0.0, 0, 0, 0.0, 0.0, False, False, f"unreadable: {exc}")

    channels = 1 if x.ndim == 1 else x.shape[1]
    x = _to_mono(x)
    duration = len(x) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0

    reason = ""
    if duration < MIN_SECONDS:
        reason = f"too short ({duration:.2f}s)"
    elif duration > MAX_SECONDS:
        reason = f"too long ({duration:.1f}s, Whisper truncates at {MAX_SECONDS:.0f}s)"
    elif rms < SILENCE_RMS:
        reason = f"silent (rms {rms:.2e})"

    ok = not reason
    resampled = sr != TARGET_SR

    if ok and dest is not None:
        y = resample_to_16k(x, sr)
        peak_y = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak_y > 1.0:  # resampling overshoot can clip on write
            y = y / peak_y
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), y, TARGET_SR, subtype="PCM_16")

    return AudioStats(
        duration_s=round(duration, 3),
        sample_rate=sr,
        channels=channels,
        rms=round(rms, 6),
        peak=round(peak, 6),
        resampled=resampled,
        ok=ok,
        reason=reason,
    )
