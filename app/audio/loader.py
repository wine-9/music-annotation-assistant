"""Safe local audio decoding and dual-path preprocessing."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from imageio_ffmpeg import get_ffmpeg_exe
from numpy.typing import NDArray
from scipy.signal import resample_poly

from app.core.exceptions import AudioDecodeError, AudioValidationError

FloatArray = NDArray[np.float32]
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class LoadedAudio:
    """Audio represented for both ML and stereo DSP paths."""

    model_mono: FloatArray
    model_sample_rate: int
    dsp_stereo: FloatArray
    dsp_sample_rate: int
    original_channels: int
    duration_seconds: float
    sha256: str
    warnings: tuple[str, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_with_ffmpeg(path: Path) -> tuple[FloatArray, int]:
    """Decode unsupported libsndfile inputs via the bundled FFmpeg binary."""

    with tempfile.TemporaryDirectory(prefix="mla-decode-") as temp_dir:
        wav_path = Path(temp_dir) / "decoded.wav"
        command = [
            get_ffmpeg_exe(),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map_metadata",
            "-1",
            "-acodec",
            "pcm_f32le",
            "-y",
            str(wav_path),
        ]
        completed = subprocess.run(command, capture_output=True, timeout=120, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-500:]
            raise AudioDecodeError(f"FFmpeg decode failed: {detail}")
        data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def _decode(path: Path) -> tuple[FloatArray, int]:
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32), int(sample_rate)
    except (RuntimeError, sf.LibsndfileError):
        try:
            return _decode_with_ffmpeg(path)
        except (AudioDecodeError, OSError, subprocess.SubprocessError) as ffmpeg_exc:
            raise AudioDecodeError("Audio is damaged or uses an unsupported codec") from ffmpeg_exc


def _to_stereo(data: FloatArray) -> FloatArray:
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] == 2:
        return data
    # Use the first two channels deterministically; preserve a warning upstream.
    return data[:, :2]


def load_audio(
    path: Path,
    *,
    model_sample_rate: int = 16_000,
    max_duration_seconds: float = 600.0,
) -> LoadedAudio:
    """Decode and validate one local audio file without uploading it."""

    if not path.is_file():
        raise AudioDecodeError("Audio file does not exist")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioDecodeError(f"Unsupported audio suffix: {path.suffix.lower()}")

    digest = sha256_file(path)
    data, sample_rate = _decode(path)
    if sample_rate < 1_000 or sample_rate > 384_000:
        raise AudioValidationError(f"Abnormal sample rate: {sample_rate}")
    if data.size == 0:
        raise AudioValidationError("Audio is empty")
    if not np.isfinite(data).all():
        raise AudioValidationError("Audio contains NaN or Inf")

    duration = data.shape[0] / sample_rate
    if duration > max_duration_seconds:
        raise AudioValidationError(
            f"Audio exceeds the configured {max_duration_seconds:.0f}s limit"
        )
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    if rms < 1e-5:
        raise AudioValidationError("Audio is silent or too quiet to analyze")

    warnings: list[str] = []
    channels = int(data.shape[1])
    if channels == 1:
        warnings.append("source_is_mono")
    elif channels > 2:
        warnings.append("multichannel_downmixed_to_first_two_channels")
    clipped_ratio = float(np.mean(np.abs(data) >= 0.999))
    if clipped_ratio > 0.01:
        warnings.append("severe_clipping_detected")

    stereo = _to_stereo(data)
    mono = np.mean(data, axis=1, dtype=np.float32)
    if sample_rate != model_sample_rate:
        divisor = int(np.gcd(sample_rate, model_sample_rate))
        mono = resample_poly(
            mono, model_sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32)

    return LoadedAudio(
        model_mono=np.ascontiguousarray(mono, dtype=np.float32),
        model_sample_rate=model_sample_rate,
        dsp_stereo=np.ascontiguousarray(stereo, dtype=np.float32),
        dsp_sample_rate=sample_rate,
        original_channels=channels,
        duration_seconds=duration,
        sha256=digest,
        warnings=tuple(warnings),
    )
