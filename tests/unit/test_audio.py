from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio.loader import load_audio
from app.audio.windows import window_audio
from app.core.exceptions import AudioDecodeError, AudioValidationError


def test_load_short_stereo_and_resample(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    audio = np.column_stack(
        (np.sin(np.linspace(0, 100, 8_000)), np.sin(np.linspace(0, 100, 8_000)))
    ).astype(np.float32)
    sf.write(path, audio, 8_000, subtype="FLOAT")
    loaded = load_audio(path)
    assert loaded.model_sample_rate == 16_000
    assert loaded.model_mono.shape[0] == 16_000
    assert loaded.dsp_stereo.shape == (8_000, 2)
    assert len(loaded.sha256) == 64


def test_silent_audio_rejected(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros((1_000, 1), dtype=np.float32), 8_000)
    with pytest.raises(AudioValidationError, match="silent"):
        load_audio(path)


def test_corrupt_audio_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wave file")
    with pytest.raises(AudioDecodeError):
        load_audio(path)


def test_short_audio_kept_as_one_window() -> None:
    audio = np.ones(16_000, dtype=np.float32)
    windows = window_audio(audio, 16_000, 10, 5)
    assert len(windows) == 1
    assert windows[0][0:2] == (0.0, 1.0)
