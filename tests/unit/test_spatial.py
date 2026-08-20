from __future__ import annotations

import numpy as np

from app.analysis.spatial import analyze_spatial


def tone(gain_left: float, gain_right: float) -> np.ndarray:
    sample_rate = 44_100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = 0.2 * np.sin(2 * np.pi * 440 * time)
    return np.column_stack((mono * gain_left, mono * gain_right)).astype(np.float32)


def test_left_right_and_center_pan() -> None:
    assert analyze_spatial(tone(1, 0.1), 44_100).pan == "left"
    assert analyze_spatial(tone(0.1, 1), 44_100).pan == "right"
    assert analyze_spatial(tone(1, 1), 44_100).pan == "center"


def test_mono_source_does_not_claim_pan() -> None:
    result = analyze_spatial(tone(1, 1), 44_100, original_channels=1)
    assert result.pan == "unknown"
    assert result.width == "mono"


def test_inverted_audio_warns_about_phase() -> None:
    result = analyze_spatial(tone(1, -1), 44_100)
    assert "possible_phase_inversion" in result.warnings
    assert result.correlation is not None and result.correlation < -0.99


def test_moving_pan_is_detected() -> None:
    sample_rate = 44_100
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 220 * time)
    position = np.sin(2 * np.pi * time / 1.5)
    stereo = np.column_stack((signal * (1 - position), signal * (1 + position)))
    result = analyze_spatial(stereo.astype(np.float32), sample_rate)
    assert result.pan == "moving"
    assert result.stability == "variable"
