from __future__ import annotations

import numpy as np

from app.analysis.timbre import analyze_timbre, extract_timbre_features


def test_timbre_features_are_finite() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    features = extract_timbre_features(audio, sample_rate)
    assert np.isfinite(list(features.model_dump().values())).all()


def test_missing_norms_returns_unknown() -> None:
    audio = np.sin(np.linspace(0, 100, 16_000)).astype(np.float32)
    result = analyze_timbre(audio, 16_000, "woodwind")
    assert result.brightness.value == "unknown"
    assert result.attack.value == "unknown"
