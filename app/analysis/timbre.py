"""Lightweight, explainable timbre candidates with per-category normalization."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft

from app.core.config import load_yaml
from app.schemas.results import Decision, TimbreAttribute, TimbreFeatureValues, TimbreResult


def _classify_pair(
    value_a: float,
    limits_a: list[float],
    value_b: float,
    limits_b: list[float],
    labels: tuple[str, str, str],
) -> TimbreAttribute:
    votes: list[str] = []
    for value, limits in ((value_a, limits_a), (value_b, limits_b)):
        if value < limits[0]:
            votes.append(labels[0])
        elif value > limits[1]:
            votes.append(labels[2])
        else:
            votes.append(labels[1])
    agreed = votes[0] == votes[1]
    return TimbreAttribute(
        value=votes[0] if agreed else labels[1],
        decision=Decision.CONFIRMED if agreed else Decision.CANDIDATE,
        supporting_indicators=2 if agreed else 1,
    )


def extract_timbre_features(
    mono: NDArray[np.float32], sample_rate: int
) -> TimbreFeatureValues:
    """Extract robust summary features without a learned timbre classifier."""

    if mono.ndim != 1 or mono.size == 0:
        raise ValueError("Timbre analysis expects non-empty mono audio")
    frequencies, _, spectrum = stft(
        mono.astype(np.float64), fs=sample_rate, nperseg=2048, noverlap=1536
    )
    magnitude = np.abs(spectrum) + 1e-12
    power = magnitude**2
    power_sum = np.sum(power, axis=0) + 1e-12
    centroid = float(np.median(np.sum(frequencies[:, None] * power, axis=0) / power_sum))
    cumulative = np.cumsum(power, axis=0)
    rolloff_indices = np.argmax(cumulative >= 0.85 * cumulative[-1], axis=0)
    rolloff = float(np.median(frequencies[rolloff_indices]))
    high_mask = frequencies >= 4_000
    hf_ratio = float(np.median(np.sum(power[high_mask], axis=0) / power_sum))
    flatness = float(
        np.median(np.exp(np.mean(np.log(magnitude), axis=0)) / np.mean(magnitude, axis=0))
    )

    frame_size = max(256, round(sample_rate * 0.05))
    hop = max(128, frame_size // 2)
    frames = [
        mono[start : start + frame_size]
        for start in range(0, max(1, len(mono) - frame_size + 1), hop)
    ]
    rms = np.array(
        [np.sqrt(np.mean(np.square(frame, dtype=np.float64))) for frame in frames]
    )
    peaks = np.array([np.max(np.abs(frame)) for frame in frames])
    crest = float(np.median(peaks / (rms + 1e-12)))
    positive_delta = np.maximum(0, np.diff(rms, prepend=rms[0]))
    onset = float(np.percentile(positive_delta, 90))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(mono)))))
    active = rms[rms > max(1e-5, np.max(rms) * 0.1)]
    dynamic_range = (
        float(
            20
            * np.log10(
                (np.percentile(active, 95) + 1e-12)
                / (np.percentile(active, 10) + 1e-12)
            )
        )
        if active.size >= 2
        else 0.0
    )
    if rms.size > 2 and np.max(rms) > 0:
        peak_index = int(np.argmax(rms))
        after = rms[peak_index:]
        threshold = np.max(rms) * 0.37
        below = np.flatnonzero(after <= threshold)
        decay = float(
            (below[0] * hop / sample_rate)
            if below.size
            else len(after) * hop / sample_rate
        )
    else:
        decay = 0.0
    return TimbreFeatureValues(
        spectral_centroid_hz=centroid,
        spectral_rolloff_hz=rolloff,
        high_frequency_energy_ratio=hf_ratio,
        spectral_flatness=flatness,
        crest_factor=crest,
        onset_strength=onset,
        zero_crossing_rate=zcr,
        temporal_decay_seconds=decay,
        dynamic_range_db=dynamic_range,
    )


def analyze_timbre(
    mono: NDArray[np.float32], sample_rate: int, canonical_label: str
) -> TimbreResult:
    """Classify three objective candidate attributes using category-specific norms."""

    features = extract_timbre_features(mono, sample_rate)
    config = load_yaml("timbre_norms.yaml")
    categories = cast(dict[str, object], config["categories"])
    raw_norms = categories.get(canonical_label)
    unknown = TimbreAttribute(
        value="unknown", decision=Decision.UNKNOWN, supporting_indicators=0
    )
    if not isinstance(raw_norms, dict):
        return TimbreResult(
            brightness=unknown, attack=unknown, sustain=unknown, features=features
        )
    norms = cast(dict[str, list[float]], raw_norms)
    brightness = _classify_pair(
        features.spectral_centroid_hz,
        norms["centroid_hz"],
        features.high_frequency_energy_ratio,
        norms["hf_ratio"],
        ("dark", "neutral", "bright"),
    )
    attack = _classify_pair(
        features.crest_factor,
        norms["crest_factor"],
        features.onset_strength,
        norms["onset_strength"],
        ("soft", "neutral", "sharp"),
    )
    decay_limits = norms["decay_seconds"]
    sustain_value = (
        "short"
        if features.temporal_decay_seconds < decay_limits[0]
        else "long"
        if features.temporal_decay_seconds > decay_limits[1]
        else "medium"
    )
    sustain = TimbreAttribute(
        value=sustain_value, decision=Decision.CANDIDATE, supporting_indicators=1
    )
    return TimbreResult(
        brightness=brightness, attack=attack, sustain=sustain, features=features
    )
