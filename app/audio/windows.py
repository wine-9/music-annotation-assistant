"""Deterministic overlapping time-window utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def window_audio(
    audio: NDArray[np.float32],
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> list[tuple[float, float, NDArray[np.float32]]]:
    """Split mono audio into overlapping windows while retaining a short final window."""

    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("Window and hop durations must be positive")
    window_samples = max(1, round(window_seconds * sample_rate))
    hop_samples = max(1, round(hop_seconds * sample_rate))
    result: list[tuple[float, float, NDArray[np.float32]]] = []
    for start in range(0, len(audio), hop_samples):
        end = min(start + window_samples, len(audio))
        if end <= start:
            break
        result.append((start / sample_rate, end / sample_rate, audio[start:end]))
        if end == len(audio):
            break
    return result
