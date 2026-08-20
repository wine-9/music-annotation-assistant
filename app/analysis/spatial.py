"""Configurable stereo spatial analysis."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.core.config import load_yaml
from app.schemas.results import SpatialResult


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric configuration value: {key}")
    return float(value)


def analyze_spatial(
    stereo: NDArray[np.float32],
    sample_rate: int,
    *,
    original_channels: int = 2,
) -> SpatialResult:
    """Analyze overall spatial properties without attributing them to instruments."""

    config = load_yaml("spatial_thresholds.yaml")
    epsilon = _number(config, "epsilon")
    silence_rms = _number(config, "silence_rms")
    pan_config = cast(dict[str, object], config["pan"])
    width_config = cast(dict[str, object], config["width"])
    frame_seconds = _number(config, "frame_seconds")
    hop_seconds = _number(config, "hop_seconds")

    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("Spatial analysis expects shape (samples, 2)")
    left = stereo[:, 0].astype(np.float64)
    right = stereo[:, 1].astype(np.float64)
    left_energy = float(np.mean(left**2))
    right_energy = float(np.mean(right**2))
    rms = float(np.sqrt((left_energy + right_energy) / 2))
    if rms < silence_rms:
        return SpatialResult(basis="low_energy", warnings=["spatial_low_energy"])

    mid = (left + right) / 2
    side = (left - right) / 2
    mid_energy = float(np.mean(mid**2))
    side_energy = float(np.mean(side**2))
    pan_value = (right_energy - left_energy) / (right_energy + left_energy + epsilon)
    width_ratio = side_energy / (mid_energy + epsilon)
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if left_std > epsilon and right_std > epsilon
        else 1.0
    )

    frame_size = max(1, round(frame_seconds * sample_rate))
    hop_size = max(1, round(hop_seconds * sample_rate))
    frame_pans: list[float] = []
    for start in range(0, max(1, len(stereo) - frame_size + 1), hop_size):
        frame = stereo[start : start + frame_size].astype(np.float64)
        le = float(np.mean(frame[:, 0] ** 2))
        re = float(np.mean(frame[:, 1] ** 2))
        if np.sqrt((le + re) / 2) >= silence_rms:
            frame_pans.append((re - le) / (re + le + epsilon))
    pan_iqr = (
        float(np.percentile(frame_pans, 75) - np.percentile(frame_pans, 25))
        if frame_pans
        else 0.0
    )
    movement_rate = (
        float(np.mean(np.abs(np.diff(frame_pans)))) if len(frame_pans) > 1 else 0.0
    )

    warnings: list[str] = []
    if correlation < _number(config, "phase_warning_correlation"):
        warnings.append("possible_phase_inversion")

    if original_channels == 1:
        pan = "unknown"
        basis = "mono_source"
    elif pan_iqr >= _number(pan_config, "moving_iqr_min"):
        pan = "moving"
        basis = "global_mix"
    elif abs(pan_value) <= _number(pan_config, "center_abs_max"):
        pan = "center"
        basis = "global_mix"
    else:
        pan = "right" if pan_value > 0 else "left"
        basis = "global_mix"

    mono_max = _number(width_config, "mono_max")
    narrow_max = _number(width_config, "narrow_max")
    medium_max = _number(width_config, "medium_max")
    if original_channels == 1 or width_ratio <= mono_max:
        width = "mono"
    elif width_ratio <= narrow_max:
        width = "narrow"
    elif width_ratio <= medium_max:
        width = "medium"
    else:
        width = "wide"

    stability = (
        "variable" if pan_iqr >= _number(pan_config, "moving_iqr_min") else "stable"
    )
    return SpatialResult(
        pan=pan,
        pan_value=pan_value,
        width=width,
        width_ratio=width_ratio,
        correlation=correlation,
        stability=stability,
        pan_iqr=pan_iqr,
        movement_rate=movement_rate,
        basis=basis,
        warnings=warnings,
    )
