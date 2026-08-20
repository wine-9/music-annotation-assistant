from __future__ import annotations

import numpy as np

from app.decision.engine import build_instrument_results, merge_time_ranges, score_statistics
from app.schemas.results import FrameInstrumentPredictions, WindowPrediction


def test_statistics_include_top_20_and_active_ratio() -> None:
    stats = score_statistics(np.array([0.1, 0.2, 0.8, 0.9, 1.0]), 0.5)
    assert stats.top_20_percent_mean == 1.0
    assert stats.active_ratio == 0.6
    assert stats.p95 > 0.9


def test_ranges_merge_small_gaps() -> None:
    ranges = merge_time_ranges(
        [0, 2, 8], [2, 4, 10], [0.7, 0.8, 0.9], threshold=0.5, merge_gap_seconds=2
    )
    assert [(item.start_seconds, item.end_seconds) for item in ranges] == [(0, 4), (8, 10)]


def test_interval_hysteresis_and_confidence() -> None:
    ranges = merge_time_ranges(
        [0.0, 5.0, 10.0, 20.0],
        [10.0, 15.0, 20.0, 30.0],
        [0.7, 0.55, 0.45, 0.2],
        threshold=0.6,
        hysteresis_ratio=0.75,
        minimum_duration_seconds=10.0,
        merge_gap_seconds=1.0,
    )
    assert len(ranges) == 1
    assert ranges[0].start_seconds == 0.0
    assert ranges[0].end_seconds == 20.0
    assert 0.5 < ranges[0].interval_confidence < 0.7


def test_unvalidated_thresholds_never_auto_confirm() -> None:
    labels = ["drums", "electricguitar", "acousticguitar"]
    predictions = FrameInstrumentPredictions(
        labels=labels,
        provider_name="test",
        provider_version="1",
        windows=[
            WindowPrediction(start_seconds=0, end_seconds=2, scores=[0.99, 0.9, 0.88]),
            WindowPrediction(start_seconds=1, end_seconds=3, scores=[0.99, 0.9, 0.88]),
        ],
    )
    results = {item.canonical_label: item for item in build_instrument_results(predictions)}
    assert results["drums"].decision == "candidate"
    assert results["electric_guitar"].decision == "candidate"
    assert results["acoustic_guitar"].decision == "candidate"
