"""Taxonomy aggregation, per-label thresholds, and cautious decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.core.config import get_active_threshold_profile_name, load_threshold_profile, load_yaml
from app.schemas.results import (
    Decision,
    FrameInstrumentPredictions,
    InstrumentResult,
    ScoreStatistics,
    SpatialResult,
    TimeRange,
)


@dataclass(frozen=True)
class CanonicalLabel:
    name: str
    display_name_zh: str
    raw_labels: tuple[str, ...]
    parent: str | None
    exclusive_group: str | None


def load_taxonomy() -> dict[str, CanonicalLabel]:
    """Load normalized labels and their raw model mappings."""

    payload = load_yaml("instrument_taxonomy.yaml")
    raw_labels = cast(dict[str, object], payload["canonical_labels"])
    result: dict[str, CanonicalLabel] = {}
    for name, raw_entry in raw_labels.items():
        entry = cast(dict[str, object], raw_entry)
        result[name] = CanonicalLabel(
            name=name,
            display_name_zh=str(entry["display_name_zh"]),
            raw_labels=tuple(str(item) for item in cast(list[object], entry["raw_labels"])),
            parent=str(entry["parent"]) if entry.get("parent") else None,
            exclusive_group=str(entry["exclusive_group"]) if entry.get("exclusive_group") else None,
        )
    return result


def score_statistics(
    scores: NDArray[np.float64], candidate_threshold: float
) -> ScoreStatistics:
    """Compute the complete set of required robust aggregate statistics."""

    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("At least one score is required")
    count = max(1, int(np.ceil(scores.size * 0.2)))
    top = np.sort(scores)[-count:]
    return ScoreStatistics(
        mean=float(np.mean(scores)),
        median=float(np.median(scores)),
        max=float(np.max(scores)),
        p90=float(np.percentile(scores, 90)),
        p95=float(np.percentile(scores, 95)),
        top_20_percent_mean=float(np.mean(top)),
        active_ratio=float(np.mean(scores >= candidate_threshold)),
    )


def merge_time_ranges(
    starts: list[float],
    ends: list[float],
    scores: list[float],
    *,
    threshold: float,
    merge_gap_seconds: float = 2.0,
    hysteresis_ratio: float = 1.0,
    minimum_duration_seconds: float = 0.0,
    minimum_active_windows: int = 1,
) -> list[TimeRange]:
    """Extract intervals using hysteresis, duration filtering, and gap merging."""

    release_threshold = threshold * hysteresis_ratio
    active: list[tuple[float, float, float]] = []
    is_active = False
    for start, end, score in zip(starts, ends, scores, strict=True):
        if not is_active and score >= threshold:
            is_active = True
        elif is_active and score < release_threshold:
            is_active = False
        if is_active:
            active.append((start, end, score))
    if not active:
        return []
    merged_raw: list[tuple[float, float, list[float]]] = []
    current_start, current_end, first_score = active[0]
    current_scores = [first_score]
    for start, end, score in active[1:]:
        if start <= current_end + merge_gap_seconds:
            current_end = max(current_end, end)
            current_scores.append(score)
        else:
            merged_raw.append((current_start, current_end, current_scores))
            current_start, current_end, current_scores = start, end, [score]
    merged_raw.append((current_start, current_end, current_scores))
    return [
        TimeRange(
            start_seconds=start,
            end_seconds=end,
            peak_score=max(interval_scores),
            interval_confidence=float(np.mean(interval_scores)),
        )
        for start, end, interval_scores in merged_raw
        if end - start >= minimum_duration_seconds
        and len(interval_scores) >= minimum_active_windows
    ]


def build_instrument_results(
    predictions: FrameInstrumentPredictions,
    *,
    threshold_profile: str | None = None,
) -> list[InstrumentResult]:
    """Map official raw labels to cautious normalized decisions."""

    taxonomy = load_taxonomy()
    threshold_payload = load_threshold_profile(
        threshold_profile or get_active_threshold_profile_name()
    )
    defaults = cast(dict[str, object], threshold_payload["defaults"])
    label_thresholds = cast(dict[str, object], threshold_payload["labels"])
    formula = cast(dict[str, object], threshold_payload["presence_formula"])
    interval_payload = load_yaml("interval_rules.yaml")
    interval_defaults = cast(dict[str, object], interval_payload["defaults"])
    interval_labels = cast(dict[str, object], interval_payload["labels"])
    raw_index = {label: index for index, label in enumerate(predictions.labels)}
    starts = [window.start_seconds for window in predictions.windows]
    ends = [window.end_seconds for window in predictions.windows]
    matrix = np.asarray([window.scores for window in predictions.windows], dtype=np.float64)
    if matrix.size == 0:
        return []

    results: list[InstrumentResult] = []
    for canonical, taxon in taxonomy.items():
        config = cast(
            dict[str, object], label_thresholds.get(canonical, defaults)
        )
        candidate = float(cast(float, config["candidate"]))
        confirmed = float(cast(float, config["confirmed"]))
        columns = [raw_index[label] for label in taxon.raw_labels if label in raw_index]
        if not columns:
            continue
        canonical_scores = np.max(matrix[:, columns], axis=1)
        stats = score_statistics(canonical_scores, candidate)
        presence_score = (
            float(cast(float, formula["top_20_percent_mean"])) * stats.top_20_percent_mean
            + float(cast(float, formula["p95"])) * stats.p95
        )
        auto_confirm = bool(config["auto_confirm_enabled"])
        if presence_score >= confirmed and auto_confirm:
            decision = Decision.CONFIRMED
        elif presence_score >= candidate:
            decision = Decision.CANDIDATE
        else:
            decision = Decision.UNKNOWN
        raw_scores = {
            predictions.labels[column]: float(np.max(matrix[:, column])) for column in columns
        }
        interval_config = cast(
            dict[str, object], interval_labels.get(canonical, interval_defaults)
        )
        transient = cast(
            dict[str, object],
            interval_config.get(
                "transient_filtering", interval_defaults["transient_filtering"]
            ),
        )
        results.append(
            InstrumentResult(
                canonical_label=canonical,
                display_name_zh=taxon.display_name_zh,
                decision=decision,
                model_score=presence_score,
                statistics=stats,
                raw_model_scores=raw_scores,
                time_ranges=merge_time_ranges(
                    starts,
                    ends,
                    canonical_scores.tolist(),
                    threshold=candidate,
                    merge_gap_seconds=float(
                        cast(
                            float,
                            interval_config.get(
                                "interval_gap_merge_seconds",
                                interval_defaults["interval_gap_merge_seconds"],
                            ),
                        )
                    ),
                    hysteresis_ratio=float(
                        cast(
                            float,
                            interval_config.get(
                                "hysteresis_threshold_ratio",
                                interval_defaults["hysteresis_threshold_ratio"],
                            ),
                        )
                    ),
                    minimum_duration_seconds=float(
                        cast(
                            float,
                            interval_config.get(
                                "minimum_duration_seconds",
                                interval_defaults["minimum_duration_seconds"],
                            ),
                        )
                    ),
                    minimum_active_windows=(
                        int(cast(int, transient.get("minimum_active_windows", 1)))
                        if bool(transient.get("enabled", True))
                        else 1
                    ),
                ),
                spatial=SpatialResult(basis="no_reliable_stem"),
            )
        )
    return _apply_subtype_ambiguity(results, taxonomy)


def _apply_subtype_ambiguity(
    results: list[InstrumentResult], taxonomy: dict[str, CanonicalLabel]
) -> list[InstrumentResult]:
    """Do not force a subtype when mutually exclusive scores are too close."""

    raw_margin = load_yaml("instrument_taxonomy.yaml")["subtype_margin"]
    if not isinstance(raw_margin, (int, float)):
        raise TypeError("subtype_margin must be numeric")
    margin = float(raw_margin)
    by_group: dict[str, list[InstrumentResult]] = {}
    for result in results:
        group = taxonomy[result.canonical_label].exclusive_group
        if group:
            by_group.setdefault(group, []).append(result)
    for members in by_group.values():
        ordered = sorted(members, key=lambda item: item.model_score, reverse=True)
        if len(ordered) >= 2 and ordered[0].model_score - ordered[1].model_score < margin:
            for item in ordered:
                if item.decision == Decision.CONFIRMED:
                    item.decision = Decision.CANDIDATE
    return results
