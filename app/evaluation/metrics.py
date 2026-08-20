"""Dependency-light binary metrics and precision-constrained threshold search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary classification metrics at one threshold."""

    samples: int
    positives: int
    negatives: int
    predicted_positive: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    pr_auc: float
    coverage: float


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Threshold selected under a minimum-precision constraint."""

    threshold: float
    precision: float
    recall: float
    f1: float
    coverage: float
    predicted_positive: int
    validation_positives: int
    meets_precision_target: bool
    auto_confirm_enabled: bool


@dataclass(frozen=True)
class DecisionCoverageMetrics:
    """Three-way product decision metrics for confirmed/candidate/unknown."""

    samples: int
    confirmed_count: int
    candidate_count: int
    unknown_count: int
    confirmed_true_positive: int
    candidate_true_positive: int
    confirmed_coverage: float
    candidate_coverage: float
    prefill_coverage: float
    unknown_coverage: float
    confirmed_precision: float
    candidate_precision: float


def _validate(
    targets: NDArray[np.int_], scores: NDArray[np.float64]
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    clean_targets = np.asarray(targets, dtype=np.int_)
    clean_scores = np.asarray(scores, dtype=np.float64)
    if clean_targets.ndim != 1 or clean_scores.ndim != 1:
        raise ValueError("Targets and scores must be one-dimensional")
    if clean_targets.size != clean_scores.size or clean_targets.size == 0:
        raise ValueError("Targets and scores must have the same non-zero length")
    if not np.isin(clean_targets, [0, 1]).all():
        raise ValueError("Targets must contain only 0 and 1")
    if not np.isfinite(clean_scores).all():
        raise ValueError("Scores must be finite")
    return clean_targets, np.clip(clean_scores, 0.0, 1.0)


def average_precision(
    targets: NDArray[np.int_], scores: NDArray[np.float64]
) -> float:
    """Compute non-interpolated PR-AUC (average precision)."""

    clean_targets, clean_scores = _validate(targets, scores)
    positive_count = int(np.sum(clean_targets))
    if positive_count == 0:
        return 0.0
    order = np.argsort(-clean_scores, kind="stable")
    ordered_targets = clean_targets[order]
    true_positives = np.cumsum(ordered_targets)
    ranks = np.arange(1, clean_targets.size + 1)
    precision_at_rank = true_positives / ranks
    return float(np.sum(precision_at_rank * ordered_targets) / positive_count)


def metrics_at_threshold(
    targets: NDArray[np.int_],
    scores: NDArray[np.float64],
    threshold: float,
) -> BinaryMetrics:
    """Calculate required metrics and auto-confirm coverage."""

    clean_targets, clean_scores = _validate(targets, scores)
    predicted = clean_scores >= threshold
    positive = clean_targets == 1
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & ~positive))
    false_negative = int(np.sum(~predicted & positive))
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    positives = int(np.sum(positive))
    recall = true_positive / positives if positives else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return BinaryMetrics(
        samples=int(clean_targets.size),
        positives=positives,
        negatives=int(clean_targets.size - positives),
        predicted_positive=predicted_positive,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=average_precision(clean_targets, clean_scores),
        coverage=predicted_positive / clean_targets.size,
    )


def find_confirmed_threshold(
    targets: NDArray[np.int_],
    scores: NDArray[np.float64],
    *,
    precision_target: float = 0.85,
    minimum_validation_positives: int = 20,
) -> ThresholdRecommendation:
    """Maximize predicted coverage among thresholds meeting the precision target."""

    clean_targets, clean_scores = _validate(targets, scores)
    positives = int(np.sum(clean_targets))
    candidates = sorted({float(value) for value in clean_scores})
    qualified: list[tuple[float, BinaryMetrics]] = []
    for threshold in candidates:
        metrics = metrics_at_threshold(clean_targets, clean_scores, threshold)
        if metrics.predicted_positive and metrics.precision >= precision_target:
            qualified.append((threshold, metrics))
    if not qualified:
        return ThresholdRecommendation(
            threshold=1.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            coverage=0.0,
            predicted_positive=0,
            validation_positives=positives,
            meets_precision_target=False,
            auto_confirm_enabled=False,
        )
    threshold, metrics = max(
        qualified,
        key=lambda item: (
            item[1].predicted_positive,
            item[1].recall,
            -item[0],
        ),
    )
    enough_positives = positives >= minimum_validation_positives
    return ThresholdRecommendation(
        threshold=threshold,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        coverage=metrics.coverage,
        predicted_positive=metrics.predicted_positive,
        validation_positives=positives,
        meets_precision_target=True,
        auto_confirm_enabled=enough_positives,
    )


def decision_coverage_metrics(
    targets: NDArray[np.int_],
    scores: NDArray[np.float64],
    *,
    candidate_threshold: float,
    confirmed_threshold: float,
    auto_confirm_enabled: bool,
) -> DecisionCoverageMetrics:
    """Measure mutually exclusive confirmed, candidate, and unknown decisions."""

    clean_targets, clean_scores = _validate(targets, scores)
    confirmed = (
        clean_scores >= confirmed_threshold
        if auto_confirm_enabled
        else np.zeros(clean_scores.shape, dtype=np.bool_)
    )
    candidate = (clean_scores >= candidate_threshold) & ~confirmed
    unknown = ~confirmed & ~candidate
    positive = clean_targets == 1
    confirmed_count = int(np.sum(confirmed))
    candidate_count = int(np.sum(candidate))
    samples = int(clean_targets.size)
    confirmed_true_positive = int(np.sum(confirmed & positive))
    candidate_true_positive = int(np.sum(candidate & positive))
    return DecisionCoverageMetrics(
        samples=samples,
        confirmed_count=confirmed_count,
        candidate_count=candidate_count,
        unknown_count=int(np.sum(unknown)),
        confirmed_true_positive=confirmed_true_positive,
        candidate_true_positive=candidate_true_positive,
        confirmed_coverage=confirmed_count / samples,
        candidate_coverage=candidate_count / samples,
        prefill_coverage=(confirmed_count + candidate_count) / samples,
        unknown_coverage=int(np.sum(unknown)) / samples,
        confirmed_precision=(
            confirmed_true_positive / confirmed_count if confirmed_count else 0.0
        ),
        candidate_precision=(
            candidate_true_positive / candidate_count if candidate_count else 0.0
        ),
    )
