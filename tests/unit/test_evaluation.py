from __future__ import annotations

import numpy as np

from app.evaluation.datasets import openmic_targets
from app.evaluation.metrics import (
    average_precision,
    decision_coverage_metrics,
    find_confirmed_threshold,
    metrics_at_threshold,
)


def test_average_precision_perfect_ranking() -> None:
    targets = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.2, 0.8, 0.1])
    assert average_precision(targets, scores) == 1.0


def test_threshold_maximizes_coverage_at_precision_target() -> None:
    targets = np.asarray([1, 1, 1, 0, 0])
    scores = np.asarray([0.95, 0.8, 0.7, 0.75, 0.1])
    result = find_confirmed_threshold(
        targets, scores, precision_target=0.75, minimum_validation_positives=3
    )
    assert result.threshold == 0.7
    assert result.precision == 0.75
    assert result.predicted_positive == 4
    assert result.auto_confirm_enabled


def test_too_few_positives_disables_auto_confirm() -> None:
    result = find_confirmed_threshold(
        np.asarray([1, 0]), np.asarray([0.9, 0.1]), minimum_validation_positives=2
    )
    assert result.meets_precision_target
    assert not result.auto_confirm_enabled


def test_metrics_coverage_is_fraction_of_all_decisions() -> None:
    result = metrics_at_threshold(
        np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.8, 0.7, 0.1]), 0.75
    )
    assert result.coverage == 0.5
    assert result.precision == 0.5


def test_openmic_partial_labels_do_not_create_fake_negatives() -> None:
    labels = [0.5] * 20
    masks = [0] * 20
    labels[6] = 1.0
    masks[6] = 1
    targets = openmic_targets(labels, masks)
    assert targets["drums"] is True
    assert targets["piano"] is None
    assert targets["electric_guitar"] is None


def test_three_way_product_coverage_is_mutually_exclusive() -> None:
    result = decision_coverage_metrics(
        np.asarray([1, 1, 0, 0]),
        np.asarray([0.9, 0.6, 0.4, 0.1]),
        candidate_threshold=0.3,
        confirmed_threshold=0.8,
        auto_confirm_enabled=True,
    )
    assert result.confirmed_count == 1
    assert result.candidate_count == 2
    assert result.unknown_count == 1
    assert result.prefill_coverage == 0.75
    assert result.confirmed_precision == 1.0
    assert result.candidate_precision == 0.5
