#!/usr/bin/env python3
"""Calibrate per-label candidate and confirmed thresholds from validation CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def calibrate(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    precision_target: float = 0.85,
    recall_target: float = 0.75,
    minimum_positives: int = 20,
) -> dict[str, object]:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    rows: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (targets == 1)))
        fp = int(np.sum(predicted & (targets == 0)))
        fn = int(np.sum(~predicted & (targets == 1)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((float(threshold), precision, recall, f1))
    confirmed_rows = [row for row in rows if row[1] >= precision_target]
    confirmed = min(confirmed_rows, key=lambda row: row[0])[0] if confirmed_rows else 1.0
    recall_rows = [row for row in rows if row[2] >= recall_target]
    candidate = (
        max(recall_rows, key=lambda row: row[3])[0]
        if recall_rows
        else max(rows, key=lambda row: row[3])[0]
    )
    positives = int(np.sum(targets == 1))
    return {
        "candidate": candidate,
        "confirmed": confirmed,
        "auto_confirm_enabled": positives >= minimum_positives and bool(confirmed_rows),
        "minimum_validation_positives": minimum_positives,
        "validation_positives": positives,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CSV columns: label,target,model_score"
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/calibrated_thresholds.json"))
    arguments = parser.parse_args()
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with arguments.input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["label"]].append((int(row["target"]), float(row["model_score"])))
    results = {
        label: calibrate(
            np.asarray([item[0] for item in items]),
            np.asarray([item[1] for item in items]),
        )
        for label, items in grouped.items()
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
