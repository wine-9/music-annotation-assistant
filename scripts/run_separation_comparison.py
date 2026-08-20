#!/usr/bin/env python3
"""Compare mix-only and optional four-stem candidate detection on public audio."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from app.audio.loader import load_audio
from app.core.config import ROOT, load_threshold_profile
from app.decision.engine import build_instrument_results
from app.evaluation.datasets import TARGET_LABELS
from app.evaluation.metrics import metrics_at_threshold
from app.providers.demucs_provider import DemucsSeparatorProvider
from app.providers.essentia_provider import EssentiaInstrumentProvider
from scripts.run_evaluation import _load_samples, _load_score_cache

EVALUATION_ROOT = ROOT / "data" / "evaluation"
OUTPUT_ROOT = ROOT / "outputs" / "evaluation"
SUPPORTED_STEM = {"drums": "drums", "percussion": "drums", "bass": "bass"}


def _energy(path: Path) -> float:
    audio = load_audio(path)
    return float(np.sqrt(np.mean(np.square(audio.dsp_stereo))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EVALUATION_ROOT / "selection_manifest.json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means the complete held-out evaluation split",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=10)
    arguments = parser.parse_args()
    samples = [
        sample
        for sample in _load_samples(arguments.manifest)
        if sample.split == "evaluation"
    ]
    if arguments.max_samples:
        samples = samples[: arguments.max_samples]
    mix_cache = _load_score_cache(OUTPUT_ROOT / "predictions.csv")
    provider = EssentiaInstrumentProvider()
    separator = DemucsSeparatorProvider(device=arguments.device)
    profile = load_threshold_profile("calibrated_public_v1")
    label_config = cast(dict[str, object], profile["labels"])
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    processed = 0
    for batch_start in range(0, len(samples), arguments.batch_size):
        batch = samples[batch_start : batch_start + arguments.batch_size]
        try:
            audio_paths = [
                sample.resolved_path(EVALUATION_ROOT) for sample in batch
            ]
            separated = separator.separate_batch(audio_paths)
            for sample, audio_path in zip(batch, audio_paths, strict=True):
                stem_paths = separated[audio_path]
                raw_energy = {
                    stem: _energy(path) for stem, path in stem_paths.items()
                }
                energy_sum = sum(raw_energy.values())
                energy = {
                    stem: value / energy_sum if energy_sum else 0.0
                    for stem, value in raw_energy.items()
                }
                stem_scores: dict[str, dict[str, float]] = {}
                for stem, path in stem_paths.items():
                    loaded = load_audio(path)
                    predictions = provider.predict(
                        loaded.model_mono, loaded.model_sample_rate
                    )
                    stem_scores[stem] = {
                        item.canonical_label: item.model_score
                        for item in build_instrument_results(
                            predictions,
                            threshold_profile="calibrated_public_v1",
                        )
                    }
                for label, target in sample.targets.items():
                    if target is None:
                        continue
                    mix_score = mix_cache[(sample.sample_id, label)]
                    supported = SUPPORTED_STEM.get(label)
                    fused_score = mix_score
                    warnings: list[str] = []
                    if supported is not None:
                        reliability = min(
                            1.0, energy.get(supported, 0.0) / 0.08
                        )
                        fused_score = max(
                            mix_score,
                            stem_scores[supported][label] * reliability,
                        )
                    else:
                        warnings.append(
                            "other_stem_not_used_for_instrument_identity"
                        )
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "dataset": sample.dataset,
                            "label": label,
                            "target": int(target),
                            "mix_score": mix_score,
                            "stem_scores": {
                                stem: scores[label]
                                for stem, scores in stem_scores.items()
                                if label in scores
                            },
                            "fused_score": fused_score,
                            "stem_energy_ratios": energy,
                            "warnings": warnings,
                        },
                    )
                processed += 1
        except Exception as exc:
            failures.extend(
                {
                    "sample_id": sample.sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                for sample in batch
            )
        print(
            f"\rSeparation: {min(batch_start + len(batch), len(samples))}/{len(samples)}",
            end="",
            flush=True,
        )
    print()
    label_results: dict[str, object] = {}
    for label in TARGET_LABELS:
        applicable = [row for row in rows if row["label"] == label]
        if not applicable:
            continue
        targets = np.asarray([row["target"] for row in applicable], dtype=np.int_)
        mix_scores = np.asarray(
            [row["mix_score"] for row in applicable], dtype=np.float64
        )
        fused_scores = np.asarray(
            [row["fused_score"] for row in applicable], dtype=np.float64
        )
        config = cast(dict[str, object], label_config[label])
        threshold = float(cast(float, config["candidate"]))
        label_results[label] = {
            "candidate_threshold": threshold,
            "mix_only": asdict(metrics_at_threshold(targets, mix_scores, threshold)),
            "with_stems": asdict(metrics_at_threshold(targets, fused_scores, threshold)),
        }
    report = {
        "schema_version": "1.0",
        "same_complete_evaluation_set": (
            not arguments.max_samples and not failures
        ),
        "audio_requested": len(samples),
        "audio_completed": processed,
        "runtime_seconds": time.perf_counter() - started,
        "device": arguments.device,
        "failures": failures,
        "labels": label_results,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "separation_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_ROOT / "separation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "dataset",
                "label",
                "target",
                "mix_score",
                "stem_scores",
                "fused_score",
                "stem_energy_ratios",
                "warnings",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "stem_scores": json.dumps(row["stem_scores"]),
                    "stem_energy_ratios": json.dumps(row["stem_energy_ratios"]),
                    "warnings": json.dumps(row["warnings"]),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
