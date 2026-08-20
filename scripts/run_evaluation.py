#!/usr/bin/env python3
"""Run real public-audio inference, calibrate thresholds, and write reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import yaml

from app.audio.loader import load_audio
from app.core.config import ROOT, load_threshold_profile
from app.decision.engine import build_instrument_results
from app.evaluation.datasets import TARGET_LABELS, EvaluationSample
from app.evaluation.metrics import (
    BinaryMetrics,
    DecisionCoverageMetrics,
    ThresholdRecommendation,
    decision_coverage_metrics,
    find_confirmed_threshold,
    metrics_at_threshold,
)
from app.providers.essentia_provider import EssentiaInstrumentProvider

EVALUATION_ROOT = ROOT / "data" / "evaluation"
OUTPUT_ROOT = ROOT / "outputs" / "evaluation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(path: Path) -> list[EvaluationSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationSample(
            sample_id=str(item["sample_id"]),
            dataset=str(item["dataset"]),
            split=str(item["split"]),
            audio_path=str(item["audio_path"]),
            source_url=str(item["source_url"]),
            source_license=str(item["source_license"]),
            targets={
                str(key): (None if value is None else bool(value))
                for key, value in cast(dict[str, object], item["targets"]).items()
            },
            source_tags=tuple(str(value) for value in item["source_tags"]),
            sha256=str(item["sha256"]),
            group_id=str(item.get("group_id", item["sample_id"])),
        )
        for item in payload["samples"]
    ]


def _product_thresholds() -> dict[str, tuple[float, float, bool]]:
    payload = load_threshold_profile("calibrated_public_v1")
    labels = cast(dict[str, object], payload["labels"])
    return {
        label: (
            float(cast(dict[str, object], labels[label])["candidate"]),
            float(cast(dict[str, object], labels[label])["confirmed"]),
            bool(cast(dict[str, object], labels[label])["auto_confirm_enabled"]),
        )
        for label in TARGET_LABELS
    }


def _metric_data(
    rows: list[dict[str, object]],
    split: str,
    label: str,
    dataset: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    applicable = [
        row
        for row in rows
        if row["split"] == split
        and row["label"] == label
        and (dataset is None or row["dataset"] == dataset)
    ]
    return (
        np.asarray([int(row["target"]) for row in applicable], dtype=np.int_),
        np.asarray([float(row["model_score"]) for row in applicable], dtype=np.float64),
    )


def _write_predictions(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "dataset",
                "split",
                "label",
                "target",
                "model_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_score_cache(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        return {}
    cache: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cache[(row["sample_id"], row["label"])] = float(row["model_score"])
    return cache


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_document(
    report: dict[str, object],
    *,
    destination: Path,
) -> None:
    results = cast(dict[str, dict[str, object]], report["labels"])
    summary = cast(dict[str, object], report["summary"])
    data_counts = cast(dict[str, object], report["data_counts"])
    by_dataset = cast(dict[str, dict[str, object]], report["by_dataset_summary"])
    lines = [
        "# Evaluation & Calibration",
        "",
        f"生成时间：{report['created_at']}",
        "",
        "## 结论",
        "",
        (
            f"在本次真实公开音乐留出评估的 {summary['applicable_label_decisions']} 个"
            f"可判定标签字段中，满足逐类别 precision ≥ 0.85 的自动确认覆盖率为 "
            f"**{_percent(float(summary['validated_auto_confirm_coverage']))}**"
            f"（{summary['validated_auto_confirm_count']} 个字段）。"
        ),
        (
            f"这些自动确认的整体 precision 为 "
            f"**{_percent(float(summary['validated_auto_confirm_precision']))}**；"
            f"覆盖了真实正标签的 "
            f"**{_percent(float(summary['validated_positive_recall']))}**。"
        ),
        (
            f"candidate coverage 为 **{_percent(float(summary['candidate_coverage']))}**，"
            f"confirmed + candidate 总预填覆盖率为 "
            f"**{_percent(float(summary['confirmed_plus_candidate_coverage']))}**，"
            f"unknown coverage 为 **{_percent(float(summary['unknown_coverage']))}**。"
        ),
        (
            f"按目标表单 10 个字段计算，公开数据能客观验证的 instrument_name 预填"
            f"占全部目标字段的保守覆盖率为 "
            f"**{_percent(float(summary['observable_target_form_prefill_coverage']))}**。"
            "其余字段没有公开真值，不把 unknown 计作已覆盖。"
        ),
        "",
        "> 这是工程规模、可复现的校准基准，不是生产精度承诺。阈值只改变决策层，模型和 "
        "Provider 均未修改。",
        "",
        "## 数据来源与数量",
        "",
        "- OpenMIC-2018：官方 test split 的 10 秒公开音乐片段；标签为人工部分标注，"
        "仅使用 mask 明确判定的正/负标签。数据集整体为 CC BY 4.0。",
        "- MTG-Jamendo instrument subset：官方 split-0 test 元数据；下载的每首音乐只"
        "保留 12 秒评估片段，原始音频采用各曲目的 Creative Commons 许可。元数据为 "
        "CC BY-NC-SA 4.0，数据集限非商业研究用途。",
        "- OpenMIC 不区分 electric/acoustic guitar，因此这两个类别只由 "
        "MTG-Jamendo 评估；OpenMIC 的 strings、brass、woodwind 由相应细类合并，只有"
        "明确标注时才计入。",
        "",
        "| 数据集 / 分区 | 音频数 | 可判定标签字段 |",
        "|---|---:|---:|",
    ]
    for key, value in data_counts.items():
        count = cast(dict[str, int], value)
        lines.append(f"| {key} | {count['audio']} | {count['decisions']} |")
    lines.extend(
        [
            "",
            "## 产品决策覆盖率",
            "",
            "| 范围 | Confirmed | Candidate | Unknown | Confirmed coverage | "
            "Candidate coverage | 总预填覆盖 | Confirmed precision | Candidate precision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| 全部 | {summary['confirmed_count']} | {summary['candidate_count']} | "
                f"{summary['unknown_count']} | {_percent(float(summary['confirmed_coverage']))} | "
                f"{_percent(float(summary['candidate_coverage']))} | "
                f"{_percent(float(summary['confirmed_plus_candidate_coverage']))} | "
                f"{_percent(float(summary['confirmed_precision']))} | "
                f"{_percent(float(summary['candidate_precision']))} |"
            ),
        ]
    )
    for dataset, metrics in by_dataset.items():
        lines.append(
            f"| {dataset} | {metrics['confirmed_count']} | {metrics['candidate_count']} | "
            f"{metrics['unknown_count']} | "
            f"{_percent(float(metrics['confirmed_coverage']))} | "
            f"{_percent(float(metrics['candidate_coverage']))} | "
            f"{_percent(float(metrics['confirmed_plus_candidate_coverage']))} | "
            f"{_percent(float(metrics['confirmed_precision']))} | "
            f"{_percent(float(metrics['candidate_precision']))} |"
        )
    lines.extend(
        [
            "",
            "## 每个类别的产品决策数量",
            "",
            "| 类别 | Confirmed | Candidate | Unknown | Confirmed precision | "
            "Candidate precision | 总预填覆盖 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in TARGET_LABELS:
        product = cast(dict[str, object], results[label]["product_decisions"])
        lines.append(
            f"| {label} | {product['confirmed_count']} | {product['candidate_count']} | "
            f"{product['unknown_count']} | "
            f"{_percent(float(product['confirmed_precision']))} | "
            f"{_percent(float(product['candidate_precision']))} | "
            f"{_percent(float(product['prefill_coverage']))} |"
        )
    lines.extend(
        [
            "",
            "## 留出评估结果与推荐阈值",
            "",
            (
                "下表“自动确认”是本次扩展评估产生的下一版建议，不会自动修改已审核的 "
                "`calibrated_public_v1`。当前 v1.0 仍只启用 bass、piano、strings、"
                "synth；例如 woodwind 的新建议需人工审核并另发配置版本后才会生效。"
            ),
            "",
            (
                "| 类别 | N（正/负） | 推荐阈值 | Precision | Recall | F1 | "
                "PR-AUC | 当前阈值覆盖 | 推荐覆盖 | 自动确认 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label in TARGET_LABELS:
        item = results[label]
        evaluation = cast(dict[str, object], item["evaluation"])
        current = cast(dict[str, object], item["current_threshold_metrics"])
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{evaluation['samples']}（{evaluation['positives']}/{evaluation['negatives']}）",
                    f"{float(item['recommended_threshold']):.4f}",
                    _percent(float(evaluation["precision"])),
                    _percent(float(evaluation["recall"])),
                    _percent(float(evaluation["f1"])),
                    f"{float(evaluation['pr_auc']):.3f}",
                    _percent(float(current["coverage"])),
                    _percent(float(evaluation["coverage"])),
                    "启用" if item["validated_auto_confirm_enabled"] else "禁用",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 校准方法",
            "",
            "1. 使用 MTG-Jamendo calibration 子集搜索每类阈值。",
            "2. 枚举真实模型分数阈值，只保留 precision ≥ 0.85 的候选。",
            "3. 在合格候选中选择预测正例数最多、即标签字段覆盖率最高的阈值。",
            "4. 验证正样本少于 20 时强制关闭 auto-confirm。",
            "5. 在未参与阈值搜索的 evaluation 子集上复核；留出 precision 未达到 "
            "0.85 的类别保持禁用。",
            "",
            "Coverage 定义为达到 confirmed threshold 的标签字段数除以全部可判定标签"
            "字段数。PR-AUC 使用 non-interpolated average precision。",
            "",
            "## 风险与限制",
            "",
            "- MTG-Jamendo 是上传者提供的弱标签；标签缺失被当作负例时可能包含漏标。",
            "- 当前模型头本身在 MTG-Jamendo instrument 数据上训练，MTG 结果属于同分布"
            "评估；OpenMIC 用于补充跨数据集检查。",
            "- 每首 MTG 音乐只抽取固定 12 秒，可能没有覆盖元数据所标注乐器真正出现的"
            "段落，因此结果偏保守。",
            "- 样本规模适合工程校准，不足以替代完整数据集上的正式论文级评测。",
            "- 推荐阈值写入 `outputs/evaluation/recommended_thresholds.yaml`，不会自动"
            "覆盖生产配置。",
            "",
            "## 可复现命令",
            "",
            "```bash",
            "make setup-evaluation",
            "make download-evaluation",
            "make evaluate",
            "```",
            "",
            "机器可读结果：`outputs/evaluation/report.json`；逐样本原始分数："
            "`outputs/evaluation/predictions.csv`。",
            "",
            "## 来源",
            "",
            "- OpenMIC-2018：https://zenodo.org/records/1432913",
            "- MTG-Jamendo：https://mtg.github.io/mtg-jamendo-dataset/",
            "- MTG-Jamendo 官方仓库：https://github.com/MTG/mtg-jamendo-dataset",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EVALUATION_ROOT / "selection_manifest.json",
    )
    arguments = parser.parse_args()
    samples = _load_samples(arguments.manifest)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_ROOT / "predictions.csv"
    score_cache = _load_score_cache(prediction_path)
    model_manifest = ROOT / "models" / "manifest.json"
    model_hash_before = _sha256(model_manifest)
    provider = EssentiaInstrumentProvider()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        audio_path = sample.resolved_path(EVALUATION_ROOT)
        if not audio_path.is_file() or _sha256(audio_path) != sample.sha256:
            failures.append({"sample_id": sample.sample_id, "error": "integrity_failure"})
            continue
        try:
            required_labels = {
                label for label, target in sample.targets.items() if target is not None
            }
            cached = all(
                (sample.sample_id, label) in score_cache for label in required_labels
            )
            if cached:
                scores = {
                    label: score_cache[(sample.sample_id, label)]
                    for label in required_labels
                }
            else:
                loaded = load_audio(audio_path)
                predictions = provider.predict(loaded.model_mono, loaded.model_sample_rate)
                instruments = {
                    result.canonical_label: result
                    for result in build_instrument_results(predictions)
                }
                scores = {
                    label: instruments[label].model_score for label in required_labels
                }
            for label, target in sample.targets.items():
                if target is None:
                    continue
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "dataset": sample.dataset,
                        "split": sample.split,
                        "label": label,
                        "target": int(target),
                        "model_score": scores[label],
                    }
                )
        except Exception as exc:
            failures.append(
                {"sample_id": sample.sample_id, "error": f"{type(exc).__name__}: {exc}"}
            )
        print(f"\rInference: {index}/{len(samples)}", end="", flush=True)
    print()
    elapsed = time.perf_counter() - started
    _write_predictions(rows, prediction_path)

    product_thresholds = _product_thresholds()
    label_results: dict[str, dict[str, object]] = {}
    product_confirmed = 0
    product_candidate = 0
    product_unknown = 0
    product_confirmed_true = 0
    product_candidate_true = 0
    total_evaluation_pairs = 0
    total_evaluation_positives = 0
    recommended_yaml: dict[str, object] = {
        "version": f"evaluation-{time.strftime('%Y-%m-%d')}",
        "precision_target": 0.85,
        "minimum_validation_positives": 20,
        "labels": {},
    }
    for label in TARGET_LABELS:
        calibration_targets, calibration_scores = _metric_data(rows, "calibration", label)
        evaluation_targets, evaluation_scores = _metric_data(rows, "evaluation", label)
        recommendation: ThresholdRecommendation = find_confirmed_threshold(
            calibration_targets,
            calibration_scores,
            precision_target=0.85,
            minimum_validation_positives=20,
        )
        evaluation: BinaryMetrics = metrics_at_threshold(
            evaluation_targets, evaluation_scores, recommendation.threshold
        )
        candidate_threshold, current_threshold, current_enabled = product_thresholds[
            label
        ]
        current_metrics = metrics_at_threshold(
            evaluation_targets, evaluation_scores, current_threshold
        )
        product_metrics: DecisionCoverageMetrics = decision_coverage_metrics(
            evaluation_targets,
            evaluation_scores,
            candidate_threshold=candidate_threshold,
            confirmed_threshold=current_threshold,
            auto_confirm_enabled=current_enabled,
        )
        validated_enabled = (
            recommendation.auto_confirm_enabled
            and evaluation.predicted_positive > 0
            and evaluation.precision >= 0.85
        )
        product_confirmed += product_metrics.confirmed_count
        product_candidate += product_metrics.candidate_count
        product_unknown += product_metrics.unknown_count
        product_confirmed_true += product_metrics.confirmed_true_positive
        product_candidate_true += product_metrics.candidate_true_positive
        total_evaluation_pairs += evaluation.samples
        total_evaluation_positives += evaluation.positives
        label_results[label] = {
            "calibration": asdict(recommendation),
            "recommended_threshold": recommendation.threshold,
            "evaluation": asdict(evaluation),
            "current_threshold": current_threshold,
            "candidate_threshold": candidate_threshold,
            "current_auto_confirm_enabled": current_enabled,
            "current_threshold_metrics": asdict(current_metrics),
            "product_decisions": asdict(product_metrics),
            "by_dataset": {
                dataset: asdict(
                    decision_coverage_metrics(
                        *_metric_data(rows, "evaluation", label, dataset),
                        candidate_threshold=candidate_threshold,
                        confirmed_threshold=current_threshold,
                        auto_confirm_enabled=current_enabled,
                    )
                )
                for dataset in ("mtg_jamendo_instrument", "openmic_2018")
                if _metric_data(rows, "evaluation", label, dataset)[0].size
            },
            "validated_auto_confirm_enabled": validated_enabled,
        }
        cast(dict[str, object], recommended_yaml["labels"])[label] = {
            "confirmed": round(recommendation.threshold, 8),
            "auto_confirm_enabled": validated_enabled,
            "calibration_positives": recommendation.validation_positives,
            "heldout_precision": round(evaluation.precision, 8),
            "heldout_coverage": round(evaluation.coverage, 8),
        }

    data_counts: dict[str, dict[str, int]] = {}
    successful_ids = {str(row["sample_id"]) for row in rows}
    for dataset in ("mtg_jamendo_instrument", "openmic_2018"):
        for split in ("calibration", "evaluation"):
            key = f"{dataset} / {split}"
            applicable = [
                row
                for row in rows
                if row["dataset"] == dataset and row["split"] == split
            ]
            audio_ids = {
                str(row["sample_id"])
                for row in applicable
                if str(row["sample_id"]) in successful_ids
            }
            data_counts[key] = {
                "audio": len(audio_ids),
                "decisions": len(applicable),
            }
    prefilled = product_confirmed + product_candidate
    summary = {
        "applicable_label_decisions": total_evaluation_pairs,
        "validated_auto_confirm_count": product_confirmed,
        "validated_auto_confirm_coverage": (
            product_confirmed / total_evaluation_pairs if total_evaluation_pairs else 0.0
        ),
        "validated_auto_confirm_precision": (
            product_confirmed_true / product_confirmed if product_confirmed else 0.0
        ),
        "validated_positive_recall": (
            product_confirmed_true / total_evaluation_positives
            if total_evaluation_positives
            else 0.0
        ),
        "confirmed_coverage": (
            product_confirmed / total_evaluation_pairs if total_evaluation_pairs else 0.0
        ),
        "candidate_coverage": (
            product_candidate / total_evaluation_pairs if total_evaluation_pairs else 0.0
        ),
        "confirmed_plus_candidate_coverage": (
            prefilled / total_evaluation_pairs if total_evaluation_pairs else 0.0
        ),
        "unknown_coverage": (
            product_unknown / total_evaluation_pairs if total_evaluation_pairs else 0.0
        ),
        "confirmed_precision": (
            product_confirmed_true / product_confirmed if product_confirmed else 0.0
        ),
        "candidate_precision": (
            product_candidate_true / product_candidate if product_candidate else 0.0
        ),
        "confirmed_count": product_confirmed,
        "candidate_count": product_candidate,
        "unknown_count": product_unknown,
        "evaluation_positives": total_evaluation_positives,
        "target_form_fields_per_instrument": 10,
        "observable_target_form_fields": total_evaluation_pairs * 10,
        "observable_target_form_prefilled": prefilled,
        "observable_target_form_prefill_coverage": (
            prefilled / (total_evaluation_pairs * 10)
            if total_evaluation_pairs
            else 0.0
        ),
    }
    by_dataset_summary: dict[str, object] = {}
    for dataset in ("mtg_jamendo_instrument", "openmic_2018"):
        dataset_metrics: list[DecisionCoverageMetrics] = []
        for label in TARGET_LABELS:
            targets, scores = _metric_data(rows, "evaluation", label, dataset)
            if not targets.size:
                continue
            candidate_threshold, confirmed_threshold, enabled = product_thresholds[
                label
            ]
            dataset_metrics.append(
                decision_coverage_metrics(
                    targets,
                    scores,
                    candidate_threshold=candidate_threshold,
                    confirmed_threshold=confirmed_threshold,
                    auto_confirm_enabled=enabled,
                )
            )
        dataset_total = sum(item.samples for item in dataset_metrics)
        dataset_confirmed = sum(item.confirmed_count for item in dataset_metrics)
        dataset_candidate = sum(item.candidate_count for item in dataset_metrics)
        dataset_unknown = sum(item.unknown_count for item in dataset_metrics)
        dataset_confirmed_true = sum(
            item.confirmed_true_positive for item in dataset_metrics
        )
        dataset_candidate_true = sum(
            item.candidate_true_positive for item in dataset_metrics
        )
        by_dataset_summary[dataset] = {
            "applicable_label_decisions": dataset_total,
            "confirmed_count": dataset_confirmed,
            "candidate_count": dataset_candidate,
            "unknown_count": dataset_unknown,
            "confirmed_coverage": (
                dataset_confirmed / dataset_total if dataset_total else 0.0
            ),
            "candidate_coverage": (
                dataset_candidate / dataset_total if dataset_total else 0.0
            ),
            "confirmed_plus_candidate_coverage": (
                (dataset_confirmed + dataset_candidate) / dataset_total
                if dataset_total
                else 0.0
            ),
            "unknown_coverage": (
                dataset_unknown / dataset_total if dataset_total else 0.0
            ),
            "confirmed_precision": (
                dataset_confirmed_true / dataset_confirmed
                if dataset_confirmed
                else 0.0
            ),
            "candidate_precision": (
                dataset_candidate_true / dataset_candidate
                if dataset_candidate
                else 0.0
            ),
        }
    report: dict[str, object] = {
        "schema_version": "1.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "provider": provider.name,
        "provider_version": provider.version,
        "model_manifest_sha256": model_hash_before,
        "model_manifest_unchanged": _sha256(model_manifest) == model_hash_before,
        "runtime_seconds": elapsed,
        "audio_samples_requested": len(samples),
        "audio_samples_failed": len(failures),
        "failures": failures,
        "data_counts": data_counts,
        "labels": label_results,
        "summary": summary,
        "by_dataset_summary": by_dataset_summary,
    }
    (OUTPUT_ROOT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "recommended_thresholds.yaml").write_text(
        yaml.safe_dump(recommended_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _write_document(report, destination=ROOT / "docs" / "EVALUATION.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
