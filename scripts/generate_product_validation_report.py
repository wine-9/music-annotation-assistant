#!/usr/bin/env python3
"""Generate the evidence-gated A/B/C product validation decision."""

from __future__ import annotations

import json
from typing import cast

from app.core.config import ROOT
from app.storage.repository import Repository


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric report value, received {type(value)}")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric report value, received {type(value)}")
    return int(value)


def main() -> int:
    evaluation_path = ROOT / "outputs" / "evaluation" / "report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = cast(dict[str, object], evaluation["summary"])
    sessions = Repository().list_review_sessions()
    runtime = json.loads(
        (ROOT / "outputs" / "evaluation" / "runtime_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    precision = _as_float(summary["confirmed_precision"])
    prefill = _as_float(summary["confirmed_plus_candidate_coverage"])
    if precision < 0.90:
        product_class = "C"
        rationale = "扩展公开测试的 confirmed precision 低于 0.90。"
    else:
        product_class = "B"
        rationale = (
            "confirmed precision 达标，但总预填覆盖率尚未达到 0.40，且没有真实人工"
            "复核时间数据，不能满足 A 的全部条件。"
        )
    separation_path = (
        ROOT / "outputs" / "evaluation" / "separation_comparison.json"
    )
    separation: dict[str, object] | None = None
    separation_runtime: dict[str, object] | None = None
    if separation_path.is_file():
        separation = json.loads(separation_path.read_text(encoding="utf-8"))
    separation_runtime_path = (
        ROOT / "outputs" / "evaluation" / "separation_runtime_benchmark.json"
    )
    if separation_runtime_path.is_file():
        separation_runtime = json.loads(
            separation_runtime_path.read_text(encoding="utf-8")
        )
    lines = [
        "# Product Validation Decision",
        "",
        f"## 当前判断：{product_class}",
        "",
        rationale,
        "",
        "## 判定证据",
        "",
        f"- confirmed precision：{_percent(precision)}",
        f"- confirmed coverage：{_percent(_as_float(summary['confirmed_coverage']))}",
        f"- candidate coverage：{_percent(_as_float(summary['candidate_coverage']))}",
        (
            "- confirmed + candidate 总预填覆盖率："
            f"{_percent(prefill)}"
        ),
        (
            "- 目标表单全部 10 字段的保守可验证预填覆盖率："
            f"{_percent(_as_float(summary['observable_target_form_prefill_coverage']))}"
        ),
        f"- 真实 Human Review 会话：{len(sessions)}",
        f"- 公开评估实际墙钟时间：{_as_float(runtime['wall_seconds']):.2f} 秒",
        (
            "- 公开评估工作流平均速度："
            f"{_as_float(runtime['wall_seconds']) / _as_int(runtime['evaluation_audio']):.2f} "
            "秒/段（含兼容分数缓存）"
        ),
        (
            "- 公开评估峰值 RSS："
            f"{_as_int(runtime['maximum_resident_set_bytes']) / 1024**3:.2f} GiB"
        ),
        "- 平均节省时间：不可判定" if not sessions else "- 平均节省时间：见可用性报告",
        "- confirmed 严重错误率：不可判定（需要人工区分一般误报与严重误导）",
        "",
        "## 分轨状态",
        "",
    ]
    if separation is None:
        lines.append("- 尚未运行 Demucs 对比。")
    else:
        lines.extend(
            [
                f"- 请求音频：{separation['audio_requested']}",
                f"- 完成音频：{separation['audio_completed']}",
                (
                    "- 与完整公开测试集一致："
                    + ("是" if separation["same_complete_evaluation_set"] else "否")
                ),
                f"- 分轨对比耗时：{_as_float(separation['runtime_seconds']):.1f} 秒",
            ]
        )
        if separation_runtime is not None:
            separation_rss_gib = (
                _as_int(separation_runtime["maximum_resident_set_bytes"])
                / 1024**3
            )
            lines.extend(
                [
                    (
                        "- 分轨真实墙钟时间："
                        f"{_as_float(separation_runtime['total_wall_seconds']):.2f} 秒"
                    ),
                    (
                        "- 分轨平均速度："
                        f"{_as_float(separation_runtime['seconds_per_audio']):.2f} 秒/段"
                    ),
                    (
                        "- 分轨峰值 RSS："
                        f"{separation_rss_gib:.2f} GiB"
                    ),
                ]
            )
        overall = cast(dict[str, dict[str, object]], separation["overall"])
        lines.extend(
            [
                "",
                "| 总体 | Precision | Recall | Coverage |",
                "|---|---:|---:|---:|",
                (
                    "| 不启用分轨 | "
                    f"{_percent(_as_float(overall['mix_only']['precision']))} | "
                    f"{_percent(_as_float(overall['mix_only']['recall']))} | "
                    f"{_percent(_as_float(overall['mix_only']['coverage']))} |"
                ),
                (
                    "| 启用分轨 | "
                    f"{_percent(_as_float(overall['with_stems']['precision']))} | "
                    f"{_percent(_as_float(overall['with_stems']['recall']))} | "
                    f"{_percent(_as_float(overall['with_stems']['coverage']))} |"
                ),
            ]
        )
        labels = cast(dict[str, dict[str, object]], separation["labels"])
        lines.extend(
            [
                "",
                "| 类别 | Mix Precision | Stem Precision | Mix Recall | Stem Recall | "
                "Mix Coverage | Stem Coverage |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, item in labels.items():
            mix = cast(dict[str, object], item["mix_only"])
            stems = cast(dict[str, object], item["with_stems"])
            lines.append(
                f"| {label} | {_percent(_as_float(mix['precision']))} | "
                f"{_percent(_as_float(stems['precision']))} | "
                f"{_percent(_as_float(mix['recall']))} | "
                f"{_percent(_as_float(stems['recall']))} | "
                f"{_percent(_as_float(mix['coverage']))} | "
                f"{_percent(_as_float(stems['coverage']))} |"
            )
    lines.extend(
        [
            "",
            "## A/B/C 条件审计",
            "",
            "| 条件 | 当前证据 |",
            "|---|---|",
            f"| confirmed precision ≥ 0.90 | {'通过' if precision >= 0.90 else '未通过'} |",
            (
                "| confirmed + candidate coverage ≥ 0.40 | "
                f"{'通过' if prefill >= 0.40 else '未通过'} |"
            ),
            "| 平均节省时间 ≥ 0.25 | 不可判定：无人工基线 |",
            "| confirmed 严重错误率 ≤ 0.05 | 不可判定：无人工严重度复核 |",
            "",
            "因此在获得真实人工复核证据前，不得升级为 A。",
            "",
        ]
    )
    destination = ROOT / "docs" / "PRODUCT_VALIDATION.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
