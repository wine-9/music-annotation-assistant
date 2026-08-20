#!/usr/bin/env python3
"""Generate an evidence-only usability report from completed human reviews."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from app.core.config import ROOT
from app.storage.repository import Repository


def _percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "不可判定"


def main() -> int:
    sessions = Repository().list_review_sessions()
    destination = ROOT / "docs" / "USABILITY_EVALUATION.md"
    lines = [
        "# Human Review Usability Evaluation",
        "",
        "本报告只统计合法公开音频上的真实人工操作，不使用模拟复核数据。",
        "",
        f"- 已完成复核会话：{len(sessions)}",
        f"- 已记录字段操作：{sum(len(session.events) for session in sessions)}",
        "",
    ]
    if not sessions:
        lines.extend(
            [
                "## 当前结论",
                "",
                "尚无用户人工复核数据，因此 confirmed 直接接受率、candidate 直接接受率、"
                "修改后可用率、漏标率、错误自动确认率、节省时间和操作次数减少比例均为"
                " **不可判定**。",
                "",
                "请在 Web 的“目标标注表单预览”中完成真实复核并点击“完成本轮人工复核”，"
                "然后重新运行：",
                "",
                "```bash",
                ".venv/bin/python scripts/generate_usability_report.py",
                "```",
                "",
            ]
        )
        destination.write_text("\n".join(lines), encoding="utf-8")
        print(destination)
        return 0

    events = [event for session in sessions for event in session.events]
    by_decision: dict[str, list[object]] = defaultdict(list)
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_field: dict[str, Counter[str]] = defaultdict(Counter)
    outcomes = Counter(event.outcome.value for event in events)
    for event in events:
        decision = (
            event.original_prediction.decision.value
            if event.original_prediction is not None
            else "unknown"
        )
        by_decision[decision].append(event)
        label = (
            event.original_prediction.instrument_label
            if event.original_prediction is not None
            else None
        )
        by_label[label or "global"][event.outcome.value] += 1
        field_name = (
            event.original_prediction.field_name
            if event.original_prediction is not None
            else event.field_id
        )
        by_field[field_name][event.outcome.value] += 1
    confirmed = by_decision["confirmed"]
    candidate = by_decision["candidate"]
    direct_confirmed = sum(
        event.outcome.value == "accept_as_is" for event in confirmed
    )
    direct_candidate = sum(
        event.outcome.value == "accept_as_is" for event in candidate
    )
    usable_after_edit = outcomes["accept_after_edit"]
    missing = outcomes["missing_label"]
    wrong_confirmed = sum(event.outcome.value == "reject" for event in confirmed)
    prefilled = len(confirmed) + len(candidate)
    timed = [
        session
        for session in sessions
        if session.baseline_manual_seconds is not None
        and session.baseline_manual_seconds > 0
    ]
    savings = [
        1 - session.total_review_seconds / float(session.baseline_manual_seconds)
        for session in timed
    ]
    lines.extend(
        [
            "## 汇总",
            "",
            f"- confirmed 直接接受率：{_percent(direct_confirmed, len(confirmed))}",
            f"- candidate 直接接受率：{_percent(direct_candidate, len(candidate))}",
            f"- 修改后可用率：{_percent(usable_after_edit, len(events))}",
            f"- 漏标率：{_percent(missing, len(events))}",
            f"- 错误自动确认率：{_percent(wrong_confirmed, len(confirmed))}",
            f"- 总预填覆盖率：{_percent(prefilled, len(events))}",
            (
                f"- 平均节省时间：{mean(savings):.1%}"
                if savings
                else "- 平均节省时间：不可判定（没有人工填写纯手工基线时间）"
            ),
            (
                f"- 中位数节省时间：{median(savings):.1%}"
                if savings
                else "- 中位数节省时间：不可判定（没有人工填写纯手工基线时间）"
            ),
            "- 操作次数减少比例：不可判定（尚未采集纯手工操作次数基线）",
            "",
            "## 各类别表现",
            "",
            "| 类别 | accept_as_is | accept_after_edit | reject | missing_label | uncertain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, counts in sorted(by_label.items()):
        lines.append(
            f"| {label} | {counts['accept_as_is']} | {counts['accept_after_edit']} | "
            f"{counts['reject']} | {counts['missing_label']} | {counts['uncertain']} |"
        )
    lines.extend(
        [
            "",
            "## 各字段表现",
            "",
            "| 字段 | accept_as_is | accept_after_edit | reject | missing_label | uncertain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for field, counts in sorted(by_field.items()):
        lines.append(
            f"| {field} | {counts['accept_as_is']} | {counts['accept_after_edit']} | "
            f"{counts['reject']} | {counts['missing_label']} | {counts['uncertain']} |"
        )
    lines.extend(
        [
            "",
            "## 时间",
            "",
            f"- 平均人工复核时间：{mean(s.total_review_seconds for s in sessions):.1f} 秒/音频",
            f"- 中位人工复核时间：{median(s.total_review_seconds for s in sessions):.1f} 秒/音频",
            f"- 平均模型分析时间：{mean(s.model_analysis_seconds for s in sessions):.1f} 秒/音频",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
