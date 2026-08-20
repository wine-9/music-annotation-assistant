"""Map internal predictions to an auditable target-platform form preview."""

from __future__ import annotations

from app.core.config import load_yaml
from app.schemas.results import (
    Decision,
    FormField,
    FormPreview,
    FormSummary,
    InstrumentResult,
)

PER_INSTRUMENT_FIELDS = (
    "arrangement_roles",
    "playing_modes",
    "pan",
    "width",
    "depth",
    "movement",
    "timbre",
    "effects",
)


def _unknown_field(field_name: str, label: str, reason: str) -> FormField:
    return FormField(
        field_id=f"{label}.{field_name}",
        field_name=field_name,
        instrument_label=label,
        value=None,
        decision=Decision.UNKNOWN,
        model_score=None,
        source="unavailable",
        requires_review=True,
        reason=reason,
    )


def _spatial_field(instrument: InstrumentResult, field_name: str) -> FormField:
    spatial = instrument.spatial
    if spatial.basis == "no_reliable_stem":
        return _unknown_field(
            field_name,
            instrument.canonical_label,
            "单乐器没有可靠 Stem；禁止使用整体混音声场代替",
        )
    value = spatial.pan if field_name == "pan" else spatial.width
    if field_name == "movement":
        value = spatial.stability
    decision = instrument.decision if value != "unknown" else Decision.UNKNOWN
    return FormField(
        field_id=f"{instrument.canonical_label}.{field_name}",
        field_name=field_name,
        instrument_label=instrument.canonical_label,
        value=value if value != "unknown" else None,
        decision=decision,
        model_score=instrument.model_score,
        source=f"instrument.spatial:{spatial.basis}",
        requires_review=decision != Decision.CONFIRMED,
        reason="来自单乐器可靠 Stem" if value != "unknown" else "单乐器声场不可判定",
    )


def map_target_form(instruments: list[InstrumentResult]) -> FormPreview:
    """Build target fields while preserving unknowns and review requirements."""

    mapping = load_yaml("platform_output_mapping.yaml")
    fields: list[FormField] = []
    detected = [
        instrument for instrument in instruments if instrument.decision != Decision.UNKNOWN
    ]
    for instrument in instruments:
        fields.append(
            FormField(
                field_id=f"{instrument.canonical_label}.instrument_name",
                field_name="instrument_name",
                instrument_label=instrument.canonical_label,
                value=(
                    instrument.canonical_label
                    if instrument.decision != Decision.UNKNOWN
                    else None
                ),
                decision=instrument.decision,
                model_score=instrument.model_score,
                source="instrument_model",
                requires_review=instrument.decision != Decision.CONFIRMED,
                reason=(
                    "达到公开校准并经留出集复核的确认阈值"
                    if instrument.decision == Decision.CONFIRMED
                    else "候选提示，需要人工确认"
                    if instrument.decision == Decision.CANDIDATE
                    else "分数未达到候选阈值"
                ),
            )
        )
    count_decision = (
        Decision.CONFIRMED
        if detected and all(item.decision == Decision.CONFIRMED for item in detected)
        else Decision.CANDIDATE
        if detected
        else Decision.UNKNOWN
    )
    fields.append(
        FormField(
            field_id="global.instrument_count",
            field_name="instrument_count",
            value=len(detected) if detected else None,
            decision=count_decision,
            model_score=max((item.model_score for item in detected), default=None),
            source="confirmed_or_candidate_instrument_count",
            requires_review=True,
            reason="混音模型只能估计可识别的乐器类别数，不能保证完整编制",
        )
    )
    for instrument in detected:
        for field_name in PER_INSTRUMENT_FIELDS:
            if field_name in {"pan", "width", "movement"}:
                fields.append(_spatial_field(instrument, field_name))
            elif field_name == "timbre" and instrument.timbre is not None:
                fields.append(
                    FormField(
                        field_id=f"{instrument.canonical_label}.timbre",
                        field_name="timbre",
                        instrument_label=instrument.canonical_label,
                        value=instrument.timbre.model_dump(mode="json"),
                        decision=instrument.decision,
                        model_score=instrument.model_score,
                        source="instrument_stem_timbre",
                        requires_review=True,
                        reason="声学特征可辅助判断，但音色词仍需人工复核",
                    )
                )
            else:
                fields.append(
                    _unknown_field(
                        field_name,
                        instrument.canonical_label,
                        "当前基础模型没有该目标字段的可靠监督信号",
                    )
                )
    confirmed = sum(field.decision == Decision.CONFIRMED for field in fields)
    candidate = sum(field.decision == Decision.CANDIDATE for field in fields)
    unknown = len(fields) - confirmed - candidate
    total = len(fields)
    return FormPreview(
        mapping_version=str(mapping["version"]),
        fields=fields,
        summary=FormSummary(
            confirmed=confirmed,
            candidate=candidate,
            unknown=unknown,
            total=total,
            confirmed_coverage=confirmed / total if total else 0.0,
            prefill_coverage=(confirmed + candidate) / total if total else 0.0,
        ),
    )
