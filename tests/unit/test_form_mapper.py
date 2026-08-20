from __future__ import annotations

from app.decision.form_mapper import map_target_form
from app.schemas.results import (
    Decision,
    InstrumentResult,
    ScoreStatistics,
    SpatialResult,
)


def _instrument() -> InstrumentResult:
    return InstrumentResult(
        canonical_label="piano",
        display_name_zh="钢琴",
        decision=Decision.CONFIRMED,
        model_score=0.9,
        statistics=ScoreStatistics(
            mean=0.5,
            median=0.5,
            max=0.9,
            p90=0.8,
            p95=0.85,
            top_20_percent_mean=0.9,
            active_ratio=0.5,
        ),
        raw_model_scores={"piano": 0.9},
        time_ranges=[],
        spatial=SpatialResult(
            pan="left",
            width="wide",
            stability="moving",
            basis="no_reliable_stem",
        ),
    )


def test_form_mapper_never_copies_global_mix_spatial() -> None:
    preview = map_target_form([_instrument()])
    by_id = {field.field_id: field for field in preview.fields}
    assert by_id["piano.instrument_name"].decision == Decision.CONFIRMED
    assert by_id["piano.pan"].decision == Decision.UNKNOWN
    assert by_id["piano.pan"].value is None
    assert "禁止" in by_id["piano.pan"].reason
