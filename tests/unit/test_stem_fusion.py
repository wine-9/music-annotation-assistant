from __future__ import annotations

from app.decision.stem_fusion import apply_stem_fusion
from app.schemas.results import (
    Decision,
    InstrumentResult,
    ScoreStatistics,
    SpatialResult,
)


def _result(label: str, score: float) -> InstrumentResult:
    return InstrumentResult(
        canonical_label=label,
        display_name_zh=label,
        decision=Decision.UNKNOWN,
        model_score=score,
        statistics=ScoreStatistics(
            mean=score,
            median=score,
            max=score,
            p90=score,
            p95=score,
            top_20_percent_mean=score,
            active_ratio=0.0,
        ),
        raw_model_scores={label: score},
        time_ranges=[],
        spatial=SpatialResult(basis="no_reliable_stem"),
    )


def test_stems_can_only_promote_supported_label_to_candidate() -> None:
    mix = [_result("drums", 0.1), _result("electric_guitar", 0.1)]
    stem_results = {
        "drums": [_result("drums", 0.9), _result("electric_guitar", 0.2)],
        "other": [_result("drums", 0.2), _result("electric_guitar", 0.99)],
    }
    fused = {
        item.canonical_label: item
        for item in apply_stem_fusion(
            mix,
            stem_results,
            {"drums": 0.5, "other": 0.5},
            {"drums": SpatialResult(basis="demucs:drums")},
            threshold_profile="calibrated_public_v1",
        )
    }
    assert fused["drums"].decision == Decision.CANDIDATE
    assert fused["drums"].decision != Decision.CONFIRMED
    assert fused["electric_guitar"].decision == Decision.UNKNOWN
    assert "other_stem_not_used_for_instrument_identity" in (
        fused["electric_guitar"].stem_fusion.warnings  # type: ignore[union-attr]
    )
