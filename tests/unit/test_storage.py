from __future__ import annotations

from pathlib import Path

from app.schemas.results import (
    AnalysisResult,
    HumanOverride,
    HumanReviewEvent,
    HumanReviewSession,
    PipelineMetadata,
    SourceMetadata,
    SpatialResult,
)
from app.storage.repository import Repository


def sample_result() -> AnalysisResult:
    return AnalysisResult(
        analysis_id="abc123",
        source=SourceMetadata(
            filename="test.wav",
            sha256="a" * 64,
            duration_seconds=1,
            sample_rate=44_100,
            channels=2,
        ),
        pipeline=PipelineMetadata(
            version="0.1.0",
            instrument_provider="test",
            instrument_model_version="1",
            threshold_config_version="test",
        ),
        instruments=[],
        global_spatial=SpatialResult(basis="global_mix"),
    )


def test_human_override_does_not_replace_model_result(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "tasks.sqlite3")
    result = sample_result()
    # Redirect output path for the test repository.
    repository.result_path = lambda analysis_id: tmp_path / analysis_id / "result.json"  # type: ignore[method-assign]
    repository.save_result = lambda value: _save(tmp_path, value)  # type: ignore[method-assign]
    repository.save_result(result)
    updated = repository.append_overrides(
        "abc123",
        [
            HumanOverride(
                field_path="instruments.0.decision",
                old_value="candidate",
                new_value="confirmed",
            )
        ],
    )
    assert updated is not None
    assert updated.instruments == []
    assert updated.human_overrides[0].new_value == "confirmed"


def test_review_session_is_append_only_and_round_trips(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "tasks.sqlite3")
    session = HumanReviewSession(
        analysis_id="abc123",
        events=[
            HumanReviewEvent(
                field_id="bass.instrument_name",
                outcome="accept_as_is",
                final_value="bass",
                threshold_profile_name="calibrated_public_v1",
                field_elapsed_seconds=2.5,
            )
        ],
        total_review_seconds=8.0,
        baseline_manual_seconds=20.0,
        model_analysis_seconds=1.2,
    )
    repository.save_review_session(session)
    loaded = repository.list_review_sessions()
    assert len(loaded) == 1
    assert loaded[0].events[0].outcome == "accept_as_is"
    assert loaded[0].total_review_seconds == 8.0


def _save(root: Path, result: AnalysisResult) -> Path:
    path = root / result.analysis_id / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path
