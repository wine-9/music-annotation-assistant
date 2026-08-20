"""Pydantic schemas for predictions and persisted results."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Decision(str, Enum):
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


class TimeRange(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    peak_score: float = Field(ge=0, le=1)
    interval_confidence: float = Field(default=0.0, ge=0, le=1)


class WindowPrediction(BaseModel):
    start_seconds: float
    end_seconds: float
    scores: list[float]


class FrameInstrumentPredictions(BaseModel):
    labels: list[str]
    windows: list[WindowPrediction]
    provider_name: str
    provider_version: str


class ScoreStatistics(BaseModel):
    mean: float
    median: float
    max: float
    p90: float
    p95: float
    top_20_percent_mean: float
    active_ratio: float


class SpatialResult(BaseModel):
    pan: str = "unknown"
    pan_value: float | None = None
    width: str = "unknown"
    width_ratio: float | None = None
    correlation: float | None = None
    stability: str = "unknown"
    pan_iqr: float | None = None
    movement_rate: float | None = None
    basis: str
    warnings: list[str] = Field(default_factory=list)


class TimbreFeatureValues(BaseModel):
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    high_frequency_energy_ratio: float
    spectral_flatness: float
    crest_factor: float
    onset_strength: float
    zero_crossing_rate: float
    temporal_decay_seconds: float
    dynamic_range_db: float


class TimbreAttribute(BaseModel):
    value: str
    decision: Decision
    supporting_indicators: int


class TimbreResult(BaseModel):
    brightness: TimbreAttribute
    attack: TimbreAttribute
    sustain: TimbreAttribute
    features: TimbreFeatureValues


class StemFusionEvidence(BaseModel):
    mix_score: float = Field(ge=0, le=1)
    stem_scores: dict[str, float] = Field(default_factory=dict)
    fused_score: float = Field(ge=0, le=1)
    stem_energy_ratios: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class InstrumentResult(BaseModel):
    canonical_label: str
    display_name_zh: str
    decision: Decision
    model_score: float
    statistics: ScoreStatistics
    raw_model_scores: dict[str, float]
    time_ranges: list[TimeRange]
    spatial: SpatialResult
    timbre: TimbreResult | None = None
    stem_fusion: StemFusionEvidence | None = None


class SourceMetadata(BaseModel):
    filename: str
    sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int


class PipelineMetadata(BaseModel):
    version: str
    instrument_provider: str
    instrument_model_version: str
    separator_provider: str | None = None
    threshold_config_version: str
    threshold_profile_name: str = "default"
    threshold_profile_generated_at: str = "unknown"
    threshold_profile_datasets: list[str] = Field(default_factory=list)


class HumanOverride(BaseModel):
    field_path: str
    old_value: object | None = None
    new_value: object
    note: str | None = None
    modified_at: datetime = Field(default_factory=datetime.now)


class ReviewOutcome(str, Enum):
    ACCEPT_AS_IS = "accept_as_is"
    ACCEPT_AFTER_EDIT = "accept_after_edit"
    REJECT = "reject"
    MISSING_LABEL = "missing_label"
    UNCERTAIN = "uncertain"


class FormField(BaseModel):
    field_id: str
    field_name: str
    instrument_label: str | None = None
    value: object | None = None
    decision: Decision
    model_score: float | None = None
    source: str
    requires_review: bool
    reason: str


class FormSummary(BaseModel):
    confirmed: int
    candidate: int
    unknown: int
    total: int
    confirmed_coverage: float
    prefill_coverage: float


class FormPreview(BaseModel):
    mapping_version: str
    fields: list[FormField]
    summary: FormSummary


class HumanReviewEvent(BaseModel):
    field_id: str
    outcome: ReviewOutcome
    original_prediction: FormField | None = None
    final_value: object | None = None
    edit_description: str | None = None
    threshold_profile_name: str
    field_elapsed_seconds: float = Field(ge=0)
    recorded_at: datetime = Field(default_factory=datetime.now)


class HumanReviewSession(BaseModel):
    analysis_id: str
    events: list[HumanReviewEvent]
    total_review_seconds: float = Field(ge=0)
    baseline_manual_seconds: float | None = Field(default=None, ge=0)
    model_analysis_seconds: float = Field(ge=0)
    completed_at: datetime = Field(default_factory=datetime.now)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    schema_version: str = "1.0.0"
    analysis_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    source: SourceMetadata
    pipeline: PipelineMetadata
    instruments: list[InstrumentResult]
    global_spatial: SpatialResult
    target_form: FormPreview | None = None
    model_analysis_seconds: float = Field(default=0.0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    human_overrides: list[HumanOverride] = Field(default_factory=list)
