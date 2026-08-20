"""End-to-end local analysis pipeline."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from app import __version__
from app.analysis.spatial import analyze_spatial
from app.audio.loader import load_audio
from app.core.config import (
    get_active_threshold_profile_name,
    get_settings,
    threshold_profile_metadata,
)
from app.decision.engine import build_instrument_results
from app.decision.form_mapper import map_target_form
from app.decision.stem_fusion import apply_stem_fusion
from app.providers.base import InstrumentProvider, SeparatorProvider
from app.schemas.results import AnalysisResult, PipelineMetadata, SourceMetadata


def analyze_file(
    path: Path,
    provider: InstrumentProvider,
    *,
    threshold_profile: str | None = None,
    separator: SeparatorProvider | None = None,
) -> AnalysisResult:
    """Run actual model inference and DSP analysis for a local file."""

    settings = get_settings()
    started = time.perf_counter()
    profile_name = threshold_profile or get_active_threshold_profile_name()
    profile_metadata = threshold_profile_metadata(profile_name)
    loaded = load_audio(
        path,
        model_sample_rate=settings.model_sample_rate,
        max_duration_seconds=settings.max_duration_seconds,
    )
    predictions = provider.predict(loaded.model_mono, loaded.model_sample_rate)
    instruments = build_instrument_results(
        predictions, threshold_profile=profile_name
    )
    separator_name: str | None = None
    if separator is not None:
        stem_paths = separator.separate(path)
        stem_results = {}
        stem_spatial = {}
        stem_energy: dict[str, float] = {}
        raw_energies: dict[str, float] = {}
        for stem_name, stem_path in stem_paths.items():
            stem_audio = load_audio(
                stem_path,
                model_sample_rate=settings.model_sample_rate,
                max_duration_seconds=settings.max_duration_seconds,
            )
            stem_predictions = provider.predict(
                stem_audio.model_mono, stem_audio.model_sample_rate
            )
            stem_results[stem_name] = build_instrument_results(
                stem_predictions, threshold_profile=profile_name
            )
            raw_energies[stem_name] = float(
                np.sqrt(np.mean(np.square(stem_audio.dsp_stereo)))
            )
            stem_spatial[stem_name] = analyze_spatial(
                stem_audio.dsp_stereo,
                stem_audio.dsp_sample_rate,
                original_channels=stem_audio.original_channels,
            ).model_copy(update={"basis": f"{separator.name}:{stem_name}"})
        energy_sum = sum(raw_energies.values())
        stem_energy = {
            name: energy / energy_sum if energy_sum else 0.0
            for name, energy in raw_energies.items()
        }
        instruments = apply_stem_fusion(
            instruments,
            stem_results,
            stem_energy,
            stem_spatial,
            threshold_profile=profile_name,
        )
        separator_name = separator.name
    global_spatial = analyze_spatial(
        loaded.dsp_stereo,
        loaded.dsp_sample_rate,
        original_channels=loaded.original_channels,
    )
    threshold_version = str(profile_metadata["version"])
    warnings = list(loaded.warnings) + list(global_spatial.warnings)
    if any(item.decision == "confirmed" for item in instruments):
        warnings.append("confirmed_results_require_human_review")
    if not any(item.timbre for item in instruments):
        warnings.append("per_instrument_timbre_requires_reliable_stems")
    return AnalysisResult(
        analysis_id=loaded.sha256[:16],
        created_at=datetime.now().astimezone(),
        source=SourceMetadata(
            filename=path.name,
            sha256=loaded.sha256,
            duration_seconds=loaded.duration_seconds,
            sample_rate=loaded.dsp_sample_rate,
            channels=loaded.original_channels,
        ),
        pipeline=PipelineMetadata(
            version=__version__,
            instrument_provider=predictions.provider_name,
            instrument_model_version=predictions.provider_version,
            separator_provider=separator_name,
            threshold_config_version=threshold_version,
            threshold_profile_name=profile_name,
            threshold_profile_generated_at=str(profile_metadata["generated_at"]),
            threshold_profile_datasets=[
                str(item)
                for item in cast(
                    list[object], profile_metadata["applicable_datasets"]
                )
            ],
        ),
        instruments=instruments,
        global_spatial=global_spatial,
        target_form=map_target_form(instruments),
        model_analysis_seconds=time.perf_counter() - started,
        warnings=warnings,
    )
