"""Conservative, training-free fusion of mix and four-stem predictions."""

from __future__ import annotations

from app.core.config import load_threshold_profile
from app.schemas.results import (
    Decision,
    InstrumentResult,
    SpatialResult,
    StemFusionEvidence,
)

SUPPORTED_STEM_BY_LABEL = {
    "drums": "drums",
    "percussion": "drums",
    "bass": "bass",
    "voice": "vocals",
}


def apply_stem_fusion(
    mix_results: list[InstrumentResult],
    stem_results: dict[str, list[InstrumentResult]],
    stem_energy_ratios: dict[str, float],
    stem_spatial: dict[str, SpatialResult],
    *,
    threshold_profile: str,
) -> list[InstrumentResult]:
    """Fuse only semantically supported stems and never promote to confirmed."""

    profile = load_threshold_profile(threshold_profile)
    label_config = profile["labels"]
    if not isinstance(label_config, dict):
        raise ValueError("Threshold profile labels must be a mapping")
    stem_by_label = {
        stem: {item.canonical_label: item for item in results}
        for stem, results in stem_results.items()
    }
    fused_results: list[InstrumentResult] = []
    for mix in mix_results:
        warnings: list[str] = []
        scores = {
            stem: items[mix.canonical_label].model_score
            for stem, items in stem_by_label.items()
            if mix.canonical_label in items
        }
        supported_stem = SUPPORTED_STEM_BY_LABEL.get(mix.canonical_label)
        fused_score = mix.model_score
        spatial = mix.spatial
        if supported_stem is not None and supported_stem in scores:
            energy = stem_energy_ratios.get(supported_stem, 0.0)
            reliability = min(1.0, energy / 0.08)
            fused_score = max(fused_score, scores[supported_stem] * reliability)
            if energy < 0.01:
                warnings.append("supported_stem_energy_too_low")
            elif supported_stem in stem_spatial:
                spatial = stem_spatial[supported_stem]
        if "other" in scores and mix.canonical_label not in SUPPORTED_STEM_BY_LABEL:
            warnings.append("other_stem_not_used_for_instrument_identity")
        config = label_config.get(mix.canonical_label, profile["defaults"])
        if not isinstance(config, dict):
            raise ValueError("Threshold label entry must be a mapping")
        candidate_threshold = float(config["candidate"])
        decision = mix.decision
        if decision == Decision.UNKNOWN and fused_score >= candidate_threshold:
            decision = Decision.CANDIDATE
        fused_results.append(
            mix.model_copy(
                update={
                    "decision": decision,
                    "model_score": fused_score,
                    "spatial": spatial,
                    "stem_fusion": StemFusionEvidence(
                        mix_score=mix.model_score,
                        stem_scores=scores,
                        fused_score=fused_score,
                        stem_energy_ratios=stem_energy_ratios,
                        warnings=warnings,
                    ),
                }
            )
        )
    return fused_results
