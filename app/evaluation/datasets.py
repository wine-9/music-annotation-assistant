"""Shared label mappings and public evaluation manifest schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

TARGET_LABELS = (
    "drums",
    "percussion",
    "bass",
    "electric_guitar",
    "acoustic_guitar",
    "piano",
    "strings",
    "brass",
    "woodwind",
    "synth",
)

MTG_TARGET_TAGS: dict[str, frozenset[str]] = {
    "drums": frozenset({"drums"}),
    "percussion": frozenset(
        {"percussion", "bongo", "conga", "djembe", "shaker", "tambourine"}
    ),
    "bass": frozenset({"bass", "acousticbassguitar", "doublebass"}),
    "electric_guitar": frozenset({"electricguitar"}),
    "acoustic_guitar": frozenset({"acousticguitar", "classicalguitar"}),
    "piano": frozenset({"piano"}),
    "strings": frozenset({"strings", "violin", "viola", "cello"}),
    "brass": frozenset({"brass", "horn", "trombone", "trumpet"}),
    "woodwind": frozenset({"clarinet", "flute", "oboe", "saxophone"}),
    "synth": frozenset({"synthesizer"}),
}

OPENMIC_LABELS = (
    "accordion",
    "banjo",
    "bass",
    "cello",
    "clarinet",
    "cymbals",
    "drums",
    "flute",
    "guitar",
    "mallet_percussion",
    "mandolin",
    "organ",
    "piano",
    "saxophone",
    "synthesizer",
    "trombone",
    "trumpet",
    "ukulele",
    "violin",
    "voice",
)

OPENMIC_TARGET_TAGS: dict[str, frozenset[str]] = {
    "drums": frozenset({"drums"}),
    "percussion": frozenset({"mallet_percussion"}),
    "bass": frozenset({"bass"}),
    "piano": frozenset({"piano"}),
    "strings": frozenset({"cello", "violin"}),
    "brass": frozenset({"trombone", "trumpet"}),
    "woodwind": frozenset({"clarinet", "flute", "saxophone"}),
    "synth": frozenset({"synthesizer"}),
}


@dataclass(frozen=True)
class EvaluationSample:
    """One local audio excerpt and its applicable binary targets."""

    sample_id: str
    dataset: str
    split: str
    audio_path: str
    source_url: str
    source_license: str
    targets: dict[str, bool | None]
    source_tags: tuple[str, ...]
    sha256: str
    group_id: str

    REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(TARGET_LABELS)

    def __post_init__(self) -> None:
        if frozenset(self.targets) != self.REQUIRED_KEYS:
            raise ValueError("Every manifest record must contain all target labels")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a stable JSON-compatible mapping."""

        return asdict(self)

    def resolved_path(self, root: Path) -> Path:
        """Resolve the manifest-relative audio path."""

        return root / self.audio_path


def mtg_targets(tags: set[str]) -> dict[str, bool | None]:
    """Map MTG-Jamendo instrument tags to the nine evaluation targets."""

    return {
        label: bool(tags & mapped_tags)
        for label, mapped_tags in MTG_TARGET_TAGS.items()
    }


def openmic_targets(
    values: list[float], masks: list[int]
) -> dict[str, bool | None]:
    """Map partial OpenMIC judgments without inventing missing negatives."""

    by_label = {
        label: (values[index] >= 0.5 if masks[index] else None)
        for index, label in enumerate(OPENMIC_LABELS)
    }
    targets: dict[str, bool | None] = {label: None for label in TARGET_LABELS}
    for target, members in OPENMIC_TARGET_TAGS.items():
        judgments = [by_label[member] for member in members]
        if any(value is True for value in judgments):
            targets[target] = True
        elif all(value is False for value in judgments):
            targets[target] = False
    return targets
