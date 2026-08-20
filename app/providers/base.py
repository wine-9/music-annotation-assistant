"""Replaceable provider interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.schemas.results import FrameInstrumentPredictions


class InstrumentProvider(Protocol):
    """Interface isolating the application from a concrete model runtime."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def predict(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> FrameInstrumentPredictions: ...


class SeparatorProvider(Protocol):
    """Interface for optional source separation implementations."""

    @property
    def name(self) -> str: ...

    def separate(self, audio_path: Path) -> dict[str, Path]: ...
