from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from app.core.config import ROOT
from app.providers.essentia_provider import (
    EMBEDDING_FILE,
    HEAD_FILE,
    HEAD_METADATA_FILE,
    EssentiaInstrumentProvider,
)


@pytest.mark.model
def test_real_model_outputs_40_scores() -> None:
    required = [ROOT / "models" / name for name in (EMBEDDING_FILE, HEAD_FILE, HEAD_METADATA_FILE)]
    if importlib.util.find_spec("essentia") is None or not all(path.is_file() for path in required):
        pytest.skip("Real Essentia runtime or model weights are not installed")
    sample_rate = 16_000
    time = np.arange(sample_rate * 8, dtype=np.float32) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    result = EssentiaInstrumentProvider().predict(audio, sample_rate)
    assert len(result.labels) == 40
    assert result.windows
    assert len(result.windows[0].scores) == 40
    assert all(0 <= value <= 1 for value in result.windows[0].scores)
