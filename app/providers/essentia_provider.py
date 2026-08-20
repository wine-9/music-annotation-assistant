"""Essentia implementation of the official MTG-Jamendo instrument model."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.core.config import ROOT
from app.core.exceptions import ModelIntegrityError, ProviderUnavailable
from app.schemas.results import FrameInstrumentPredictions, WindowPrediction

EMBEDDING_FILE = "discogs-effnet-bs64-1.pb"
HEAD_FILE = "mtg_jamendo_instrument-discogs-effnet-1.pb"
HEAD_METADATA_FILE = "mtg_jamendo_instrument-discogs-effnet-1.json"


class EssentiaInstrumentProvider:
    """Lazily load and run the official Essentia TensorFlow graph pair."""

    name = "essentia_mtg_jamendo"
    version = "1"

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or ROOT / "models"
        self._embedding_model: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None
        self._head_model: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None
        self._labels: list[str] | None = None

    def _load(self) -> None:
        if self._embedding_model is not None:
            return
        embedding_path = self.model_dir / EMBEDDING_FILE
        head_path = self.model_dir / HEAD_FILE
        metadata_path = self.model_dir / HEAD_METADATA_FILE
        missing = [
            path.name
            for path in (embedding_path, head_path, metadata_path)
            if not path.is_file()
        ]
        if missing:
            raise ModelIntegrityError(
                "Missing model artifacts: " + ", ".join(missing) + ". Run make download-models."
            )
        try:
            from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs
        except (ImportError, OSError) as exc:
            raise ProviderUnavailable(
                "essentia-tensorflow is unavailable; install the model extra"
            ) from exc
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = metadata["classes"]
            if not isinstance(labels, list) or len(labels) != 40:
                raise ValueError("Expected official 40-class label list")
            self._labels = [str(label) for label in labels]
            self._embedding_model = TensorflowPredictEffnetDiscogs(
                graphFilename=str(embedding_path),
                output="PartitionedCall:1",
                batchSize=64,
            )
            self._head_model = TensorflowPredict2D(
                graphFilename=str(head_path),
                input="model/Placeholder",
                output="model/Sigmoid",
            )
        except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._embedding_model = None
            self._head_model = None
            raise ProviderUnavailable(f"Could not initialize Essentia model: {exc}") from exc

    def embed(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        """Return frozen Discogs EffNet frame embeddings without running a head."""

        if sample_rate != 16_000:
            raise ValueError("Essentia provider requires mono 16 kHz audio")
        self._load()
        assert self._embedding_model is not None
        try:
            embeddings = np.asarray(self._embedding_model(audio), dtype=np.float32)
        except (RuntimeError, ValueError) as exc:
            raise ProviderUnavailable(f"Essentia embedding inference failed: {exc}") from exc
        if embeddings.ndim != 2 or embeddings.shape[1] != 1280:
            raise ProviderUnavailable(
                f"Unexpected embedding shape {embeddings.shape}; expected (*, 1280)"
            )
        return embeddings

    def score_embeddings(
        self, embeddings: NDArray[np.float32]
    ) -> tuple[list[str], NDArray[np.float32]]:
        """Run the frozen official head on precomputed Discogs EffNet embeddings."""

        self._load()
        assert self._head_model is not None
        assert self._labels is not None
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 1280:
            raise ValueError(f"Expected (*, 1280) embeddings, got {values.shape}")
        try:
            predictions = np.asarray(self._head_model(values), dtype=np.float32)
        except (RuntimeError, ValueError) as exc:
            raise ProviderUnavailable(f"Essentia inference failed: {exc}") from exc
        if predictions.ndim != 2 or predictions.shape[1] != len(self._labels):
            raise ProviderUnavailable(
                f"Unexpected prediction shape {predictions.shape}; expected (*, 40)"
            )
        return list(self._labels), predictions

    def predict(
        self, audio: NDArray[np.float32], sample_rate: int
    ) -> FrameInstrumentPredictions:
        """Return genuine model scores at the embedding model's ~1 Hz cadence."""

        embeddings = self.embed(audio, sample_rate)
        labels, predictions = self.score_embeddings(embeddings)
        duration = len(audio) / sample_rate
        hop_seconds = 62 * 256 / sample_rate
        patch_seconds = 128 * 256 / sample_rate
        windows = [
            WindowPrediction(
                start_seconds=index * hop_seconds,
                end_seconds=min(duration, index * hop_seconds + patch_seconds),
                scores=[float(value) for value in row],
            )
            for index, row in enumerate(predictions)
        ]
        return FrameInstrumentPredictions(
            labels=labels,
            windows=windows,
            provider_name=self.name,
            provider_version=self.version,
        )
