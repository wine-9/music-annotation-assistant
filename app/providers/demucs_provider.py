"""Optional four-stem Demucs adapter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from app.core.exceptions import ProviderUnavailable


class DemucsSeparatorProvider:
    """Run Demucs in an isolated temporary directory with a hard timeout."""

    def __init__(self, *, timeout_seconds: int = 900, device: str = "cpu") -> None:
        self.timeout_seconds = timeout_seconds
        self.device = device

    @property
    def name(self) -> str:
        return "demucs_htdemucs_4stem"

    def separate(self, audio_path: Path) -> dict[str, Path]:
        return self.separate_batch([audio_path])[audio_path]

    def separate_batch(
        self, audio_paths: list[Path]
    ) -> dict[Path, dict[str, Path]]:
        """Separate a bounded batch while loading Demucs weights only once."""

        if not audio_paths:
            return {}
        if importlib.util.find_spec("demucs") is None:
            raise ProviderUnavailable("Demucs is not installed")
        # The temporary directory deliberately survives only while copying/consuming stems.
        # Callers needing persistent cache should copy verified stems into their cache.
        temp_dir = tempfile.TemporaryDirectory(prefix="mla-demucs-")
        output_dir = Path(temp_dir.name)
        command = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "-d",
            self.device,
            "-o",
            str(output_dir),
            *(str(path) for path in audio_paths),
        ]
        completed = subprocess.run(
            command, capture_output=True, timeout=self.timeout_seconds, check=False
        )
        if completed.returncode != 0 and self.device == "mps":
            command[command.index("mps")] = "cpu"
            completed = subprocess.run(
                command, capture_output=True, timeout=self.timeout_seconds, check=False
            )
        if completed.returncode != 0:
            temp_dir.cleanup()
            raise ProviderUnavailable("Demucs separation failed")
        results: dict[Path, dict[str, Path]] = {}
        for audio_path in audio_paths:
            stem_root = output_dir / "htdemucs" / audio_path.stem
            stems = {
                name: stem_root / f"{name}.wav"
                for name in ("drums", "bass", "vocals", "other")
            }
            if not all(path.is_file() for path in stems.values()):
                temp_dir.cleanup()
                raise ProviderUnavailable("Demucs did not produce all four stems")
            results[audio_path] = stems
        # Keep the TemporaryDirectory owner reachable from returned Path objects' provider.
        self._active_temp_dir = temp_dir
        return results
