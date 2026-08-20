#!/usr/bin/env python3
"""Measure real local runtime and peak RSS for a supplied file."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from app.core.pipeline import analyze_file
from app.providers.essentia_provider import EssentiaInstrumentProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path, nargs="?")
    arguments = parser.parse_args()
    if arguments.audio is None:
        print("Provide a legal local audio file: make benchmark ARGS=/path/file.wav")
        return 2
    start = time.perf_counter()
    result = analyze_file(arguments.audio, EssentiaInstrumentProvider())
    elapsed = time.perf_counter() - start
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if platform.system() != "Darwin":
        peak_bytes *= 1024
    report = {
        "device": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "audio_duration_seconds": result.source.duration_seconds,
        "processing_seconds": elapsed,
        "real_time_factor": elapsed / result.source.duration_seconds,
        "peak_memory_gib": peak_bytes / 1024**3,
        "mps_enabled": False,
        "instrument_provider": result.pipeline.instrument_provider,
    }
    output = Path("outputs/benchmark.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
