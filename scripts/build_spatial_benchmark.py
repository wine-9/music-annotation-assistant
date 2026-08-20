#!/usr/bin/env python3
"""Create deterministic synthetic stereo fixtures and exact spatial ground truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

OUTPUT = Path("data/spatial_benchmark")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_rate = 44_100
    time = np.arange(sample_rate * 3) / sample_rate
    mono = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    cases = {
        "left": (1.0, 0.0),
        "right": (0.0, 1.0),
        "center": (1.0, 1.0),
        "narrow": (1.0, 0.85),
        "wide": (1.0, -0.25),
        "inverted": (1.0, -1.0),
    }
    truth: dict[str, object] = {"sample_rate": sample_rate, "cases": {}}
    for name, gains in cases.items():
        stereo = np.column_stack((mono * gains[0], mono * gains[1]))
        sf.write(OUTPUT / f"{name}.wav", stereo, sample_rate, subtype="FLOAT")
        truth["cases"][name] = {"left_gain": gains[0], "right_gain": gains[1]}
    (OUTPUT / "ground_truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
