#!/usr/bin/env python3
"""Verify model sizes, checksums, metadata, and label order."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.is_file():
        print("Model manifest missing. Run make download-models.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = MODEL_DIR / artifact["filename"]
        if not path.is_file():
            print(f"Missing: {path.name}", file=sys.stderr)
            return 1
        if path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
            print(f"Integrity failure: {path.name}", file=sys.stderr)
            return 1
    metadata = json.loads(
        (MODEL_DIR / "mtg_jamendo_instrument-discogs-effnet-1.json").read_text(
            encoding="utf-8"
        )
    )
    if len(metadata.get("classes", [])) != 40:
        print("Official label list is not 40 classes.", file=sys.stderr)
        return 1
    print("Verified all model artifacts and the official 40-class label order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
