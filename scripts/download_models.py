#!/usr/bin/env python3
"""Download only the two official model graphs and metadata with atomic writes."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MAX_BYTES = 2 * 1024**3
ARTIFACTS = {
    "discogs-effnet-bs64-1.pb": (
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
        "discogs-effnet-bs64-1.pb"
    ),
    "discogs-effnet-bs64-1.json": (
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
        "discogs-effnet-bs64-1.json"
    ),
    "mtg_jamendo_instrument-discogs-effnet-1.pb": (
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/"
        "mtg_jamendo_instrument-discogs-effnet-1.pb"
    ),
    "mtg_jamendo_instrument-discogs-effnet-1.json": (
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/"
        "mtg_jamendo_instrument-discogs-effnet-1.json"
    ),
}


def download(name: str, url: str) -> dict[str, object]:
    destination = MODEL_DIR / name
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "MusicLabelAssistant/0.1"})
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        declared = int(response.headers.get("Content-Length", "0"))
        if declared > MAX_BYTES:
            raise RuntimeError(f"{name} exceeds the 2GB safety limit")
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > MAX_BYTES:
                raise RuntimeError(f"{name} exceeded the 2GB safety limit")
            digest.update(chunk)
            output.write(chunk)
            print(f"\r{name}: {received / 1024**2:.1f} MiB", end="", flush=True)
    print()
    temporary.replace(destination)
    return {
        "filename": name,
        "url": url,
        "bytes": received,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    records = [download(name, url) for name, url in ARTIFACTS.items()]
    manifest = {
        "source": "Essentia official model repository",
        "model_family": "discogs-effnet-bs64-1 + mtg_jamendo_instrument head",
        "artifacts": records,
    }
    (MODEL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
