#!/usr/bin/env python3
"""Document safe public-dataset metadata entry points without bulk audio downloads."""

from __future__ import annotations

import json
from pathlib import Path

SOURCES = {
    "mtg_jamendo": "https://github.com/MTG/mtg-jamendo-dataset",
    "openmic_2018": "https://zenodo.org/records/1432913",
    "medleydb": "https://medleydb.weebly.com/",
}


def main() -> int:
    output = Path("data/public_metadata_sources.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "notice": "Only source metadata is recorded; no dataset audio was downloaded.",
                "sources": SOURCES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
