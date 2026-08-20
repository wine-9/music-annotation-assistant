#!/usr/bin/env python3
"""Analyze one local audio file and persist JSON/CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.pipeline import analyze_file
from app.providers.essentia_provider import EssentiaInstrumentProvider
from app.storage.repository import Repository, export_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    arguments = parser.parse_args()
    result = analyze_file(arguments.audio, EssentiaInstrumentProvider())
    repository = Repository()
    result_path = repository.save_result(result)
    csv_path = result_path.with_suffix(".csv")
    csv_path.write_text(export_csv(result), encoding="utf-8")
    print(result_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
