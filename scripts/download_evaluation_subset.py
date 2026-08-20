#!/usr/bin/env python3
"""Download deterministic, bounded public evaluation excerpts and metadata."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from imageio_ffmpeg import get_ffmpeg_exe

from app.evaluation.datasets import (
    TARGET_LABELS,
    EvaluationSample,
    mtg_targets,
    openmic_targets,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = ROOT / "data" / "evaluation"
METADATA_DIR = EVALUATION_ROOT / "metadata"
AUDIO_DIR = EVALUATION_ROOT / "audio"
MAX_DOWNLOAD_BYTES = 2 * 1024**3
MTG_SPLIT_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
    "data/splits/split-0/autotagging_instrument-test.tsv"
)
MTG_LICENSES_URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
    "audio_licenses.txt"
)
OPENMIC_ROWS_URL = "https://datasets-server.huggingface.co/rows"
OPENMIC_SOURCE_URL = "https://zenodo.org/records/1432913"
OPENMIC_PARQUET_URL = (
    "https://huggingface.co/datasets/CPJKU/openmic/resolve/main/"
    "data/shard_test_0.parquet"
)


@dataclass(frozen=True)
class MtgTrack:
    track_id: str
    numeric_id: int
    path: str
    duration: float
    tags: frozenset[str]
    targets: dict[str, bool | None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_bytes(url: str, *, attempts: int = 3, timeout: int = 120) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "MusicLabelAssistant-Evaluation/0.1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                declared = int(response.headers.get("Content-Length", "0"))
                if declared > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("Download exceeds the 2GB safety limit")
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("Download exceeded the 2GB safety limit")
                return data
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable retry state")


def _download_metadata(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(_request_bytes(url))
    temporary.replace(destination)
    return destination


def _stream_download(url: str, destination: Path) -> Path:
    """Stream a bounded temporary artifact without holding it in memory."""

    request = urllib.request.Request(
        url, headers={"User-Agent": "MusicLabelAssistant-Evaluation/0.1"}
    )
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        declared = int(response.headers.get("Content-Length", "0"))
        if declared > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Download exceeds the 2GB safety limit")
        with destination.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("Download exceeded the 2GB safety limit")
                handle.write(chunk)
    return destination


def _parse_mtg_tracks(path: Path) -> list[MtgTrack]:
    tracks: list[MtgTrack] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            tags = frozenset(
                tag.removeprefix("instrument---") for tag in row["TAGS"].split(",")
            )
            numeric_id = int(row["TRACK_ID"].removeprefix("track_"))
            tracks.append(
                MtgTrack(
                    track_id=row["TRACK_ID"],
                    numeric_id=numeric_id,
                    path=row["PATH"],
                    duration=float(row["DURATION"]),
                    tags=tags,
                    targets=mtg_targets(set(tags)),
                )
            )
    return tracks


def _parse_mtg_licenses(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    licenses: dict[str, str] = {}
    for index, line in enumerate(lines):
        if line.endswith(".mp3") and index + 2 < len(lines):
            licenses[line] = lines[index + 2]
    return licenses


def _ensure_quota(
    selected: dict[str, dict[str, bool]],
    candidates: list[MtgTrack],
    *,
    label: str,
    target: bool,
    quota: int,
    excluded: set[str],
    allow_shortfall: bool = False,
) -> int:
    current = sum(
        selected_targets.get(label) is target
        for selected_targets in selected.values()
    )
    if current >= quota:
        return current
    for track in candidates:
        if track.track_id in excluded:
            continue
        if track.targets[label] is not target:
            continue
        selected_targets = selected.setdefault(track.track_id, {})
        if label in selected_targets:
            continue
        selected_targets[label] = target
        current += 1
        if current >= quota:
            return current
    if allow_shortfall and current:
        return current
    raise RuntimeError(f"Not enough MTG-Jamendo samples for {label}={target}")


def _select_mtg(
    tracks: list[MtgTrack],
    *,
    calibration_per_state: int,
    evaluation_per_state: int,
    seed: int,
) -> tuple[
    list[tuple[MtgTrack, dict[str, bool]]],
    list[tuple[MtgTrack, dict[str, bool]]],
]:
    shuffled = list(tracks)
    random.Random(seed).shuffle(shuffled)
    calibration: dict[str, dict[str, bool]] = {}
    for label in TARGET_LABELS:
        for target in (True, False):
            _ensure_quota(
                calibration,
                shuffled,
                label=label,
                target=target,
                quota=calibration_per_state,
                excluded=set(),
            )
    evaluation: dict[str, dict[str, bool]] = {}
    for label in TARGET_LABELS:
        for target in (True, False):
            _ensure_quota(
                evaluation,
                shuffled,
                label=label,
                target=target,
                quota=evaluation_per_state,
                excluded=set(calibration),
                allow_shortfall=target,
            )
    by_id = {track.track_id: track for track in tracks}
    return (
        [
            (by_id[track_id], calibration[track_id])
            for track_id in sorted(calibration)
        ],
        [
            (by_id[track_id], evaluation[track_id])
            for track_id in sorted(evaluation)
        ],
    )


def _download_mtg_clip(
    track: MtgTrack,
    *,
    split: str,
    applicable_targets: dict[str, bool],
    clip_seconds: float,
    licenses: dict[str, str],
) -> EvaluationSample | None:
    output_dir = AUDIO_DIR / "mtg_jamendo" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{track.track_id}.mp3"
    source_url = (
        f"https://prod-1.storage.jamendo.com/?trackid={track.numeric_id}&format=mp32"
    )
    if not output.is_file():
        temporary = output_dir / f".{track.track_id}.part.mp3"
        start_seconds = max(0.0, min(30.0, track.duration / 2 - clip_seconds / 2))
        command = [
            get_ffmpeg_exe(),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            source_url,
            "-t",
            f"{clip_seconds:.3f}",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "96k",
            "-y",
            str(temporary),
        ]
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=180
        )
        if completed.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            return None
        temporary.replace(output)
    return EvaluationSample(
        sample_id=track.track_id,
        dataset="mtg_jamendo_instrument",
        split=split,
        audio_path=str(output.relative_to(EVALUATION_ROOT)),
        source_url=source_url,
        source_license=licenses.get(
            track.path, "Creative Commons; exact license unavailable in parsed metadata"
        ),
        targets={
            label: applicable_targets.get(label)
            for label in TARGET_LABELS
        },
        source_tags=tuple(sorted(track.tags)),
        sha256=_sha256(output),
        group_id=track.track_id,
    )


def _openmic_rows(offset: int, length: int) -> list[dict[str, object]]:
    query = urllib.parse.urlencode(
        {
            "dataset": "CPJKU/openmic",
            "config": "default",
            "split": "test",
            "offset": offset,
            "length": length,
        }
    )
    payload = json.loads(_request_bytes(f"{OPENMIC_ROWS_URL}?{query}"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected OpenMIC rows response")
    return rows


def _select_openmic(
    *,
    per_state: int,
    max_scanned_rows: int,
) -> list[EvaluationSample]:
    selected: list[EvaluationSample] = []
    counts = {
        (label, target): 0
        for label in TARGET_LABELS
        for target in (True, False)
    }
    output_dir = AUDIO_DIR / "openmic" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    page_size = 20
    failed_pages = 0
    for offset in range(0, max_scanned_rows, page_size):
        try:
            rows = _openmic_rows(
                offset, min(page_size, max_scanned_rows - offset)
            )
        except (urllib.error.HTTPError, urllib.error.URLError):
            failed_pages += 1
            continue
        for wrapped in rows:
            raw_row = wrapped.get("row")
            if not isinstance(raw_row, dict):
                continue
            values = [float(value) for value in raw_row["true"]]
            masks = [int(value) for value in raw_row["mask"]]
            targets = openmic_targets(values, masks)
            helps = any(
                target is not None and counts[(label, target)] < per_state
                for label, target in targets.items()
            )
            if not helps:
                continue
            sample_id = str(raw_row["filename"])
            audio_bytes = base64.b64decode(str(raw_row["mp3_bytes"]))
            output = output_dir / f"{sample_id}.mp3"
            if not output.is_file():
                output.write_bytes(audio_bytes)
            selected.append(
                EvaluationSample(
                    sample_id=sample_id,
                    dataset="openmic_2018",
                    split="evaluation",
                    audio_path=str(output.relative_to(EVALUATION_ROOT)),
                    source_url=OPENMIC_SOURCE_URL,
                    source_license="CC BY 4.0",
                    targets=targets,
                    source_tags=tuple(
                        label for label, target in targets.items() if target is True
                    ),
                    sha256=_sha256(output),
                    group_id=sample_id,
                )
            )
            for label, target in targets.items():
                if target is not None:
                    counts[(label, target)] += 1
        applicable = [
            (label, target)
            for label in TARGET_LABELS
            if label not in {"electric_guitar", "acoustic_guitar"}
            for target in (True, False)
        ]
        if all(counts[key] >= per_state for key in applicable):
            break
        if offset and offset % 500 == 0:
            print(
                f"\rOpenMIC scanned {offset}/{max_scanned_rows}; "
                f"selected {len(selected)}; skipped pages {failed_pages}",
                end="",
                flush=True,
            )
    print()
    return selected


def _select_openmic_parquet(
    path: Path,
    *,
    per_state: int,
) -> list[EvaluationSample]:
    """Select only needed OpenMIC rows from a bounded local public shard."""

    parquet = pq.ParquetFile(path)
    metadata_rows = parquet.read(columns=["filename", "true", "mask"]).to_pylist()
    counts = {
        (label, target): 0
        for label in TARGET_LABELS
        for target in (True, False)
    }
    chosen: dict[int, tuple[str, dict[str, bool | None]]] = {}
    for index, row in enumerate(metadata_rows):
        targets = openmic_targets(
            [float(value) for value in row["true"]],
            [int(value) for value in row["mask"]],
        )
        helps = any(
            target is not None and counts[(label, target)] < per_state
            for label, target in targets.items()
        )
        if not helps:
            continue
        sample_id = str(row["filename"])
        chosen[index] = (sample_id, targets)
        for label, target in targets.items():
            if target is not None:
                counts[(label, target)] += 1
        applicable = [
            (label, target)
            for label in TARGET_LABELS
            if label not in {"electric_guitar", "acoustic_guitar"}
            for target in (True, False)
        ]
        if all(counts[key] >= per_state for key in applicable):
            break

    output_dir = AUDIO_DIR / "openmic" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[EvaluationSample] = []
    by_group: dict[int, list[int]] = {}
    row_start = 0
    for group_index in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(group_index).num_rows
        local_indices = [
            index - row_start
            for index in chosen
            if row_start <= index < row_start + row_count
        ]
        if local_indices:
            by_group[group_index] = local_indices
        row_start += row_count
    row_start = 0
    for group_index in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(group_index).num_rows
        if group_index not in by_group:
            row_start += row_count
            continue
        rows = parquet.read_row_group(
            group_index, columns=["filename", "mp3_bytes"]
        ).to_pylist()
        for local_index in by_group[group_index]:
            global_index = row_start + local_index
            sample_id, targets = chosen[global_index]
            output = output_dir / f"{sample_id}.mp3"
            if not output.is_file():
                output.write_bytes(bytes(rows[local_index]["mp3_bytes"]))
            selected.append(
                EvaluationSample(
                    sample_id=sample_id,
                    dataset="openmic_2018",
                    split="evaluation",
                    audio_path=str(output.relative_to(EVALUATION_ROOT)),
                    source_url=OPENMIC_SOURCE_URL,
                    source_license="CC BY 4.0",
                    targets=targets,
                    source_tags=tuple(
                        label for label, target in targets.items() if target is True
                    ),
                    sha256=_sha256(output),
                    group_id=sample_id,
                )
            )
        row_start += row_count
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-per-state", type=int, default=20)
    parser.add_argument("--evaluation-per-state", type=int, default=50)
    parser.add_argument("--openmic-per-state", type=int, default=40)
    parser.add_argument("--openmic-max-scan", type=int, default=5000)
    parser.add_argument(
        "--openmic-parquet",
        type=Path,
        help="Optional existing official-mirror test parquet; otherwise use a temporary copy",
    )
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    arguments = parser.parse_args()

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    split_path = _download_metadata(
        MTG_SPLIT_URL, METADATA_DIR / "mtg_jamendo_instrument_split0_test.tsv"
    )
    licenses_path = _download_metadata(
        MTG_LICENSES_URL, METADATA_DIR / "mtg_jamendo_audio_licenses.txt"
    )
    tracks = _parse_mtg_tracks(split_path)
    licenses = _parse_mtg_licenses(licenses_path)
    calibration, evaluation = _select_mtg(
        tracks,
        calibration_per_state=arguments.calibration_per_state,
        evaluation_per_state=arguments.evaluation_per_state,
        seed=arguments.seed,
    )

    jobs = [(track, targets, "calibration") for track, targets in calibration] + [
        (track, targets, "evaluation") for track, targets in evaluation
    ]
    mtg_samples: list[EvaluationSample] = []
    failures: list[str] = []
    print(f"Downloading {len(jobs)} bounded MTG-Jamendo excerpts...")
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        future_map = {
            executor.submit(
                _download_mtg_clip,
                track,
                split=split,
                applicable_targets=targets,
                clip_seconds=arguments.clip_seconds,
                licenses=licenses,
            ): track.track_id
            for track, targets, split in jobs
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            sample = future.result()
            if sample is None:
                failures.append(future_map[future])
            else:
                mtg_samples.append(sample)
            print(f"\rMTG-Jamendo: {index}/{len(jobs)}", end="", flush=True)
    print()

    print("Selecting bounded OpenMIC test excerpts...")
    if arguments.openmic_parquet:
        openmic_samples = _select_openmic_parquet(
            arguments.openmic_parquet,
            per_state=arguments.openmic_per_state,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="mla-openmic-public-") as temp_dir:
            temporary_parquet = Path(temp_dir) / "openmic_test.parquet"
            print("Downloading bounded 581 MiB OpenMIC test shard temporarily...")
            _stream_download(OPENMIC_PARQUET_URL, temporary_parquet)
            openmic_samples = _select_openmic_parquet(
                temporary_parquet,
                per_state=arguments.openmic_per_state,
            )
    samples = sorted(
        [*mtg_samples, *openmic_samples],
        key=lambda item: (item.dataset, item.split, item.sample_id),
    )
    groups_by_split: dict[str, set[str]] = {}
    for sample in samples:
        groups_by_split.setdefault(sample.split, set()).add(sample.group_id)
    if groups_by_split.get("calibration", set()) & groups_by_split.get(
        "evaluation", set()
    ):
        raise RuntimeError("Source group leakage detected between splits")
    manifest = {
        "schema_version": "1.0",
        "created_at_epoch": time.time(),
        "seed": arguments.seed,
        "clip_seconds": arguments.clip_seconds,
        "sources": {
            "mtg_jamendo_split": MTG_SPLIT_URL,
            "mtg_jamendo_licenses": MTG_LICENSES_URL,
            "openmic": OPENMIC_SOURCE_URL,
            "openmic_mirror": "https://huggingface.co/datasets/CPJKU/openmic",
        },
        "download_failures": failures,
        "samples": [sample.to_dict() for sample in samples],
    }
    manifest_path = EVALUATION_ROOT / "selection_manifest.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=EVALUATION_ROOT,
        prefix=".selection-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(manifest_path)
    total_bytes = sum(
        sample.resolved_path(EVALUATION_ROOT).stat().st_size for sample in samples
    )
    print(f"Saved {len(samples)} excerpts ({total_bytes / 1024**2:.1f} MiB)")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
