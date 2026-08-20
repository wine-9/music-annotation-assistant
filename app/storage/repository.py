"""SQLite task and immutable model-result storage."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from app.core.config import ROOT
from app.schemas.results import AnalysisResult, HumanOverride, HumanReviewSession


class Repository:
    """Small local repository; no external database or network service."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path or ROOT / "outputs" / "tasks.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    upload_path TEXT NOT NULL,
                    file_sha256 TEXT,
                    result_path TEXT,
                    error TEXT,
                    model_version TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS human_review_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                )
                """
            )

    def create_task(self, analysis_id: str, upload_path: Path) -> None:
        now = datetime.now().astimezone().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses
                (analysis_id, status, stage, progress, upload_path, created_at, updated_at)
                VALUES (?, 'queued', '等待分析', 0, ?, ?, ?)
                """,
                (analysis_id, str(upload_path), now, now),
            )

    def update_task(
        self,
        analysis_id: str,
        *,
        status: str,
        stage: str,
        progress: float,
        error: str | None = None,
        file_sha256: str | None = None,
        result_path: Path | None = None,
        model_version: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE analyses
                SET status=?, stage=?, progress=?, error=COALESCE(?, error),
                    file_sha256=COALESCE(?, file_sha256),
                    result_path=COALESCE(?, result_path),
                    model_version=COALESCE(?, model_version),
                    updated_at=?
                WHERE analysis_id=?
                """,
                (
                    status,
                    stage,
                    progress,
                    error,
                    file_sha256,
                    str(result_path) if result_path else None,
                    model_version,
                    datetime.now().astimezone().isoformat(),
                    analysis_id,
                ),
            )

    def get_task(self, analysis_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT analysis_id, status, stage, progress, error, created_at, updated_at
                FROM analyses WHERE analysis_id=?
                """,
                (analysis_id,),
            ).fetchone()
        return dict(row) if row else None

    def result_path(self, analysis_id: str) -> Path:
        return ROOT / "outputs" / analysis_id / "result.json"

    def save_result(self, result: AnalysisResult) -> Path:
        output_dir = ROOT / "outputs" / result.analysis_id
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "result.json"
        temporary = output_dir / ".result.json.tmp"
        temporary.write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def load_result(self, analysis_id: str) -> AnalysisResult | None:
        path = self.result_path(analysis_id)
        if not path.is_file():
            return None
        return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))

    def append_overrides(
        self, analysis_id: str, overrides: list[HumanOverride]
    ) -> AnalysisResult | None:
        result = self.load_result(analysis_id)
        if result is None:
            return None
        # Human edits are audit events and never mutate instruments or raw model scores.
        result.human_overrides.extend(overrides)
        self.save_result(result)
        return result

    def save_review_session(self, session: HumanReviewSession) -> None:
        """Append an immutable field-level usability review session."""

        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO human_review_sessions
                (analysis_id, payload_json, completed_at)
                VALUES (?, ?, ?)
                """,
                (
                    session.analysis_id,
                    session.model_dump_json(),
                    session.completed_at.isoformat(),
                ),
            )

    def list_review_sessions(self) -> list[HumanReviewSession]:
        """Load all completed review sessions for usability reporting."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM human_review_sessions ORDER BY id"
            ).fetchall()
        return [
            HumanReviewSession.model_validate_json(str(row["payload_json"]))
            for row in rows
        ]


def export_csv(result: AnalysisResult) -> str:
    """Return a spreadsheet-friendly CSV without applying human overrides."""

    import csv
    import io

    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(
        ["canonical_label", "display_name_zh", "decision", "model_score", "time_ranges"]
    )
    for instrument in result.instruments:
        ranges = "; ".join(
            f"{item.start_seconds:.2f}-{item.end_seconds:.2f}" for item in instrument.time_ranges
        )
        writer.writerow(
            [
                instrument.canonical_label,
                instrument.display_name_zh,
                instrument.decision,
                f"{instrument.model_score:.6f}",
                ranges,
            ]
        )
    return handle.getvalue()
