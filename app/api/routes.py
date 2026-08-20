"""FastAPI routes for local analysis jobs and review."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.core.config import (
    ROOT,
    THRESHOLD_PROFILES,
    get_active_threshold_profile_name,
    get_settings,
    set_active_threshold_profile_name,
    threshold_profile_metadata,
)
from app.core.exceptions import MusicLabelAssistantError
from app.core.pipeline import analyze_file
from app.providers.demucs_provider import DemucsSeparatorProvider
from app.providers.essentia_provider import EssentiaInstrumentProvider
from app.schemas.results import AnalysisResult, HumanOverride, HumanReviewSession
from app.storage.repository import Repository, export_csv

router = APIRouter()
repository = Repository()
provider = EssentiaInstrumentProvider()



def _run_analysis(
    task_id: str,
    upload_path: Path,
    threshold_profile: str,
    enable_separation: bool,
) -> None:
    repository.update_task(
        task_id, status="running", stage="解码与模型分析", progress=0.1
    )
    try:
        result = analyze_file(
            upload_path,
            provider,
            threshold_profile=threshold_profile,
            separator=DemucsSeparatorProvider() if enable_separation else None,
        )
        # The stable SHA-derived result ID replaces the temporary task ID in the payload.
        result.analysis_id = task_id
        result_path = repository.save_result(result)
        repository.update_task(
            task_id,
            status="completed",
            stage="完成",
            progress=1.0,
            file_sha256=result.source.sha256,
            result_path=result_path,
            model_version=result.pipeline.instrument_model_version,
        )
    except MusicLabelAssistantError as exc:
        repository.update_task(
            task_id,
            status="failed",
            stage="分析失败",
            progress=1.0,
            error=str(exc),
        )
    except Exception as exc:
        repository.update_task(
            task_id,
            status="failed",
            stage="内部错误",
            progress=1.0,
            error=f"{type(exc).__name__}: {exc}",
        )


@router.post("/api/v1/analyses", status_code=202)
def create_analysis(
    background_tasks: BackgroundTasks,
    audio: Annotated[UploadFile, File()],
    enable_separation: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Save an upload locally and queue analysis in the same process."""

    profile_name = get_active_threshold_profile_name()
    if enable_separation and not get_settings().enable_separation:
        raise HTTPException(status_code=400, detail="四轨分离功能未启用")
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}:
        raise HTTPException(status_code=415, detail="不支持的音频格式")
    upload_dir = ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = upload_dir / f".{uuid.uuid4().hex}.upload"
    copied = 0
    digest = hashlib.sha256()
    with temporary_path.open("wb") as destination:
        while chunk := audio.file.read(1024 * 1024):
            copied += len(chunk)
            if copied > get_settings().max_upload_bytes:
                destination.close()
                temporary_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="文件超过本地上传限制")
            digest.update(chunk)
            destination.write(chunk)
    source_sha256 = digest.hexdigest()
    task_key = f"{source_sha256}:{profile_name}:{int(enable_separation)}"
    task_id = hashlib.sha256(task_key.encode()).hexdigest()[:16]
    existing = repository.get_task(task_id)
    if existing is not None:
        temporary_path.unlink(missing_ok=True)
        return {"analysis_id": task_id, "status": existing["status"], "cached": True}
    upload_path = upload_dir / f"{task_id}{suffix}"
    temporary_path.replace(upload_path)
    repository.create_task(task_id, upload_path)
    background_tasks.add_task(
        _run_analysis,
        task_id,
        upload_path,
        profile_name,
        enable_separation,
    )
    return {"analysis_id": task_id, "status": "queued", "cached": False}



@router.get("/api/v1/threshold-profile")
def get_threshold_profile() -> dict[str, object]:
    """Return active profile metadata and the allow-listed rollback targets."""

    return {
        "active": threshold_profile_metadata(),
        "available": [
            threshold_profile_metadata(name) for name in THRESHOLD_PROFILES
        ],
    }


@router.put("/api/v1/threshold-profile/{profile_name}")
def set_threshold_profile(profile_name: str) -> dict[str, object]:
    """Switch only to a reviewed, local allow-listed threshold profile."""

    if profile_name not in THRESHOLD_PROFILES:
        raise HTTPException(status_code=400, detail="不允许的阈值配置")
    set_active_threshold_profile_name(profile_name)
    return {
        "active": threshold_profile_metadata(profile_name),
        "rollback_available": profile_name != "default",
    }


@router.get("/api/v1/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, object]:
    task = repository.get_task(analysis_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.get("/api/v1/analyses/{analysis_id}/result", response_model=AnalysisResult)
def get_result(analysis_id: str) -> AnalysisResult:
    result = repository.load_result(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="结果尚未生成")
    return result


@router.patch("/api/v1/analyses/{analysis_id}/result", response_model=AnalysisResult)
def patch_result(
    analysis_id: str, overrides: list[HumanOverride]
) -> AnalysisResult:
    result = repository.append_overrides(analysis_id, overrides)
    if result is None:
        raise HTTPException(status_code=404, detail="结果不存在")
    return result


@router.post("/api/v1/analyses/{analysis_id}/review")
def save_review(
    analysis_id: str, session: HumanReviewSession
) -> dict[str, object]:
    """Persist one complete, real human-review session."""

    if session.analysis_id != analysis_id:
        raise HTTPException(status_code=400, detail="复核任务 ID 不一致")
    if repository.load_result(analysis_id) is None:
        raise HTTPException(status_code=404, detail="结果不存在")
    repository.save_review_session(session)
    return {"saved": True, "event_count": len(session.events)}


@router.get("/api/v1/analyses/{analysis_id}/result.csv", response_class=PlainTextResponse)
def get_result_csv(analysis_id: str) -> PlainTextResponse:
    result = repository.load_result(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="结果不存在")
    return PlainTextResponse(
        export_csv(result),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{analysis_id}.csv"'},
    )
