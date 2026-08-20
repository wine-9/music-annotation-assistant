"""Application entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api.routes import router
from app.core.config import ROOT, get_settings
from app.providers.essentia_provider import (
    EMBEDDING_FILE,
    HEAD_FILE,
    HEAD_METADATA_FILE,
)

app = FastAPI(
    title="音乐音频智能预标注助手",
    version=__version__,
    description="所有音频均在本机处理。",
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "ui" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "app" / "ui" / "templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")



@app.get("/health")
def health() -> dict[str, object]:
    model_dir = ROOT / "models"
    model_files = [EMBEDDING_FILE, HEAD_FILE, HEAD_METADATA_FILE]
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = Path(get_ffmpeg_exe()).is_file()
    except (ImportError, RuntimeError):
        ffmpeg = False
    try:
        import essentia

        model_runtime: str | None = str(essentia.__version__)
    except (ImportError, OSError):
        model_runtime = None
    return {
        "status": "ok",
        "local_only": True,
        "ffmpeg": ffmpeg,
        "instrument_runtime": model_runtime,
        "models": {name: (model_dir / name).is_file() for name in model_files},
        "separation_enabled": get_settings().enable_separation,
        "separation_available": importlib.util.find_spec("demucs") is not None,
    }


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
