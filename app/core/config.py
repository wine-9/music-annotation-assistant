"""Centralized configuration loading."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import TypeVar, cast

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RUNTIME_SETTINGS_PATH = ROOT / "outputs" / "runtime_settings.json"
THRESHOLD_PROFILES = ("default", "calibrated_public_v1")
T = TypeVar("T", bound=BaseModel)
_profile_lock = RLock()


class Settings(BaseSettings):
    """Runtime settings, optionally overridden with MLA_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="MLA_")

    host: str = "127.0.0.1"
    port: int = 8000
    max_duration_seconds: float = 600.0
    window_seconds: float = 10.0
    hop_seconds: float = 5.0
    enable_separation: bool = True
    model_sample_rate: int = 16_000
    max_upload_bytes: int = 1_000_000_000
    threshold_profile: str = "calibrated_public_v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide validated settings."""

    return Settings()


@lru_cache(maxsize=16)
def load_yaml(name: str) -> dict[str, object]:
    """Load a mapping from the configuration directory."""

    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {name}")
    return cast(dict[str, object], payload)


def load_threshold_profile(name: str) -> dict[str, object]:
    """Load one allow-listed threshold profile without modifying the source config."""

    if name not in THRESHOLD_PROFILES:
        raise ValueError(f"Unknown threshold profile: {name}")
    if name == "default":
        payload = dict(load_yaml("thresholds.yaml"))
        payload.update(
            {
                "profile_name": "default",
                "generated_at": "project-default",
                "applicable_datasets": [],
            }
        )
        return payload
    return dict(load_yaml(f"threshold_profiles/{name}.yaml"))


def get_active_threshold_profile_name() -> str:
    """Return the persisted active profile, falling back to the safe configured default."""

    with _profile_lock:
        if RUNTIME_SETTINGS_PATH.is_file():
            try:
                payload = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
                name = str(payload.get("threshold_profile", ""))
                if name in THRESHOLD_PROFILES:
                    return name
            except (json.JSONDecodeError, OSError):
                pass
        configured = get_settings().threshold_profile
        return configured if configured in THRESHOLD_PROFILES else "calibrated_public_v1"


def set_active_threshold_profile_name(name: str) -> dict[str, object]:
    """Persist an allow-listed profile selection in outputs for safe rollback."""

    profile = load_threshold_profile(name)
    with _profile_lock:
        RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = RUNTIME_SETTINGS_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"threshold_profile": name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(RUNTIME_SETTINGS_PATH)
    return profile


def threshold_profile_metadata(name: str | None = None) -> dict[str, object]:
    """Return UI-safe metadata for the selected threshold profile."""

    selected = name or get_active_threshold_profile_name()
    payload = load_threshold_profile(selected)
    return {
        "name": selected,
        "version": str(payload["version"]),
        "generated_at": str(payload.get("generated_at", "unknown")),
        "applicable_datasets": [
            str(item)
            for item in cast(list[object], payload.get("applicable_datasets", []))
        ],
    }
