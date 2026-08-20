from __future__ import annotations

from pathlib import Path

from app.core import config


def test_profile_switch_is_allow_listed_and_does_not_touch_default(
    tmp_path: Path, monkeypatch: object
) -> None:
    source = config.CONFIG_DIR / "thresholds.yaml"
    before = source.read_bytes()
    # pytest's monkeypatch fixture is intentionally used through its runtime API.
    monkeypatch.setattr(  # type: ignore[attr-defined]
        config, "RUNTIME_SETTINGS_PATH", tmp_path / "runtime_settings.json"
    )
    config.set_active_threshold_profile_name("default")
    assert config.get_active_threshold_profile_name() == "default"
    config.set_active_threshold_profile_name("calibrated_public_v1")
    assert config.get_active_threshold_profile_name() == "calibrated_public_v1"
    assert source.read_bytes() == before
