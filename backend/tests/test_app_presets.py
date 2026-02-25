import os

from fastapi import HTTPException
import pytest

os.environ.setdefault("SPOTIFY_CLIENT_ID", "test-client")
os.environ.setdefault("SPOTIFY_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SPOTIFY_REDIRECT_URI", "http://localhost:8000/callback")

from backend.app import BUILTIN_PRESETS, OptimizeRequest, apply_builtin_preset


def test_apply_builtin_preset_populates_profile_defaults():
    payload = OptimizeRequest(playlist="abc123", preset_id="warmup")
    resolved = apply_builtin_preset(payload)
    expected = BUILTIN_PRESETS["warmup"]["config"]

    assert resolved.preset_id == "warmup"
    assert resolved.flow_curve == expected["flow_curve"]
    assert resolved.flow_profile == expected["flow_profile"]
    assert resolved.mix_mode == expected["mix_mode"]


def test_apply_builtin_preset_keeps_explicit_overrides():
    payload = OptimizeRequest(
        playlist="abc123",
        preset_id="peak_hour",
        flow_profile="cooldown",
        minimax_passes=1,
    )
    resolved = apply_builtin_preset(payload)

    assert resolved.flow_profile == "cooldown"
    assert resolved.minimax_passes == 1


def test_apply_builtin_preset_rejects_unknown_preset():
    payload = OptimizeRequest(playlist="abc123", preset_id="missing")
    with pytest.raises(HTTPException) as exc_info:
        apply_builtin_preset(payload)
    exc = exc_info.value
    assert exc.status_code == 400
