import os

from fastapi import HTTPException
import pytest

os.environ.setdefault("SPOTIFY_CLIENT_ID", "test-client")
os.environ.setdefault("SPOTIFY_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SPOTIFY_REDIRECT_URI", "http://localhost:8000/callback")

from backend.app import (
    BUILTIN_PRESETS,
    OptimizeRequest,
    apply_builtin_preset,
    transition_summary,
)


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


def test_transition_summary_counts_reason_codes_and_worst_edges():
    transitions = [
        {"reason_code": "tempo_mismatch", "score": 0.6, "from_track": "A", "to_track": "B"},
        {"reason_code": "tempo_mismatch", "score": 0.5, "from_track": "C", "to_track": "D"},
        {"reason_code": "harmonic_mismatch", "score": 0.9, "from_track": "E", "to_track": "F"},
    ]
    summary = transition_summary(transitions)
    assert summary["dominant_penalties"][0]["reason_code"] == "tempo_mismatch"
    assert summary["worst_edges"][0]["score"] == 0.9
