import os
import time
import uuid

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

os.environ.setdefault("SPOTIFY_CLIENT_ID", "test-client")
os.environ.setdefault("SPOTIFY_CLIENT_SECRET", "test-secret")
os.environ.setdefault("SPOTIFY_REDIRECT_URI", "http://localhost:8000/callback")

import backend.app as app_module
from backend.app import (
    BUILTIN_PRESETS,
    app,
    SESSION_COOKIE,
    STORE,
    OptimizeRequest,
    apply_builtin_preset,
    edge_score_diff,
    transition_summary,
    validate_optimize_payload,
)


def make_authenticated_client(user_id: str = "tester") -> TestClient:
    client = TestClient(app)
    sid = f"sid-{uuid.uuid4().hex}"
    STORE.set_session(
        sid,
        {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_at": time.time() + 3600,
            "user_id": user_id,
        },
    )
    client.cookies.set(SESSION_COOKIE, sid)
    return client


def wait_for_training_job(client: TestClient, job_id: str, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    last_payload: dict = {}
    while time.time() < deadline:
        response = client.get(f"/model/train/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        last_payload = payload
        if payload.get("status") in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    return last_payload


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


def test_edge_score_diff_reports_improvements_and_regressions():
    baseline = [
        {"score": 0.7, "from_track": "A", "to_track": "B", "reason_code": "tempo_mismatch"},
        {"score": 0.3, "from_track": "B", "to_track": "C", "reason_code": "energy_shift"},
    ]
    candidate = [
        {"score": 0.4, "from_track": "A", "to_track": "B", "reason_code": "tempo_mismatch"},
        {"score": 0.5, "from_track": "B", "to_track": "C", "reason_code": "energy_shift"},
    ]
    payload = edge_score_diff(baseline, candidate, max_edges=3)
    assert payload["most_improved"][0]["score_delta"] < 0
    assert payload["most_regressed"][0]["score_delta"] > 0


def test_validate_optimize_payload_rejects_invalid_anneal_temps():
    payload = OptimizeRequest(
        playlist="abc123",
        solver_mode="hybrid",
        anneal_temp_start=0.01,
        anneal_temp_end=0.02,
    )
    with pytest.raises(HTTPException):
        validate_optimize_payload(payload)


def test_model_status_endpoint_requires_authentication(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = TestClient(app)
    response = client.get("/model/status")
    assert response.status_code == 401


def test_model_status_endpoint_requires_admin_role(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = make_authenticated_client("normal-user")
    response = client.get("/model/status")
    assert response.status_code == 403


def test_model_status_endpoint_returns_expected_shape_for_admin(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = make_authenticated_client("admin-user")
    response = client.get("/model/status")
    assert response.status_code == 200
    payload = response.json()
    assert "active_version" in payload
    assert "available_versions" in payload


def test_model_train_endpoint_returns_graceful_response_when_data_is_small(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = make_authenticated_client("admin-user")
    response = client.post("/model/train", json={"owner_scope": "all", "activate": True, "min_samples": 5000})
    assert response.status_code == 200
    queued = response.json()
    assert queued["status"] in {"queued", "running"}
    assert "job_id" in queued

    final_payload = wait_for_training_job(client, queued["job_id"])
    assert final_payload.get("status") == "completed"
    result = final_payload.get("result") or {}
    assert result.get("trained") is False


def test_model_train_endpoint_rejects_non_admin(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = make_authenticated_client("normal-user")
    response = client.post("/model/train", json={"owner_scope": "all", "activate": True, "min_samples": 5000})
    assert response.status_code == 403


def test_model_train_status_rejects_non_admin(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    admin_client = make_authenticated_client("admin-user")
    normal_client = make_authenticated_client("normal-user")
    response = admin_client.post("/model/train", json={"owner_scope": "all", "activate": True, "min_samples": 5000})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    forbidden = normal_client.get(f"/model/train/{job_id}")
    assert forbidden.status_code == 403
