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
    FEEDBACK_STORE,
    MODEL_REGISTRY,
    MODEL_STATE,
    MODEL_DIR,
    RUN_HISTORY,
    app,
    SESSION_COOKIE,
    STORE,
    OptimizeRequest,
    apply_builtin_preset,
    build_optimize_config_hash,
    execute_transition_training,
    edge_score_diff,
    normalize_optimize_payload,
    run_optimize_tracks_for_payload,
    run_single_optimization,
    resolve_optimize_payload,
    transition_summary,
    validate_optimize_payload,
)
from backend.model_store import save_model_artifact
from backend.modeling import MODEL_FEATURE_KEYS, TransitionModel


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


def clear_model_state() -> None:
    for version, _ in list(MODEL_REGISTRY.items()):
        MODEL_REGISTRY.pop(version, None)
    MODEL_STATE.pop("active_version", None)
    MODEL_STATE.pop("last_auto_train_at", None)


def clear_feedback_data(prefix: str) -> None:
    for run_id, _ in list(RUN_HISTORY.items()):
        if run_id.startswith(prefix):
            RUN_HISTORY.pop(run_id, None)
    for feedback_id, _ in list(FEEDBACK_STORE.items()):
        if feedback_id.startswith(prefix):
            FEEDBACK_STORE.pop(feedback_id, None)


def test_background_services_start_and_stop_scheduler_thread(monkeypatch):
    def fake_scheduler_loop():
        while not app_module.SCHEDULER_STOP.is_set():
            app_module.SCHEDULER_STOP.wait(0.01)

    monkeypatch.setattr(app_module, "scheduler_loop", fake_scheduler_loop)
    app_module.stop_background_services(timeout_seconds=0.2)
    app_module.start_background_services()

    thread = getattr(app.state, "scheduler_thread", None)
    assert thread is not None
    assert thread.is_alive()

    app_module.stop_background_services(timeout_seconds=0.5)
    assert getattr(app.state, "scheduler_thread", None) is None
    assert app_module.SCHEDULER_STOP.is_set()


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


def test_normalize_optimize_payload_sorts_guardrails_and_curve_points():
    payload = OptimizeRequest(
        playlist="abc123",
        bpm_guardrails=[128, 110, 128, 0],
        mood_curve_points=[
            {"position": 0.8, "energy": 0.5},
            {"position": 0.2, "energy": 0.4},
        ],
    )
    normalized = normalize_optimize_payload(payload)

    assert normalized.bpm_guardrails == [110.0, 128.0]
    assert [point.position for point in (normalized.mood_curve_points or [])] == [0.2, 0.8]


def test_build_optimize_config_hash_is_stable_for_equivalent_payloads():
    left = OptimizeRequest(
        playlist="abc123",
        name="mix A",
        bpm_guardrails=[128, 110],
        mood_curve_points=[
            {"position": 0.8, "energy": 0.5},
            {"position": 0.2, "energy": 0.4},
        ],
    )
    right = OptimizeRequest(
        playlist="abc123",
        name="mix B",
        bpm_guardrails=[110, 128],
        mood_curve_points=[
            {"position": 0.2, "energy": 0.4},
            {"position": 0.8, "energy": 0.5},
        ],
    )

    assert build_optimize_config_hash(left) == build_optimize_config_hash(right)


def test_resolve_optimize_payload_applies_preset_and_normalization():
    payload = OptimizeRequest(
        playlist="abc123",
        preset_id="warmup",
        bpm_guardrails=[130, 115],
    )
    resolved = resolve_optimize_payload(payload)

    assert resolved.flow_curve is True
    assert resolved.flow_profile == "gentle"
    assert resolved.bpm_guardrails == [115.0, 130.0]


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


def test_run_optimize_tracks_for_payload_uses_default_solver_budget(monkeypatch):
    captured: dict = {}

    def fake_optimize_tracks(**kwargs):
        captured.update(kwargs)
        return "Playlist", [], 0.0, [], [], {"elapsed_ms": 1.0}

    monkeypatch.setattr(app_module, "DEFAULT_MAX_SOLVER_MS", 4200)
    monkeypatch.setattr(app_module, "optimize_tracks", fake_optimize_tracks)

    payload = OptimizeRequest(playlist="abc123")
    run_optimize_tracks_for_payload(
        sp=object(),
        payload=payload,
        seed=42,
        feedback_offsets={},
        model_payload={},
        cache_path="cache/audio_features.json",
    )

    assert captured.get("max_solver_ms") == 4200


def test_run_single_optimization_includes_solver_diagnostics_when_debug_enabled(monkeypatch):
    run_id = f"diag-{uuid.uuid4().hex[:8]}"
    solver_diag = {"anneal_runs": 2, "elapsed_ms": 12.3}

    monkeypatch.setattr(app_module, "OPTIMIZE_DIAGNOSTICS_DEBUG", True)
    monkeypatch.setattr(app_module, "owner_feedback_offsets", lambda owner_id: {})
    monkeypatch.setattr(app_module, "active_model_payload", lambda: {"version": None, "alpha": 0.0})
    monkeypatch.setattr(app_module, "fetch_playlist_track_ids", lambda sp, playlist_id: ["t1", "t2"])
    monkeypatch.setattr(app_module, "create_playlist_with_items", lambda **kwargs: "playlist123")
    monkeypatch.setattr(
        app_module,
        "run_optimize_tracks_for_payload",
        lambda **kwargs: ("abc123", "My Playlist", [], 0.123, [], [], solver_diag),
    )

    response = run_single_optimization(
        sp=object(),
        owner_id="tester",
        payload=OptimizeRequest(playlist="abc123"),
        seed=42,
        run_id=run_id,
    )

    assert response["solver_diagnostics"] == solver_diag
    assert RUN_HISTORY[run_id]["solver_diagnostics"] == solver_diag

    snapshot_id = RUN_HISTORY[run_id].get("snapshot_id")
    RUN_HISTORY.pop(run_id, None)
    if snapshot_id:
        app_module.SNAPSHOT_STORE.pop(snapshot_id, None)


def test_run_single_optimization_hides_solver_diagnostics_when_debug_disabled(monkeypatch):
    run_id = f"diag-{uuid.uuid4().hex[:8]}"
    solver_diag = {"anneal_runs": 2, "elapsed_ms": 12.3}

    monkeypatch.setattr(app_module, "OPTIMIZE_DIAGNOSTICS_DEBUG", False)
    monkeypatch.setattr(app_module, "owner_feedback_offsets", lambda owner_id: {})
    monkeypatch.setattr(app_module, "active_model_payload", lambda: {"version": None, "alpha": 0.0})
    monkeypatch.setattr(app_module, "fetch_playlist_track_ids", lambda sp, playlist_id: ["t1", "t2"])
    monkeypatch.setattr(app_module, "create_playlist_with_items", lambda **kwargs: "playlist123")
    monkeypatch.setattr(
        app_module,
        "run_optimize_tracks_for_payload",
        lambda **kwargs: ("abc123", "My Playlist", [], 0.123, [], [], solver_diag),
    )

    response = run_single_optimization(
        sp=object(),
        owner_id="tester",
        payload=OptimizeRequest(playlist="abc123"),
        seed=42,
        run_id=run_id,
    )

    assert "solver_diagnostics" not in response
    assert RUN_HISTORY[run_id]["solver_diagnostics"] == solver_diag

    snapshot_id = RUN_HISTORY[run_id].get("snapshot_id")
    RUN_HISTORY.pop(run_id, None)
    if snapshot_id:
        app_module.SNAPSHOT_STORE.pop(snapshot_id, None)


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
    assert "quality_gate_thresholds" in payload
    assert "promotion_thresholds" in payload


def test_model_evaluation_endpoint_requires_admin_role(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = make_authenticated_client("normal-user")
    response = client.get("/model/evaluation")
    assert response.status_code == 403


def test_model_evaluation_endpoint_rejects_invalid_days(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    client = make_authenticated_client("admin-user")
    response = client.get("/model/evaluation?days=0")
    assert response.status_code == 400


def test_model_evaluation_endpoint_returns_version_breakdown(monkeypatch):
    clear_model_state()
    prefix = f"eval-{uuid.uuid4().hex[:8]}"
    clear_feedback_data(prefix)
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})

    active_version = f"transition_eval_{uuid.uuid4().hex[:8]}"
    MODEL_STATE["active_version"] = active_version
    MODEL_REGISTRY[active_version] = {
        "artifact_path": "",
        "owner_scope": "all",
        "metrics": {"accuracy": 0.7, "loss": 0.4},
        "validation_metrics": {"accuracy": 0.74, "loss": 0.35},
        "sample_count": 90,
        "created_at": time.time(),
    }

    run_id_active = f"{prefix}-run-active"
    RUN_HISTORY[run_id_active] = {
        "model_version": active_version,
        "transitions": [
            {"reason_code": "tempo_mismatch"},
            {"reason_code": "harmonic_mismatch"},
        ],
        "created_at": time.time(),
    }
    FEEDBACK_STORE[f"{prefix}-fb-1"] = {
        "run_id": run_id_active,
        "edge_index": 0,
        "rating": -2,
        "owner_id": "tester",
        "created_at": time.time(),
    }
    FEEDBACK_STORE[f"{prefix}-fb-2"] = {
        "run_id": run_id_active,
        "edge_index": 1,
        "rating": 2,
        "owner_id": "tester",
        "created_at": time.time(),
    }
    run_id_unknown_time = f"{prefix}-run-unknown-time"
    RUN_HISTORY[run_id_unknown_time] = {
        "model_version": active_version,
        "transitions": [{"reason_code": "tempo_mismatch"}],
    }
    FEEDBACK_STORE[f"{prefix}-fb-unknown-time"] = {
        "run_id": run_id_unknown_time,
        "edge_index": 0,
        "rating": -1,
        "owner_id": "tester",
    }

    run_id_heuristic = f"{prefix}-run-heuristic"
    RUN_HISTORY[run_id_heuristic] = {
        "model_version": None,
        "transitions": [{"reason_code": "energy_shift"}],
        "created_at": time.time(),
    }
    FEEDBACK_STORE[f"{prefix}-fb-3"] = {
        "run_id": run_id_heuristic,
        "edge_index": 0,
        "rating": -1,
        "owner_id": "tester",
        "created_at": time.time(),
    }

    client = make_authenticated_client("admin-user")
    response = client.get("/model/evaluation?days=30")
    assert response.status_code == 200
    payload = response.json()
    assert payload["active_version"] == active_version

    versions_by_id = {item["version"]: item for item in payload["versions"]}
    assert versions_by_id[active_version]["sample_count"] == 2
    assert versions_by_id[active_version]["rough_rate"] == 0.5
    assert versions_by_id["heuristic"]["sample_count"] == 1
    assert payload["active_metrics"]["version"] == active_version

    clear_feedback_data(prefix)
    clear_model_state()


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


def test_execute_transition_training_blocks_activation_when_quality_gate_fails(monkeypatch):
    clear_model_state()
    monkeypatch.setattr(app_module, "MODEL_MIN_ACCURACY", 0.9)
    monkeypatch.setattr(app_module, "MODEL_MAX_LOSS", 0.2)
    version = f"transition_gate_{uuid.uuid4().hex[:8]}"

    def fake_train(*args, **kwargs):
        model = TransitionModel(
            version=version,
            weights={key: 0.01 for key in MODEL_FEATURE_KEYS},
            bias=0.0,
            trained_at=time.time(),
            sample_count=64,
        )
        details = {
            "trained": True,
            "sample_count": 64,
            "metrics": {"accuracy": 0.78, "loss": 0.44},
            "validation_metrics": {"accuracy": 0.48, "loss": 0.83},
            "positive_ratio": 0.4,
            "version": version,
        }
        return model, details

    monkeypatch.setattr(app_module, "train_transition_model_from_feedback", fake_train)

    result = execute_transition_training(owner_id=None, min_samples=10, activate=True)

    assert result["trained"] is True
    assert result["activated"] is False
    assert result["quality_gate"]["passed"] is False
    assert MODEL_STATE.get("active_version") != version
    clear_model_state()


def test_model_activate_endpoint_rejects_versions_that_fail_quality_gate(monkeypatch):
    clear_model_state()
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    monkeypatch.setattr(app_module, "MODEL_MIN_ACCURACY", 0.8)
    monkeypatch.setattr(app_module, "MODEL_MAX_LOSS", 0.3)

    version = f"transition_gate_{uuid.uuid4().hex[:8]}"
    artifact_path = save_model_artifact(
        MODEL_DIR,
        TransitionModel(
            version=version,
            weights={key: 0.01 for key in MODEL_FEATURE_KEYS},
            bias=0.0,
            trained_at=time.time(),
            sample_count=80,
        ).to_dict(),
    )
    MODEL_REGISTRY[version] = {
        "artifact_path": artifact_path,
        "owner_scope": "all",
        "metrics": {"accuracy": 0.6, "loss": 0.45},
        "validation_metrics": {"accuracy": 0.52, "loss": 0.56},
        "sample_count": 80,
        "created_at": time.time(),
    }

    client = make_authenticated_client("admin-user")
    response = client.post(f"/model/activate/{version}")
    assert response.status_code == 409
    assert "quality gate failed" in response.json().get("detail", "")

    clear_model_state()


def test_execute_transition_training_blocks_activation_when_promotion_gate_fails(monkeypatch):
    clear_model_state()
    monkeypatch.setattr(app_module, "MODEL_MIN_ACCURACY", 0.5)
    monkeypatch.setattr(app_module, "MODEL_MAX_LOSS", 0.9)
    monkeypatch.setattr(app_module, "MODEL_MIN_ACCURACY_DELTA", 0.02)
    monkeypatch.setattr(app_module, "MODEL_MAX_LOSS_DELTA", 0.0)

    active_version = f"transition_active_{uuid.uuid4().hex[:8]}"
    MODEL_STATE["active_version"] = active_version
    MODEL_REGISTRY[active_version] = {
        "artifact_path": "",
        "owner_scope": "all",
        "metrics": {"accuracy": 0.72, "loss": 0.36},
        "validation_metrics": {"accuracy": 0.78, "loss": 0.31},
        "sample_count": 120,
        "created_at": time.time(),
    }

    candidate_version = f"transition_candidate_{uuid.uuid4().hex[:8]}"

    def fake_train(*args, **kwargs):
        model = TransitionModel(
            version=candidate_version,
            weights={key: 0.01 for key in MODEL_FEATURE_KEYS},
            bias=0.0,
            trained_at=time.time(),
            sample_count=64,
        )
        details = {
            "trained": True,
            "sample_count": 64,
            "metrics": {"accuracy": 0.75, "loss": 0.35},
            "validation_metrics": {"accuracy": 0.77, "loss": 0.33},
            "positive_ratio": 0.4,
            "version": candidate_version,
        }
        return model, details

    monkeypatch.setattr(app_module, "train_transition_model_from_feedback", fake_train)

    result = execute_transition_training(owner_id=None, min_samples=10, activate=True)

    assert result["trained"] is True
    assert result["quality_gate"]["passed"] is True
    assert result["promotion_gate"]["passed"] is False
    assert result["activated"] is False
    assert MODEL_STATE.get("active_version") == active_version
    clear_model_state()


def test_model_activate_endpoint_rejects_versions_that_fail_promotion_gate(monkeypatch):
    clear_model_state()
    monkeypatch.setattr(app_module, "MODEL_ADMIN_USER_IDS", {"admin-user"})
    monkeypatch.setattr(app_module, "MODEL_MIN_ACCURACY", 0.5)
    monkeypatch.setattr(app_module, "MODEL_MAX_LOSS", 0.9)
    monkeypatch.setattr(app_module, "MODEL_MIN_ACCURACY_DELTA", 0.01)
    monkeypatch.setattr(app_module, "MODEL_MAX_LOSS_DELTA", 0.0)

    active_version = f"transition_active_{uuid.uuid4().hex[:8]}"
    active_artifact_path = save_model_artifact(
        MODEL_DIR,
        TransitionModel(
            version=active_version,
            weights={key: 0.01 for key in MODEL_FEATURE_KEYS},
            bias=0.0,
            trained_at=time.time(),
            sample_count=80,
        ).to_dict(),
    )
    MODEL_STATE["active_version"] = active_version
    MODEL_REGISTRY[active_version] = {
        "artifact_path": active_artifact_path,
        "owner_scope": "all",
        "metrics": {"accuracy": 0.71, "loss": 0.36},
        "validation_metrics": {"accuracy": 0.79, "loss": 0.31},
        "sample_count": 80,
        "created_at": time.time(),
    }

    candidate_version = f"transition_candidate_{uuid.uuid4().hex[:8]}"
    candidate_artifact_path = save_model_artifact(
        MODEL_DIR,
        TransitionModel(
            version=candidate_version,
            weights={key: 0.01 for key in MODEL_FEATURE_KEYS},
            bias=0.0,
            trained_at=time.time(),
            sample_count=80,
        ).to_dict(),
    )
    MODEL_REGISTRY[candidate_version] = {
        "artifact_path": candidate_artifact_path,
        "owner_scope": "all",
        "metrics": {"accuracy": 0.68, "loss": 0.37},
        "validation_metrics": {"accuracy": 0.77, "loss": 0.33},
        "sample_count": 80,
        "created_at": time.time(),
    }

    client = make_authenticated_client("admin-user")
    response = client.post(f"/model/activate/{candidate_version}")
    assert response.status_code == 409
    assert "promotion gate failed" in response.json().get("detail", "")
    assert MODEL_STATE.get("active_version") == active_version

    clear_model_state()
