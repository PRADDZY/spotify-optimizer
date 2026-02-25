import json
import hashlib
import logging
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from redis import Redis
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .optimizer_core import (
    enrich_audio_features,
    fetch_playlist_tracks,
    optimized_name,
    optimize_tracks,
    parse_playlist_id,
    spotify_call_with_retry,
)
from .model_store import load_model_artifact, save_model_artifact
from .modeling import transition_model_from_dict
from .state_store import DurableDict, DurableEventBuffer, SQLiteStateStore
from .training import train_transition_model_from_feedback

load_dotenv()

SESSION_COOKIE = "spotify_opt_sid"
ENV = os.getenv("ENV", "development").lower()
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))
STATE_TTL_SECONDS = int(os.getenv("STATE_TTL_SECONDS", "600"))
COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE")
if COOKIE_SECURE is None:
    COOKIE_SECURE = "true" if ENV == "production" else "false"
COOKIE_SECURE = COOKIE_SECURE.lower() == "true"
COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax").lower()
COOKIE_DOMAIN = os.getenv("SESSION_COOKIE_DOMAIN")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()
RATE_LIMIT_LOGIN = os.getenv("RATE_LIMIT_LOGIN", "10/minute")
RATE_LIMIT_OPTIMIZE = os.getenv("RATE_LIMIT_OPTIMIZE", "5/minute")
RATE_LIMIT_GLOBAL = os.getenv("RATE_LIMIT_GLOBAL", "60/minute")
RATE_LIMIT_MODEL_STATUS = os.getenv("RATE_LIMIT_MODEL_STATUS", "20/minute")
RATE_LIMIT_MODEL_WRITE = os.getenv("RATE_LIMIT_MODEL_WRITE", "3/minute")
TRANSITION_LOG_PATH = os.getenv("TRANSITION_LOG_PATH")
STATE_RETENTION_DAYS = int(os.getenv("STATE_RETENTION_DAYS", "30"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
SECURITY_HEADERS_ENABLED = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_BLEND_ALPHA = float(os.getenv("MODEL_BLEND_ALPHA", "0.2"))
MODEL_MIN_SAMPLES = int(os.getenv("MODEL_MIN_SAMPLES", "20"))
MODEL_RETRAIN_INTERVAL_MINUTES = int(os.getenv("MODEL_RETRAIN_INTERVAL_MINUTES", "240"))
MODEL_MIN_ACCURACY = float(os.getenv("MODEL_MIN_ACCURACY", "0.55"))
MODEL_MAX_LOSS = float(os.getenv("MODEL_MAX_LOSS", "0.72"))
MODEL_MIN_ACCURACY_DELTA = float(os.getenv("MODEL_MIN_ACCURACY_DELTA", "0.0"))
MODEL_MAX_LOSS_DELTA = float(os.getenv("MODEL_MAX_LOSS_DELTA", "0.0"))
MODEL_EVAL_WINDOW_DAYS = int(os.getenv("MODEL_EVAL_WINDOW_DAYS", "30"))
OPTIMIZE_CONFIG_HASH_DEBUG = os.getenv("OPTIMIZE_CONFIG_HASH_DEBUG", "false").lower() == "true"
MODEL_ADMIN_USER_IDS = {
    value.strip()
    for value in os.getenv("MODEL_ADMIN_USER_IDS", "").split(",")
    if value.strip()
}


class WeightConfig(BaseModel):
    bpm: float = Field(0.32, ge=0.0, le=2.0)
    key: float = Field(0.28, ge=0.0, le=2.0)
    energy: float = Field(0.12, ge=0.0, le=2.0)
    valence: float = Field(0.06, ge=0.0, le=2.0)
    dance: float = Field(0.06, ge=0.0, le=2.0)
    loudness: float = Field(0.06, ge=0.0, le=2.0)
    acousticness: float = Field(0.03, ge=0.0, le=2.0)
    instrumentalness: float = Field(0.02, ge=0.0, le=2.0)
    speechiness: float = Field(0.02, ge=0.0, le=2.0)
    liveness: float = Field(0.01, ge=0.0, le=2.0)
    time_signature: float = Field(0.02, ge=0.0, le=2.0)


class MoodPoint(BaseModel):
    position: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)


class OptimizeRequest(BaseModel):
    playlist: str
    name: Optional[str] = None
    public: bool = False
    preset_id: Optional[str] = None
    mix_mode: str = "balanced"
    flow_curve: bool = False
    flow_profile: str = "peak"
    key_lock_window: int = Field(3, ge=1, le=12)
    tempo_ramp_weight: float = Field(0.08, ge=0.0, le=1.0)
    minimax_passes: int = Field(2, ge=0, le=10)
    locked_first_track_id: Optional[str] = None
    locked_last_track_id: Optional[str] = None
    locked_blocks: Optional[list[list[str]]] = None
    artist_gap: int = Field(0, ge=0, le=20)
    album_gap: int = Field(0, ge=0, le=20)
    explicit_mode: str = "allow"
    duration_target_sec: Optional[int] = Field(None, ge=60, le=43200)
    duration_tolerance_sec: int = Field(90, ge=0, le=1800)
    genre_cluster_strength: float = Field(0.0, ge=0.0, le=1.0)
    mood_curve_points: Optional[list[MoodPoint]] = None
    bpm_guardrails: Optional[list[float]] = None
    harmonic_strict: bool = False
    smoothness_weight: float = Field(1.0, ge=0.0, le=5.0)
    variety_weight: float = Field(0.0, ge=0.0, le=5.0)
    max_bpm_jump: Optional[float] = Field(None, ge=0.0, le=240.0)
    min_key_compatibility: Optional[float] = Field(None, ge=0.0, le=1.0)
    no_repeat_artist_within: int = Field(0, ge=0, le=20)
    solver_mode: str = "hybrid"
    beam_width: int = Field(8, ge=1, le=24)
    anneal_steps: int = Field(140, ge=0, le=1500)
    anneal_temp_start: float = Field(0.08, ge=0.0001, le=2.0)
    anneal_temp_end: float = Field(0.004, ge=0.0001, le=2.0)
    max_solver_ms: Optional[int] = Field(None, ge=50, le=120000)
    lookahead_horizon: int = Field(3, ge=1, le=8)
    lookahead_decay: float = Field(0.6, ge=0.05, le=0.99)
    bpm_window: float = Field(0.08, ge=0.0, le=0.5)
    restarts: int = Field(12, ge=1, le=100)
    two_opt_passes: int = Field(2, ge=1, le=10)
    missing: str = "append"
    weights: Optional[WeightConfig] = None


class OptimizeResponse(BaseModel):
    run_id: str
    playlist_id: str
    playlist_name: str
    playlist_url: str
    transition_score: float
    roughest: list
    model_version: Optional[str] = None
    solver_diagnostics: Optional[dict] = None


class QuickFixRequest(BaseModel):
    minimax_boost: int = Field(2, ge=1, le=10)
    public: Optional[bool] = None


class CompareRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    include_edge_diff: bool = True
    max_edges: int = Field(10, ge=1, le=50)


class PresetRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    config: dict
    schema_version: int = 1


class PresetPatchRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    config: Optional[dict] = None
    schema_version: Optional[int] = None


class SnapshotCreateRequest(BaseModel):
    source_playlist_id: str
    source_playlist_name: str
    source_track_ids: list[str]
    optimized_track_ids: Optional[list[str]] = None
    note: Optional[str] = None


class RollbackRequest(BaseModel):
    public: bool = False
    name: Optional[str] = None


class BatchRequest(BaseModel):
    playlists: list[str]
    name_prefix: Optional[str] = None
    public: bool = False
    options: Optional[dict] = None


class ScheduleRequest(BaseModel):
    cron: str
    batch: BatchRequest
    enabled: bool = True


class SchedulePatchRequest(BaseModel):
    cron: Optional[str] = None
    enabled: Optional[bool] = None
    batch: Optional[BatchRequest] = None


class AnchorSuggestRequest(BaseModel):
    playlist: str
    count: int = Field(3, ge=1, le=10)


class ManualFeedbackRequest(BaseModel):
    run_id: str
    edge_index: int = Field(..., ge=0)
    rating: int = Field(..., ge=-2, le=2)
    feature: Optional[str] = None
    note: Optional[str] = None


class ModelTrainRequest(BaseModel):
    owner_scope: str = "all"
    min_samples: Optional[int] = Field(None, ge=5, le=5000)
    activate: bool = True


BUILTIN_PRESETS: dict[str, dict] = {
    "warmup": {
        "name": "Warmup",
        "description": "Gentle energy ramp with harmonic-safe transitions.",
        "config": {
            "mix_mode": "balanced",
            "flow_curve": True,
            "flow_profile": "gentle",
            "key_lock_window": 4,
            "tempo_ramp_weight": 0.1,
            "minimax_passes": 2,
            "smoothness_weight": 1.15,
            "variety_weight": 0.2,
        },
    },
    "peak_hour": {
        "name": "Peak Hour",
        "description": "Maximizes momentum and punch while keeping cuts controlled.",
        "config": {
            "mix_mode": "harmonic",
            "flow_curve": True,
            "flow_profile": "peak",
            "key_lock_window": 5,
            "tempo_ramp_weight": 0.12,
            "minimax_passes": 3,
            "smoothness_weight": 1.3,
            "variety_weight": 0.3,
        },
    },
    "cooldown_set": {
        "name": "Cooldown",
        "description": "Gradually lowers intensity and preserves vibe continuity.",
        "config": {
            "mix_mode": "vibe",
            "flow_curve": True,
            "flow_profile": "cooldown",
            "tempo_ramp_weight": 0.08,
            "minimax_passes": 2,
            "smoothness_weight": 1.05,
            "variety_weight": 0.25,
        },
    },
    "workout": {
        "name": "Workout",
        "description": "Keeps tempo stable with tighter BPM guardrails and less downtime.",
        "config": {
            "mix_mode": "balanced",
            "flow_curve": True,
            "flow_profile": "peak",
            "tempo_ramp_weight": 0.15,
            "minimax_passes": 3,
            "max_bpm_jump": 12,
            "smoothness_weight": 1.25,
            "variety_weight": 0.2,
        },
    },
    "chill": {
        "name": "Chill",
        "description": "Low-contrast transitions prioritizing mood consistency.",
        "config": {
            "mix_mode": "vibe",
            "flow_curve": True,
            "flow_profile": "gentle",
            "tempo_ramp_weight": 0.06,
            "minimax_passes": 2,
            "max_bpm_jump": 8,
            "smoothness_weight": 1.1,
            "variety_weight": 0.15,
        },
    },
}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("spotify_optimizer")
    logger.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler()

    if LOG_FORMAT == "json":
        formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.propagate = False
    return logger


LOGGER = setup_logging()
if ENV == "production" and not MODEL_ADMIN_USER_IDS:
    LOGGER.error("model_admin_users_not_configured")
if ENV != "production" and not MODEL_ADMIN_USER_IDS:
    LOGGER.warning("model_admin_fallback_enabled", extra={"env": ENV})
STATE_DB_PATH = os.getenv(
    "STATE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "state.db"),
)
STATE_STORE = SQLiteStateStore(STATE_DB_PATH)
RUN_HISTORY = DurableDict(STATE_STORE, "run_history")
COMPARISON_HISTORY = DurableDict(STATE_STORE, "comparison_history")
PRESET_STORE = DurableDict(STATE_STORE, "preset_store")
SNAPSHOT_STORE = DurableDict(STATE_STORE, "snapshot_store")
BATCH_STORE = DurableDict(STATE_STORE, "batch_store")
SCHEDULE_STORE = DurableDict(STATE_STORE, "schedule_store")
SCHEDULER_STOP = threading.Event()
FEEDBACK_STORE = DurableDict(STATE_STORE, "feedback_store")
FEEDBACK_WEIGHT_OFFSETS = DurableDict(STATE_STORE, "feedback_weight_offsets")
RUN_TASK_STATUS = DurableDict(STATE_STORE, "run_task_status")
EVENT_STORE = DurableEventBuffer(STATE_STORE, "run_event_buffer")
REPORT_STORE = DurableDict(STATE_STORE, "report_store")
IDEMPOTENCY_STORE = DurableDict(STATE_STORE, "idempotency_store")
MODEL_REGISTRY = DurableDict(STATE_STORE, "model_registry")
MODEL_STATE = DurableDict(STATE_STORE, "model_state")
TRAINING_JOB_STORE = DurableDict(STATE_STORE, "training_job_store")

rate_limit_storage = os.getenv("RATE_LIMIT_REDIS_URL") or os.getenv("REDIS_URL") or "memory://"
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_GLOBAL], storage_uri=rate_limit_storage)

app = FastAPI(title="Spotify Mix Optimizer")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "path"],
)

ACTIVE_MODEL_CACHE: dict[str, object] = {
    "version": None,
    "weights": None,
    "bias": 0.0,
    "sample_count": 0,
    "trained_at": None,
}


def refresh_active_model_cache() -> None:
    active_version = MODEL_STATE.get("active_version")
    if not active_version:
        ACTIVE_MODEL_CACHE.update(
            {
                "version": None,
                "weights": None,
                "bias": 0.0,
                "sample_count": 0,
                "trained_at": None,
            }
        )
        return

    record = MODEL_REGISTRY.get(active_version)
    if not record:
        ACTIVE_MODEL_CACHE.update(
            {
                "version": None,
                "weights": None,
                "bias": 0.0,
                "sample_count": 0,
                "trained_at": None,
            }
        )
        return

    artifact = load_model_artifact(record.get("artifact_path", ""))
    if not artifact:
        ACTIVE_MODEL_CACHE.update(
            {
                "version": None,
                "weights": None,
                "bias": 0.0,
                "sample_count": 0,
                "trained_at": None,
            }
        )
        return

    model = transition_model_from_dict(artifact)
    ACTIVE_MODEL_CACHE.update(
        {
            "version": model.version,
            "weights": model.weights,
            "bias": model.bias,
            "sample_count": model.sample_count,
            "trained_at": model.trained_at,
        }
    )


def active_model_payload() -> dict[str, object]:
    if not ACTIVE_MODEL_CACHE.get("weights"):
        return {
            "version": None,
            "weights": None,
            "bias": 0.0,
            "alpha": 0.0,
            "sample_count": 0,
        }
    return {
        "version": ACTIVE_MODEL_CACHE.get("version"),
        "weights": ACTIVE_MODEL_CACHE.get("weights"),
        "bias": float(ACTIVE_MODEL_CACHE.get("bias") or 0.0),
        "alpha": max(0.0, min(1.0, MODEL_BLEND_ALPHA)),
        "sample_count": int(ACTIVE_MODEL_CACHE.get("sample_count") or 0),
    }


def model_quality_thresholds() -> dict[str, float]:
    return {
        "min_accuracy": max(0.0, min(1.0, float(MODEL_MIN_ACCURACY))),
        "max_loss": max(0.0, float(MODEL_MAX_LOSS)),
    }


def evaluate_quality_gate(
    metrics: Optional[dict],
    validation_metrics: Optional[dict] = None,
) -> dict[str, object]:
    thresholds = model_quality_thresholds()
    metric_source = "validation" if isinstance(validation_metrics, dict) and validation_metrics else "train"
    source_metrics = validation_metrics if metric_source == "validation" else metrics
    source_metrics = source_metrics if isinstance(source_metrics, dict) else {}

    raw_accuracy = source_metrics.get("accuracy")
    raw_loss = source_metrics.get("loss")

    accuracy = None if raw_accuracy is None else float(raw_accuracy)
    loss = None if raw_loss is None else float(raw_loss)

    reasons: list[str] = []
    if accuracy is None:
        reasons.append("accuracy metric missing")
    elif accuracy < thresholds["min_accuracy"]:
        reasons.append(
            f"accuracy {accuracy:.4f} below minimum {thresholds['min_accuracy']:.4f}"
        )

    if loss is None:
        reasons.append("loss metric missing")
    elif loss > thresholds["max_loss"]:
        reasons.append(f"loss {loss:.4f} above maximum {thresholds['max_loss']:.4f}")

    return {
        "passed": len(reasons) == 0,
        "metric_source": metric_source,
        "metrics": {
            "accuracy": accuracy,
            "loss": loss,
        },
        "thresholds": thresholds,
        "reasons": reasons,
    }


def quality_gate_for_record(record: Optional[dict]) -> dict[str, object]:
    data = record if isinstance(record, dict) else {}
    return evaluate_quality_gate(
        metrics=data.get("metrics"),
        validation_metrics=data.get("validation_metrics"),
    )


def model_promotion_thresholds() -> dict[str, float]:
    return {
        "min_accuracy_delta": float(MODEL_MIN_ACCURACY_DELTA),
        "max_loss_delta": float(MODEL_MAX_LOSS_DELTA),
    }


def _metrics_from_record(record: Optional[dict]) -> tuple[str, dict]:
    data = record if isinstance(record, dict) else {}
    validation_metrics = data.get("validation_metrics")
    if isinstance(validation_metrics, dict) and validation_metrics:
        return "validation", validation_metrics
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return "train", metrics
    return "none", {}


def evaluate_promotion_gate(
    candidate_record: Optional[dict],
    active_record: Optional[dict],
) -> dict[str, object]:
    thresholds = model_promotion_thresholds()
    candidate_source, _ = _metrics_from_record(candidate_record)
    if not isinstance(active_record, dict):
        return {
            "passed": True,
            "thresholds": thresholds,
            "candidate_metric_source": candidate_source,
            "active_metric_source": None,
            "deltas": {"accuracy": None, "loss": None},
            "reasons": [],
        }

    _, candidate_metrics = _metrics_from_record(candidate_record)
    active_source, active_metrics = _metrics_from_record(active_record)

    candidate_accuracy_raw = candidate_metrics.get("accuracy")
    candidate_loss_raw = candidate_metrics.get("loss")
    active_accuracy_raw = active_metrics.get("accuracy")
    active_loss_raw = active_metrics.get("loss")

    candidate_accuracy = None if candidate_accuracy_raw is None else float(candidate_accuracy_raw)
    candidate_loss = None if candidate_loss_raw is None else float(candidate_loss_raw)
    active_accuracy = None if active_accuracy_raw is None else float(active_accuracy_raw)
    active_loss = None if active_loss_raw is None else float(active_loss_raw)

    accuracy_delta = (
        None
        if candidate_accuracy is None or active_accuracy is None
        else candidate_accuracy - active_accuracy
    )
    loss_delta = None if candidate_loss is None or active_loss is None else candidate_loss - active_loss

    reasons: list[str] = []
    if accuracy_delta is None:
        reasons.append("missing accuracy metrics for promotion comparison")
    elif accuracy_delta < thresholds["min_accuracy_delta"]:
        reasons.append(
            f"accuracy delta {accuracy_delta:.4f} below minimum {thresholds['min_accuracy_delta']:.4f}"
        )

    if loss_delta is None:
        reasons.append("missing loss metrics for promotion comparison")
    elif loss_delta > thresholds["max_loss_delta"]:
        reasons.append(f"loss delta {loss_delta:.4f} above maximum {thresholds['max_loss_delta']:.4f}")

    return {
        "passed": len(reasons) == 0,
        "thresholds": thresholds,
        "candidate_metric_source": candidate_source,
        "active_metric_source": active_source,
        "deltas": {
            "accuracy": accuracy_delta,
            "loss": loss_delta,
        },
        "reasons": reasons,
    }


def build_model_feedback_evaluation(days: int) -> dict[str, object]:
    window_days = max(1, min(365, int(days)))
    cutoff = time.time() - (window_days * 86400)

    runs = {run_id: row for run_id, row in RUN_HISTORY.items()}
    active_version = MODEL_STATE.get("active_version")
    active_record = MODEL_REGISTRY.get(active_version) if active_version else None

    version_buckets: dict[str, dict[str, object]] = {}
    total_labeled_feedback = 0

    for _, feedback in FEEDBACK_STORE.items():
        if not isinstance(feedback, dict):
            continue
        rating = int(feedback.get("rating", 0))
        if rating == 0:
            continue

        run_id = feedback.get("run_id")
        run = runs.get(run_id)
        if not isinstance(run, dict):
            continue

        created_at = float(feedback.get("created_at", 0.0) or 0.0)
        if created_at <= 0:
            created_at = float(run.get("created_at", 0.0) or 0.0)
        if created_at <= 0 or created_at < cutoff:
            continue

        transitions = run.get("transitions") or []
        edge_index = int(feedback.get("edge_index", -1))
        reason_code = "unknown"
        if 0 <= edge_index < len(transitions):
            transition = transitions[edge_index] if isinstance(transitions[edge_index], dict) else {}
            reason_code = str(transition.get("reason_code") or "unknown")

        model_version = str(run.get("model_version") or "heuristic")
        bucket = version_buckets.setdefault(
            model_version,
            {
                "version": model_version,
                "sample_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "rating_sum": 0.0,
                "reason_counts": {},
                "latest_feedback_at": 0.0,
            },
        )

        bucket["sample_count"] = int(bucket.get("sample_count", 0)) + 1
        bucket["rating_sum"] = float(bucket.get("rating_sum", 0.0)) + float(rating)
        if rating > 0:
            bucket["positive_count"] = int(bucket.get("positive_count", 0)) + 1
        elif rating < 0:
            bucket["negative_count"] = int(bucket.get("negative_count", 0)) + 1

        reason_counts = bucket.setdefault("reason_counts", {})
        reason_counts[reason_code] = int(reason_counts.get(reason_code, 0)) + 1

        latest_feedback_at = float(bucket.get("latest_feedback_at", 0.0) or 0.0)
        if created_at > latest_feedback_at:
            bucket["latest_feedback_at"] = created_at

        total_labeled_feedback += 1

    versions = []
    for version, bucket in version_buckets.items():
        sample_count = int(bucket.get("sample_count", 0))
        if sample_count <= 0:
            continue

        reason_counts = bucket.get("reason_counts", {})
        dominant_reason_codes = [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in sorted(
                reason_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        ]

        record = MODEL_REGISTRY.get(version) if version != "heuristic" else None
        quality_gate = quality_gate_for_record(record) if record else None
        promotion_gate = (
            evaluate_promotion_gate(
                record,
                active_record if version != active_version else None,
            )
            if record
            else None
        )

        versions.append(
            {
                "version": version,
                "sample_count": sample_count,
                "positive_ratio": round(int(bucket.get("positive_count", 0)) / sample_count, 6),
                "rough_rate": round(int(bucket.get("negative_count", 0)) / sample_count, 6),
                "mean_rating": round(float(bucket.get("rating_sum", 0.0)) / sample_count, 6),
                "latest_feedback_at": float(bucket.get("latest_feedback_at", 0.0) or 0.0),
                "dominant_reason_codes": dominant_reason_codes,
                "quality_gate": quality_gate,
                "promotion_gate": promotion_gate,
            }
        )

    versions.sort(
        key=lambda row: (
            int(row.get("sample_count", 0)),
            float(row.get("latest_feedback_at", 0.0)),
        ),
        reverse=True,
    )
    active_metrics = next((item for item in versions if item.get("version") == active_version), None)

    return {
        "window_days": window_days,
        "total_labeled_feedback": total_labeled_feedback,
        "active_version": active_version,
        "active_metrics": active_metrics,
        "versions": versions[:20],
    }


def update_training_job(job_id: str, **updates) -> dict:
    current = dict(TRAINING_JOB_STORE.get(job_id, {}))
    current.update(updates)
    current["updated_at"] = time.time()
    TRAINING_JOB_STORE[job_id] = current
    return current


def has_active_training_job() -> bool:
    for _, row in TRAINING_JOB_STORE.items():
        if row.get("status") in {"queued", "running"}:
            return True
    return False


def execute_transition_training(owner_id: Optional[str], min_samples: int, activate: bool) -> dict:
    started_at = time.time()
    active_version_before = MODEL_STATE.get("active_version")
    active_record_before = (
        MODEL_REGISTRY.get(active_version_before) if active_version_before else None
    )
    model, details = train_transition_model_from_feedback(
        RUN_HISTORY.items(),
        FEEDBACK_STORE.items(),
        owner_id=owner_id,
        min_samples=min_samples,
    )

    if not model:
        return {
            "trained": False,
            "owner_scope": owner_id or "all",
            "started_at": started_at,
            "finished_at": time.time(),
            **details,
        }

    artifact_path = save_model_artifact(MODEL_DIR, model.to_dict())
    quality_gate = evaluate_quality_gate(
        metrics=details.get("metrics"),
        validation_metrics=details.get("validation_metrics"),
    )
    promotion_gate = evaluate_promotion_gate(
        {
            "metrics": details.get("metrics"),
            "validation_metrics": details.get("validation_metrics"),
        },
        active_record_before,
    )
    should_activate = bool(activate and quality_gate.get("passed") and promotion_gate.get("passed"))
    MODEL_REGISTRY[model.version] = {
        "artifact_path": artifact_path,
        "owner_scope": owner_id or "all",
        "metrics": details.get("metrics", {}),
        "validation_metrics": details.get("validation_metrics", {}),
        "quality_gate": quality_gate,
        "promotion_gate": promotion_gate,
        "promotion_baseline_version": active_version_before,
        "sample_count": details.get("sample_count", 0),
        "created_at": time.time(),
    }

    if should_activate:
        MODEL_STATE["active_version"] = model.version
        MODEL_STATE["last_auto_train_at"] = time.time()
        refresh_active_model_cache()

    return {
        "trained": True,
        "owner_scope": owner_id or "all",
        "version": model.version,
        "artifact_path": artifact_path,
        "activate": activate,
        "activated": should_activate,
        "quality_gate": quality_gate,
        "promotion_gate": promotion_gate,
        "started_at": started_at,
        "finished_at": time.time(),
        **details,
    }


def run_training_job(
    job_id: str,
    owner_id: Optional[str],
    min_samples: int,
    activate: bool,
    trigger: str,
) -> None:
    update_training_job(job_id, status="running", progress=20, started_at=time.time())
    try:
        result = execute_transition_training(owner_id=owner_id, min_samples=min_samples, activate=activate)
        update_training_job(
            job_id,
            status="completed",
            progress=100,
            finished_at=time.time(),
            trigger=trigger,
            result=result,
            error=None,
        )
    except Exception as exc:
        LOGGER.exception("model_training_failed", extra={"job_id": job_id, "trigger": trigger})
        update_training_job(
            job_id,
            status="failed",
            progress=100,
            finished_at=time.time(),
            trigger=trigger,
            error=str(exc),
        )


def queue_training_job(
    owner_id: Optional[str],
    min_samples: int,
    activate: bool,
    trigger: str = "manual",
) -> dict:
    job_id = uuid.uuid4().hex
    created_at = time.time()
    TRAINING_JOB_STORE[job_id] = {
        "status": "queued",
        "progress": 0,
        "owner_scope": owner_id or "all",
        "min_samples": min_samples,
        "activate": bool(activate),
        "trigger": trigger,
        "created_at": created_at,
        "updated_at": created_at,
    }
    worker = threading.Thread(
        target=run_training_job,
        args=(job_id, owner_id, min_samples, activate, trigger),
        daemon=True,
        name=f"model-train-{job_id[:8]}",
    )
    worker.start()
    return {"job_id": job_id, **TRAINING_JOB_STORE.get(job_id, {})}


def maybe_auto_retrain_model() -> None:
    interval = max(0, MODEL_RETRAIN_INTERVAL_MINUTES)
    if interval <= 0:
        return
    now = time.time()
    last_attempt = float(MODEL_STATE.get("last_auto_train_at", 0.0))
    if now - last_attempt < interval * 60:
        return

    if has_active_training_job():
        return

    MODEL_STATE["last_auto_train_at"] = now
    queue_training_job(owner_id=None, min_samples=MODEL_MIN_SAMPLES, activate=True, trigger="auto")


def cleanup_state_retention() -> None:
    if STATE_RETENTION_DAYS <= 0:
        return
    cutoff = time.time() - (STATE_RETENTION_DAYS * 86400)
    namespaces = [
        "run_history",
        "comparison_history",
        "feedback_store",
        "run_task_status",
        "report_store",
        "idempotency_store",
        "training_job_store",
    ]
    for namespace in namespaces:
        try:
            STATE_STORE.delete_older_than(namespace, cutoff)
        except Exception:
            continue
    try:
        STATE_STORE.delete_events_older_than("run_event_buffer", cutoff)
    except Exception:
        pass

frontend_urls = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")
frontend_urls = [url.strip() for url in frontend_urls if url.strip()]
frontend_redirect = os.getenv("FRONTEND_REDIRECT_URL") or (
    frontend_urls[0] if frontend_urls else "http://localhost:3000"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_urls,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def check_rate_limit_store() -> Optional[bool]:
    if rate_limit_storage.startswith("redis://") or rate_limit_storage.startswith("rediss://"):
        try:
            return bool(Redis.from_url(rate_limit_storage, decode_responses=True).ping())
        except Exception:
            return False
    return None


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    LOGGER.exception(
        "unhandled_error",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", None)
    LOGGER.warning(
        "validation_error",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "errors": exc.errors(),
        },
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": request_id},
    )


@app.middleware("http")
async def request_guardrails(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid content-length header"})

    response = await call_next(request)
    if SECURITY_HEADERS_ENABLED:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.monotonic()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = int((time.monotonic() - start) * 1000)
        status = response.status_code if response else 500
        if response is not None:
            response.headers["X-Request-ID"] = request_id
        user_id = None
        try:
            sid = get_session_id(request)
            if sid:
                session = STORE.get_session(sid)
                if session:
                    user_id = session.get("user_id")
        except Exception:
            user_id = None
        REQUEST_COUNT.labels(request.method, request.url.path, str(status)).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration_ms / 1000.0)
        LOGGER.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": duration_ms,
                "user_id": user_id,
            },
        )

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8000/callback")
scopes = "playlist-read-private playlist-read-collaborative playlist-modify-private playlist-modify-public"

if not client_id or not client_secret:
    raise RuntimeError("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment.")


def build_oauth() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scopes,
        open_browser=False,
        cache_path=None,
        show_dialog=True,
    )


class SessionStore:
    def get_session(self, sid: str) -> Optional[dict]:
        raise NotImplementedError

    def set_session(self, sid: str, token_info: dict) -> None:
        raise NotImplementedError

    def delete_session(self, sid: str) -> None:
        raise NotImplementedError

    def set_state(self, state: str, sid: str) -> None:
        raise NotImplementedError

    def pop_state(self, state: str) -> Optional[str]:
        raise NotImplementedError

    def ping(self) -> bool:
        raise NotImplementedError


class InMemoryStore(SessionStore):
    def __init__(self, ttl: int, state_ttl: int) -> None:
        self.ttl = ttl
        self.state_ttl = state_ttl
        self.sessions: dict[str, tuple[dict, float]] = {}
        self.states: dict[str, tuple[str, float]] = {}

    def _set(self, store: dict, key: str, value, ttl: int) -> None:
        store[key] = (value, time.time() + ttl)

    def _get(self, store: dict, key: str):
        item = store.get(key)
        if not item:
            return None
        value, exp = item
        if exp <= time.time():
            store.pop(key, None)
            return None
        return value

    def get_session(self, sid: str) -> Optional[dict]:
        return self._get(self.sessions, sid)

    def set_session(self, sid: str, token_info: dict) -> None:
        self._set(self.sessions, sid, token_info, self.ttl)

    def delete_session(self, sid: str) -> None:
        self.sessions.pop(sid, None)

    def set_state(self, state: str, sid: str) -> None:
        self._set(self.states, state, sid, self.state_ttl)

    def pop_state(self, state: str) -> Optional[str]:
        value = self._get(self.states, state)
        if value is not None:
            self.states.pop(state, None)
        return value

    def ping(self) -> bool:
        return True


class RedisStore(SessionStore):
    def __init__(self, url: str, ttl: int, state_ttl: int) -> None:
        self.redis = Redis.from_url(url, decode_responses=True)
        self.ttl = ttl
        self.state_ttl = state_ttl

    def get_session(self, sid: str) -> Optional[dict]:
        data = self.redis.get(f"session:{sid}")
        if not data:
            return None
        return json.loads(data)

    def set_session(self, sid: str, token_info: dict) -> None:
        self.redis.setex(f"session:{sid}", self.ttl, json.dumps(token_info))

    def delete_session(self, sid: str) -> None:
        self.redis.delete(f"session:{sid}")

    def set_state(self, state: str, sid: str) -> None:
        self.redis.setex(f"state:{state}", self.state_ttl, sid)

    def pop_state(self, state: str) -> Optional[str]:
        key = f"state:{state}"
        data = self.redis.get(key)
        if data is not None:
            self.redis.delete(key)
        return data

    def ping(self) -> bool:
        try:
            return bool(self.redis.ping())
        except Exception:
            return False


redis_url = os.getenv("REDIS_URL")
if redis_url:
    STORE: SessionStore = RedisStore(redis_url, SESSION_TTL_SECONDS, STATE_TTL_SECONDS)
else:
    STORE = InMemoryStore(SESSION_TTL_SECONDS, STATE_TTL_SECONDS)


def get_session_id(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def get_token_for_session(request: Request) -> Dict:
    sid = get_session_id(request)
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token_info = STORE.get_session(sid)
    if not token_info:
        raise HTTPException(status_code=401, detail="Not authenticated")
    oauth = build_oauth()
    if oauth.is_token_expired(token_info):
        user_id = token_info.get("user_id")
        if "refresh_token" not in token_info:
            raise HTTPException(status_code=401, detail="Session expired")
        token_info = oauth.refresh_access_token(token_info["refresh_token"])
        if user_id:
            token_info["user_id"] = user_id
        STORE.set_session(sid, token_info)

    return token_info


def spotify_for_session(request: Request) -> spotipy.Spotify:
    token_info = get_token_for_session(request)
    return spotipy.Spotify(auth=token_info["access_token"], requests_timeout=10)


@app.get("/health")
@limiter.exempt
def health():
    return {"status": "ok"}


@app.get("/ready")
@limiter.exempt
def readiness():
    checks = {
        "sessions": STORE.ping(),
        "state_store": STATE_STORE.ping(),
    }
    rate_limit_ok = check_rate_limit_store()
    if rate_limit_ok is not None:
        checks["rate_limit"] = rate_limit_ok

    ok = all(checks.values())
    status = "ok" if ok else "degraded"
    code = 200 if ok else 503
    return JSONResponse(status_code=code, content={"status": status, "checks": checks})


@app.get("/metrics")
@limiter.exempt
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/login")
@limiter.limit(RATE_LIMIT_LOGIN)
def login(request: Request):
    sid = get_session_id(request) or secrets.token_urlsafe(16)
    state = secrets.token_urlsafe(16)
    STORE.set_state(state, sid)

    oauth = build_oauth()
    auth_url = oauth.get_authorize_url(state=state)

    response = RedirectResponse(auth_url)
    response.set_cookie(
        SESSION_COOKIE,
        sid,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        domain=COOKIE_DOMAIN,
    )
    return response


@app.get("/callback")
def callback(request: Request, code: Optional[str] = None, state: Optional[str] = None):
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code/state")

    sid = get_session_id(request)
    expected_sid = STORE.pop_state(state)
    if not sid or not expected_sid or sid != expected_sid:
        raise HTTPException(status_code=400, detail="Invalid state")

    oauth = build_oauth()
    token_info = oauth.get_access_token(code)
    try:
        sp = spotipy.Spotify(auth=token_info["access_token"], requests_timeout=10)
        profile = sp.current_user()
        if profile and profile.get("id"):
            token_info["user_id"] = profile.get("id")
    except Exception:
        pass
    STORE.set_session(sid, token_info)

    response = RedirectResponse(frontend_redirect)
    response.set_cookie(
        SESSION_COOKIE,
        sid,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        domain=COOKIE_DOMAIN,
    )
    return response


@app.get("/logout")
def logout(request: Request):
    sid = get_session_id(request)
    if sid:
        STORE.delete_session(sid)
    response = RedirectResponse(frontend_redirect)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/me")
def me(request: Request):
    sp = spotify_for_session(request)
    owner_id = current_owner_id(request)
    feedback_offsets = owner_feedback_offsets(owner_id)
    profile = sp.current_user()
    return {"id": profile.get("id"), "display_name": profile.get("display_name")}


@app.post("/optimize/suggest-anchors")
def suggest_anchors(request: Request, payload: AnchorSuggestRequest):
    sp = spotify_for_session(request)
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    playlist_id = parse_playlist_id(payload.playlist)
    _, tracks = fetch_playlist_tracks(sp, playlist_id)
    if not tracks:
        raise HTTPException(status_code=404, detail="no playable tracks found")
    enrich_audio_features(sp, tracks, cache_path)

    def opener_score(track):
        features = track.features or {}
        energy = float(features.get("energy") or 0.5)
        tempo = float(features.get("tempo") or 120.0)
        valence = float(features.get("valence") or 0.5)
        return abs(tempo - 110.0) * 0.01 + energy * 0.6 + valence * 0.2

    def closer_score(track):
        features = track.features or {}
        energy = float(features.get("energy") or 0.5)
        valence = float(features.get("valence") or 0.5)
        dance = float(features.get("danceability") or 0.5)
        return -(energy * 0.5 + valence * 0.2 + dance * 0.3)

    opener_candidates = sorted(tracks, key=opener_score)[: payload.count]
    closer_candidates = sorted(tracks, key=closer_score)[: payload.count]
    return {
        "playlist_id": playlist_id,
        "openers": [{"id": track.id, "name": track.name, "artists": track.artists} for track in opener_candidates],
        "closers": [{"id": track.id, "name": track.name, "artists": track.artists} for track in closer_candidates],
    }


def apply_builtin_preset(payload: OptimizeRequest) -> OptimizeRequest:
    if not payload.preset_id:
        return payload
    preset = BUILTIN_PRESETS.get(payload.preset_id)
    if not preset:
        raise HTTPException(status_code=400, detail="unknown preset_id")

    merged = payload.model_dump()
    merged.update(preset.get("config", {}))
    merged.update(payload.model_dump(exclude_unset=True))
    merged["preset_id"] = payload.preset_id
    return OptimizeRequest(**merged)


def normalize_optimize_payload(payload: OptimizeRequest) -> OptimizeRequest:
    data = payload.model_dump()
    data["bpm_guardrails"] = sorted({float(value) for value in (data.get("bpm_guardrails") or []) if float(value) > 0})
    data["locked_blocks"] = data.get("locked_blocks") or None

    mood_points = data.get("mood_curve_points") or []
    if mood_points:
        data["mood_curve_points"] = sorted(
            mood_points,
            key=lambda point: (float(point.get("position", 0.0)), float(point.get("energy", 0.0))),
        )
    return OptimizeRequest(**data)


def build_optimize_config_hash(payload: OptimizeRequest) -> str:
    normalized = normalize_optimize_payload(payload)
    raw = normalized.model_dump()
    raw.pop("name", None)
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def resolve_optimize_payload(payload: OptimizeRequest) -> OptimizeRequest:
    resolved = apply_builtin_preset(payload)
    resolved = normalize_optimize_payload(resolved)
    validate_optimize_payload(resolved)
    return resolved


def validate_optimize_payload(payload: OptimizeRequest) -> None:
    if payload.missing not in {"append", "drop"}:
        raise HTTPException(status_code=400, detail="missing must be append or drop")
    if payload.mix_mode not in {"balanced", "harmonic", "vibe"}:
        raise HTTPException(status_code=400, detail="mix_mode must be balanced, harmonic, or vibe")
    if payload.flow_profile not in {"peak", "gentle", "cooldown"}:
        raise HTTPException(status_code=400, detail="flow_profile must be peak, gentle, or cooldown")
    if payload.explicit_mode not in {"allow", "prefer_clean", "clean_only"}:
        raise HTTPException(status_code=400, detail="explicit_mode must be allow, prefer_clean, or clean_only")
    if payload.solver_mode not in {"classic", "hybrid"}:
        raise HTTPException(status_code=400, detail="solver_mode must be classic or hybrid")
    if payload.anneal_temp_end > payload.anneal_temp_start:
        raise HTTPException(
            status_code=400,
            detail="anneal_temp_end must be less than or equal to anneal_temp_start",
        )


@app.get("/presets/builtin")
def list_builtin_presets():
    return {
        "items": [
            {"preset_id": preset_id, **value}
            for preset_id, value in BUILTIN_PRESETS.items()
        ]
    }


@app.post("/optimize", response_model=OptimizeResponse)
@limiter.limit(RATE_LIMIT_OPTIMIZE)
def optimize(request: Request, payload: OptimizeRequest):
    payload = resolve_optimize_payload(payload)

    owner_id = current_owner_id(request)
    cached = get_idempotency_response(owner_id, "optimize", request)
    if cached:
        return cached.get("payload")

    sp = spotify_for_session(request)
    result = run_single_optimization(sp=sp, owner_id=owner_id, payload=payload, seed=42)
    RUN_TASK_STATUS[result["run_id"]] = {
        "status": "completed",
        "progress": 100,
        "result": result,
        "updated_at": time.time(),
    }
    set_idempotency_response(owner_id, "optimize", request, result)
    return result


def async_optimize_job(run_id: str, sid: str, owner_id: str, payload_dict: dict) -> None:
    try:
        RUN_TASK_STATUS[run_id] = {"status": "running", "progress": 5, "updated_at": time.time()}
        emit_run_event(run_id, "queued", 5, "Queued async optimization")
        sp = spotify_for_sid(sid)
        if not sp:
            raise RuntimeError("session expired or missing for async run")
        payload = resolve_optimize_payload(OptimizeRequest(**payload_dict))
        result = run_single_optimization(
            sp=sp,
            owner_id=owner_id,
            payload=payload,
            seed=45,
            run_id=run_id,
        )
        RUN_TASK_STATUS[run_id] = {
            "status": "completed",
            "progress": 100,
            "result": result,
            "updated_at": time.time(),
        }
    except Exception as exc:
        RUN_TASK_STATUS[run_id] = {
            "status": "failed",
            "progress": 100,
            "error": str(exc),
            "updated_at": time.time(),
        }
        emit_run_event(run_id, "failed", 100, "Run failed", {"error": str(exc)})


@app.post("/optimize/async")
@limiter.limit(RATE_LIMIT_OPTIMIZE)
def optimize_async(request: Request, payload: OptimizeRequest):
    payload = resolve_optimize_payload(payload)
    sid = get_session_id(request)
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    owner_id = current_owner_id(request)
    cached = get_idempotency_response(owner_id, "optimize_async", request)
    if cached:
        return cached.get("payload")

    run_id = uuid.uuid4().hex
    RUN_TASK_STATUS[run_id] = {
        "status": "queued",
        "progress": 0,
        "updated_at": time.time(),
    }
    emit_run_event(run_id, "queued", 0, "Async run queued")
    worker = threading.Thread(
        target=async_optimize_job,
        args=(run_id, sid, owner_id, payload.model_dump()),
        daemon=True,
        name=f"optimize-async-{run_id[:8]}",
    )
    worker.start()
    response_payload = {"run_id": run_id, "status": "queued"}
    set_idempotency_response(owner_id, "optimize_async", request, response_payload)
    return response_payload


@app.get("/optimize/{run_id}")
def optimize_status(run_id: str):
    status = RUN_TASK_STATUS.get(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, **status}


@app.get("/events/{run_id}")
def run_events(run_id: str):
    def event_generator():
        cursor = 0
        idle_ticks = 0
        yield "retry: 2000\n\n"
        initial = json.dumps({"event": "stream_open", "run_id": run_id})
        yield f"data: {initial}\n\n"
        while idle_ticks < 600:
            events = EVENT_STORE.list_after(run_id, after_seq=cursor)
            for seq, event in events:
                payload = json.dumps(event)
                cursor = seq
                yield f"data: {payload}\n\n"
                idle_ticks = 0
            status = RUN_TASK_STATUS.get(run_id, {})
            if status.get("status") in {"completed", "failed"} and not events:
                break
            if idle_ticks % 30 == 0:
                heartbeat = json.dumps({"event": "heartbeat", "run_id": run_id, "cursor": cursor})
                yield f"data: {heartbeat}\n\n"
            idle_ticks += 1
            time.sleep(0.5)
        end_payload = json.dumps({"event": "stream_end", "run_id": run_id})
        yield f"data: {end_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/optimize/{run_id}/transitions")
def run_transitions(run_id: str):
    run = RUN_HISTORY.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    transitions = run.get("transitions", [])
    return {
        "run_id": run_id,
        "playlist_id": run.get("playlist_id"),
        "playlist_name": run.get("playlist_name"),
        "transition_score": run.get("transition_score"),
        "summary": transition_summary(transitions),
        "transitions": transitions,
    }


def run_summary_metrics(run: dict) -> dict:
    transitions = run.get("transitions", [])
    scores = [float(item.get("score", 0.0)) for item in transitions]
    if not scores:
        return {"mean_edge_score": 0.0, "max_edge_score": 0.0, "edge_count": 0}
    return {
        "mean_edge_score": round(sum(scores) / len(scores), 6),
        "max_edge_score": round(max(scores), 6),
        "edge_count": len(scores),
    }


def transition_summary(transitions: list[dict]) -> dict:
    if not transitions:
        return {"dominant_penalties": [], "worst_edges": []}

    dominant_counts: dict[str, int] = {}
    for item in transitions:
        dominant = item.get("reason_code") or item.get("dominant_component") or "mixed_factors"
        dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1

    dominant_penalties = sorted(
        [{"reason_code": key, "count": value} for key, value in dominant_counts.items()],
        key=lambda row: row["count"],
        reverse=True,
    )[:5]
    worst_edges = sorted(transitions, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:5]
    return {"dominant_penalties": dominant_penalties, "worst_edges": worst_edges}


def build_run_report(run_id: str, run: dict) -> dict:
    transitions = run.get("transitions", [])
    top = sorted(transitions, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:10]
    return {
        "run_id": run_id,
        "created_at": run.get("created_at"),
        "source_playlist_id": run.get("source_playlist_id"),
        "playlist_id": run.get("playlist_id"),
        "playlist_name": run.get("playlist_name"),
        "transition_score": run.get("transition_score"),
        "model_version": run.get("model_version"),
        "model_alpha": run.get("model_alpha"),
        "metrics": run_summary_metrics(run),
        "transition_summary": transition_summary(transitions),
        "roughest": run.get("roughest", []),
        "top_transitions": top,
        "request": run.get("request", {}),
    }


def edge_score_diff(
    baseline_transitions: list[dict],
    candidate_transitions: list[dict],
    max_edges: int,
) -> dict:
    size = min(len(baseline_transitions), len(candidate_transitions))
    if size <= 0:
        return {"edge_diffs": [], "most_improved": [], "most_regressed": []}

    rows: list[dict] = []
    for idx in range(size):
        baseline_edge = baseline_transitions[idx]
        candidate_edge = candidate_transitions[idx]
        baseline_score = float(baseline_edge.get("score", 0.0))
        candidate_score = float(candidate_edge.get("score", 0.0))
        rows.append(
            {
                "index": idx,
                "baseline_score": round(baseline_score, 6),
                "candidate_score": round(candidate_score, 6),
                "score_delta": round(candidate_score - baseline_score, 6),
                "baseline_edge": {
                    "from_track": baseline_edge.get("from_track"),
                    "to_track": baseline_edge.get("to_track"),
                    "reason_code": baseline_edge.get("reason_code"),
                },
                "candidate_edge": {
                    "from_track": candidate_edge.get("from_track"),
                    "to_track": candidate_edge.get("to_track"),
                    "reason_code": candidate_edge.get("reason_code"),
                },
            }
        )

    rows.sort(key=lambda item: abs(float(item.get("score_delta", 0.0))), reverse=True)
    improved = sorted(rows, key=lambda item: float(item.get("score_delta", 0.0)))[:max_edges]
    regressed = sorted(rows, key=lambda item: float(item.get("score_delta", 0.0)), reverse=True)[:max_edges]
    return {
        "edge_diffs": rows[:max_edges],
        "most_improved": improved,
        "most_regressed": regressed,
    }


def build_simple_pdf(lines: list[str]) -> bytes:
    sanitized = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    text_ops = []
    y = 760
    for line in sanitized:
        text_ops.append(f"1 0 0 1 40 {y} Tm ({line}) Tj")
        y -= 14
        if y < 40:
            break
    stream = "BT /F1 10 Tf " + " ".join(text_ops) + " ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n",
        f"4 0 obj << /Length {len(stream.encode('latin-1'))} >> stream\n{stream}\nendstream endobj\n",
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj.encode("latin-1"))
    xref_pos = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    output.extend(
        (
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(output)


@app.post("/compare")
def compare_runs(payload: CompareRequest):
    baseline = RUN_HISTORY.get(payload.baseline_run_id)
    candidate = RUN_HISTORY.get(payload.candidate_run_id)
    if not baseline or not candidate:
        raise HTTPException(status_code=404, detail="one or both runs not found")

    baseline_metrics = run_summary_metrics(baseline)
    candidate_metrics = run_summary_metrics(candidate)
    delta = {
        "mean_edge_score_delta": round(
            candidate_metrics["mean_edge_score"] - baseline_metrics["mean_edge_score"], 6
        ),
        "max_edge_score_delta": round(
            candidate_metrics["max_edge_score"] - baseline_metrics["max_edge_score"], 6
        ),
        "transition_score_delta": round(
            float(candidate.get("transition_score", 0.0)) - float(baseline.get("transition_score", 0.0)),
            6,
        ),
    }
    comparison_id = uuid.uuid4().hex
    transitions = {}
    if payload.include_edge_diff:
        transitions = edge_score_diff(
            baseline.get("transitions", []),
            candidate.get("transitions", []),
            max_edges=payload.max_edges,
        )
    COMPARISON_HISTORY[comparison_id] = {
        "baseline_run_id": payload.baseline_run_id,
        "candidate_run_id": payload.candidate_run_id,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
        **transitions,
        "created_at": time.time(),
    }
    return {"comparison_id": comparison_id, **COMPARISON_HISTORY[comparison_id]}


@app.get("/compare/{comparison_id}")
def get_comparison(comparison_id: str):
    comparison = COMPARISON_HISTORY.get(comparison_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="comparison not found")
    return {"comparison_id": comparison_id, **comparison}


@app.get("/reports/{run_id}")
def report_json(request: Request, run_id: str):
    run = RUN_HISTORY.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    report = build_run_report(run_id, run)
    REPORT_STORE[run_id] = {
        "owner_id": current_owner_id(request),
        "created_at": time.time(),
        "report": report,
    }
    return report


@app.get("/reports/{run_id}.pdf")
def report_pdf(request: Request, run_id: str):
    run = RUN_HISTORY.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    report = build_run_report(run_id, run)
    lines = [
        f"Spotify Optimizer Report - Run {run_id}",
        f"Playlist: {report.get('playlist_name', '')}",
        f"Source Playlist ID: {report.get('source_playlist_id', '')}",
        f"Output Playlist ID: {report.get('playlist_id', '')}",
        f"Transition Score: {report.get('transition_score', '')}",
        f"Mean Edge Score: {report['metrics'].get('mean_edge_score', '')}",
        f"Max Edge Score: {report['metrics'].get('max_edge_score', '')}",
        "Top Transitions:",
    ]
    for item in report.get("top_transitions", [])[:8]:
        lines.append(f"- {item.get('from_track')} -> {item.get('to_track')} (score {item.get('score')})")
    pdf_bytes = build_simple_pdf(lines)
    headers = {"Content-Disposition": f"attachment; filename=report_{run_id}.pdf"}
    REPORT_STORE[run_id] = {
        "owner_id": current_owner_id(request),
        "created_at": time.time(),
        "report": report,
        "pdf_bytes": len(pdf_bytes),
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


@app.get("/reports/{run_id}/manifest")
def report_manifest(request: Request, run_id: str):
    record = REPORT_STORE.get(run_id)
    if not record:
        run = RUN_HISTORY.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        report = build_run_report(run_id, run)
        record = {
            "owner_id": current_owner_id(request),
            "created_at": time.time(),
            "report": report,
            "pdf_bytes": None,
        }
        REPORT_STORE[run_id] = record
    return {
        "run_id": run_id,
        "created_at": record.get("created_at"),
        "has_json": bool(record.get("report")),
        "has_pdf": record.get("pdf_bytes") is not None,
        "pdf_bytes": record.get("pdf_bytes"),
    }


@app.post("/feedback/manual")
def manual_feedback(request: Request, payload: ManualFeedbackRequest):
    run = RUN_HISTORY.get(payload.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    transitions = run.get("transitions", [])
    if payload.edge_index >= len(transitions):
        raise HTTPException(status_code=400, detail="edge_index out of range")

    owner_id = current_owner_id(request)
    transition = transitions[payload.edge_index]
    components = transition.get("components", {})
    inferred = None
    if components:
        inferred = max(components.items(), key=lambda item: float(item[1]))[0]
    feature = payload.feature or inferred
    if not feature:
        raise HTTPException(status_code=400, detail="feature could not be inferred")

    feedback_id = uuid.uuid4().hex
    FEEDBACK_STORE[feedback_id] = {
        "owner_id": owner_id,
        "run_id": payload.run_id,
        "edge_index": payload.edge_index,
        "rating": payload.rating,
        "feature": feature,
        "note": payload.note,
        "created_at": time.time(),
    }
    update_feedback_offsets(owner_id, feature, payload.rating)
    return {
        "saved": True,
        "owner_id": owner_id,
        "feature": feature,
        "offsets": owner_feedback_offsets(owner_id),
    }


def require_authenticated_owner_id(request: Request) -> str:
    sid = get_session_id(request)
    if not sid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    session = STORE.get_session(sid)
    if not session or not session.get("user_id"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return str(session["user_id"])


def require_model_admin(request: Request) -> str:
    owner_id = require_authenticated_owner_id(request)
    if owner_id in MODEL_ADMIN_USER_IDS:
        return owner_id
    if ENV != "production" and not MODEL_ADMIN_USER_IDS:
        # Local/dev convenience: no configured admin list means any authenticated user can manage models.
        return owner_id
    raise HTTPException(status_code=403, detail="Model admin role required")


@app.get("/model/status")
@limiter.limit(RATE_LIMIT_MODEL_STATUS)
def model_status(request: Request):
    _ = require_model_admin(request)
    refresh_active_model_cache()
    active = active_model_payload()
    active_record = MODEL_REGISTRY.get(str(active.get("version"))) if active.get("version") else None
    versions = []
    for version, record in MODEL_REGISTRY.items():
        row = {"version": version, **record}
        row["quality_gate"] = quality_gate_for_record(record)
        row["promotion_gate"] = evaluate_promotion_gate(
            record,
            active_record if version != active.get("version") else None,
        )
        versions.append(row)
    versions.sort(key=lambda row: row.get("created_at", 0), reverse=True)
    return {
        "active_version": active.get("version"),
        "alpha": active.get("alpha"),
        "sample_count": active.get("sample_count"),
        "min_samples": MODEL_MIN_SAMPLES,
        "quality_gate_thresholds": model_quality_thresholds(),
        "promotion_thresholds": model_promotion_thresholds(),
        "active_quality_gate": quality_gate_for_record(active_record) if active_record else None,
        "available_versions": versions[:20],
    }


@app.get("/model/evaluation")
@limiter.limit(RATE_LIMIT_MODEL_STATUS)
def model_evaluation(request: Request, days: int = MODEL_EVAL_WINDOW_DAYS):
    _ = require_model_admin(request)
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")
    return build_model_feedback_evaluation(days)


@app.post("/model/train")
@limiter.limit(RATE_LIMIT_MODEL_WRITE)
def model_train(request: Request, payload: ModelTrainRequest):
    _ = require_model_admin(request)
    if payload.owner_scope not in {"all", "me"}:
        raise HTTPException(status_code=400, detail="owner_scope must be all or me")
    owner_id = require_authenticated_owner_id(request) if payload.owner_scope == "me" else None
    min_samples = payload.min_samples or MODEL_MIN_SAMPLES
    return queue_training_job(
        owner_id=owner_id,
        min_samples=min_samples,
        activate=payload.activate,
        trigger="manual",
    )


@app.get("/model/train/{job_id}")
@limiter.limit(RATE_LIMIT_MODEL_STATUS)
def model_train_status(request: Request, job_id: str):
    _ = require_model_admin(request)
    job = TRAINING_JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="training job not found")
    return {"job_id": job_id, **job}


@app.post("/model/activate/{version}")
@limiter.limit(RATE_LIMIT_MODEL_WRITE)
def model_activate(request: Request, version: str):
    _ = require_model_admin(request)
    record = MODEL_REGISTRY.get(version)
    if not record:
        raise HTTPException(status_code=404, detail="model version not found")
    artifact = load_model_artifact(record.get("artifact_path", ""))
    if not artifact:
        raise HTTPException(status_code=404, detail="model artifact missing")
    quality_gate = quality_gate_for_record(record)
    if not quality_gate.get("passed"):
        reasons = quality_gate.get("reasons") or []
        reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "unknown gate failure"
        raise HTTPException(status_code=409, detail=f"quality gate failed: {reason_text}")
    active_version = MODEL_STATE.get("active_version")
    active_record = (
        MODEL_REGISTRY.get(active_version)
        if active_version and active_version != version
        else None
    )
    promotion_gate = evaluate_promotion_gate(record, active_record)
    if not promotion_gate.get("passed"):
        reasons = promotion_gate.get("reasons") or []
        reason_text = ", ".join(str(reason) for reason in reasons) if reasons else "unknown promotion gate failure"
        raise HTTPException(status_code=409, detail=f"promotion gate failed: {reason_text}")
    MODEL_STATE["active_version"] = version
    refresh_active_model_cache()
    return {
        "activated": True,
        "version": version,
        "quality_gate": quality_gate,
        "promotion_gate": promotion_gate,
    }


def current_owner_id(request: Request) -> str:
    sid = get_session_id(request)
    if sid:
        session = STORE.get_session(sid)
        if session and session.get("user_id"):
            return str(session["user_id"])
    return "anonymous"


def owner_feedback_offsets(owner_id: str) -> dict[str, float]:
    return FEEDBACK_WEIGHT_OFFSETS.get(owner_id, {})


def update_feedback_offsets(owner_id: str, feature: str, rating: int) -> None:
    key_map = {
        "bpm": "bpm",
        "key": "key",
        "energy": "energy",
        "valence": "valence",
        "dance": "dance",
        "danceability": "dance",
        "loudness": "loudness",
    }
    mapped = key_map.get(feature)
    if not mapped:
        return
    direction = 1.0 if rating > 0 else -1.0
    step = min(0.05, abs(rating) * 0.02)
    offsets = FEEDBACK_WEIGHT_OFFSETS.setdefault(owner_id, {})
    offsets[mapped] = max(-0.25, min(0.25, offsets.get(mapped, 0.0) + direction * step))


def idempotency_record_key(owner_id: str, action: str, key: str) -> str:
    return f"{owner_id}:{action}:{key}"


def get_idempotency_response(owner_id: str, action: str, request: Request) -> Optional[dict]:
    key = request.headers.get("Idempotency-Key")
    if not key:
        return None
    return IDEMPOTENCY_STORE.get(idempotency_record_key(owner_id, action, key))


def set_idempotency_response(owner_id: str, action: str, request: Request, response_payload: dict) -> None:
    key = request.headers.get("Idempotency-Key")
    if not key:
        return
    IDEMPOTENCY_STORE[idempotency_record_key(owner_id, action, key)] = {
        "payload": response_payload,
        "created_at": time.time(),
    }


def fetch_playlist_track_ids(sp: spotipy.Spotify, playlist_id: str) -> list[str]:
    results = spotify_call_with_retry(
        sp.playlist_items,
        playlist_id,
        fields="items(track(id)),next",
        additional_types=["track"],
        limit=100,
    )
    track_ids: list[str] = []
    while results:
        for item in results.get("items", []):
            track = item.get("track")
            if track and track.get("id"):
                track_ids.append(track["id"])
        if results.get("next"):
            results = spotify_call_with_retry(sp.next, results)
        else:
            break
    return track_ids


def create_playlist_with_items(
    sp: spotipy.Spotify,
    name: str,
    public: bool,
    track_ids: list[str],
    description: str,
) -> str:
    playlist_id = spotify_call_with_retry(
        sp.user_playlist_create,
        spotify_call_with_retry(sp.current_user)["id"],
        name=name,
        public=public,
        description=description,
    )["id"]
    for i in range(0, len(track_ids), 100):
        spotify_call_with_retry(sp.playlist_add_items, playlist_id, track_ids[i : i + 100])
    return playlist_id


def cron_field_matches(value: int, field: str) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(cron_field_matches(value, part.strip()) for part in field.split(","))
    if "-" in field:
        left, right = field.split("-", 1)
        return int(left) <= value <= int(right)
    return value == int(field)


def cron_matches_now(cron_expr: str, now: datetime) -> bool:
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        return False
    minute, hour, day, month, weekday = parts
    return (
        cron_field_matches(now.minute, minute)
        and cron_field_matches(now.hour, hour)
        and cron_field_matches(now.day, day)
        and cron_field_matches(now.month, month)
        and cron_field_matches((now.weekday() + 1) % 7, weekday)
    )


def serialize_mood_curve_points(points: Optional[list[MoodPoint]]) -> list[dict]:
    rows: list[dict] = []
    for point in points or []:
        if hasattr(point, "model_dump"):
            rows.append(point.model_dump())
        elif isinstance(point, dict):
            rows.append(point)
    return rows


def run_optimize_tracks_for_payload(
    sp: spotipy.Spotify,
    payload: OptimizeRequest,
    *,
    seed: int,
    feedback_offsets: dict[str, float],
    model_payload: dict[str, object],
    cache_path: str,
    source_playlist_id: Optional[str] = None,
) -> tuple[str, str, list, float, list[dict], list[dict], dict]:
    source_playlist_id = source_playlist_id or parse_playlist_id(payload.playlist)
    weights = payload.weights.model_dump() if payload.weights else {}

    playlist_name, ordered_tracks, cost, roughest, explainability, solver_diagnostics = optimize_tracks(
        sp=sp,
        playlist_id=source_playlist_id,
        cache_path=cache_path,
        weights=weights,
        bpm_window=payload.bpm_window,
        restarts=payload.restarts,
        two_opt_passes=payload.two_opt_passes,
        missing=payload.missing,
        seed=seed,
        mix_mode=payload.mix_mode,
        flow_curve=payload.flow_curve,
        flow_profile=payload.flow_profile,
        key_lock_window=payload.key_lock_window,
        tempo_ramp_weight=payload.tempo_ramp_weight,
        minimax_passes=payload.minimax_passes,
        locked_first_track_id=payload.locked_first_track_id,
        locked_last_track_id=payload.locked_last_track_id,
        locked_blocks=payload.locked_blocks,
        artist_gap=payload.artist_gap,
        album_gap=payload.album_gap,
        explicit_mode=payload.explicit_mode,
        duration_target_sec=payload.duration_target_sec,
        duration_tolerance_sec=payload.duration_tolerance_sec,
        genre_cluster_strength=payload.genre_cluster_strength,
        mood_curve_points=serialize_mood_curve_points(payload.mood_curve_points),
        bpm_guardrails=payload.bpm_guardrails or [],
        harmonic_strict=payload.harmonic_strict,
        feedback_offsets=feedback_offsets,
        smoothness_weight=payload.smoothness_weight,
        variety_weight=payload.variety_weight,
        max_bpm_jump=payload.max_bpm_jump,
        min_key_compatibility=payload.min_key_compatibility,
        no_repeat_artist_within=payload.no_repeat_artist_within,
        solver_mode=payload.solver_mode,
        beam_width=payload.beam_width,
        anneal_steps=payload.anneal_steps,
        anneal_temp_start=payload.anneal_temp_start,
        anneal_temp_end=payload.anneal_temp_end,
        max_solver_ms=payload.max_solver_ms,
        lookahead_horizon=payload.lookahead_horizon,
        lookahead_decay=payload.lookahead_decay,
        model_weights=model_payload.get("weights"),
        model_bias=float(model_payload.get("bias") or 0.0),
        model_alpha=float(model_payload.get("alpha") or 0.0),
        model_version=model_payload.get("version"),
        transition_log_path=TRANSITION_LOG_PATH,
    )
    return source_playlist_id, playlist_name, ordered_tracks, cost, roughest, explainability, solver_diagnostics


def run_batch_optimization(sp: spotipy.Spotify, owner_id: str, payload: BatchRequest, batch_source: str) -> tuple[str, dict]:
    batch_id = uuid.uuid4().hex
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    options = payload.options or {}
    feedback_offsets = owner_feedback_offsets(owner_id)
    model_payload = active_model_payload()
    items = []

    for index, playlist in enumerate(payload.playlists):
        try:
            cfg = dict(options)
            cfg.pop("playlist", None)
            cfg_payload = OptimizeRequest(playlist=playlist, **cfg)
            cfg_payload = resolve_optimize_payload(cfg_payload)
            source_playlist_id, playlist_name, ordered_tracks, cost, roughest, explainability, solver_diagnostics = run_optimize_tracks_for_payload(
                sp=sp,
                payload=cfg_payload,
                seed=44 + index,
                feedback_offsets=feedback_offsets,
                model_payload=model_payload,
                cache_path=cache_path,
            )
            config_hash = build_optimize_config_hash(cfg_payload)

            base_name = cfg_payload.name or playlist_name or f"Batch {index+1}"
            if payload.name_prefix:
                base_name = f"{payload.name_prefix}_{index+1}_{base_name}"
            new_name = optimized_name(base_name)
            ordered_ids = [track.id for track in ordered_tracks]
            output_playlist_id = create_playlist_with_items(
                sp=sp,
                name=new_name,
                public=payload.public,
                track_ids=ordered_ids,
                description=f"{batch_source} optimized item {index+1}",
            )

            run_id = uuid.uuid4().hex
            RUN_HISTORY[run_id] = {
                "source_playlist_id": source_playlist_id,
                "playlist_id": output_playlist_id,
                "playlist_name": playlist_name,
                "transition_score": round(cost, 4),
                "roughest": roughest,
                "transitions": explainability,
                "request": cfg_payload.model_dump(),
                "config_hash": config_hash,
                "public": payload.public,
                "batch_id": batch_id,
                "model_version": model_payload.get("version"),
                "model_alpha": model_payload.get("alpha"),
                "solver_diagnostics": solver_diagnostics,
                "created_at": time.time(),
            }
            items.append(
                {
                    "index": index,
                    "status": "completed",
                    "source_playlist_id": source_playlist_id,
                    "run_id": run_id,
                    "playlist_id": output_playlist_id,
                    "playlist_name": new_name,
                    "playlist_url": f"https://open.spotify.com/playlist/{output_playlist_id}",
                    "transition_score": round(cost, 4),
                    "model_version": model_payload.get("version"),
                    "config_hash": config_hash,
                    "solver_diagnostics": solver_diagnostics,
                }
            )
        except Exception as exc:
            items.append(
                {
                    "index": index,
                    "status": "failed",
                    "source_playlist_id": parse_playlist_id(playlist),
                    "error": str(exc),
                }
            )

    batch_record = {
        "owner_id": owner_id,
        "status": "completed",
        "source": batch_source,
        "created_at": time.time(),
        "updated_at": time.time(),
        "total": len(payload.playlists),
        "completed": sum(1 for item in items if item.get("status") == "completed"),
        "failed": sum(1 for item in items if item.get("status") == "failed"),
        "items": items,
    }
    BATCH_STORE[batch_id] = batch_record
    return batch_id, batch_record


def emit_run_event(run_id: str, event: str, progress: int, message: str, data: Optional[dict] = None) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "progress": max(0, min(100, progress)),
        "message": message,
        "data": data or {},
    }
    EVENT_STORE.append(run_id, payload)
    RUN_TASK_STATUS.setdefault(run_id, {})
    RUN_TASK_STATUS[run_id]["progress"] = max(0, min(100, progress))


def run_single_optimization(
    sp: spotipy.Spotify,
    owner_id: str,
    payload: OptimizeRequest,
    seed: int,
    parent_run_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    run_id = run_id or uuid.uuid4().hex
    emit_run_event(run_id, "start", 2, "Starting optimization")
    config_hash = build_optimize_config_hash(payload)

    feedback_offsets = owner_feedback_offsets(owner_id)
    model_payload = active_model_payload()
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    source_playlist_id = parse_playlist_id(payload.playlist)
    source_track_ids = fetch_playlist_track_ids(sp, source_playlist_id)
    emit_run_event(run_id, "ingest", 15, "Fetched playlist tracks", {"count": len(source_track_ids)})

    source_playlist_id, playlist_name, ordered_tracks, cost, roughest, explainability, solver_diagnostics = run_optimize_tracks_for_payload(
        sp=sp,
        payload=payload,
        seed=seed,
        feedback_offsets=feedback_offsets,
        model_payload=model_payload,
        cache_path=cache_path,
        source_playlist_id=source_playlist_id,
    )
    emit_run_event(run_id, "search", 65, "Optimization complete", {"transitions": len(explainability)})

    base_name = payload.name or playlist_name or "Playlist"
    new_name = optimized_name(base_name)
    ordered_ids = [track.id for track in ordered_tracks]
    playlist_id = create_playlist_with_items(
        sp=sp,
        name=new_name,
        public=payload.public,
        track_ids=ordered_ids,
        description="Optimized playlist order",
    )
    emit_run_event(run_id, "publish", 90, "Created optimized playlist", {"playlist_id": playlist_id})

    snapshot_id = uuid.uuid4().hex
    SNAPSHOT_STORE[snapshot_id] = {
        "owner_id": owner_id,
        "source_playlist_id": source_playlist_id,
        "source_playlist_name": playlist_name,
        "source_track_ids": source_track_ids,
        "optimized_track_ids": ordered_ids,
        "output_playlist_id": playlist_id,
        "note": "auto_snapshot",
        "created_at": time.time(),
    }

    RUN_HISTORY[run_id] = {
        "source_playlist_id": source_playlist_id,
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "transition_score": round(cost, 4),
        "roughest": roughest,
        "transitions": explainability,
        "request": payload.model_dump(),
        "config_hash": config_hash,
        "public": payload.public,
        "snapshot_id": snapshot_id,
        "parent_run_id": parent_run_id,
        "model_version": model_payload.get("version"),
        "model_alpha": model_payload.get("alpha"),
        "solver_diagnostics": solver_diagnostics,
        "created_at": time.time(),
    }

    emit_run_event(run_id, "done", 100, "Run completed", {"snapshot_id": snapshot_id})
    response_payload = {
        "run_id": run_id,
        "playlist_id": playlist_id,
        "playlist_name": new_name,
        "playlist_url": f"https://open.spotify.com/playlist/{playlist_id}",
        "transition_score": round(cost, 4),
        "roughest": roughest,
        "model_version": model_payload.get("version"),
        "solver_diagnostics": solver_diagnostics,
    }
    if OPTIMIZE_CONFIG_HASH_DEBUG:
        response_payload["config_hash"] = config_hash
    return response_payload


@app.post("/presets")
def create_preset(request: Request, payload: PresetRequest):
    preset_id = uuid.uuid4().hex
    owner_id = current_owner_id(request)
    PRESET_STORE[preset_id] = {
        "owner_id": owner_id,
        "name": payload.name,
        "config": payload.config,
        "schema_version": payload.schema_version,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return {"preset_id": preset_id, **PRESET_STORE[preset_id]}


@app.get("/presets")
def list_presets(request: Request):
    owner_id = current_owner_id(request)
    rows = []
    for preset_id, value in PRESET_STORE.items():
        if value.get("owner_id") != owner_id:
            continue
        rows.append({"preset_id": preset_id, **value})
    rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return {"items": rows}


@app.patch("/presets/{preset_id}")
def patch_preset(request: Request, preset_id: str, payload: PresetPatchRequest):
    preset = PRESET_STORE.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="preset not found")
    owner_id = current_owner_id(request)
    if preset.get("owner_id") != owner_id:
        raise HTTPException(status_code=403, detail="forbidden")
    if payload.name is not None:
        preset["name"] = payload.name
    if payload.config is not None:
        preset["config"] = payload.config
    if payload.schema_version is not None:
        preset["schema_version"] = payload.schema_version
    preset["updated_at"] = time.time()
    return {"preset_id": preset_id, **preset}


@app.delete("/presets/{preset_id}")
def delete_preset(request: Request, preset_id: str):
    preset = PRESET_STORE.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="preset not found")
    owner_id = current_owner_id(request)
    if preset.get("owner_id") != owner_id:
        raise HTTPException(status_code=403, detail="forbidden")
    PRESET_STORE.pop(preset_id, None)
    return {"deleted": True, "preset_id": preset_id}


@app.post("/snapshots")
def create_snapshot(request: Request, payload: SnapshotCreateRequest):
    snapshot_id = uuid.uuid4().hex
    SNAPSHOT_STORE[snapshot_id] = {
        "owner_id": current_owner_id(request),
        "source_playlist_id": parse_playlist_id(payload.source_playlist_id),
        "source_playlist_name": payload.source_playlist_name,
        "source_track_ids": payload.source_track_ids,
        "optimized_track_ids": payload.optimized_track_ids or [],
        "output_playlist_id": None,
        "note": payload.note,
        "created_at": time.time(),
    }
    return {"snapshot_id": snapshot_id, **SNAPSHOT_STORE[snapshot_id]}


@app.get("/snapshots")
def list_snapshots(request: Request):
    owner_id = current_owner_id(request)
    rows = []
    for snapshot_id, value in SNAPSHOT_STORE.items():
        if value.get("owner_id") != owner_id:
            continue
        rows.append({"snapshot_id": snapshot_id, **value})
    rows.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return {"items": rows}


@app.post("/snapshots/{snapshot_id}/rollback")
def rollback_snapshot(request: Request, snapshot_id: str, payload: RollbackRequest):
    snapshot = SNAPSHOT_STORE.get(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="snapshot not found")
    if snapshot.get("owner_id") != current_owner_id(request):
        raise HTTPException(status_code=403, detail="forbidden")
    source_track_ids = snapshot.get("source_track_ids") or []
    if not source_track_ids:
        raise HTTPException(status_code=400, detail="snapshot has no source tracks")
    sp = spotify_for_session(request)
    base_name = payload.name or snapshot.get("source_playlist_name") or "Playlist"
    rollback_name = optimized_name(f"{base_name}_rollback")
    rollback_playlist_id = create_playlist_with_items(
        sp=sp,
        name=rollback_name,
        public=payload.public,
        track_ids=source_track_ids,
        description=f"Rollback from snapshot {snapshot_id}",
    )
    return {
        "snapshot_id": snapshot_id,
        "playlist_id": rollback_playlist_id,
        "playlist_name": rollback_name,
        "playlist_url": f"https://open.spotify.com/playlist/{rollback_playlist_id}",
    }


@app.post("/batch")
def create_batch(request: Request, payload: BatchRequest):
    if not payload.playlists:
        raise HTTPException(status_code=400, detail="playlists list cannot be empty")
    owner_id = current_owner_id(request)
    sp = spotify_for_session(request)
    batch_id, record = run_batch_optimization(sp, owner_id, payload, batch_source="manual")
    return {"batch_id": batch_id, **record}


@app.get("/batch/{batch_id}")
def get_batch(request: Request, batch_id: str):
    batch = BATCH_STORE.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="batch not found")
    if batch.get("owner_id") != current_owner_id(request):
        raise HTTPException(status_code=403, detail="forbidden")
    data = dict(batch)
    data.pop("items", None)
    return {"batch_id": batch_id, **data}


@app.get("/batch/{batch_id}/items")
def get_batch_items(request: Request, batch_id: str):
    batch = BATCH_STORE.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="batch not found")
    if batch.get("owner_id") != current_owner_id(request):
        raise HTTPException(status_code=403, detail="forbidden")
    return {"batch_id": batch_id, "items": batch.get("items", [])}


def spotify_for_sid(sid: str) -> Optional[spotipy.Spotify]:
    token_info = STORE.get_session(sid)
    if not token_info:
        return None
    oauth = build_oauth()
    if oauth.is_token_expired(token_info):
        if "refresh_token" not in token_info:
            return None
        refreshed = oauth.refresh_access_token(token_info["refresh_token"])
        if token_info.get("user_id"):
            refreshed["user_id"] = token_info.get("user_id")
        STORE.set_session(sid, refreshed)
        token_info = refreshed
    return spotipy.Spotify(auth=token_info["access_token"], requests_timeout=10)


def run_scheduled_job(schedule_id: str) -> None:
    schedule = SCHEDULE_STORE.get(schedule_id)
    if not schedule or not schedule.get("enabled"):
        return
    sid = schedule.get("session_id")
    if not sid:
        schedule["last_error"] = "missing session id"
        schedule["updated_at"] = time.time()
        return
    sp = spotify_for_sid(sid)
    if not sp:
        schedule["last_error"] = "session expired or missing"
        schedule["updated_at"] = time.time()
        return
    batch_payload = BatchRequest(**schedule["batch"])
    batch_id, _ = run_batch_optimization(sp, schedule["owner_id"], batch_payload, batch_source="scheduled")
    schedule["last_batch_id"] = batch_id
    schedule["last_run_at"] = datetime.now(timezone.utc).isoformat()
    schedule["last_error"] = None
    schedule["updated_at"] = time.time()


def scheduler_loop() -> None:
    last_cleanup = 0.0
    while not SCHEDULER_STOP.is_set():
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        current_tick = now.isoformat()
        maybe_auto_retrain_model()
        if time.time() - last_cleanup > 6 * 3600:
            cleanup_state_retention()
            last_cleanup = time.time()
        for schedule_id, schedule in list(SCHEDULE_STORE.items()):
            if not schedule.get("enabled"):
                continue
            if schedule.get("last_tick") == current_tick:
                continue
            schedule["last_tick"] = current_tick
            try:
                if cron_matches_now(schedule.get("cron", "* * * * *"), now):
                    run_scheduled_job(schedule_id)
            except Exception as exc:
                schedule["last_error"] = str(exc)
                schedule["updated_at"] = time.time()
        SCHEDULER_STOP.wait(20)


@app.on_event("startup")
def start_scheduler() -> None:
    cleanup_state_retention()
    refresh_active_model_cache()
    if getattr(app.state, "scheduler_thread", None):
        return
    thread = threading.Thread(target=scheduler_loop, daemon=True, name="spotify-optimizer-scheduler")
    app.state.scheduler_thread = thread
    thread.start()


@app.on_event("shutdown")
def stop_scheduler() -> None:
    SCHEDULER_STOP.set()


@app.post("/schedules")
def create_schedule(request: Request, payload: ScheduleRequest):
    sid = get_session_id(request)
    if not sid:
        raise HTTPException(status_code=401, detail="not authenticated")
    try:
        cron_matches_now(payload.cron, datetime.now(timezone.utc))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid cron expression")

    schedule_id = uuid.uuid4().hex
    SCHEDULE_STORE[schedule_id] = {
        "owner_id": current_owner_id(request),
        "session_id": sid,
        "cron": payload.cron,
        "enabled": payload.enabled,
        "batch": payload.batch.model_dump(),
        "last_run_at": None,
        "last_batch_id": None,
        "last_error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return {"schedule_id": schedule_id, **SCHEDULE_STORE[schedule_id]}


@app.get("/schedules")
def list_schedules(request: Request):
    owner_id = current_owner_id(request)
    rows = []
    for schedule_id, value in SCHEDULE_STORE.items():
        if value.get("owner_id") != owner_id:
            continue
        rows.append({"schedule_id": schedule_id, **value})
    rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return {"items": rows}


@app.patch("/schedules/{schedule_id}")
def patch_schedule(request: Request, schedule_id: str, payload: SchedulePatchRequest):
    schedule = SCHEDULE_STORE.get(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="schedule not found")
    if schedule.get("owner_id") != current_owner_id(request):
        raise HTTPException(status_code=403, detail="forbidden")
    if payload.cron is not None:
        try:
            cron_matches_now(payload.cron, datetime.now(timezone.utc))
        except Exception:
            raise HTTPException(status_code=400, detail="invalid cron expression")
        schedule["cron"] = payload.cron
    if payload.enabled is not None:
        schedule["enabled"] = payload.enabled
    if payload.batch is not None:
        schedule["batch"] = payload.batch.model_dump()
    schedule["updated_at"] = time.time()
    return {"schedule_id": schedule_id, **schedule}


@app.delete("/schedules/{schedule_id}")
def delete_schedule(request: Request, schedule_id: str):
    schedule = SCHEDULE_STORE.get(schedule_id)
    if not schedule:
        raise HTTPException(status_code=404, detail="schedule not found")
    if schedule.get("owner_id") != current_owner_id(request):
        raise HTTPException(status_code=403, detail="forbidden")
    SCHEDULE_STORE.pop(schedule_id, None)
    return {"deleted": True, "schedule_id": schedule_id}


@app.post("/optimize/{run_id}/quick-fix", response_model=OptimizeResponse)
def quick_fix_optimize(request: Request, run_id: str, payload: QuickFixRequest):
    run = RUN_HISTORY.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")

    raw_request = run.get("request")
    if not raw_request:
        raise HTTPException(status_code=400, detail="run has no replayable request payload")

    cfg = dict(raw_request)
    cfg["minimax_passes"] = int(cfg.get("minimax_passes", 2)) + payload.minimax_boost
    cfg["playlist"] = run.get("source_playlist_id")
    if payload.public is not None:
        cfg["public"] = payload.public
    replay = resolve_optimize_payload(OptimizeRequest(**cfg))
    config_hash = build_optimize_config_hash(replay)

    sp = spotify_for_session(request)
    owner_id = current_owner_id(request)
    feedback_offsets = owner_feedback_offsets(owner_id)
    model_payload = active_model_payload()
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    source_playlist_id, playlist_name, ordered_tracks, cost, roughest, explainability, solver_diagnostics = run_optimize_tracks_for_payload(
        sp=sp,
        payload=replay,
        seed=43,
        feedback_offsets=feedback_offsets,
        model_payload=model_payload,
        cache_path=cache_path,
    )

    base_name = replay.name or playlist_name or "Playlist"
    quick_name = optimized_name(f"{base_name}_quickfix")
    ordered_ids = [track.id for track in ordered_tracks]
    new_playlist_id = create_playlist_with_items(
        sp=sp,
        name=quick_name,
        public=replay.public,
        track_ids=ordered_ids,
        description=f"Quick-fix optimized from run {run_id}",
    )

    new_run_id = uuid.uuid4().hex
    RUN_HISTORY[new_run_id] = {
        "source_playlist_id": source_playlist_id,
        "playlist_id": new_playlist_id,
        "playlist_name": playlist_name,
        "transition_score": round(cost, 4),
        "roughest": roughest,
        "transitions": explainability,
        "request": replay.model_dump(),
        "config_hash": config_hash,
        "public": replay.public,
        "parent_run_id": run_id,
        "model_version": model_payload.get("version"),
        "model_alpha": model_payload.get("alpha"),
        "solver_diagnostics": solver_diagnostics,
        "created_at": time.time(),
    }

    response_payload = {
        "run_id": new_run_id,
        "playlist_id": new_playlist_id,
        "playlist_name": quick_name,
        "playlist_url": f"https://open.spotify.com/playlist/{new_playlist_id}",
        "transition_score": round(cost, 4),
        "roughest": roughest,
        "model_version": model_payload.get("version"),
        "solver_diagnostics": solver_diagnostics,
    }
    if OPTIMIZE_CONFIG_HASH_DEBUG:
        response_payload["config_hash"] = config_hash
    return response_payload
