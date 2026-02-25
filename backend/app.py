import json
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
from .state_store import DurableDict, DurableEventBuffer, SQLiteStateStore

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
TRANSITION_LOG_PATH = os.getenv("TRANSITION_LOG_PATH")
STATE_RETENTION_DAYS = int(os.getenv("STATE_RETENTION_DAYS", "30"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
SECURITY_HEADERS_ENABLED = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"


class WeightConfig(BaseModel):
    bpm: float = 0.32
    key: float = 0.28
    energy: float = 0.12
    valence: float = 0.06
    dance: float = 0.06


class MoodPoint(BaseModel):
    position: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)


class OptimizeRequest(BaseModel):
    playlist: str
    name: Optional[str] = None
    public: bool = False
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


class QuickFixRequest(BaseModel):
    minimax_boost: int = Field(2, ge=1, le=10)
    public: Optional[bool] = None


class CompareRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str


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


@app.post("/optimize", response_model=OptimizeResponse)
@limiter.limit(RATE_LIMIT_OPTIMIZE)
def optimize(request: Request, payload: OptimizeRequest):
    if payload.missing not in {"append", "drop"}:
        raise HTTPException(status_code=400, detail="missing must be append or drop")
    if payload.mix_mode not in {"balanced", "harmonic", "vibe"}:
        raise HTTPException(status_code=400, detail="mix_mode must be balanced, harmonic, or vibe")
    if payload.flow_profile not in {"peak", "gentle", "cooldown"}:
        raise HTTPException(status_code=400, detail="flow_profile must be peak, gentle, or cooldown")
    if payload.explicit_mode not in {"allow", "prefer_clean", "clean_only"}:
        raise HTTPException(status_code=400, detail="explicit_mode must be allow, prefer_clean, or clean_only")

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
        payload = OptimizeRequest(**payload_dict)
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
    return {
        "run_id": run_id,
        "playlist_id": run.get("playlist_id"),
        "playlist_name": run.get("playlist_name"),
        "transition_score": run.get("transition_score"),
        "transitions": run.get("transitions", []),
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
        "metrics": run_summary_metrics(run),
        "roughest": run.get("roughest", []),
        "top_transitions": top,
        "request": run.get("request", {}),
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
    COMPARISON_HISTORY[comparison_id] = {
        "baseline_run_id": payload.baseline_run_id,
        "candidate_run_id": payload.candidate_run_id,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta": delta,
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


def run_batch_optimization(sp: spotipy.Spotify, owner_id: str, payload: BatchRequest, batch_source: str) -> tuple[str, dict]:
    batch_id = uuid.uuid4().hex
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    options = payload.options or {}
    feedback_offsets = owner_feedback_offsets(owner_id)
    items = []

    for index, playlist in enumerate(payload.playlists):
        try:
            cfg = dict(options)
            cfg.pop("playlist", None)
            cfg_payload = OptimizeRequest(playlist=playlist, **cfg)
            weights = cfg_payload.weights.model_dump() if cfg_payload.weights else {}
            source_playlist_id = parse_playlist_id(cfg_payload.playlist)

            playlist_name, ordered_tracks, cost, roughest, explainability = optimize_tracks(
                sp=sp,
                playlist_id=source_playlist_id,
                cache_path=cache_path,
                weights=weights,
                bpm_window=cfg_payload.bpm_window,
                restarts=cfg_payload.restarts,
                two_opt_passes=cfg_payload.two_opt_passes,
                missing=cfg_payload.missing,
                seed=44 + index,
                mix_mode=cfg_payload.mix_mode,
                flow_curve=cfg_payload.flow_curve,
                flow_profile=cfg_payload.flow_profile,
                key_lock_window=cfg_payload.key_lock_window,
                tempo_ramp_weight=cfg_payload.tempo_ramp_weight,
                minimax_passes=cfg_payload.minimax_passes,
                locked_first_track_id=cfg_payload.locked_first_track_id,
                locked_last_track_id=cfg_payload.locked_last_track_id,
                locked_blocks=cfg_payload.locked_blocks,
                artist_gap=cfg_payload.artist_gap,
                album_gap=cfg_payload.album_gap,
                explicit_mode=cfg_payload.explicit_mode,
                duration_target_sec=cfg_payload.duration_target_sec,
                duration_tolerance_sec=cfg_payload.duration_tolerance_sec,
                genre_cluster_strength=cfg_payload.genre_cluster_strength,
                mood_curve_points=[point.model_dump() for point in cfg_payload.mood_curve_points or []],
                bpm_guardrails=cfg_payload.bpm_guardrails or [],
                harmonic_strict=cfg_payload.harmonic_strict,
                feedback_offsets=feedback_offsets,
                smoothness_weight=cfg_payload.smoothness_weight,
                variety_weight=cfg_payload.variety_weight,
                transition_log_path=TRANSITION_LOG_PATH,
            )

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
                "public": payload.public,
                "batch_id": batch_id,
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

    weights = payload.weights.model_dump() if payload.weights else {}
    feedback_offsets = owner_feedback_offsets(owner_id)
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    source_playlist_id = parse_playlist_id(payload.playlist)
    source_track_ids = fetch_playlist_track_ids(sp, source_playlist_id)
    emit_run_event(run_id, "ingest", 15, "Fetched playlist tracks", {"count": len(source_track_ids)})

    playlist_name, ordered_tracks, cost, roughest, explainability = optimize_tracks(
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
        mood_curve_points=[point.model_dump() for point in payload.mood_curve_points or []],
        bpm_guardrails=payload.bpm_guardrails or [],
        harmonic_strict=payload.harmonic_strict,
        feedback_offsets=feedback_offsets,
        smoothness_weight=payload.smoothness_weight,
        variety_weight=payload.variety_weight,
        transition_log_path=TRANSITION_LOG_PATH,
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
        "public": payload.public,
        "snapshot_id": snapshot_id,
        "parent_run_id": parent_run_id,
        "created_at": time.time(),
    }

    emit_run_event(run_id, "done", 100, "Run completed", {"snapshot_id": snapshot_id})
    return {
        "run_id": run_id,
        "playlist_id": playlist_id,
        "playlist_name": new_name,
        "playlist_url": f"https://open.spotify.com/playlist/{playlist_id}",
        "transition_score": round(cost, 4),
        "roughest": roughest,
    }


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
    replay = OptimizeRequest(**cfg)

    sp = spotify_for_session(request)
    owner_id = current_owner_id(request)
    feedback_offsets = owner_feedback_offsets(owner_id)
    weights = replay.weights.model_dump() if replay.weights else {}
    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    source_playlist_id = parse_playlist_id(replay.playlist)

    playlist_name, ordered_tracks, cost, roughest, explainability = optimize_tracks(
        sp=sp,
        playlist_id=source_playlist_id,
        cache_path=cache_path,
        weights=weights,
        bpm_window=replay.bpm_window,
        restarts=replay.restarts,
        two_opt_passes=replay.two_opt_passes,
        missing=replay.missing,
        seed=43,
        mix_mode=replay.mix_mode,
        flow_curve=replay.flow_curve,
        flow_profile=replay.flow_profile,
        key_lock_window=replay.key_lock_window,
        tempo_ramp_weight=replay.tempo_ramp_weight,
        minimax_passes=replay.minimax_passes,
        locked_first_track_id=replay.locked_first_track_id,
        locked_last_track_id=replay.locked_last_track_id,
        locked_blocks=replay.locked_blocks,
        artist_gap=replay.artist_gap,
        album_gap=replay.album_gap,
        explicit_mode=replay.explicit_mode,
        duration_target_sec=replay.duration_target_sec,
        duration_tolerance_sec=replay.duration_tolerance_sec,
        genre_cluster_strength=replay.genre_cluster_strength,
        mood_curve_points=[point.model_dump() for point in replay.mood_curve_points or []],
        bpm_guardrails=replay.bpm_guardrails or [],
        harmonic_strict=replay.harmonic_strict,
        feedback_offsets=feedback_offsets,
        smoothness_weight=replay.smoothness_weight,
        variety_weight=replay.variety_weight,
        transition_log_path=TRANSITION_LOG_PATH,
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
        "public": replay.public,
        "parent_run_id": run_id,
        "created_at": time.time(),
    }

    return {
        "run_id": new_run_id,
        "playlist_id": new_playlist_id,
        "playlist_name": quick_name,
        "playlist_url": f"https://open.spotify.com/playlist/{new_playlist_id}",
        "transition_score": round(cost, 4),
        "roughest": roughest,
    }
