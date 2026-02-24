import json
import logging
import os
import secrets
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pythonjsonlogger import jsonlogger
from redis import Redis
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .optimizer_core import (
    optimized_name,
    optimize_tracks,
    parse_playlist_id,
)

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


class WeightConfig(BaseModel):
    bpm: float = 0.32
    key: float = 0.28
    energy: float = 0.12
    valence: float = 0.06
    dance: float = 0.06


class OptimizeRequest(BaseModel):
    playlist: str
    name: Optional[str] = None
    public: bool = False
    mix_mode: str = "balanced"
    flow_curve: bool = False
    bpm_window: float = Field(0.08, ge=0.0, le=0.5)
    restarts: int = Field(12, ge=1, le=100)
    two_opt_passes: int = Field(2, ge=1, le=10)
    missing: str = "append"
    weights: Optional[WeightConfig] = None


class OptimizeResponse(BaseModel):
    playlist_id: str
    playlist_name: str
    playlist_url: str
    transition_score: float
    roughest: list


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

rate_limit_storage = os.getenv("RATE_LIMIT_REDIS_URL") or os.getenv("REDIS_URL") or "memory://"
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_GLOBAL], storage_uri=rate_limit_storage)

app = FastAPI(title="Spotify Mix Optimizer")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
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
    }
    rate_limit_ok = check_rate_limit_store()
    if rate_limit_ok is not None:
        checks["rate_limit"] = rate_limit_ok

    ok = all(checks.values())
    status = "ok" if ok else "degraded"
    code = 200 if ok else 503
    return JSONResponse(status_code=code, content={"status": status, "checks": checks})


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
    profile = sp.current_user()
    return {"id": profile.get("id"), "display_name": profile.get("display_name")}


@app.post("/optimize", response_model=OptimizeResponse)
@limiter.limit(RATE_LIMIT_OPTIMIZE)
def optimize(request: Request, payload: OptimizeRequest):
    if payload.missing not in {"append", "drop"}:
        raise HTTPException(status_code=400, detail="missing must be append or drop")
    if payload.mix_mode not in {"balanced", "harmonic", "vibe"}:
        raise HTTPException(status_code=400, detail="mix_mode must be balanced, harmonic, or vibe")

    sp = spotify_for_session(request)

    weights = payload.weights.model_dump() if payload.weights else {}

    cache_path = os.path.join(os.path.dirname(__file__), "cache", "audio_features.json")
    playlist_id = parse_playlist_id(payload.playlist)

    playlist_name, ordered_tracks, cost, roughest = optimize_tracks(
        sp=sp,
        playlist_id=playlist_id,
        cache_path=cache_path,
        weights=weights,
        bpm_window=payload.bpm_window,
        restarts=payload.restarts,
        two_opt_passes=payload.two_opt_passes,
        missing=payload.missing,
        seed=42,
        mix_mode=payload.mix_mode,
        flow_curve=payload.flow_curve,
    )

    base_name = payload.name or playlist_name or "Playlist"
    new_name = optimized_name(base_name)

    ordered_ids = [t.id for t in ordered_tracks]
    playlist_id = sp.user_playlist_create(
        sp.current_user()["id"],
        name=new_name,
        public=payload.public,
        description="Optimized playlist order",
    )["id"]

    for i in range(0, len(ordered_ids), 100):
        sp.playlist_add_items(playlist_id, ordered_ids[i : i + 100])

    return {
        "playlist_id": playlist_id,
        "playlist_name": new_name,
        "playlist_url": f"https://open.spotify.com/playlist/{playlist_id}",
        "transition_score": round(cost, 4),
        "roughest": roughest,
    }
