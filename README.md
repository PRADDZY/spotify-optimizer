# Spotify Mix Optimizer

A local web app + API that reorders a Spotify playlist for smoother transitions using BPM, musical key (Camelot wheel), energy, loudness, and vibe features. The optimizer **creates a new playlist** named `<name>_optimized`. You can choose harmonic/vibe/balanced mix presets, flow profile (`peak`, `gentle`, `cooldown`), key-lock windowing, tempo-ramp weighting, and minimax smoothing passes.

## Project Layout

- `backend/` FastAPI service + optimizer engine
- `frontend/` Next.js UI
- `optimizer.py` CLI (creates a new optimized playlist; no CSV output)

## Backend Setup

1. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Create `.env` in `backend/` (copy `backend/.env.example`):

```bash
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
SPOTIFY_REDIRECT_URI=http://localhost:8000/callback
FRONTEND_URLS=http://localhost:3000
FRONTEND_REDIRECT_URL=http://localhost:3000
ENV=development
SESSION_TTL_SECONDS=604800
STATE_TTL_SECONDS=600
SESSION_COOKIE_SECURE=false
SESSION_COOKIE_SAMESITE=lax
SESSION_COOKIE_DOMAIN=
# REDIS_URL=redis://localhost:6379/0
```

3. Run the API:

```bash
uvicorn backend.app:app --reload --port 8000
```

## Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

3. Run the UI:

```bash
npm run dev
```

Open `http://localhost:3000` and click **Connect Spotify**, then **Optimize**.

## CLI Usage (Optional)

Install root dependencies and run:

```bash
pip install -r requirements.txt
python optimizer.py optimize --playlist "https://open.spotify.com/playlist/PLAYLIST_ID"
```

Use `--dry-run` to avoid creating a playlist.

Optional flags:

```bash
python optimizer.py optimize \
  --playlist "https://open.spotify.com/playlist/PLAYLIST_ID" \
  --mix-mode harmonic \
  --flow-curve \
  --flow-profile peak \
  --key-lock-window 4 \
  --tempo-ramp-weight 0.12 \
  --minimax-passes 3 \
  --transition-log logs/transitions.jsonl
```

## Docker Compose (Local Prod-like)

Create `backend/.env` (see `backend/.env.example`), then run:

```bash
docker compose up --build
```

Frontend: `http://localhost:3000`  
Backend: `http://localhost:8000`

`docker-compose.yml` now persists backend runtime state and cache using named volumes (`backend_state`, `backend_cache`).

## Notes

- For production, set `ENV=production` and `SESSION_COOKIE_SECURE=true`, and provide a Redis URL to share sessions across instances.
- Rate limiting and JSON request logs are enabled by default. Adjust `RATE_LIMIT_*` and `LOG_*` in `backend/.env`.
- Set `TRANSITION_LOG_PATH` in `backend/.env` to append optimization transition diagnostics as JSONL.
- Runtime state is now persisted in SQLite (`STATE_DB_PATH`) so runs/presets/schedules/snapshots survive restarts.
- Old runtime records/events are pruned by retention policy (`STATE_RETENTION_DAYS`, default 30).
- Spotify API calls use automatic retry/backoff (`SPOTIFY_API_RETRIES`, `SPOTIFY_API_BACKOFF`) for transient failures.
- API guardrails include request size limits (`MAX_REQUEST_BYTES`) and default security headers (`SECURITY_HEADERS_ENABLED`).
- Use `Idempotency-Key` request header on `POST /optimize` and `POST /optimize/async` to avoid duplicate runs.
- `/ready` returns readiness and checks Redis connectivity when enabled.
- `/metrics` exposes Prometheus-compatible metrics.
- Spotify audio-features and audio-analysis endpoints are marked deprecated in their docs. If they are removed, you will need another feature source.
- Spotify's developer terms prohibit using Spotify content to train ML models; this tool is heuristic-based.

## VPS Deploy

See `deploy/README.md` for the VPS setup script and templates.
