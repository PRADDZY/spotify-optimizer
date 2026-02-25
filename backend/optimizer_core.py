
import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from spotipy.exceptions import SpotifyException

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CAMELOT_MAP: Dict[Tuple[str, int], Tuple[int, str]] = {
    ("C", 1): (8, "B"),
    ("C", 0): (5, "A"),
    ("C#", 1): (3, "B"),
    ("C#", 0): (12, "A"),
    ("D", 1): (10, "B"),
    ("D", 0): (7, "A"),
    ("D#", 1): (5, "B"),
    ("D#", 0): (2, "A"),
    ("E", 1): (12, "B"),
    ("E", 0): (9, "A"),
    ("F", 1): (7, "B"),
    ("F", 0): (4, "A"),
    ("F#", 1): (2, "B"),
    ("F#", 0): (11, "A"),
    ("G", 1): (9, "B"),
    ("G", 0): (6, "A"),
    ("G#", 1): (4, "B"),
    ("G#", 0): (1, "A"),
    ("A", 1): (11, "B"),
    ("A", 0): (8, "A"),
    ("A#", 1): (6, "B"),
    ("A#", 0): (3, "A"),
    ("B", 1): (1, "B"),
    ("B", 0): (10, "A"),
}

FEATURE_KEYS = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bpm": 0.32,
    "key": 0.28,
    "energy": 0.12,
    "valence": 0.06,
    "dance": 0.06,
    "loudness": 0.06,
    "acousticness": 0.03,
    "instrumentalness": 0.02,
    "speechiness": 0.02,
    "liveness": 0.01,
    "time_signature": 0.02,
}

HARMONIC_WEIGHTS: Dict[str, float] = {
    "bpm": 0.4,
    "key": 0.34,
    "energy": 0.08,
    "valence": 0.04,
    "dance": 0.05,
    "loudness": 0.04,
    "acousticness": 0.02,
    "instrumentalness": 0.01,
    "speechiness": 0.01,
    "liveness": 0.0,
    "time_signature": 0.01,
}

VIBE_WEIGHTS: Dict[str, float] = {
    "bpm": 0.24,
    "key": 0.18,
    "energy": 0.2,
    "valence": 0.12,
    "dance": 0.12,
    "loudness": 0.08,
    "acousticness": 0.03,
    "instrumentalness": 0.02,
    "speechiness": 0.02,
    "liveness": 0.0,
    "time_signature": 0.01,
}

FLOW_CURVE_WEIGHT = 0.2


@dataclass
class Track:
    id: str
    name: str
    artists: str
    features: Optional[Dict]
    artist_ids: List[str] = field(default_factory=list)
    album_id: Optional[str] = None
    explicit: bool = False
    duration_ms: int = 0
    genres: List[str] = field(default_factory=list)


@dataclass
class FeatureContext:
    scales: Dict[str, float]


@dataclass
class OptimizationConfig:
    flow_curve: bool = False
    flow_profile: str = "peak"
    flow_weight: float = FLOW_CURVE_WEIGHT
    key_lock_window: int = 3
    tempo_ramp_weight: float = 0.08
    minimax_passes: int = 2
    artist_gap: int = 0
    album_gap: int = 0
    explicit_mode: str = "allow"
    genre_cluster_strength: float = 0.0
    bpm_guardrails: List[float] = field(default_factory=list)
    harmonic_strict: bool = False
    smoothness_weight: float = 1.0
    variety_weight: float = 0.0
    max_bpm_jump: Optional[float] = None
    min_key_compatibility: Optional[float] = None
    no_repeat_artist_within: int = 0
    lookahead_horizon: int = 3
    lookahead_decay: float = 0.6


def parse_playlist_id(value: str) -> str:
    if value.startswith("spotify:playlist:"):
        return value.split(":")[-1]
    match = re.search(r"playlist/([a-zA-Z0-9]+)", value)
    if match:
        return match.group(1)
    return value


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_spotify_client(auth_mode: str, redirect_uri: str, scopes: str) -> spotipy.Spotify:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment.")

    if auth_mode == "app":
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    else:
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scopes,
            cache_path=".cache-spotify",
            open_browser=True,
        )

    return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10)


def spotify_call_with_retry(fn: Callable, *args, **kwargs):
    retries = max(1, int(os.getenv("SPOTIFY_API_RETRIES", "4")))
    backoff = max(0.1, float(os.getenv("SPOTIFY_API_BACKOFF", "0.8")))
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except SpotifyException as exc:
            status = getattr(exc, "http_status", None)
            if status not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            wait_time = backoff * (2 ** attempt) + random.uniform(0, 0.2)
            time.sleep(wait_time)
        except Exception:
            if attempt == retries - 1:
                raise
            wait_time = backoff * (2 ** attempt) + random.uniform(0, 0.2)
            time.sleep(wait_time)


def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> Tuple[str, List[Track]]:
    results = spotify_call_with_retry(
        sp.playlist_items,
        playlist_id,
        fields="items(track(id,name,artists(id,name),album(id),explicit,duration_ms),is_local),next,total",
        additional_types=["track"],
        limit=100,
    )

    tracks: List[Track] = []
    playlist_name = ""

    while results:
        for item in results.get("items", []):
            track = item.get("track")
            if item.get("is_local") or not track or not track.get("id"):
                continue
            artists_payload = track.get("artists", [])
            artists = ", ".join([a.get("name", "") for a in artists_payload])
            artist_ids = [a.get("id") for a in artists_payload if a.get("id")]
            tracks.append(
                Track(
                    id=track["id"],
                    name=track.get("name", ""),
                    artists=artists,
                    features=None,
                    artist_ids=artist_ids,
                    album_id=(track.get("album") or {}).get("id"),
                    explicit=bool(track.get("explicit", False)),
                    duration_ms=int(track.get("duration_ms") or 0),
                )
            )
        if results.get("next"):
            results = spotify_call_with_retry(sp.next, results)
        else:
            break

    try:
        playlist_name = spotify_call_with_retry(sp.playlist, playlist_id, fields="name").get("name", "")
    except Exception:
        playlist_name = ""

    return playlist_name, tracks


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def enrich_audio_features(sp: spotipy.Spotify, tracks: List[Track], cache_path: str) -> None:
    cache = load_json(cache_path)
    missing_ids = [t.id for t in tracks if t.id not in cache]

    for batch in chunked(missing_ids, 100):
        features = spotify_call_with_retry(sp.audio_features, batch)
        for track_id, feat in zip(batch, features):
            cache[track_id] = feat

    save_json(cache_path, cache)

    for track in tracks:
        track.features = cache.get(track.id)


def enrich_artist_genres(sp: spotipy.Spotify, tracks: List[Track], cache_path: str) -> None:
    cache = load_json(cache_path)
    artist_ids = sorted({artist_id for track in tracks for artist_id in track.artist_ids if artist_id})
    missing = [artist_id for artist_id in artist_ids if artist_id not in cache]

    for batch in chunked(missing, 50):
        artists_payload = spotify_call_with_retry(sp.artists, batch).get("artists", [])
        for artist in artists_payload:
            if not artist:
                continue
            cache[artist.get("id")] = artist.get("genres", [])

    save_json(cache_path, cache)

    for track in tracks:
        genres: List[str] = []
        for artist_id in track.artist_ids:
            genres.extend(cache.get(artist_id, []))
        track.genres = sorted(set(genres))


def track_has_essential_features(features: Optional[Dict]) -> bool:
    if not features:
        return False
    key = features.get("key")
    mode = features.get("mode")
    tempo = features.get("tempo")
    if key is None or key < 0:
        return False
    if mode is None:
        return False
    if tempo is None or tempo <= 0:
        return False
    return True


def key_name(key: Optional[int], mode: Optional[int]) -> str:
    if key is None or key < 0 or mode is None:
        return "?"
    name = KEY_NAMES[key % 12]
    return f"{name}{'m' if mode == 0 else ''}"


def camelot_of(key: Optional[int], mode: Optional[int]) -> Optional[Tuple[int, str]]:
    if key is None or key < 0 or mode is None:
        return None
    name = KEY_NAMES[key % 12]
    return CAMELOT_MAP.get((name, mode))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.5
    pct = max(0.0, min(1.0, pct))
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * pct
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_vals[lower]
    weight = idx - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def robust_scale(values: List[float], minimum: float = 0.05) -> float:
    if len(values) < 2:
        return 1.0
    p10 = percentile(values, 0.1)
    p90 = percentile(values, 0.9)
    p25 = percentile(values, 0.25)
    p75 = percentile(values, 0.75)
    iqr = p75 - p25
    spread = max(p90 - p10, iqr * 1.5)
    return max(minimum, spread)


def build_feature_context(tracks: List[Track]) -> FeatureContext:
    scales: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        values = []
        for track in tracks:
            features = track.features or {}
            val = features.get(key)
            if val is not None:
                values.append(float(val))
        if key == "loudness":
            scales[key] = robust_scale(values, minimum=1.5)
        else:
            scales[key] = robust_scale(values, minimum=0.05)
    return FeatureContext(scales=scales)


def build_energy_curve(energies: List[float], n: int, profile: str = "peak") -> List[float]:
    if n <= 0:
        return []
    if not energies:
        return [0.5 for _ in range(n)]

    if profile == "gentle":
        start = percentile(energies, 0.3)
        peak = percentile(energies, 0.7)
        end = percentile(energies, 0.45)
        peak_pos = 0.55
    elif profile == "cooldown":
        start = percentile(energies, 0.75)
        peak = percentile(energies, 0.8)
        end = percentile(energies, 0.3)
        peak_pos = 0.2
    else:
        start = percentile(energies, 0.2)
        peak = percentile(energies, 0.85)
        end = percentile(energies, 0.35)
        peak_pos = 0.6

    targets = []
    for i in range(n):
        pos = 0.0 if n == 1 else i / (n - 1)
        if pos <= peak_pos:
            t = pos / peak_pos if peak_pos > 0 else 0.0
            target = start + (peak - start) * t
        else:
            t = (pos - peak_pos) / (1 - peak_pos) if peak_pos < 1 else 0.0
            target = peak + (end - peak) * t
        targets.append(clamp01(target))

    return targets


def build_custom_curve(points: List[Dict[str, float]], n: int, fallback: List[float]) -> List[float]:
    if n <= 0:
        return []
    if not points:
        return fallback

    normalized: List[Tuple[float, float]] = []
    for point in points:
        try:
            x = clamp01(float(point.get("position", 0.0)))
            y = clamp01(float(point.get("energy", point.get("value", 0.5))))
            normalized.append((x, y))
        except Exception:
            continue

    if len(normalized) < 2:
        return fallback

    normalized.sort(key=lambda item: item[0])
    curve: List[float] = []
    for i in range(n):
        pos = 0.0 if n == 1 else i / (n - 1)
        left = normalized[0]
        right = normalized[-1]
        for j in range(len(normalized) - 1):
            a = normalized[j]
            b = normalized[j + 1]
            if a[0] <= pos <= b[0]:
                left, right = a, b
                break
        if abs(right[0] - left[0]) < 1e-9:
            value = left[1]
        else:
            t = (pos - left[0]) / (right[0] - left[0])
            value = left[1] + (right[1] - left[1]) * t
        curve.append(clamp01(value))
    return curve


def build_energy_order(tracks: List[Track], energy_targets: List[float]) -> List[int]:
    n = len(tracks)
    if n == 0:
        return []
    if not energy_targets:
        return list(range(n))

    def track_energy(idx: int) -> float:
        features = tracks[idx].features or {}
        energy = features.get("energy")
        return float(energy) if energy is not None else 0.5

    remaining = set(range(n))
    order: List[int] = []
    for target in energy_targets:
        pick = min(remaining, key=lambda i: abs(track_energy(i) - target))
        order.append(pick)
        remaining.remove(pick)

    return order


def build_cluster_seed_order(tracks: List[Track], flow_profile: str) -> List[int]:
    if not tracks:
        return []

    indexed = []
    for idx, track in enumerate(tracks):
        features = track.features or {}
        energy = features.get("energy")
        tempo = features.get("tempo")
        indexed.append(
            (
                idx,
                float(energy) if energy is not None else 0.5,
                float(tempo) if tempo is not None else 120.0,
            )
        )

    indexed.sort(key=lambda item: item[1])
    n = len(indexed)
    low = indexed[: max(1, n // 3)]
    mid = indexed[max(1, n // 3) : max(2, (2 * n) // 3)]
    high = indexed[max(2, (2 * n) // 3) :]

    for segment in (low, mid, high):
        segment.sort(key=lambda item: item[2])

    if flow_profile == "cooldown":
        ordered_segments = [high, mid, low]
    else:
        ordered_segments = [low, mid, high]

    return [item[0] for segment in ordered_segments for item in segment]


def build_genre_seed_order(tracks: List[Track]) -> List[int]:
    if not tracks:
        return []

    buckets: Dict[str, List[Tuple[int, float, float]]] = {}
    for idx, track in enumerate(tracks):
        genre = (track.genres[0] if track.genres else "unknown").lower()
        features = track.features or {}
        energy = float(features.get("energy") or 0.5)
        tempo = float(features.get("tempo") or 120.0)
        buckets.setdefault(genre, []).append((idx, energy, tempo))

    ordered_groups = sorted(
        buckets.items(),
        key=lambda item: sum(value[1] for value in item[1]) / max(1, len(item[1])),
    )

    order: List[int] = []
    for _, values in ordered_groups:
        values.sort(key=lambda item: item[2])
        order.extend(index for index, _, _ in values)
    return order


def bpm_distance(t1: Optional[float], t2: Optional[float], window: float) -> float:
    if t1 is None or t2 is None or t1 <= 0 or t2 <= 0:
        return 1.0

    ratio = t2 / t1
    if ratio <= 0:
        return 1.0

    # Log2 ratio handles half/double-time relationships naturally.
    log_ratio = math.log(ratio, 2)
    log_ratio -= round(log_ratio)
    diff = abs(log_ratio)

    if window <= 0:
        return clamp01(diff)

    scale = math.log(1 + window, 2)
    if scale <= 0:
        return clamp01(diff)

    return clamp01(diff / scale)


def key_distance(k1: Optional[int], m1: Optional[int], k2: Optional[int], m2: Optional[int]) -> float:
    c1 = camelot_of(k1, m1)
    c2 = camelot_of(k2, m2)
    if not c1 or not c2:
        return 1.0

    n1, l1 = c1
    n2, l2 = c2

    if n1 == n2 and l1 == l2:
        return 0.0
    if n1 == n2 and l1 != l2:
        return 0.18

    diff = abs((n1 - n2) % 12)
    diff = min(diff, 12 - diff)

    if diff == 1 and l1 == l2:
        return 0.28
    if diff == 1 and l1 != l2:
        return 0.45
    if diff == 2 and l1 == l2:
        return 0.62
    if diff == 2 and l1 != l2:
        return 0.78

    return 1.0


def loudness_distance(a: Optional[float], b: Optional[float], loudness_scale: float) -> float:
    if a is None or b is None:
        return 1.0
    return clamp01(abs(float(a) - float(b)) / max(loudness_scale, 1e-6))


def time_signature_distance(a: Optional[int], b: Optional[int]) -> float:
    if a is None or b is None:
        return 0.5
    return 0.0 if a == b else 0.4


def scaled_diff(a: Optional[float], b: Optional[float], scale: float, missing_penalty: float = 1.0) -> float:
    if a is None or b is None:
        return missing_penalty
    return clamp01(abs(float(a) - float(b)) / max(scale, 1e-6))


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return weights
    return {k: max(0.0, v) / total for k, v in weights.items()}


def resolve_weights(weights: Dict[str, float], mix_mode: str) -> Dict[str, float]:
    if mix_mode == "harmonic":
        base = HARMONIC_WEIGHTS
    elif mix_mode == "vibe":
        base = VIBE_WEIGHTS
    else:
        base = DEFAULT_WEIGHTS

    merged = base.copy()
    for key, value in weights.items():
        if value is not None:
            merged[key] = float(value)

    return normalize_weights(merged)


def apply_weight_offsets(weights: Dict[str, float], offsets: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not offsets:
        return weights
    adjusted = dict(weights)
    for key, delta in offsets.items():
        if key not in adjusted:
            continue
        adjusted[key] = max(0.0, adjusted[key] + float(delta))
    return normalize_weights(adjusted)


def track_distance(
    t1: Track,
    t2: Track,
    weights: Dict[str, float],
    bpm_window: float,
    context: FeatureContext,
) -> float:
    f1 = t1.features or {}
    f2 = t2.features or {}

    bpm_d = bpm_distance(f1.get("tempo"), f2.get("tempo"), bpm_window)
    key_d = key_distance(f1.get("key"), f1.get("mode"), f2.get("key"), f2.get("mode"))
    energy_d = scaled_diff(f1.get("energy"), f2.get("energy"), context.scales.get("energy", 0.1))
    valence_d = scaled_diff(f1.get("valence"), f2.get("valence"), context.scales.get("valence", 0.1))
    dance_d = scaled_diff(
        f1.get("danceability"),
        f2.get("danceability"),
        context.scales.get("danceability", 0.1),
    )
    acoustic_d = scaled_diff(
        f1.get("acousticness"),
        f2.get("acousticness"),
        context.scales.get("acousticness", 0.1),
    )
    instr_d = scaled_diff(
        f1.get("instrumentalness"),
        f2.get("instrumentalness"),
        context.scales.get("instrumentalness", 0.1),
    )
    speech_d = scaled_diff(
        f1.get("speechiness"),
        f2.get("speechiness"),
        context.scales.get("speechiness", 0.1),
    )
    live_d = scaled_diff(
        f1.get("liveness"),
        f2.get("liveness"),
        context.scales.get("liveness", 0.1),
    )
    loud_d = loudness_distance(
        f1.get("loudness"),
        f2.get("loudness"),
        context.scales.get("loudness", 6.0),
    )
    ts_d = time_signature_distance(f1.get("time_signature"), f2.get("time_signature"))

    score = (
        weights.get("bpm", 0.0) * bpm_d
        + weights.get("key", 0.0) * key_d
        + weights.get("energy", 0.0) * energy_d
        + weights.get("valence", 0.0) * valence_d
        + weights.get("dance", 0.0) * dance_d
        + weights.get("acousticness", 0.0) * acoustic_d
        + weights.get("instrumentalness", 0.0) * instr_d
        + weights.get("speechiness", 0.0) * speech_d
        + weights.get("liveness", 0.0) * live_d
        + weights.get("loudness", 0.0) * loud_d
        + weights.get("time_signature", 0.0) * ts_d
    )

    if bpm_d < 0.15 and key_d < 0.35:
        score *= 0.88
    if energy_d < 0.18 and loud_d < 0.22:
        score *= 0.93

    if bpm_d > 0.8:
        score += 0.08
    if key_d > 0.8:
        score += 0.06
    if energy_d > 0.75:
        score += 0.05
    if loud_d > 0.75:
        score += 0.05

    return clamp01(score)


def build_distance_matrix(
    tracks: List[Track],
    weights: Dict[str, float],
    bpm_window: float,
    context: FeatureContext,
) -> List[List[float]]:
    n = len(tracks)
    dist = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = track_distance(tracks[i], tracks[j], weights, bpm_window, context)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def total_cost(order: List[int], dist: List[List[float]]) -> float:
    if len(order) < 2:
        return 0.0
    return sum(dist[order[i]][order[i + 1]] for i in range(len(order) - 1))

def robust_bounds(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    low = percentile(values, 0.1)
    high = percentile(values, 0.9)
    if high - low < 1e-6:
        low = min(values)
        high = max(values)
    if high - low < 1e-6:
        high = low + 1.0
    return (low, high)


def to_unit(value: Optional[float], bounds: Tuple[float, float], default: float = 0.5) -> float:
    if value is None:
        return default
    low, high = bounds
    if high <= low:
        return default
    return clamp01((float(value) - low) / (high - low))


def build_tempo_curve(tempos: List[float], n: int, profile: str = "peak") -> List[float]:
    if n <= 0:
        return []
    if not tempos:
        return [0.5 for _ in range(n)]

    low, high = robust_bounds(tempos)
    norm_tempos = [to_unit(t, (low, high), 0.5) for t in tempos]
    return build_energy_curve(norm_tempos, n, profile=profile)


def key_lock_penalty(order: List[int], tracks: List[Track], window: int) -> float:
    if window <= 1 or len(order) < 3:
        return 0.0

    penalty = 0.0
    denom = 0.0
    max_step = max(2, window)

    for i in range(len(order)):
        fi = tracks[order[i]].features or {}
        for step in range(2, max_step + 1):
            j = i + step
            if j >= len(order):
                break
            fj = tracks[order[j]].features or {}
            kd = key_distance(fi.get("key"), fi.get("mode"), fj.get("key"), fj.get("mode"))
            weight = (max_step + 1 - step) / max_step
            penalty += kd * weight
            denom += weight

    if denom <= 0:
        return 0.0
    return penalty / denom


def repetition_gap_penalty(
    order: List[int],
    tracks: List[Track],
    gap: int,
    key_fn: Callable[[Track], Optional[str]],
) -> float:
    if gap <= 0 or len(order) < 2:
        return 0.0

    penalty = 0.0
    checks = 0.0
    for i in range(len(order)):
        left = tracks[order[i]]
        left_key = key_fn(left)
        if not left_key:
            continue
        for step in range(1, gap + 1):
            j = i + step
            if j >= len(order):
                break
            right_key = key_fn(tracks[order[j]])
            checks += 1.0
            if left_key == right_key:
                penalty += (gap + 1 - step) / (gap + 1)

    if checks <= 0:
        return 0.0
    return penalty / checks


def explicit_penalty(order: List[int], tracks: List[Track], mode: str) -> float:
    if mode != "prefer_clean" or not order:
        return 0.0
    explicit_count = sum(1 for idx in order if tracks[idx].explicit)
    return explicit_count / max(1, len(order))


def genre_switch_penalty(order: List[int], tracks: List[Track]) -> float:
    if len(order) < 2:
        return 0.0
    switches = 0
    total = 0
    for i in range(len(order) - 1):
        left = tracks[order[i]].genres[0] if tracks[order[i]].genres else "unknown"
        right = tracks[order[i + 1]].genres[0] if tracks[order[i + 1]].genres else "unknown"
        total += 1
        if left != right:
            switches += 1
    return switches / max(1, total)


def bpm_band_index(tempo: Optional[float], bands: List[float]) -> int:
    if tempo is None:
        return -1
    index = 0
    for band in bands:
        if tempo >= band:
            index += 1
        else:
            break
    return index


def bpm_guardrail_penalty(order: List[int], tracks: List[Track], bands: List[float]) -> float:
    if len(order) < 2 or not bands:
        return 0.0

    bands = sorted(float(b) for b in bands)
    penalty = 0.0
    checks = 0.0
    for i in range(len(order) - 1):
        left_tempo = (tracks[order[i]].features or {}).get("tempo")
        right_tempo = (tracks[order[i + 1]].features or {}).get("tempo")
        left_band = bpm_band_index(left_tempo, bands)
        right_band = bpm_band_index(right_tempo, bands)
        if left_band < 0 or right_band < 0:
            continue
        checks += 1.0
        jump = abs(left_band - right_band)
        if jump > 1:
            penalty += (jump - 1)

    if checks <= 0:
        return 0.0
    return penalty / checks


def harmonic_strict_penalty(order: List[int], tracks: List[Track]) -> float:
    if len(order) < 2:
        return 0.0
    violations = 0
    total = 0
    for i in range(len(order) - 1):
        left = tracks[order[i]].features or {}
        right = tracks[order[i + 1]].features or {}
        kd = key_distance(left.get("key"), left.get("mode"), right.get("key"), right.get("mode"))
        total += 1
        if kd > 0.45:
            violations += 1
    return violations / max(1, total)


def variety_penalty(order: List[int], tracks: List[Track]) -> float:
    if len(order) < 2:
        return 0.0
    same_artist_adjacent = 0.0
    same_genre_adjacent = 0.0
    total = 0.0
    for i in range(len(order) - 1):
        left = tracks[order[i]]
        right = tracks[order[i + 1]]
        total += 1.0
        left_artist = left.artist_ids[0] if left.artist_ids else left.artists
        right_artist = right.artist_ids[0] if right.artist_ids else right.artists
        if left_artist and right_artist and left_artist == right_artist:
            same_artist_adjacent += 1.0
        left_genre = left.genres[0] if left.genres else "unknown"
        right_genre = right.genres[0] if right.genres else "unknown"
        if left_genre == right_genre:
            same_genre_adjacent += 1.0
    return (same_artist_adjacent * 0.7 + same_genre_adjacent * 0.3) / max(1.0, total)


def hard_constraint_penalty(
    order: List[int],
    tracks: List[Track],
    max_bpm_jump: Optional[float],
    min_key_compatibility: Optional[float],
    no_repeat_artist_within: int,
) -> float:
    if len(order) < 2:
        return 0.0

    checks = 0.0
    violations = 0.0

    for i in range(len(order) - 1):
        left = tracks[order[i]]
        right = tracks[order[i + 1]]
        f1 = left.features or {}
        f2 = right.features or {}

        if max_bpm_jump is not None and max_bpm_jump > 0:
            tempo_a = f1.get("tempo")
            tempo_b = f2.get("tempo")
            if tempo_a is not None and tempo_b is not None:
                checks += 1.0
                jump = abs(float(tempo_a) - float(tempo_b))
                if jump > max_bpm_jump:
                    violations += (jump - max_bpm_jump) / max(1.0, max_bpm_jump)

        if min_key_compatibility is not None:
            checks += 1.0
            compatibility = 1.0 - key_distance(
                f1.get("key"),
                f1.get("mode"),
                f2.get("key"),
                f2.get("mode"),
            )
            if compatibility < min_key_compatibility:
                violations += (min_key_compatibility - compatibility) / max(0.05, min_key_compatibility)

    if no_repeat_artist_within > 0:
        for i in range(len(order)):
            left = tracks[order[i]]
            left_artist = left.artist_ids[0] if left.artist_ids else (left.artists or "")
            if not left_artist:
                continue
            for step in range(1, no_repeat_artist_within + 1):
                j = i + step
                if j >= len(order):
                    break
                right = tracks[order[j]]
                right_artist = right.artist_ids[0] if right.artist_ids else (right.artists or "")
                if not right_artist:
                    continue
                checks += 1.0
                if left_artist == right_artist:
                    violations += (no_repeat_artist_within + 1 - step) / (no_repeat_artist_within + 1)

    if checks <= 0:
        return 0.0
    return violations / checks


def lookahead_penalty(order: List[int], dist: List[List[float]], horizon: int, decay: float) -> float:
    if len(order) < 3 or horizon <= 1:
        return 0.0
    decay = min(0.99, max(0.05, float(decay)))
    horizon = max(2, int(horizon))

    weighted_sum = 0.0
    weight_total = 0.0
    for i in range(len(order) - 1):
        for step in range(2, horizon + 1):
            edge_pos = i + step - 1
            if edge_pos >= len(order) - 1:
                break
            weight = decay ** (step - 2)
            weighted_sum += weight * dist[order[edge_pos]][order[edge_pos + 1]]
            weight_total += weight

    if weight_total <= 1e-9:
        return 0.0
    return weighted_sum / weight_total


def order_cost(
    order: List[int],
    dist: List[List[float]],
    tracks: List[Track],
    energy_targets: Optional[List[float]],
    tempo_targets: Optional[List[float]],
    tempo_unit_values: Optional[List[float]],
    config: OptimizationConfig,
) -> float:
    cost = total_cost(order, dist)

    if energy_targets and config.flow_curve and config.flow_weight > 0:
        for pos, idx in enumerate(order):
            features = tracks[idx].features or {}
            energy = features.get("energy")
            if energy is None:
                energy = 0.5
            cost += config.flow_weight * abs(float(energy) - energy_targets[pos])

    if tempo_targets and tempo_unit_values and config.tempo_ramp_weight > 0:
        for pos, idx in enumerate(order):
            cost += config.tempo_ramp_weight * abs(tempo_unit_values[idx] - tempo_targets[pos])

    if config.key_lock_window > 1:
        cost += 0.12 * key_lock_penalty(order, tracks, config.key_lock_window)
    if config.artist_gap > 0:
        cost += 0.18 * repetition_gap_penalty(
            order,
            tracks,
            config.artist_gap,
            key_fn=lambda track: track.artist_ids[0] if track.artist_ids else (track.artists or None),
        )
    if config.album_gap > 0:
        cost += 0.14 * repetition_gap_penalty(
            order,
            tracks,
            config.album_gap,
            key_fn=lambda track: track.album_id,
        )
    if config.explicit_mode == "prefer_clean":
        cost += 0.08 * explicit_penalty(order, tracks, config.explicit_mode)
    if config.genre_cluster_strength > 0:
        cost += config.genre_cluster_strength * 0.12 * genre_switch_penalty(order, tracks)
    if config.bpm_guardrails:
        cost += 0.11 * bpm_guardrail_penalty(order, tracks, config.bpm_guardrails)
    if config.harmonic_strict:
        cost += 0.35 * harmonic_strict_penalty(order, tracks)
    if (
        config.max_bpm_jump is not None
        or config.min_key_compatibility is not None
        or config.no_repeat_artist_within > 0
    ):
        cost += 2.4 * hard_constraint_penalty(
            order,
            tracks,
            max_bpm_jump=config.max_bpm_jump,
            min_key_compatibility=config.min_key_compatibility,
            no_repeat_artist_within=config.no_repeat_artist_within,
        )
    if config.lookahead_horizon > 1:
        cost += 0.28 * lookahead_penalty(
            order,
            dist,
            horizon=config.lookahead_horizon,
            decay=config.lookahead_decay,
        )

    if config.variety_weight > 0:
        return config.smoothness_weight * cost + config.variety_weight * variety_penalty(order, tracks)
    return config.smoothness_weight * cost


def nearest_neighbor(dist: List[List[float]], start: int, rng: Optional[random.Random] = None, k: int = 3) -> List[int]:
    n = len(dist)
    unvisited = set(range(n))
    order = [start]
    unvisited.remove(start)

    while unvisited:
        last = order[-1]
        candidates = sorted(unvisited, key=lambda j: dist[last][j])
        if rng is None or k <= 1:
            next_idx = candidates[0]
        else:
            top = candidates[: min(k, len(candidates))]
            weights = [1.0 / (i + 1) for i in range(len(top))]
            next_idx = rng.choices(top, weights=weights, k=1)[0]
        order.append(next_idx)
        unvisited.remove(next_idx)

    return order


def two_opt(
    order: List[int],
    dist: List[List[float]],
    max_passes: int = 2,
    objective_fn: Optional[Callable[[List[int]], float]] = None,
) -> List[int]:
    n = len(order)
    if n < 4:
        return order

    if objective_fn is None:
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, n - 2):
                for k in range(i + 1, n - 1):
                    a, b = order[i - 1], order[i]
                    c, d = order[k], order[k + 1]
                    delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
                    if delta < -1e-9:
                        order[i : k + 1] = reversed(order[i : k + 1])
                        improved = True
            if not improved:
                break
        return order

    passes = 0
    improved = True
    best_cost = objective_fn(order)
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                order[i : k + 1] = reversed(order[i : k + 1])
                cost = objective_fn(order)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    improved = True
                    break
                order[i : k + 1] = reversed(order[i : k + 1])
            if improved:
                break

    return order


def best_relocation(
    order: List[int],
    dist: List[List[float]],
    seg_len: int,
    objective_fn: Optional[Callable[[List[int]], float]] = None,
) -> bool:
    n = len(order)
    if seg_len <= 0 or seg_len >= n:
        return False

    if objective_fn is None:
        best_delta = 0.0
        best_i = None
        best_j = None

        for i in range(0, n - seg_len + 1):
            segment = order[i : i + seg_len]
            first = segment[0]
            last = segment[-1]
            prev = order[i - 1] if i > 0 else None
            nxt = order[i + seg_len] if i + seg_len < n else None

            removal = 0.0
            if prev is not None:
                removal -= dist[prev][first]
            if nxt is not None:
                removal -= dist[last][nxt]
            if prev is not None and nxt is not None:
                removal += dist[prev][nxt]

            base = order[:i] + order[i + seg_len :]
            for j in range(0, len(base) + 1):
                before = base[j - 1] if j > 0 else None
                after = base[j] if j < len(base) else None

                insertion = 0.0
                if before is not None:
                    insertion -= dist[before][after] if after is not None else 0.0
                    insertion += dist[before][first]
                if after is not None:
                    insertion += dist[last][after]

                delta = removal + insertion
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i is None or best_j is None:
            return False

        segment = order[best_i : best_i + seg_len]
        remaining = order[:best_i] + order[best_i + seg_len :]
        order[:] = remaining[:best_j] + segment + remaining[best_j:]
        return True

    best_cost = objective_fn(order)
    best_order = None

    for i in range(0, n - seg_len + 1):
        segment = order[i : i + seg_len]
        base = order[:i] + order[i + seg_len :]
        for j in range(0, len(base) + 1):
            new_order = base[:j] + segment + base[j:]
            cost = objective_fn(new_order)
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_order = new_order

    if best_order is None:
        return False

    order[:] = best_order
    return True

def best_segment_reversal(
    order: List[int],
    objective_fn: Callable[[List[int]], float],
    max_len: int = 7,
) -> bool:
    n = len(order)
    if n < 4:
        return False

    best_cost = objective_fn(order)
    best_slice = None

    for seg_len in range(3, min(max_len, n) + 1):
        for i in range(0, n - seg_len + 1):
            j = i + seg_len
            candidate = order[:i] + list(reversed(order[i:j])) + order[j:]
            cost = objective_fn(candidate)
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_slice = (i, j)

    if not best_slice:
        return False

    i, j = best_slice
    order[i:j] = reversed(order[i:j])
    return True


def local_search(
    order: List[int],
    dist: List[List[float]],
    two_opt_passes: int,
    objective_fn: Optional[Callable[[List[int]], float]] = None,
) -> List[int]:
    passes = 0
    improved = True

    while improved and passes < 5:
        improved = False
        passes += 1

        before = objective_fn(order) if objective_fn else total_cost(order, dist)
        order = two_opt(order, dist, max_passes=two_opt_passes, objective_fn=objective_fn)
        after = objective_fn(order) if objective_fn else total_cost(order, dist)
        if after + 1e-9 < before:
            improved = True

        for seg_len in (1, 2, 3):
            if best_relocation(order, dist, seg_len, objective_fn=objective_fn):
                improved = True

        if objective_fn and best_segment_reversal(order, objective_fn):
            improved = True

    return order


def order_max_edge(order: List[int], dist: List[List[float]]) -> float:
    if len(order) < 2:
        return 0.0
    return max(dist[order[i]][order[i + 1]] for i in range(len(order) - 1))


def worst_edge_index(order: List[int], dist: List[List[float]]) -> int:
    if len(order) < 2:
        return 0
    return max(range(len(order) - 1), key=lambda i: dist[order[i]][order[i + 1]])


def is_better_lexicographic(
    max_edge: float,
    total: float,
    best_max_edge: float,
    best_total: float,
) -> bool:
    if max_edge < best_max_edge - 1e-9:
        return True
    if abs(max_edge - best_max_edge) <= 1e-9 and total < best_total - 1e-9:
        return True
    return False


def minimax_refine(
    order: List[int],
    dist: List[List[float]],
    objective_fn: Callable[[List[int]], float],
    passes: int,
) -> List[int]:
    n = len(order)
    if n < 4 or passes <= 0:
        return order

    for _ in range(passes):
        w = worst_edge_index(order, dist)
        current_total = objective_fn(order)
        current_max = order_max_edge(order, dist)

        best_order = None
        best_total = current_total
        best_max = current_max

        left = max(0, w - 3)
        right = min(n - 1, w + 4)

        # Reversal candidates around the roughest edge.
        for i in range(left, w + 1):
            for j in range(w + 1, right + 1):
                if j - i < 2:
                    continue
                candidate = list(order)
                candidate[i : j + 1] = reversed(candidate[i : j + 1])
                cand_max = order_max_edge(candidate, dist)
                cand_total = objective_fn(candidate)
                if is_better_lexicographic(cand_max, cand_total, best_max, best_total):
                    best_order = candidate
                    best_max = cand_max
                    best_total = cand_total

        # Relocate local nodes to a broader area.
        for pos in range(left, right + 1):
            node = order[pos]
            base = order[:pos] + order[pos + 1 :]
            for ins in range(0, len(base) + 1):
                if ins == pos or ins == pos + 1:
                    continue
                candidate = base[:ins] + [node] + base[ins:]
                cand_max = order_max_edge(candidate, dist)
                cand_total = objective_fn(candidate)
                if is_better_lexicographic(cand_max, cand_total, best_max, best_total):
                    best_order = candidate
                    best_max = cand_max
                    best_total = cand_total

        # Swap one side of the rough edge with nearby nodes.
        swap_targets = {w, w + 1}
        for a in swap_targets:
            if a < 0 or a >= n:
                continue
            for b in range(max(0, a - 6), min(n, a + 7)):
                if b == a:
                    continue
                candidate = list(order)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                cand_max = order_max_edge(candidate, dist)
                cand_total = objective_fn(candidate)
                if is_better_lexicographic(cand_max, cand_total, best_max, best_total):
                    best_order = candidate
                    best_max = cand_max
                    best_total = cand_total

        if best_order is None:
            break

        order = best_order

    return order


def pick_start_indices(dist: List[List[float]], tracks: List[Track], rng: random.Random, restarts: int) -> List[int]:
    n = len(dist)
    if n == 0:
        return []

    avg_dist = [sum(row) for row in dist]
    medoid = min(range(n), key=lambda i: avg_dist[i])
    starts = [medoid]

    def add_index(idx: Optional[int]) -> None:
        if idx is not None and idx not in starts:
            starts.append(idx)

    if tracks:
        tempos = [t.features.get("tempo") if t.features else None for t in tracks]
        energies = [t.features.get("energy") if t.features else None for t in tracks]

        tempo_pairs = [(i, v) for i, v in enumerate(tempos) if v is not None]
        energy_pairs = [(i, v) for i, v in enumerate(energies) if v is not None]

        if tempo_pairs:
            add_index(min(tempo_pairs, key=lambda x: x[1])[0])
            add_index(max(tempo_pairs, key=lambda x: x[1])[0])
        if energy_pairs:
            add_index(min(energy_pairs, key=lambda x: x[1])[0])
            add_index(max(energy_pairs, key=lambda x: x[1])[0])

    while len(starts) < max(1, restarts):
        idx = rng.randrange(n)
        if idx not in starts:
            starts.append(idx)

    return starts


def beam_search_order(dist: List[List[float]], start: int, width: int) -> List[int]:
    n = len(dist)
    if n == 0:
        return []
    width = max(1, width)
    beams: List[Tuple[List[int], set[int], float]] = [([start], set(range(n)) - {start}, 0.0)]

    while True:
        expanded: List[Tuple[List[int], set[int], float]] = []
        has_remaining = False
        for order, remaining, cost in beams:
            if not remaining:
                expanded.append((order, remaining, cost))
                continue
            has_remaining = True
            last = order[-1]
            ranked = sorted(remaining, key=lambda idx: dist[last][idx])[: max(2, width * 2)]
            for nxt in ranked:
                next_remaining = set(remaining)
                next_remaining.remove(nxt)
                expanded.append((order + [nxt], next_remaining, cost + dist[last][nxt]))

        if not has_remaining:
            break
        expanded.sort(key=lambda item: item[2])
        beams = expanded[:width]
        if not beams:
            break

    if not beams:
        return [start]
    best = min(beams, key=lambda item: item[2])
    return best[0]


def anneal_refine(
    order: List[int],
    objective_fn: Callable[[List[int]], float],
    rng: random.Random,
    steps: int,
    temp_start: float,
    temp_end: float,
) -> List[int]:
    if len(order) < 3 or steps <= 0:
        return order

    current = list(order)
    current_cost = objective_fn(current)
    best = list(current)
    best_cost = current_cost

    start_temp = max(float(temp_start), 1e-6)
    end_temp = max(float(temp_end), 1e-6)

    for step in range(steps):
        i, j = sorted(rng.sample(range(len(current)), 2))
        if i == j:
            continue

        candidate = list(current)
        if rng.random() < 0.55 and j - i >= 2:
            candidate[i : j + 1] = reversed(candidate[i : j + 1])
        else:
            candidate[i], candidate[j] = candidate[j], candidate[i]

        candidate_cost = objective_fn(candidate)
        progress = step / max(1, steps - 1)
        temp = start_temp + (end_temp - start_temp) * progress
        temp = max(end_temp, temp)
        delta = candidate_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost

    return best

def optimize_order(
    dist: List[List[float]],
    tracks: List[Track],
    restarts: int,
    seed: int,
    two_opt_passes: int,
    objective_fn: Callable[[List[int]], float],
    flow_profile: str,
    energy_targets: Optional[List[float]],
    minimax_passes: int,
    genre_cluster_strength: float = 0.0,
    solver_mode: str = "hybrid",
    beam_width: int = 8,
    anneal_steps: int = 140,
    anneal_temp_start: float = 0.08,
    anneal_temp_end: float = 0.004,
) -> Tuple[List[int], float]:
    n = len(dist)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    rng = random.Random(seed)
    best_order: Optional[List[int]] = None
    best_cost = math.inf

    starts = pick_start_indices(dist, tracks, rng, restarts)

    candidate_orders: List[List[int]] = []
    for idx, start in enumerate(starts):
        candidate_orders.append(nearest_neighbor(dist, start, rng if idx > 0 else None, k=4))

    if solver_mode == "hybrid":
        beam_starts = starts[: max(2, min(5, beam_width))]
        for start in beam_starts:
            candidate_orders.append(beam_search_order(dist, start, width=max(2, beam_width)))

    cluster_seed = build_cluster_seed_order(tracks, flow_profile)
    if cluster_seed:
        candidate_orders.append(cluster_seed)

    if genre_cluster_strength > 0:
        genre_seed = build_genre_seed_order(tracks)
        if genre_seed:
            candidate_orders.append(genre_seed)

    if energy_targets:
        energy_seed = build_energy_order(tracks, energy_targets)
        if energy_seed:
            candidate_orders.append(energy_seed)

    for order in candidate_orders:
        working = list(order)
        if solver_mode == "hybrid":
            working = anneal_refine(
                working,
                objective_fn=objective_fn,
                rng=rng,
                steps=max(0, anneal_steps),
                temp_start=max(1e-6, anneal_temp_start),
                temp_end=max(1e-6, anneal_temp_end),
            )

        working = local_search(working, dist, two_opt_passes=two_opt_passes, objective_fn=objective_fn)

        if minimax_passes > 0:
            working = minimax_refine(working, dist, objective_fn=objective_fn, passes=minimax_passes)
            working = local_search(working, dist, two_opt_passes=1, objective_fn=objective_fn)

        cost = objective_fn(working)
        if cost < best_cost:
            best_cost = cost
            best_order = list(working)

    if best_order is None:
        best_order = list(range(n))
        best_cost = objective_fn(best_order)

    return best_order, best_cost


def summarize_transitions(tracks: List[Track], dist: List[List[float]], order: List[int], limit: int = 5) -> List[Dict]:
    if len(order) < 2:
        return []

    scored = []
    for i in range(len(order) - 1):
        a = order[i]
        b = order[i + 1]
        fa = tracks[a].features or {}
        fb = tracks[b].features or {}
        scored.append(
            {
                "score": dist[a][b],
                "from": tracks[a].name,
                "to": tracks[b].name,
                "from_bpm": fa.get("tempo"),
                "to_bpm": fb.get("tempo"),
                "from_key": key_name(fa.get("key"), fa.get("mode")),
                "to_key": key_name(fb.get("key"), fb.get("mode")),
                "crossfade_seconds": recommend_crossfade_seconds(tracks[a], tracks[b]),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def apply_fixed_endpoints(
    order: List[int],
    tracks: List[Track],
    locked_first_track_id: Optional[str],
    locked_last_track_id: Optional[str],
) -> List[int]:
    if not order:
        return order

    by_id = {tracks[idx].id: idx for idx in order}
    fixed_order = list(order)

    first_idx = by_id.get(locked_first_track_id) if locked_first_track_id else None
    if first_idx is not None and first_idx in fixed_order:
        fixed_order.remove(first_idx)
        fixed_order.insert(0, first_idx)

    last_idx = by_id.get(locked_last_track_id) if locked_last_track_id else None
    if last_idx is not None and last_idx in fixed_order:
        fixed_order.remove(last_idx)
        fixed_order.append(last_idx)

    return fixed_order


def apply_locked_blocks(
    order: List[int],
    tracks: List[Track],
    locked_blocks: Optional[List[List[str]]],
) -> List[int]:
    if not order or not locked_blocks:
        return order

    idx_by_id = {tracks[idx].id: idx for idx in order}
    original_pos = {idx: pos for pos, idx in enumerate(order)}

    block_indexes: List[List[int]] = []
    blocked_set: set[int] = set()

    for block in locked_blocks:
        indexes: List[int] = []
        for track_id in block:
            idx = idx_by_id.get(track_id)
            if idx is None or idx in blocked_set:
                continue
            indexes.append(idx)
        if len(indexes) < 2:
            continue
        block_indexes.append(indexes)
        blocked_set.update(indexes)

    if not block_indexes:
        return order

    remaining = [idx for idx in order if idx not in blocked_set]
    merged = list(remaining)
    for block in block_indexes:
        anchor = min(original_pos[idx] for idx in block)
        insert_at = 0
        while insert_at < len(merged) and original_pos[merged[insert_at]] < anchor:
            insert_at += 1
        merged[insert_at:insert_at] = block

    return merged


def apply_explicit_filter(tracks: List[Track], explicit_mode: str) -> List[Track]:
    if explicit_mode != "clean_only":
        return tracks
    return [track for track in tracks if not track.explicit]


def apply_duration_target(
    tracks: List[Track],
    duration_target_sec: Optional[int],
    duration_tolerance_sec: int,
) -> List[Track]:
    if not tracks or not duration_target_sec or duration_target_sec <= 0:
        return tracks

    upper_bound = duration_target_sec + max(0, duration_tolerance_sec)
    current = list(tracks)
    total_sec = sum((track.duration_ms or 0) for track in current) / 1000.0
    if total_sec <= upper_bound:
        return current

    # Drop the longest tracks first to get near the upper bound while preserving order.
    indexed = list(enumerate(current))
    indexed.sort(key=lambda item: item[1].duration_ms or 0, reverse=True)
    dropped: set[int] = set()

    for index, track in indexed:
        if len(current) - len(dropped) <= 2:
            break
        if total_sec <= upper_bound:
            break
        total_sec -= (track.duration_ms or 0) / 1000.0
        dropped.add(index)

    return [track for i, track in enumerate(current) if i not in dropped]


def transition_record(from_track: Track, to_track: Track, score: float, index: int) -> Dict:
    from_features = from_track.features or {}
    to_features = to_track.features or {}
    return {
        "index": index,
        "score": round(float(score), 6),
        "from_track_id": from_track.id,
        "from_track": from_track.name,
        "to_track_id": to_track.id,
        "to_track": to_track.name,
        "from_tempo": from_features.get("tempo"),
        "to_tempo": to_features.get("tempo"),
        "from_key": key_name(from_features.get("key"), from_features.get("mode")),
        "to_key": key_name(to_features.get("key"), to_features.get("mode")),
        "from_energy": from_features.get("energy"),
        "to_energy": to_features.get("energy"),
    }


def transition_component_breakdown(
    from_track: Track,
    to_track: Track,
    weights: Dict[str, float],
    bpm_window: float,
    context: FeatureContext,
) -> Dict[str, float]:
    f1 = from_track.features or {}
    f2 = to_track.features or {}
    return {
        "bpm": weights.get("bpm", 0.0) * bpm_distance(f1.get("tempo"), f2.get("tempo"), bpm_window),
        "key": weights.get("key", 0.0) * key_distance(f1.get("key"), f1.get("mode"), f2.get("key"), f2.get("mode")),
        "energy": weights.get("energy", 0.0) * scaled_diff(f1.get("energy"), f2.get("energy"), context.scales.get("energy", 0.1)),
        "valence": weights.get("valence", 0.0) * scaled_diff(f1.get("valence"), f2.get("valence"), context.scales.get("valence", 0.1)),
        "dance": weights.get("dance", 0.0) * scaled_diff(f1.get("danceability"), f2.get("danceability"), context.scales.get("danceability", 0.1)),
        "loudness": weights.get("loudness", 0.0) * loudness_distance(f1.get("loudness"), f2.get("loudness"), context.scales.get("loudness", 6.0)),
    }


def normalize_component_contributions(components: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in components.values())
    if total <= 1e-9:
        return {key: 0.0 for key in components}
    return {key: round(max(0.0, float(value)) / total, 6) for key, value in components.items()}


def transition_reason_code(components: Dict[str, float]) -> str:
    if not components:
        return "balanced"
    top_metric = max(components.items(), key=lambda item: item[1])[0]
    mapping = {
        "bpm": "tempo_mismatch",
        "key": "harmonic_mismatch",
        "energy": "energy_shift",
        "valence": "mood_shift",
        "dance": "groove_shift",
        "loudness": "loudness_gap",
    }
    return mapping.get(top_metric, "mixed_factors")


def transition_reason(components: Dict[str, float]) -> str:
    if not components:
        return "Balanced transition profile."
    top_metric = max(components.items(), key=lambda item: item[1])[0]
    reasons = {
        "bpm": "Tempo mismatch drives this transition cost.",
        "key": "Harmonic/key mismatch is the main issue.",
        "energy": "Energy shift is the largest contributor.",
        "valence": "Mood/valence change contributes most.",
        "dance": "Danceability contrast is dominant.",
        "loudness": "Perceived loudness gap is dominant.",
    }
    return reasons.get(top_metric, "Multiple factors contribute to this transition.")


def recommend_crossfade_seconds(from_track: Track, to_track: Track) -> float:
    left = from_track.features or {}
    right = to_track.features or {}
    tempo_a = float(left.get("tempo") or 120.0)
    tempo_b = float(right.get("tempo") or 120.0)
    energy_a = float(left.get("energy") or 0.5)
    energy_b = float(right.get("energy") or 0.5)
    key_d = key_distance(left.get("key"), left.get("mode"), right.get("key"), right.get("mode"))
    tempo_ratio = abs(tempo_a - tempo_b) / max(1.0, max(tempo_a, tempo_b))
    energy_delta = abs(energy_a - energy_b)
    crossfade = 3.5 + tempo_ratio * 10.0 + energy_delta * 4.0 + key_d * 1.8
    return round(max(2.0, min(12.0, crossfade)), 1)


def build_transition_explainability(
    tracks: List[Track],
    order: List[int],
    dist: List[List[float]],
    weights: Dict[str, float],
    bpm_window: float,
    context: FeatureContext,
) -> List[Dict]:
    details: List[Dict] = []
    if len(order) < 2:
        return details

    for pos in range(len(order) - 1):
        left_idx = order[pos]
        right_idx = order[pos + 1]
        left_track = tracks[left_idx]
        right_track = tracks[right_idx]
        components = transition_component_breakdown(left_track, right_track, weights, bpm_window, context)
        dominant_component = max(components.items(), key=lambda item: item[1])[0] if components else None
        details.append(
            {
                **transition_record(left_track, right_track, dist[left_idx][right_idx], pos),
                "components": components,
                "component_share": normalize_component_contributions(components),
                "dominant_component": dominant_component,
                "reason_code": transition_reason_code(components),
                "reason": transition_reason(components),
                "crossfade_seconds": recommend_crossfade_seconds(left_track, right_track),
            }
        )
    return details


def append_transition_log(
    path: Optional[str],
    playlist_id: str,
    playlist_name: str,
    tracks: List[Track],
    dist: List[List[float]],
    order: List[int],
    score: float,
    config: OptimizationConfig,
    weights: Dict[str, float],
    bpm_window: float,
) -> None:
    if not path:
        return

    ensure_dir(os.path.dirname(path))

    transitions = []
    for i in range(len(order) - 1):
        a = order[i]
        b = order[i + 1]
        transitions.append(transition_record(tracks[a], tracks[b], dist[a][b], i))

    payload = {
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "track_count": len(order),
        "score": round(float(score), 6),
        "weights": weights,
        "bpm_window": bpm_window,
        "config": {
            "flow_curve": config.flow_curve,
            "flow_profile": config.flow_profile,
            "flow_weight": config.flow_weight,
            "key_lock_window": config.key_lock_window,
            "tempo_ramp_weight": config.tempo_ramp_weight,
            "minimax_passes": config.minimax_passes,
            "lookahead_horizon": config.lookahead_horizon,
            "lookahead_decay": config.lookahead_decay,
        },
        "transitions": transitions,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def optimized_name(base_name: str) -> str:
    base = base_name.strip() if base_name else "Optimized"
    if base.lower().endswith("_optimized"):
        return base
    return f"{base}_optimized"


def create_playlist(sp: spotipy.Spotify, name: str, track_ids: List[str], public: bool) -> str:
    user_id = spotify_call_with_retry(sp.current_user)["id"]
    playlist = spotify_call_with_retry(
        sp.user_playlist_create,
        user_id,
        name=name,
        public=public,
        description="Optimized playlist order",
    )
    playlist_id = playlist["id"]
    for batch in chunked(track_ids, 100):
        spotify_call_with_retry(sp.playlist_add_items, playlist_id, batch)
    return playlist_id


def optimize_tracks(
    sp: spotipy.Spotify,
    playlist_id: str,
    cache_path: str,
    weights: Dict[str, float],
    bpm_window: float,
    restarts: int,
    two_opt_passes: int,
    missing: str,
    seed: int,
    mix_mode: str = "balanced",
    flow_curve: bool = False,
    flow_profile: str = "peak",
    key_lock_window: int = 3,
    tempo_ramp_weight: float = 0.08,
    minimax_passes: int = 2,
    locked_first_track_id: Optional[str] = None,
    locked_last_track_id: Optional[str] = None,
    locked_blocks: Optional[List[List[str]]] = None,
    artist_gap: int = 0,
    album_gap: int = 0,
    explicit_mode: str = "allow",
    duration_target_sec: Optional[int] = None,
    duration_tolerance_sec: int = 90,
    genre_cluster_strength: float = 0.0,
    mood_curve_points: Optional[List[Dict[str, float]]] = None,
    bpm_guardrails: Optional[List[float]] = None,
    harmonic_strict: bool = False,
    feedback_offsets: Optional[Dict[str, float]] = None,
    smoothness_weight: float = 1.0,
    variety_weight: float = 0.0,
    max_bpm_jump: Optional[float] = None,
    min_key_compatibility: Optional[float] = None,
    no_repeat_artist_within: int = 0,
    solver_mode: str = "hybrid",
    beam_width: int = 8,
    anneal_steps: int = 140,
    anneal_temp_start: float = 0.08,
    anneal_temp_end: float = 0.004,
    lookahead_horizon: int = 3,
    lookahead_decay: float = 0.6,
    transition_log_path: Optional[str] = None,
) -> Tuple[str, List[Track], float, List[Dict], List[Dict]]:
    playlist_name, tracks = fetch_playlist_tracks(sp, playlist_id)
    if not tracks:
        raise RuntimeError("No playable tracks found in playlist.")

    enrich_audio_features(sp, tracks, cache_path)
    enrich_artist_genres(sp, tracks, cache_path.replace("audio_features.json", "artist_genres.json"))

    filtered_tracks = apply_explicit_filter(tracks, explicit_mode)
    filtered_tracks = apply_duration_target(
        filtered_tracks,
        duration_target_sec=duration_target_sec,
        duration_tolerance_sec=duration_tolerance_sec,
    )
    with_features = [t for t in filtered_tracks if track_has_essential_features(t.features)]
    missing_tracks = [t for t in filtered_tracks if not track_has_essential_features(t.features)]

    if not with_features:
        raise RuntimeError("No tracks had the required key/tempo features to optimize.")

    resolved_weights = resolve_weights(weights, mix_mode)
    resolved_weights = apply_weight_offsets(resolved_weights, feedback_offsets)
    context = build_feature_context(with_features)
    dist = build_distance_matrix(with_features, resolved_weights, bpm_window, context)

    config = OptimizationConfig(
        flow_curve=flow_curve,
        flow_profile=flow_profile,
        flow_weight=FLOW_CURVE_WEIGHT,
        key_lock_window=max(1, key_lock_window),
        tempo_ramp_weight=max(0.0, tempo_ramp_weight),
        minimax_passes=max(0, minimax_passes),
        artist_gap=max(0, artist_gap),
        album_gap=max(0, album_gap),
        explicit_mode=explicit_mode,
        genre_cluster_strength=max(0.0, genre_cluster_strength),
        bpm_guardrails=sorted([float(value) for value in (bpm_guardrails or []) if float(value) > 0]),
        harmonic_strict=bool(harmonic_strict),
        smoothness_weight=max(0.0, float(smoothness_weight)),
        variety_weight=max(0.0, float(variety_weight)),
        max_bpm_jump=float(max_bpm_jump) if max_bpm_jump is not None else None,
        min_key_compatibility=float(min_key_compatibility)
        if min_key_compatibility is not None
        else None,
        no_repeat_artist_within=max(0, int(no_repeat_artist_within)),
        lookahead_horizon=max(1, int(lookahead_horizon)),
        lookahead_decay=min(0.99, max(0.05, float(lookahead_decay))),
    )

    energy_targets: Optional[List[float]] = None
    if flow_curve:
        energies = [
            float(t.features.get("energy"))
            for t in with_features
            if t.features and t.features.get("energy") is not None
        ]
        energy_targets = build_energy_curve(energies, len(with_features), profile=flow_profile)
        energy_targets = build_custom_curve(mood_curve_points or [], len(with_features), energy_targets)

    tempo_targets: Optional[List[float]] = None
    tempo_unit_values: Optional[List[float]] = None
    if tempo_ramp_weight > 0:
        tempos = [
            float(t.features.get("tempo"))
            for t in with_features
            if t.features and t.features.get("tempo") is not None
        ]
        tempo_targets = build_tempo_curve(tempos, len(with_features), profile=flow_profile)
        tempo_targets = build_custom_curve(mood_curve_points or [], len(with_features), tempo_targets)
        bounds = robust_bounds(tempos)
        tempo_unit_values = [
            to_unit((t.features or {}).get("tempo"), bounds, default=0.5)
            for t in with_features
        ]

    objective_fn = lambda order: order_cost(
        order,
        dist,
        with_features,
        energy_targets,
        tempo_targets,
        tempo_unit_values,
        config,
    )

    order, cost = optimize_order(
        dist=dist,
        tracks=with_features,
        restarts=restarts,
        seed=seed,
        two_opt_passes=two_opt_passes,
        objective_fn=objective_fn,
        flow_profile=flow_profile,
        energy_targets=energy_targets,
        minimax_passes=config.minimax_passes,
        genre_cluster_strength=config.genre_cluster_strength,
        solver_mode=solver_mode,
        beam_width=max(1, int(beam_width)),
        anneal_steps=max(0, int(anneal_steps)),
        anneal_temp_start=max(1e-6, float(anneal_temp_start)),
        anneal_temp_end=max(1e-6, float(anneal_temp_end)),
    )

    order = apply_fixed_endpoints(
        order,
        with_features,
        locked_first_track_id=locked_first_track_id,
        locked_last_track_id=locked_last_track_id,
    )
    order = apply_locked_blocks(order, with_features, locked_blocks)
    cost = objective_fn(order)

    ordered_tracks = [with_features[i] for i in order]
    if missing == "append" and missing_tracks:
        ordered_tracks.extend(missing_tracks)

    roughest = summarize_transitions(with_features, dist, order, limit=5)
    explainability = build_transition_explainability(
        tracks=with_features,
        order=order,
        dist=dist,
        weights=resolved_weights,
        bpm_window=bpm_window,
        context=context,
    )

    append_transition_log(
        path=transition_log_path,
        playlist_id=playlist_id,
        playlist_name=playlist_name,
        tracks=with_features,
        dist=dist,
        order=order,
        score=cost,
        config=config,
        weights=resolved_weights,
        bpm_window=bpm_window,
    )

    return playlist_name, ordered_tracks, cost, roughest, explainability
