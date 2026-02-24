import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

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


def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_id: str) -> Tuple[str, List[Track]]:
    results = sp.playlist_items(
        playlist_id,
        fields="items(track(id,name,artists(name)),is_local),next,total",
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
            artists = ", ".join([a.get("name", "") for a in track.get("artists", [])])
            tracks.append(Track(id=track["id"], name=track.get("name", ""), artists=artists, features=None))
        if results.get("next"):
            results = sp.next(results)
        else:
            break

    try:
        playlist_name = sp.playlist(playlist_id, fields="name").get("name", "")
    except Exception:
        playlist_name = ""

    return playlist_name, tracks


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def enrich_audio_features(sp: spotipy.Spotify, tracks: List[Track], cache_path: str) -> None:
    cache = load_json(cache_path)
    missing_ids = [t.id for t in tracks if t.id not in cache]

    for batch in chunked(missing_ids, 100):
        features = sp.audio_features(batch)
        for track_id, feat in zip(batch, features):
            cache[track_id] = feat

    save_json(cache_path, cache)

    for track in tracks:
        track.features = cache.get(track.id)


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


def build_energy_curve(energies: List[float], n: int) -> List[float]:
    if n <= 0:
        return []
    if not energies:
        return [0.5 for _ in range(n)]

    low = percentile(energies, 0.2)
    peak = percentile(energies, 0.85)
    end = percentile(energies, 0.35)
    peak_pos = 0.6

    targets = []
    for i in range(n):
        pos = 0.0 if n == 1 else i / (n - 1)
        if pos <= peak_pos:
            t = pos / peak_pos if peak_pos > 0 else 0.0
            target = low + (peak - low) * t
        else:
            t = (pos - peak_pos) / (1 - peak_pos) if peak_pos < 1 else 0.0
            target = peak + (end - peak) * t
        targets.append(clamp01(target))

    return targets


def build_energy_order(tracks: List[Track], energy_targets: List[float]) -> List[int]:
    n = len(tracks)
    if n == 0:
        return []
    if not energy_targets:
        return list(range(n))

    def track_energy(idx: int) -> float:
        features = tracks[idx].features or {}
        energy = features.get("energy")
        return energy if energy is not None else 0.5

    remaining = set(range(n))
    order: List[int] = []
    for target in energy_targets:
        pick = min(remaining, key=lambda i: abs(track_energy(i) - target))
        order.append(pick)
        remaining.remove(pick)

    return order


def bpm_distance(t1: Optional[float], t2: Optional[float], window: float) -> float:
    if t1 is None or t2 is None or t1 <= 0 or t2 <= 0:
        return 1.0
    ratio = t2 / t1
    if ratio <= 0:
        return 1.0
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


def loudness_distance(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 1.0
    return clamp01(abs(a - b) / 10.0)


def time_signature_distance(a: Optional[int], b: Optional[int]) -> float:
    if a is None or b is None:
        return 0.5
    return 0.0 if a == b else 0.4


def safe_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 1.0
    return clamp01(abs(a - b))


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return weights
    return {k: max(0.0, v) / total for k, v in weights.items()}


def resolve_weights(weights: Dict[str, float], mix_mode: str) -> Dict[str, float]:
    base = DEFAULT_WEIGHTS
    if mix_mode == "harmonic":
        base = HARMONIC_WEIGHTS
    elif mix_mode == "vibe":
        base = VIBE_WEIGHTS

    merged = base.copy()
    for key, value in weights.items():
        if value is not None:
            merged[key] = value
    return normalize_weights(merged)


def track_distance(t1: Track, t2: Track, weights: Dict[str, float], bpm_window: float) -> float:
    f1 = t1.features or {}
    f2 = t2.features or {}
    bpm_d = bpm_distance(f1.get("tempo"), f2.get("tempo"), bpm_window)
    key_d = key_distance(f1.get("key"), f1.get("mode"), f2.get("key"), f2.get("mode"))
    energy_d = safe_diff(f1.get("energy"), f2.get("energy"))
    valence_d = safe_diff(f1.get("valence"), f2.get("valence"))
    dance_d = safe_diff(f1.get("danceability"), f2.get("danceability"))
    acoustic_d = safe_diff(f1.get("acousticness"), f2.get("acousticness"))
    instr_d = safe_diff(f1.get("instrumentalness"), f2.get("instrumentalness"))
    speech_d = safe_diff(f1.get("speechiness"), f2.get("speechiness"))
    live_d = safe_diff(f1.get("liveness"), f2.get("liveness"))
    loud_d = loudness_distance(f1.get("loudness"), f2.get("loudness"))
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
        score *= 0.9
    if energy_d < 0.1 and loud_d < 0.1:
        score *= 0.95

    if bpm_d > 0.8:
        score += 0.08
    if key_d > 0.8:
        score += 0.05
    if energy_d > 0.55:
        score += 0.05
    if loud_d > 0.7:
        score += 0.05
    if valence_d > 0.6:
        score += 0.03

    return clamp01(score)


def build_distance_matrix(tracks: List[Track], weights: Dict[str, float], bpm_window: float) -> List[List[float]]:
    n = len(tracks)
    dist = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = track_distance(tracks[i], tracks[j], weights, bpm_window)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def total_cost(order: List[int], dist: List[List[float]]) -> float:
    if len(order) < 2:
        return 0.0
    return sum(dist[order[i]][order[i + 1]] for i in range(len(order) - 1))


def order_cost(
    order: List[int],
    dist: List[List[float]],
    tracks: List[Track],
    energy_targets: Optional[List[float]],
    flow_weight: float,
) -> float:
    cost = total_cost(order, dist)
    if not energy_targets or flow_weight <= 0:
        return cost

    for pos, idx in enumerate(order):
        features = tracks[idx].features or {}
        energy = features.get("energy")
        if energy is None:
            energy = 0.5
        cost += flow_weight * abs(energy - energy_targets[pos])

    return cost


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
    cost_fn: Optional[Callable[[List[int]], float]] = None,
) -> List[int]:
    n = len(order)
    if n < 4:
        return order

    if cost_fn is None:
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
    best_cost = cost_fn(order)
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                order[i : k + 1] = reversed(order[i : k + 1])
                cost = cost_fn(order)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    improved = True
                    break
                order[i : k + 1] = reversed(order[i : k + 1])
            if improved:
                break
        if not improved:
            break
    return order


def best_relocation(
    order: List[int],
    dist: List[List[float]],
    seg_len: int,
    cost_fn: Optional[Callable[[List[int]], float]] = None,
) -> bool:
    n = len(order)
    if seg_len <= 0 or seg_len >= n:
        return False

    if cost_fn is None:
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
        new_order = remaining[:best_j] + segment + remaining[best_j:]
        order[:] = new_order
        return True

    best_cost = cost_fn(order)
    best_order = None
    for i in range(0, n - seg_len + 1):
        segment = order[i : i + seg_len]
        base = order[:i] + order[i + seg_len :]
        for j in range(0, len(base) + 1):
            new_order = base[:j] + segment + base[j:]
            cost = cost_fn(new_order)
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_order = new_order

    if best_order is None:
        return False

    order[:] = best_order
    return True


def local_search(
    order: List[int],
    dist: List[List[float]],
    two_opt_passes: int,
    cost_fn: Optional[Callable[[List[int]], float]] = None,
) -> List[int]:
    passes = 0
    improved = True
    while improved and passes < 4:
        improved = False
        passes += 1

        before = cost_fn(order) if cost_fn else total_cost(order, dist)
        order = two_opt(order, dist, max_passes=two_opt_passes, cost_fn=cost_fn)
        after = cost_fn(order) if cost_fn else total_cost(order, dist)
        if after + 1e-9 < before:
            improved = True

        for seg_len in (1, 2, 3):
            if best_relocation(order, dist, seg_len, cost_fn=cost_fn):
                improved = True

    return order


def pick_start_indices(dist: List[List[float]], tracks: List[Track], rng: random.Random, restarts: int) -> List[int]:
    n = len(dist)
    if n == 0:
        return []

    avg_dist = [sum(row) for row in dist]
    medoid = min(range(n), key=lambda i: avg_dist[i])

    starts = [medoid]

    def add_index(idx: Optional[int]) -> None:
        if idx is None:
            return
        if idx not in starts:
            starts.append(idx)

    if tracks:
        tempos = [t.features.get("tempo") if t.features else None for t in tracks]
        energies = [t.features.get("energy") if t.features else None for t in tracks]
        tempos = [v for v in tempos if v is not None]
        energies = [v for v in energies if v is not None]

        if tempos:
            min_tempo = min(
                range(n),
                key=lambda i: tracks[i].features.get("tempo") if tracks[i].features else math.inf,
            )
            max_tempo = max(
                range(n),
                key=lambda i: tracks[i].features.get("tempo") if tracks[i].features else -math.inf,
            )
            add_index(min_tempo)
            add_index(max_tempo)
        if energies:
            min_energy = min(
                range(n),
                key=lambda i: tracks[i].features.get("energy") if tracks[i].features else math.inf,
            )
            max_energy = max(
                range(n),
                key=lambda i: tracks[i].features.get("energy") if tracks[i].features else -math.inf,
            )
            add_index(min_energy)
            add_index(max_energy)

    while len(starts) < max(1, restarts):
        idx = rng.randrange(n)
        if idx not in starts:
            starts.append(idx)

    return starts


def optimize_order(
    dist: List[List[float]],
    tracks: List[Track],
    restarts: int,
    seed: int,
    two_opt_passes: int,
    cost_fn: Optional[Callable[[List[int]], float]] = None,
) -> Tuple[List[int], float]:
    n = len(dist)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], 0.0

    rng = random.Random(seed)
    best_order = None
    best_cost = math.inf

    starts = pick_start_indices(dist, tracks, rng, restarts)

    for idx, start in enumerate(starts):
        order = nearest_neighbor(dist, start, rng if idx > 0 else None, k=4)
        order = local_search(order, dist, two_opt_passes=two_opt_passes, cost_fn=cost_fn)
        cost = cost_fn(order) if cost_fn else total_cost(order, dist)
        if cost < best_cost:
            best_order, best_cost = list(order), cost

    return best_order or list(range(n)), best_cost


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
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def optimized_name(base_name: str) -> str:
    base = base_name.strip() if base_name else "Optimized"
    if base.lower().endswith("_optimized"):
        return base
    return f"{base}_optimized"


def create_playlist(sp: spotipy.Spotify, name: str, track_ids: List[str], public: bool) -> str:
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(user_id, name=name, public=public, description="Optimized playlist order")
    playlist_id = playlist["id"]
    for batch in chunked(track_ids, 100):
        sp.playlist_add_items(playlist_id, batch)
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
) -> Tuple[str, List[Track], float, List[Dict]]:
    playlist_name, tracks = fetch_playlist_tracks(sp, playlist_id)
    if not tracks:
        raise RuntimeError("No playable tracks found in playlist.")

    enrich_audio_features(sp, tracks, cache_path)

    with_features = [t for t in tracks if track_has_essential_features(t.features)]
    missing_tracks = [t for t in tracks if not track_has_essential_features(t.features)]

    weights = resolve_weights(weights, mix_mode)
    dist = build_distance_matrix(with_features, weights, bpm_window)

    energy_targets = None
    cost_fn = None
    if flow_curve:
        energies = [
            t.features.get("energy")
            for t in with_features
            if t.features and t.features.get("energy") is not None
        ]
        energy_targets = build_energy_curve(energies, len(with_features))
        cost_fn = lambda order: order_cost(
            order, dist, with_features, energy_targets, FLOW_CURVE_WEIGHT
        )

    order, cost = optimize_order(
        dist,
        with_features,
        restarts,
        seed,
        two_opt_passes,
        cost_fn=cost_fn,
    )

    if flow_curve and cost_fn and energy_targets:
        curve_order = build_energy_order(with_features, energy_targets)
        curve_order = local_search(curve_order, dist, two_opt_passes=two_opt_passes, cost_fn=cost_fn)
        curve_cost = cost_fn(curve_order)
        if curve_cost < cost:
            order, cost = curve_order, curve_cost

    ordered_tracks = [with_features[i] for i in order]
    if missing == "append" and missing_tracks:
        ordered_tracks.extend(missing_tracks)

    roughest = summarize_transitions(with_features, dist, order, limit=5)

    return playlist_name, ordered_tracks, cost, roughest
