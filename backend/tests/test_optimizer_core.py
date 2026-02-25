import json

from backend.optimizer_core import (
    OptimizationConfig,
    Track,
    apply_fixed_endpoints,
    apply_locked_blocks,
    append_transition_log,
    build_energy_curve,
    minimax_refine,
    order_cost,
    order_max_edge,
)


def make_track(track_id: str, energy: float, tempo: float, key: int = 0, mode: int = 1) -> Track:
    return Track(
        id=track_id,
        name=track_id,
        artists="tester",
        artist_ids=[track_id.split("-")[0]],
        features={
            "energy": energy,
            "tempo": tempo,
            "key": key,
            "mode": mode,
            "time_signature": 4,
            "loudness": -8.0,
            "valence": 0.5,
            "danceability": 0.6,
        },
    )


def test_build_energy_curve_profiles_have_distinct_shapes():
    energies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    peak = build_energy_curve(energies, 9, profile="peak")
    gentle = build_energy_curve(energies, 9, profile="gentle")
    cooldown = build_energy_curve(energies, 9, profile="cooldown")

    assert len(peak) == len(gentle) == len(cooldown) == 9
    assert 2 <= peak.index(max(peak)) <= 6
    assert cooldown[0] > cooldown[-1]
    assert (max(peak) - min(peak)) > (max(gentle) - min(gentle))


def test_order_cost_prefers_tracks_that_follow_energy_and_tempo_targets():
    tracks = [
        make_track("t1", 0.2, 100.0),
        make_track("t2", 0.4, 110.0),
        make_track("t3", 0.6, 120.0),
        make_track("t4", 0.8, 130.0),
    ]
    dist = [[0.0 for _ in tracks] for _ in tracks]
    config = OptimizationConfig(flow_curve=True, flow_weight=0.25, key_lock_window=1, tempo_ramp_weight=0.25)

    energy_targets = [0.2, 0.4, 0.6, 0.8]
    tempo_targets = [0.2, 0.4, 0.6, 0.8]
    tempo_unit_values = [0.2, 0.4, 0.6, 0.8]

    good_order = [0, 1, 2, 3]
    bad_order = [3, 2, 1, 0]

    good = order_cost(good_order, dist, tracks, energy_targets, tempo_targets, tempo_unit_values, config)
    bad = order_cost(bad_order, dist, tracks, energy_targets, tempo_targets, tempo_unit_values, config)
    assert good < bad


def test_minimax_refine_reduces_worst_transition():
    dist = [
        [0.0, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.0, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.0, 1.0, 0.1],
        [0.2, 0.2, 1.0, 0.0, 0.1],
        [0.2, 0.2, 0.1, 0.1, 0.0],
    ]
    order = [0, 1, 2, 3, 4]
    objective = lambda value: sum(dist[value[i]][value[i + 1]] for i in range(len(value) - 1))

    before = order_max_edge(order, dist)
    optimized = minimax_refine(order, dist, objective_fn=objective, passes=3)
    after = order_max_edge(optimized, dist)

    assert after < before


def test_append_transition_log_writes_jsonl(tmp_path):
    path = tmp_path / "transitions.jsonl"
    tracks = [make_track("t1", 0.4, 118.0), make_track("t2", 0.45, 120.0)]
    dist = [[0.0, 0.12], [0.12, 0.0]]
    order = [0, 1]
    config = OptimizationConfig(flow_curve=True, flow_profile="peak", key_lock_window=3)

    append_transition_log(
        path=str(path),
        playlist_id="abc123",
        playlist_name="Test Playlist",
        tracks=tracks,
        dist=dist,
        order=order,
        score=0.12,
        config=config,
        weights={"bpm": 0.4, "key": 0.3},
        bpm_window=0.08,
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["playlist_id"] == "abc123"
    assert payload["score"] == 0.12
    assert len(payload["transitions"]) == 1
    assert payload["transitions"][0]["from_track"] == "t1"
    assert payload["transitions"][0]["to_track"] == "t2"


def test_apply_fixed_endpoints_keeps_requested_start_and_end():
    tracks = [
        make_track("a", 0.2, 100.0),
        make_track("b", 0.4, 110.0),
        make_track("c", 0.6, 120.0),
        make_track("d", 0.8, 130.0),
    ]
    order = [1, 2, 0, 3]
    fixed = apply_fixed_endpoints(
        order,
        tracks,
        locked_first_track_id="a",
        locked_last_track_id="c",
    )
    assert tracks[fixed[0]].id == "a"
    assert tracks[fixed[-1]].id == "c"


def test_apply_locked_blocks_keeps_block_contiguous():
    tracks = [
        make_track("a", 0.2, 100.0),
        make_track("b", 0.4, 110.0),
        make_track("c", 0.6, 120.0),
        make_track("d", 0.8, 130.0),
        make_track("e", 0.5, 115.0),
    ]
    order = [0, 2, 1, 4, 3]
    fixed = apply_locked_blocks(order, tracks, [["b", "d"]])
    ids = [tracks[idx].id for idx in fixed]
    joined = ",".join(ids)
    assert "b,d" in joined


def test_artist_gap_penalty_prefers_spaced_artists():
    tracks = [
        make_track("artist1-a", 0.2, 100.0),
        make_track("artist1-b", 0.4, 110.0),
        make_track("artist2-a", 0.6, 120.0),
    ]
    dist = [[0.0 for _ in tracks] for _ in tracks]
    config = OptimizationConfig(artist_gap=2)
    crowded = order_cost([0, 1, 2], dist, tracks, None, None, None, config)
    spaced = order_cost([0, 2, 1], dist, tracks, None, None, None, config)
    assert spaced < crowded
