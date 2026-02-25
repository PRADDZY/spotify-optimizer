import os

from backend.model_store import load_model_artifact, save_model_artifact
from backend.modeling import TransitionModel, transition_features_from_transition
from backend.training import train_transition_model_from_feedback


def test_transition_features_from_transition_extracts_expected_keys():
    transition = {
        "component_share": {
            "bpm": 0.4,
            "key": 0.3,
            "energy": 0.2,
            "valence": 0.05,
            "dance": 0.03,
            "loudness": 0.02,
        },
        "from_tempo": 100,
        "to_tempo": 120,
        "from_energy": 0.3,
        "to_energy": 0.8,
    }
    features = transition_features_from_transition(transition)
    assert "bpm_component" in features
    assert "key_component" in features
    assert 0.0 <= features["tempo_jump"] <= 1.0


def test_train_transition_model_from_feedback_produces_model_when_sample_count_is_met():
    run_id = "run-alpha"
    run_history_items = [
        (
            run_id,
            {
                "transitions": [
                    {
                        "component_share": {
                            "bpm": 0.6,
                            "key": 0.2,
                            "energy": 0.1,
                            "valence": 0.05,
                            "dance": 0.03,
                            "loudness": 0.02,
                        },
                        "from_tempo": 95,
                        "to_tempo": 130,
                        "from_energy": 0.2,
                        "to_energy": 0.85,
                    },
                    {
                        "component_share": {
                            "bpm": 0.1,
                            "key": 0.1,
                            "energy": 0.45,
                            "valence": 0.15,
                            "dance": 0.1,
                            "loudness": 0.1,
                        },
                        "from_tempo": 110,
                        "to_tempo": 112,
                        "from_energy": 0.6,
                        "to_energy": 0.62,
                    },
                ]
            },
        )
    ]

    feedback_items = []
    for i in range(24):
        feedback_items.append(
            (
                f"fb-{i}",
                {
                    "run_id": run_id,
                    "edge_index": 0 if i % 2 == 0 else 1,
                    "rating": -2 if i % 2 == 0 else 2,
                    "owner_id": "tester",
                },
            )
        )

    model, details = train_transition_model_from_feedback(
        run_history_items,
        feedback_items,
        owner_id="tester",
        min_samples=10,
    )

    assert model is not None
    assert details["trained"] is True
    assert model.sample_count >= 10


def test_model_store_round_trip(tmp_path):
    model = TransitionModel(
        version="transition_test_v1",
        weights={
            "bpm_component": 0.3,
            "key_component": 0.2,
            "energy_component": 0.1,
            "valence_component": 0.1,
            "dance_component": 0.1,
            "loudness_component": 0.1,
            "tempo_jump": 0.05,
            "energy_jump": 0.05,
        },
        bias=-0.2,
        trained_at=1.0,
        sample_count=42,
    )

    artifact_path = save_model_artifact(str(tmp_path), model.to_dict())
    loaded = load_model_artifact(artifact_path)

    assert loaded is not None
    assert loaded["version"] == "transition_test_v1"
    assert os.path.exists(artifact_path)
