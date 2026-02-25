import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .optimizer_core import MODEL_FEATURE_KEYS


def normalize_component_map(components: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in components.values())
    if total <= 1e-9:
        return {key: 0.0 for key in components}
    return {key: max(0.0, float(value)) / total for key, value in components.items()}


def transition_features_from_transition(transition: Dict) -> Dict[str, float]:
    component_share = transition.get("component_share") or {}
    if not component_share:
        component_share = normalize_component_map(transition.get("components", {}))

    from_tempo = transition.get("from_tempo")
    to_tempo = transition.get("to_tempo")
    if from_tempo is None or to_tempo is None:
        tempo_jump = 0.5
    else:
        tempo_jump = abs(float(from_tempo) - float(to_tempo)) / max(
            80.0,
            max(float(from_tempo), float(to_tempo)),
        )

    from_energy = transition.get("from_energy")
    to_energy = transition.get("to_energy")
    if from_energy is None or to_energy is None:
        energy_jump = 0.5
    else:
        energy_jump = abs(float(from_energy) - float(to_energy))

    features = {
        "bpm_component": float(component_share.get("bpm", 0.0)),
        "key_component": float(component_share.get("key", 0.0)),
        "energy_component": float(component_share.get("energy", 0.0)),
        "valence_component": float(component_share.get("valence", 0.0)),
        "dance_component": float(component_share.get("dance", 0.0)),
        "loudness_component": float(component_share.get("loudness", 0.0)),
        "tempo_jump": max(0.0, min(1.0, float(tempo_jump))),
        "energy_jump": max(0.0, min(1.0, float(energy_jump))),
    }
    return {key: float(features.get(key, 0.0)) for key in MODEL_FEATURE_KEYS}


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


@dataclass
class TransitionModel:
    version: str
    weights: Dict[str, float]
    bias: float
    trained_at: float
    sample_count: int

    def predict(self, features: Dict[str, float]) -> float:
        score = float(self.bias)
        for key in MODEL_FEATURE_KEYS:
            score += float(self.weights.get(key, 0.0)) * float(features.get(key, 0.0))
        return sigmoid(score)

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "weights": {key: float(self.weights.get(key, 0.0)) for key in MODEL_FEATURE_KEYS},
            "bias": float(self.bias),
            "trained_at": float(self.trained_at),
            "sample_count": int(self.sample_count),
        }


def transition_model_from_dict(payload: Dict) -> TransitionModel:
    return TransitionModel(
        version=str(payload.get("version") or "unknown"),
        weights={
            key: float((payload.get("weights") or {}).get(key, 0.0))
            for key in MODEL_FEATURE_KEYS
        },
        bias=float(payload.get("bias", 0.0)),
        trained_at=float(payload.get("trained_at", time.time())),
        sample_count=int(payload.get("sample_count", 0)),
    )


def average_prediction(model: TransitionModel, rows: Iterable[Dict[str, float]]) -> float:
    scores: List[float] = [model.predict(row) for row in rows]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
