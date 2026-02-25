import math
import time
from typing import Dict, Iterable, List, Tuple

from .model_store import build_model_version
from .modeling import MODEL_FEATURE_KEYS, TransitionModel, transition_features_from_transition


def build_feedback_dataset(
    run_history_items: Iterable[Tuple[str, Dict]],
    feedback_items: Iterable[Tuple[str, Dict]],
    owner_id: str | None = None,
) -> tuple[List[Dict[str, float]], List[int], List[float]]:
    runs = {run_id: row for run_id, row in run_history_items}

    features: List[Dict[str, float]] = []
    labels: List[int] = []
    sample_weights: List[float] = []

    for _, feedback in feedback_items:
        if owner_id and feedback.get("owner_id") != owner_id:
            continue
        rating = int(feedback.get("rating", 0))
        if rating == 0:
            continue
        run_id = feedback.get("run_id")
        edge_index = int(feedback.get("edge_index", -1))
        run = runs.get(run_id)
        if not run:
            continue
        transitions = run.get("transitions") or []
        if edge_index < 0 or edge_index >= len(transitions):
            continue

        transition = transitions[edge_index]
        features.append(transition_features_from_transition(transition))
        # Target 1 means rough transition, 0 means good transition.
        labels.append(1 if rating < 0 else 0)
        sample_weights.append(max(1.0, abs(float(rating))))

    return features, labels, sample_weights


def train_logistic_model(
    rows: List[Dict[str, float]],
    labels: List[int],
    sample_weights: List[float],
    *,
    epochs: int = 320,
    learning_rate: float = 0.22,
    l2: float = 0.001,
) -> tuple[Dict[str, float], float, Dict[str, float]]:
    weights = {key: 0.0 for key in MODEL_FEATURE_KEYS}
    bias = 0.0

    if not rows:
        return weights, bias, {"loss": 0.0, "accuracy": 0.0}

    for _ in range(max(1, epochs)):
        grad_w = {key: 0.0 for key in MODEL_FEATURE_KEYS}
        grad_b = 0.0
        total_weight = 0.0

        for row, label, sample_weight in zip(rows, labels, sample_weights):
            z = bias
            for key in MODEL_FEATURE_KEYS:
                z += weights[key] * float(row.get(key, 0.0))
            pred = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

            error = (pred - float(label)) * sample_weight
            total_weight += sample_weight
            for key in MODEL_FEATURE_KEYS:
                grad_w[key] += error * float(row.get(key, 0.0)) + (l2 * weights[key])
            grad_b += error

        if total_weight <= 0:
            break

        step = learning_rate / total_weight
        for key in MODEL_FEATURE_KEYS:
            weights[key] -= step * grad_w[key]
        bias -= step * grad_b

    loss = 0.0
    correct = 0.0
    total_weight = 0.0
    for row, label, sample_weight in zip(rows, labels, sample_weights):
        z = bias + sum(weights[key] * float(row.get(key, 0.0)) for key in MODEL_FEATURE_KEYS)
        pred = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        pred = min(1 - 1e-8, max(1e-8, pred))
        loss += sample_weight * (
            -float(label) * math.log(pred) - (1.0 - float(label)) * math.log(1.0 - pred)
        )
        if (pred >= 0.5 and label == 1) or (pred < 0.5 and label == 0):
            correct += sample_weight
        total_weight += sample_weight

    metrics = {
        "loss": round(loss / max(1.0, total_weight), 6),
        "accuracy": round(correct / max(1.0, total_weight), 6),
    }
    return weights, bias, metrics


def train_transition_model_from_feedback(
    run_history_items: Iterable[Tuple[str, Dict]],
    feedback_items: Iterable[Tuple[str, Dict]],
    *,
    owner_id: str | None = None,
    min_samples: int = 20,
) -> tuple[TransitionModel | None, Dict]:
    rows, labels, sample_weights = build_feedback_dataset(
        run_history_items,
        feedback_items,
        owner_id=owner_id,
    )
    if len(rows) < min_samples:
        return None, {
            "trained": False,
            "sample_count": len(rows),
            "reason": f"Need at least {min_samples} labeled transitions",
        }

    weights, bias, metrics = train_logistic_model(rows, labels, sample_weights)
    version = build_model_version()
    model = TransitionModel(
        version=version,
        weights=weights,
        bias=bias,
        trained_at=time.time(),
        sample_count=len(rows),
    )
    payload = {
        "trained": True,
        "sample_count": len(rows),
        "metrics": metrics,
        "positive_ratio": round(sum(labels) / max(1, len(labels)), 6),
        "version": version,
    }
    return model, payload
