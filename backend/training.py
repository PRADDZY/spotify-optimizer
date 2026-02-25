import math
import hashlib
import json
import time
from typing import Dict, Iterable, List, Tuple

from .model_store import build_model_version
from .modeling import MODEL_FEATURE_KEYS, TransitionModel, transition_features_from_transition


def _logistic_probability(weights: Dict[str, float], bias: float, row: Dict[str, float]) -> float:
    z = float(bias)
    for key in MODEL_FEATURE_KEYS:
        z += float(weights.get(key, 0.0)) * float(row.get(key, 0.0))
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def evaluate_logistic_model(
    rows: List[Dict[str, float]],
    labels: List[int],
    sample_weights: List[float],
    weights: Dict[str, float],
    bias: float,
) -> Dict[str, float]:
    if not rows:
        return {"loss": 0.0, "accuracy": 0.0}

    loss = 0.0
    correct = 0.0
    total_weight = 0.0
    for row, label, sample_weight in zip(rows, labels, sample_weights):
        pred = _logistic_probability(weights, bias, row)
        pred = min(1 - 1e-8, max(1e-8, pred))
        loss += sample_weight * (
            -float(label) * math.log(pred) - (1.0 - float(label)) * math.log(1.0 - pred)
        )
        if (pred >= 0.5 and label == 1) or (pred < 0.5 and label == 0):
            correct += sample_weight
        total_weight += sample_weight

    return {
        "loss": round(loss / max(1.0, total_weight), 6),
        "accuracy": round(correct / max(1.0, total_weight), 6),
    }


def _row_fingerprint(index: int, row: Dict[str, float], label: int) -> str:
    payload = {
        "index": index,
        "label": int(label),
        "features": {key: round(float(row.get(key, 0.0)), 6) for key in MODEL_FEATURE_KEYS},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _slice_rows(
    rows: List[Dict[str, float]],
    labels: List[int],
    sample_weights: List[float],
    indices: List[int],
) -> tuple[List[Dict[str, float]], List[int], List[float]]:
    return (
        [rows[index] for index in indices],
        [labels[index] for index in indices],
        [sample_weights[index] for index in indices],
    )


def split_feedback_dataset(
    rows: List[Dict[str, float]],
    labels: List[int],
    sample_weights: List[float],
    *,
    validation_ratio: float = 0.2,
) -> tuple[List[Dict[str, float]], List[int], List[float], List[Dict[str, float]], List[int], List[float]]:
    total = len(rows)
    if total < 2:
        return rows, labels, sample_weights, [], [], []

    safe_ratio = max(0.05, min(0.5, validation_ratio))
    target_validation = int(round(total * safe_ratio))
    target_validation = max(1, min(total - 1, target_validation))

    label_buckets: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        label_buckets.setdefault(int(label), []).append(index)

    validation_indices: set[int] = set()
    for _, bucket in sorted(label_buckets.items(), key=lambda item: item[0]):
        if not bucket:
            continue
        proportional = int(round(target_validation * (len(bucket) / total)))
        if target_validation > 0 and proportional == 0:
            proportional = 1
        proportional = min(len(bucket), max(0, proportional))
        ranked = sorted(
            bucket,
            key=lambda idx: _row_fingerprint(idx, rows[idx], labels[idx]),
        )
        validation_indices.update(ranked[:proportional])

    if len(validation_indices) > target_validation:
        ranked_validation = sorted(
            validation_indices,
            key=lambda idx: _row_fingerprint(idx, rows[idx], labels[idx]),
        )
        validation_indices = set(ranked_validation[:target_validation])
    elif len(validation_indices) < target_validation:
        remaining = [
            idx
            for idx in range(total)
            if idx not in validation_indices
        ]
        ranked_remaining = sorted(
            remaining,
            key=lambda idx: _row_fingerprint(idx, rows[idx], labels[idx]),
        )
        for idx in ranked_remaining[: target_validation - len(validation_indices)]:
            validation_indices.add(idx)

    if not validation_indices or len(validation_indices) >= total:
        split = max(1, min(total - 1, target_validation))
        ranked_all = sorted(
            range(total),
            key=lambda idx: _row_fingerprint(idx, rows[idx], labels[idx]),
        )
        validation_indices = set(ranked_all[:split])

    train_indices = [idx for idx in range(total) if idx not in validation_indices]
    validation_indices_sorted = sorted(validation_indices)
    if not train_indices:
        train_indices = [validation_indices_sorted.pop()]

    train_rows, train_labels, train_sample_weights = _slice_rows(
        rows,
        labels,
        sample_weights,
        train_indices,
    )
    validation_rows, validation_labels, validation_sample_weights = _slice_rows(
        rows,
        labels,
        sample_weights,
        validation_indices_sorted,
    )
    return (
        train_rows,
        train_labels,
        train_sample_weights,
        validation_rows,
        validation_labels,
        validation_sample_weights,
    )


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
            pred = _logistic_probability(weights, bias, row)

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

    metrics = evaluate_logistic_model(rows, labels, sample_weights, weights, bias)
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

    (
        train_rows,
        train_labels,
        train_sample_weights,
        validation_rows,
        validation_labels,
        validation_sample_weights,
    ) = split_feedback_dataset(rows, labels, sample_weights)

    weights, bias, metrics = train_logistic_model(train_rows, train_labels, train_sample_weights)
    validation_metrics = evaluate_logistic_model(
        validation_rows,
        validation_labels,
        validation_sample_weights,
        weights,
        bias,
    )
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
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "positive_ratio": round(sum(labels) / max(1, len(labels)), 6),
        "version": version,
    }
    return model, payload
