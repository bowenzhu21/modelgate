"""Inspectable JSON softmax models, trained with deterministic full-batch gradients."""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifacts import IntegrityError, digest, finite_tree, require_keys
from .data import LABELS, SLICES, dataset_manifest, training_records

FEATURES = {
    "axis-v1": ["extent_x", "extent_y", "extent_z", "variance_x", "variance_y", "variance_z", "mean_radius"],
    "invariant-v2": ["eigen_min", "eigen_mid", "eigen_max", "radius_p10", "radius_p25", "radius_p50",
                     "radius_p75", "radius_p90", "radius_cv", "radius_p99", "fitted_radius_cv"],
}


def features(cloud: Any, mode: str) -> np.ndarray:
    points = np.asarray(cloud, dtype=np.float64)
    points = points - points.mean(axis=0)
    radius = np.linalg.norm(points, axis=1)
    if mode == "axis-v1":
        result = np.concatenate((np.ptp(points, axis=0), np.var(points, axis=0), [radius.mean()]))
    elif mode == "invariant-v2":
        eig = np.linalg.eigvalsh(np.einsum("ni,nj->ij", points, points) / len(points))
        eig /= max(eig.sum(), 1e-12)
        normalized_radius = radius / max(float(np.sqrt(np.mean(radius ** 2))), 1e-12)
        # Algebraic sphere fit estimates a center without the finite-sample centroid
        # shift. Its radial residual also separates curved and planar surfaces.
        design = np.column_stack((2 * points, np.ones(len(points))))
        fitted_center = np.linalg.lstsq(design, np.sum(points ** 2, axis=1), rcond=None)[0][:3]
        fitted_radius = np.linalg.norm(points - fitted_center, axis=1)
        result = np.concatenate((eig, np.quantile(normalized_radius, [0.1, 0.25, 0.5, 0.75, 0.9]),
                                 [radius.std() / max(radius.mean(), 1e-12), np.quantile(normalized_radius, 0.99),
                                  fitted_radius.std() / max(fitted_radius.mean(), 1e-12)]))
    else:
        raise IntegrityError(f"Unsupported feature version: {mode}")
    if not np.isfinite(result).all():
        raise IntegrityError("Non-finite extracted feature")
    return result


def feature_matrix(records: list[dict[str, Any]], mode: str) -> np.ndarray:
    return np.vstack([features(record["points"], mode) for record in records])


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - logits.max(axis=1, keepdims=True)
    exponents = np.exp(stable)
    return exponents / exponents.sum(axis=1, keepdims=True)


def train(data: dict[str, Any], mode: str, epochs: int = 700, fit_slices: tuple[str, ...] | None = None) -> dict[str, Any]:
    if not isinstance(mode, str) or mode not in FEATURES or type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError("Unknown feature version or invalid training epochs")
    manifest = dataset_manifest(data)
    fit_slices = fit_slices or ("clean",)
    if any(s not in SLICES for s in fit_slices):
        raise ValueError("Unknown training slice")
    records = training_records(data, fit_slices)
    raw = feature_matrix(records, mode)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std = np.maximum(std, 1e-8)
    x = (raw - mean) / std
    y = np.array([LABELS.index(record["label"]) for record in records])
    target = np.eye(len(LABELS))[y]
    weights = np.zeros((x.shape[1], len(LABELS)))
    bias = np.zeros(len(LABELS))
    learning_rate, regularization = 0.12, 0.003
    for _ in range(epochs):
        residual = (_softmax(np.einsum("ij,jk->ik", x, weights) + bias) - target) / len(x)
        weights -= learning_rate * (np.einsum("ni,nj->ij", x, residual) + regularization * weights)
        bias -= learning_rate * residual.sum(axis=0)
    model = {
        "schema": "modelgate.softmax.v1", "name": "baseline" if mode == "axis-v1" else "candidate",
        "labels": list(LABELS), "feature_version": mode, "feature_names": list(FEATURES[mode]),
        "normalizer": {"mean": mean.tolist(), "std": std.tolist()},
        "weights": weights.tolist(), "bias": bias.tolist(),
        "training": {"dataset_sha256": manifest["sha256"], "train_sha256": manifest["train_sha256"],
                     "fit_sha256": digest(records), "sample_count": len(records), "split": "train",
                     "slices": list(fit_slices), "epochs": epochs,
                     "learning_rate": learning_rate, "l2": regularization, "initialization": "zeros"},
    }
    validate_model(model)
    return model


def validate_model(model: Any) -> None:
    require_keys(model, {"schema", "name", "labels", "feature_version", "feature_names", "normalizer", "weights", "bias", "training"}, "Model")
    finite_tree(model)
    if model["schema"] != "modelgate.softmax.v1" or model["labels"] != LABELS or not isinstance(model["name"], str):
        raise IntegrityError("Unsupported model schema or label order")
    mode = model["feature_version"]
    if not isinstance(mode, str) or mode not in FEATURES or model["feature_names"] != FEATURES[mode]:
        raise IntegrityError("Unknown feature schema")
    require_keys(model["normalizer"], {"mean", "std"}, "Normalizer")
    require_keys(model["training"], {"dataset_sha256", "train_sha256", "fit_sha256", "sample_count", "split", "slices", "epochs", "learning_rate", "l2", "initialization"}, "Training")
    training = model["training"]
    if training["split"] != "train" or training["initialization"] != "zeros":
        raise IntegrityError("Unsupported training provenance")
    if not isinstance(training["slices"], list) or not training["slices"] or any(not isinstance(s, str) or s not in SLICES for s in training["slices"]):
        raise IntegrityError("Invalid training slices")
    if len(training["slices"]) != len(set(training["slices"])):
        raise IntegrityError("Duplicate training slices")
    for key in ("sample_count", "epochs"):
        if type(training[key]) is not int or not 1 <= training[key] <= 10000:
            raise IntegrityError(f"Invalid training {key}")
    for key, low, high in (("learning_rate", 0, 1), ("l2", 0, 1)):
        if type(training[key]) not in (int, float) or not low < training[key] <= high:
            raise IntegrityError(f"Invalid training {key}")
    for key in ("dataset_sha256", "train_sha256", "fit_sha256"):
        value = model["training"][key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise IntegrityError(f"Invalid {key}")
    dimension = len(FEATURES[mode])
    for label, value, shape in [("weights", model["weights"], (dimension, 3)), ("bias", model["bias"], (3,)),
                                ("mean", model["normalizer"]["mean"], (dimension,)),
                                ("std", model["normalizer"]["std"], (dimension,))]:
        try:
            array = np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise IntegrityError(f"Invalid model {label}") from exc
        if array.shape != shape or not np.isfinite(array).all() or np.max(np.abs(array)) > 1e12:
            raise IntegrityError(f"Invalid model {label} dimensions or values")
        flattened = [v for row in value for v in row] if label == "weights" else value
        if any(type(v) not in (int, float) for v in flattened):
            raise IntegrityError(f"Model {label} must contain JSON numbers")
        if label == "std" and np.any(array < 1e-8):
            raise IntegrityError("Normalizer standard deviations must be at least 1e-8")


def predict(model: dict[str, Any], records: list[dict[str, Any]]) -> np.ndarray:
    validate_model(model)
    raw = feature_matrix(records, model["feature_version"])
    x = (raw - np.asarray(model["normalizer"]["mean"])) / np.asarray(model["normalizer"]["std"])
    result = _softmax(np.einsum("ij,jk->ik", x, np.asarray(model["weights"])) + np.asarray(model["bias"]))
    if not np.isfinite(result).all():
        raise IntegrityError("Model produced non-finite probabilities")
    return result


def broken_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Deliberately damage a real trained artifact to exercise rejection."""
    import copy
    result = copy.deepcopy(candidate)
    result["name"] = "broken-candidate"
    result["weights"] = np.zeros_like(result["weights"]).tolist()
    result["bias"] = [9.0, 0.0, 0.0]
    return result
