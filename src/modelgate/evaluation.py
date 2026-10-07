"""Holdout evidence and a versioned policy that requires every metric to be valid."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .artifacts import IntegrityError, digest, finite_tree, require_keys
from .data import LABELS, SLICES, dataset_manifest, training_records
from .model import feature_matrix, predict, validate_model

POLICY = {
    "version": "geometry-gate-v1", "minimum_accuracy": 0.88, "minimum_slice_accuracy": 0.82,
    "minimum_class_recall": 0.80, "minimum_accuracy_delta": 0.0,
    "maximum_log_loss_increase": 0.03, "minimum_group_bootstrap_lower_delta": -0.03,
    "maximum_mean_psi": 4.0, "minimum_holdout_groups": 24,
}


def classification_metrics(records: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    expected = np.asarray([LABELS.index(r["label"]) for r in records])
    actual = probabilities.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (expected, actual), 1)
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise IntegrityError("Evaluation slice is missing a class")
    return {"samples": len(records), "accuracy": float(np.mean(expected == actual)),
            "log_loss": float(-np.log(np.maximum(probabilities[np.arange(len(records)), expected], 1e-15)).mean()),
            "class_recall": {label: float(matrix[i, i] / support[i]) for i, label in enumerate(LABELS)},
            "confusion_matrix": matrix.tolist()}


def _model_metrics(records: list[dict[str, Any]], probabilities: np.ndarray) -> dict[str, Any]:
    slices = {}
    for name in SLICES:
        indices = [i for i, record in enumerate(records) if record["slice"] == name]
        slices[name] = classification_metrics([records[i] for i in indices], probabilities[indices])
    return {"overall": classification_metrics(records, probabilities), "slices": slices}


def _drift(reference: np.ndarray, observed: np.ndarray, names: list[str]) -> dict[str, Any]:
    """PSI with train-only quantile bins and 0.5 pseudo-count smoothing."""
    values = []
    for index, name in enumerate(names):
        ref, obs = reference[:, index], observed[:, index]
        cuts = np.unique(np.quantile(ref, [0.2, 0.4, 0.6, 0.8]))
        if np.ptp(ref) < 1e-9:
            cuts = np.array([ref.mean() - 1e-8, ref.mean() + 1e-8])
        bins = np.concatenate(([-np.inf], cuts, [np.inf]))
        a, b = np.histogram(ref, bins)[0] + 0.5, np.histogram(obs, bins)[0] + 0.5
        a, b = a / a.sum(), b / b.sum()
        psi = float(np.sum((b - a) * np.log(b / a)))
        shift = float(abs(obs.mean() - ref.mean()) / max(ref.std(), 1e-8))
        values.append({"feature": name, "psi": psi, "standardized_mean_shift": shift})
    return {"mean_psi": float(np.mean([r["psi"] for r in values])),
            "maximum_mean_shift": float(max(r["standardized_mean_shift"] for r in values)), "features": values}


def _validate_metric(metric: Any, label: str) -> None:
    require_keys(metric, {"samples", "accuracy", "log_loss", "class_recall", "confusion_matrix"}, label)
    if type(metric["samples"]) is not int or metric["samples"] <= 0:
        raise IntegrityError(f"{label} has invalid sample count")
    for key in ("accuracy", "log_loss"):
        value = metric[key]
        if type(value) not in (int, float) or value < 0 or (key == "accuracy" and value > 1):
            raise IntegrityError(f"{label} has invalid {key}")
    require_keys(metric["class_recall"], set(LABELS), f"{label} recall")
    if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in metric["class_recall"].values()):
        raise IntegrityError(f"{label} has invalid class recall")
    try:
        matrix = np.asarray(metric["confusion_matrix"])
    except (ValueError, TypeError) as exc:
        raise IntegrityError(f"{label} has malformed confusion matrix") from exc
    if matrix.shape != (3, 3) or not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0) or int(matrix.sum()) != metric["samples"]:
        raise IntegrityError(f"{label} has inconsistent confusion matrix")
    if np.any(matrix.sum(axis=1) == 0):
        raise IntegrityError(f"{label} is missing a class")
    if not np.isclose(np.trace(matrix) / matrix.sum(), metric["accuracy"], rtol=0, atol=1e-12):
        raise IntegrityError(f"{label} accuracy does not match confusion matrix")
    for i, name in enumerate(LABELS):
        if not np.isclose(matrix[i, i] / matrix[i].sum(), metric["class_recall"][name], rtol=0, atol=1e-12):
            raise IntegrityError(f"{label} class recall does not match confusion matrix")


def assess(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return explicit blocking reasons; missing/NaN/inconsistent data raises."""
    finite_tree(evidence)
    require_keys(evidence, {"baseline", "candidate", "comparison", "drift", "holdout_groups"}, "Evidence")
    for model_name in ("baseline", "candidate"):
        require_keys(evidence[model_name], {"overall", "slices"}, model_name)
        _validate_metric(evidence[model_name]["overall"], f"{model_name}/overall")
        require_keys(evidence[model_name]["slices"], set(SLICES), f"{model_name}/slices")
        for name, metric in evidence[model_name]["slices"].items():
            _validate_metric(metric, f"{model_name}/{name}")
        aggregate = evidence[model_name]["overall"]
        slices = list(evidence[model_name]["slices"].values())
        if sum(m["samples"] for m in slices) != aggregate["samples"] or not np.array_equal(
            np.sum([m["confusion_matrix"] for m in slices], axis=0), aggregate["confusion_matrix"]
        ):
            raise IntegrityError("Overall evidence is inconsistent with slices")
        if not np.isclose(sum(m["log_loss"] * m["samples"] for m in slices) / aggregate["samples"], aggregate["log_loss"], rtol=0, atol=1e-12):
            raise IntegrityError("Overall log loss is inconsistent with slices")
    comparison = evidence["comparison"]
    require_keys(comparison, {"accuracy_delta", "log_loss_delta", "group_bootstrap_95pct"}, "Comparison")
    require_keys(comparison["group_bootstrap_95pct"], {"low", "high", "replicates", "seed", "unit"}, "Bootstrap")
    for key in ("accuracy_delta", "log_loss_delta"):
        if type(comparison[key]) not in (int, float):
            raise IntegrityError(f"Invalid comparison {key}")
    interval = comparison["group_bootstrap_95pct"]
    if type(interval["low"]) not in (int, float) or type(interval["high"]) not in (int, float) or not -1 <= interval["low"] <= interval["high"] <= 1:
        raise IntegrityError("Invalid paired bootstrap interval")
    if interval["replicates"] != 1000 or interval["seed"] != 917 or interval["unit"] != "latent-shape-group":
        raise IntegrityError("Unexpected bootstrap method")
    candidate, baseline = evidence["candidate"]["overall"], evidence["baseline"]["overall"]
    for name in ("accuracy", "log_loss"):
        if not np.isclose(comparison[f"{name}_delta"], candidate[name] - baseline[name], rtol=0, atol=1e-12):
            raise IntegrityError("Comparison delta does not match metrics")
    require_keys(evidence["drift"], set(SLICES), "Drift")
    for name, stats in evidence["drift"].items():
        require_keys(stats, {"mean_psi", "maximum_mean_shift", "features"}, f"Drift/{name}")
        if type(stats["mean_psi"]) not in (int, float) or stats["mean_psi"] < 0:
            raise IntegrityError("Invalid PSI drift metric")
        if type(stats["maximum_mean_shift"]) not in (int, float) or stats["maximum_mean_shift"] < 0 or not stats["features"]:
            raise IntegrityError("Missing drift evidence")
        for feature in stats["features"]:
            require_keys(feature, {"feature", "psi", "standardized_mean_shift"}, "Drift feature")
            if not isinstance(feature["feature"], str) or any(type(feature[key]) not in (int, float) or feature[key] < 0 for key in ("psi", "standardized_mean_shift")):
                raise IntegrityError("Invalid feature drift metric")
        if not np.isclose(stats["mean_psi"], np.mean([f["psi"] for f in stats["features"]]), rtol=0, atol=1e-12):
            raise IntegrityError("Inconsistent PSI aggregate")
    if type(evidence["holdout_groups"]) is not int or evidence["holdout_groups"] < 1:
        raise IntegrityError("Missing holdout group count")
    reasons = []
    def minimum(value: float, threshold: float, description: str) -> None:
        if value < threshold:
            reasons.append(f"{description}: {value:.4f} < {threshold:.4f}")
    minimum(candidate["accuracy"], POLICY["minimum_accuracy"], "Overall accuracy")
    minimum(evidence["holdout_groups"], POLICY["minimum_holdout_groups"], "Independent holdout groups")
    minimum(comparison["accuracy_delta"], POLICY["minimum_accuracy_delta"], "Accuracy delta")
    minimum(interval["low"], POLICY["minimum_group_bootstrap_lower_delta"], "Paired group bootstrap lower delta")
    if comparison["log_loss_delta"] > POLICY["maximum_log_loss_increase"]:
        reasons.append(f"Log loss increase: {comparison['log_loss_delta']:.4f} > {POLICY['maximum_log_loss_increase']:.4f}")
    for label, recall in candidate["class_recall"].items():
        minimum(recall, POLICY["minimum_class_recall"], f"Overall {label} recall")
    for name, metric in evidence["candidate"]["slices"].items():
        minimum(metric["accuracy"], POLICY["minimum_slice_accuracy"], f"{name} slice accuracy")
        for label, recall in metric["class_recall"].items():
            minimum(recall, POLICY["minimum_class_recall"], f"{name} {label} recall")
    for name, stats in evidence["drift"].items():
        if stats["mean_psi"] > POLICY["maximum_mean_psi"]:
            reasons.append(f"{name} mean PSI: {stats['mean_psi']:.4f} > {POLICY['maximum_mean_psi']:.4f}")
    return {"allowed": not reasons, "decision": "PASS" if not reasons else "BLOCK", "reasons": reasons}


def evaluate(data: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    manifest = dataset_manifest(data)
    for model in (baseline, candidate):
        validate_model(model)
        if model["training"]["dataset_sha256"] != manifest["sha256"] or model["training"]["train_sha256"] != manifest["train_sha256"]:
            raise IntegrityError("Dataset hash mismatch against model training provenance")
        fitted = training_records(data, model["training"]["slices"])
        if model["training"]["fit_sha256"] != digest(fitted) or model["training"]["sample_count"] != len(fitted):
            raise IntegrityError("Training subset hash or sample count mismatch")
    records = [r for r in data["records"] if r["split"] == "test"]
    baseline_prob, candidate_prob = predict(baseline, records), predict(candidate, records)
    baseline_metrics, candidate_metrics = _model_metrics(records, baseline_prob), _model_metrics(records, candidate_prob)
    expected = np.array([LABELS.index(r["label"]) for r in records])
    delta = (candidate_prob.argmax(axis=1) == expected).astype(float) - (baseline_prob.argmax(axis=1) == expected).astype(float)
    groups: dict[str, list[float]] = defaultdict(list)
    for record, value in zip(records, delta):
        groups[record["group"]].append(float(value))
    group_delta = np.array([np.mean(groups[name]) for name in sorted(groups)])
    rng = np.random.default_rng(917)
    bootstrap = rng.choice(group_delta, size=(1000, len(group_delta)), replace=True).mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    mode = candidate["feature_version"]
    reference = feature_matrix(training_records(data), mode)
    drift = {name: _drift(reference, feature_matrix([r for r in records if r["slice"] == name], mode), candidate["feature_names"]) for name in SLICES}
    evidence = {
        "baseline": baseline_metrics, "candidate": candidate_metrics, "holdout_groups": len(groups), "drift": drift,
        "comparison": {"accuracy_delta": candidate_metrics["overall"]["accuracy"] - baseline_metrics["overall"]["accuracy"],
                       "log_loss_delta": candidate_metrics["overall"]["log_loss"] - baseline_metrics["overall"]["log_loss"],
                       "group_bootstrap_95pct": {"low": float(low), "high": float(high), "replicates": 1000,
                                                 "seed": 917, "unit": "latent-shape-group"}},
    }
    return {"schema": "modelgate.evaluation.v1", "evaluator": {"modelgate": "1.0.0", "numpy": np.__version__}, "dataset": manifest,
            "models": {"baseline_sha256": digest(baseline), "candidate_sha256": digest(candidate),
                       "baseline_features": baseline["feature_version"], "candidate_features": candidate["feature_version"]},
            "policy": dict(POLICY), "evidence": evidence, "gate": assess(evidence)}
