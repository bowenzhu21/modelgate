"""Synthetic surface point clouds with a group-preserving evaluation split."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .artifacts import IntegrityError, digest, finite_tree, require_keys

LABELS = ["cube", "sphere", "cylinder"]
SLICES = ["clean", "noisy", "rotated", "scale"]
SCHEMA = "modelgate.point-cloud.v1"


def _surface(label: str, rng: np.random.Generator, count: int) -> np.ndarray:
    if label == "sphere":
        points = rng.normal(size=(count, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    elif label == "cube":
        points = rng.uniform(-1, 1, size=(count, 3))
        axes = rng.integers(0, 3, count)
        points[np.arange(count), axes] = rng.choice([-1.0, 1.0], count)
    else:
        # Unit cylinder surface: side area 4*pi, cap area 2*pi at height 2.
        angles = rng.uniform(0, 2 * np.pi, count)
        points = np.column_stack((np.cos(angles), np.sin(angles), rng.uniform(-1, 1, count)))
        caps = rng.random(count) < 1 / 3
        radii = np.sqrt(rng.random(np.count_nonzero(caps)))
        points[caps, :2] *= radii[:, None]
        points[caps, 2] = rng.choice([-1.0, 1.0], np.count_nonzero(caps))
        points[:, 2] *= rng.uniform(0.85, 1.15)
    return points * rng.uniform(0.8, 1.2)


def generate_dataset(seed: int = 42, train_groups: int = 36, test_groups: int = 18, points: int = 256) -> dict[str, Any]:
    if train_groups < 8 or test_groups < 8 or points < 32:
        raise ValueError("Use at least 8 train/test groups per class and 32 points")
    records: list[dict[str, Any]] = []
    for class_index, label in enumerate(LABELS):
        order = np.random.default_rng(seed + class_index * 1009).permutation(train_groups + test_groups)
        train_ids = set(int(x) for x in order[:train_groups])
        for index in range(train_groups + test_groups):
            # Local streams make each latent shape independent of iteration order.
            rng = np.random.default_rng(np.random.SeedSequence([seed, class_index, index]))
            cloud = _surface(label, rng, points)
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            variants = {
                "clean": cloud,
                "noisy": cloud + rng.normal(0, 0.055, size=cloud.shape),
                "rotated": cloud @ rotation,
                "scale": cloud * (0.35 if index % 2 else 2.8),
            }
            group = f"{label}-{index:04d}"
            for slice_name, sample in variants.items():
                records.append({
                    "id": f"{group}/{slice_name}", "group": group, "label": label,
                    "split": "train" if index in train_ids else "test", "slice": slice_name,
                    "points": np.round(sample, 10).tolist(),
                })
    result = {
        "schema": SCHEMA,
        "generator": {"name": "surface-cloud-v1", "seed": seed, "points_per_cloud": points,
                      "train_groups_per_class": train_groups, "test_groups_per_class": test_groups},
        "labels": list(LABELS), "slices": list(SLICES), "records": records,
    }
    validate_dataset(result)
    return result


def validate_dataset(data: Any) -> None:
    require_keys(data, {"schema", "generator", "labels", "slices", "records"}, "Dataset")
    finite_tree(data)
    if data["schema"] != SCHEMA or data["labels"] != LABELS or data["slices"] != SLICES:
        raise IntegrityError("Unsupported dataset schema, labels, or slices")
    generator = data["generator"]
    require_keys(generator, {"name", "seed", "points_per_cloud", "train_groups_per_class", "test_groups_per_class"}, "Generator")
    if generator["name"] != "surface-cloud-v1":
        raise IntegrityError("Unsupported generator")
    for key in ("seed", "points_per_cloud", "train_groups_per_class", "test_groups_per_class"):
        if type(generator[key]) is not int or generator[key] < 0:
            raise IntegrityError(f"Invalid generator {key}")
    if generator["points_per_cloud"] < 32 or min(generator["train_groups_per_class"], generator["test_groups_per_class"]) < 8:
        raise IntegrityError("Dataset is too small for this gate")
    if not isinstance(data["records"], list) or not data["records"]:
        raise IntegrityError("Dataset has no records")
    ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in data["records"]:
        require_keys(record, {"id", "group", "label", "split", "slice", "points"}, "Record")
        if not isinstance(record["id"], str) or record["id"] in ids:
            raise IntegrityError("Duplicate or invalid sample ID")
        if not isinstance(record["group"], str) or not record["group"]:
            raise IntegrityError("Invalid shape group")
        ids.add(record["id"])
        if any(not isinstance(record[key], str) for key in ("label", "slice", "split")) or record["label"] not in LABELS or record["slice"] not in SLICES or record["split"] not in {"train", "test"}:
            raise IntegrityError("Invalid record label, slice, or split")
        if record["id"] != f"{record['group']}/{record['slice']}":
            raise IntegrityError("Sample ID does not match its group/slice")
        try:
            array = np.asarray(record["points"], dtype=float)
        except (ValueError, TypeError) as exc:
            raise IntegrityError("Invalid point coordinates") from exc
        if array.shape != (generator["points_per_cloud"], 3) or not np.isfinite(array).all():
            raise IntegrityError("Point cloud has wrong dimensions or non-finite coordinates")
        if np.max(np.abs(array)) > 1e6 or np.linalg.norm(array.std(axis=0)) < 1e-8:
            raise IntegrityError("Point cloud is degenerate or out of schema bounds")
        # JSON strings and booleans must not silently become numerical coordinates.
        if any(type(v) not in (int, float) for row in record["points"] for v in row):
            raise IntegrityError("Point coordinates must be JSON numbers")
        groups[record["group"]].append(record)
    counts: Counter[tuple[str, str]] = Counter()
    for group, records in groups.items():
        if len({r["split"] for r in records}) != 1:
            raise IntegrityError(f"Train/test leakage within shape group {group}")
        if len({r["label"] for r in records}) != 1:
            raise IntegrityError(f"Inconsistent group label: {group}")
        if sorted(r["slice"] for r in records) != sorted(SLICES):
            raise IntegrityError(f"Group {group} must contain each slice exactly once")
        counts[(records[0]["split"], records[0]["label"])] += 1
    for split in ("train", "test"):
        for label in LABELS:
            if counts[(split, label)] != generator[f"{split}_groups_per_class"]:
                raise IntegrityError(f"Unexpected {split}/{label} group count")


def training_records(data: dict[str, Any], slices: tuple[str, ...] | list[str] = ("clean",)) -> list[dict[str, Any]]:
    return [r for r in data["records"] if r["split"] == "train" and r["slice"] in slices]


def dataset_manifest(data: dict[str, Any]) -> dict[str, Any]:
    validate_dataset(data)
    return {"sha256": digest(data), "schema": data["schema"], "generator": data["generator"],
            "records": len(data["records"]), "train_sha256": digest(training_records(data, SLICES)),
            "holdout_groups": data["generator"]["test_groups_per_class"] * len(LABELS)}
