"""A local content-addressed registry with locked, atomic release pointers."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .artifacts import IntegrityError, canonical, digest, load_json, require_keys, write_json
from .data import LABELS, SCHEMA, SLICES
from .evaluation import POLICY, assess, evaluate
from .model import validate_model


class PolicyBlocked(ValueError):
    """Valid evidence did not satisfy the release policy."""


class Registry:
    """Local POSIX registry. Treat the directory as trusted storage, not a security boundary."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "objects").mkdir(exist_ok=True)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.root / ".lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validate_address(sha: str) -> None:
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise IntegrityError("Invalid content address")

    def _get(self, sha: str) -> dict[str, Any]:
        self._validate_address(sha)
        value = load_json(self.root / "objects" / f"{sha}.json")
        if digest(value) != sha:
            raise IntegrityError(f"Registry object hash mismatch: {sha}")
        return value

    def _store(self, value: dict[str, Any]) -> str:
        sha = digest(value)
        path = self.root / "objects" / f"{sha}.json"
        if path.exists():
            self._get(sha)
        else:
            # Only called under the registry's writer lock. Content is published atomically.
            write_json(path, value)
            os.chmod(path, 0o444)
        return sha

    def _load_release(self, sha: str) -> dict[str, Any]:
        try:
            return self._validate_release(sha)
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise IntegrityError(f"Malformed stored release evidence: {exc}") from exc

    def _validate_release(self, sha: str) -> dict[str, Any]:
        """Validate one complete release before exposing it or restoring its pointer.

        Parent addresses are validated syntactically, not recursively: historical
        corruption must not prevent reading an otherwise intact active release.
        The target's full model/baseline/evaluation closure is checked on rollback.
        """
        release = self._get(sha)
        if not isinstance(release, dict):
            raise IntegrityError("Release must be an object")
        fields = {"schema", "model_sha256", "baseline_sha256", "evaluation_sha256",
                  "dataset_sha256", "parent_release_sha256", "operation"}
        if release.get("operation") == "rollback":
            fields.add("restored_release_sha256")
        require_keys(release, fields, "Release")
        if release["schema"] != "modelgate.release.v1" or not isinstance(release["operation"], str) or release["operation"] not in {"promote", "rollback"}:
            raise IntegrityError("Unsupported release schema")
        for name in ("model_sha256", "baseline_sha256", "evaluation_sha256", "dataset_sha256"):
            self._validate_address(release[name])
        if release["parent_release_sha256"] is not None:
            self._validate_address(release["parent_release_sha256"])
        if release["operation"] == "rollback":
            self._validate_address(release["restored_release_sha256"])
            if release["parent_release_sha256"] is None:
                raise IntegrityError("Rollback release must retain its parent")
        model = self._get(release["model_sha256"])
        validate_model(model)
        baseline = self._get(release["baseline_sha256"])
        validate_model(baseline)
        evaluation = self._get(release["evaluation_sha256"])
        require_keys(evaluation, {"schema", "evaluator", "dataset", "models", "policy", "evidence", "gate"}, "Evaluation")
        if evaluation["schema"] != "modelgate.evaluation.v1":
            raise IntegrityError("Unsupported evaluation schema")
        require_keys(evaluation["evaluator"], {"modelgate", "numpy"}, "Evaluator")
        if evaluation["evaluator"]["modelgate"] != "1.0.0" or not isinstance(evaluation["evaluator"]["numpy"], str) or not evaluation["evaluator"]["numpy"]:
            raise IntegrityError("Unsupported evaluator identity")
        if canonical(evaluation["policy"]) != canonical(POLICY):
            raise IntegrityError("Stored evaluation uses an unsupported release policy")
        bindings = evaluation["models"]
        require_keys(bindings, {"candidate_sha256", "baseline_sha256", "candidate_features", "baseline_features"}, "Evaluation models")
        if bindings["candidate_sha256"] != release["model_sha256"] or bindings["baseline_sha256"] != release["baseline_sha256"]:
            raise IntegrityError("Release model/baseline bindings do not match evaluation")
        if bindings["candidate_features"] != model["feature_version"] or bindings["baseline_features"] != baseline["feature_version"]:
            raise IntegrityError("Evaluation feature bindings do not match models")
        manifest = evaluation["dataset"]
        require_keys(manifest, {"sha256", "schema", "generator", "records", "train_sha256", "holdout_groups"}, "Evaluation dataset")
        self._validate_address(manifest["sha256"])
        self._validate_address(manifest["train_sha256"])
        if manifest["schema"] != SCHEMA or manifest["sha256"] != release["dataset_sha256"]:
            raise IntegrityError("Release dataset binding does not match evaluation")
        for artifact in (model, baseline):
            if artifact["training"]["dataset_sha256"] != manifest["sha256"] or artifact["training"]["train_sha256"] != manifest["train_sha256"]:
                raise IntegrityError("Stored model provenance does not match evaluation dataset")
        generator = manifest["generator"]
        require_keys(generator, {"name", "seed", "points_per_cloud", "train_groups_per_class", "test_groups_per_class"}, "Evaluation generator")
        if generator["name"] != "surface-cloud-v1" or any(type(generator[k]) is not int or generator[k] < 0 for k in ("seed", "points_per_cloud", "train_groups_per_class", "test_groups_per_class")):
            raise IntegrityError("Invalid evaluation generator")
        if generator["points_per_cloud"] < 32 or min(generator["train_groups_per_class"], generator["test_groups_per_class"]) < 8:
            raise IntegrityError("Stored evaluation dataset is too small")
        groups = generator["test_groups_per_class"] * len(LABELS)
        records = (generator["train_groups_per_class"] + generator["test_groups_per_class"]) * len(LABELS) * len(SLICES)
        if type(manifest["records"]) is not int or type(manifest["holdout_groups"]) is not int or manifest["records"] != records or manifest["holdout_groups"] != groups:
            raise IntegrityError("Stored evaluation dataset counts are inconsistent")
        gate = assess(evaluation["evidence"])
        if canonical(gate) != canonical(evaluation["gate"]) or not gate["allowed"]:
            raise IntegrityError("Stored evaluation is not backed by passing policy evidence")
        if evaluation["evidence"]["holdout_groups"] != groups or any(evaluation["evidence"][name]["overall"]["samples"] != groups * len(SLICES) for name in ("candidate", "baseline")):
            raise IntegrityError("Stored evaluation support does not match dataset manifest")
        return {"release_sha256": sha, "release": release, "model": model}

    def current(self) -> dict[str, Any] | None:
        path = self.root / "current.json"
        if not path.exists():
            return None
        pointer = load_json(path)
        require_keys(pointer, {"release_sha256"}, "Current release pointer")
        return self._load_release(pointer["release_sha256"])

    def promote(self, data: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
        # Re-run the evaluator; never trust scores or a hand-edited PASS flag.
        recomputed = evaluate(data, baseline, candidate)
        if canonical(recomputed) != canonical(report):
            raise IntegrityError("Evaluation evidence mismatch: report differs from re-evaluation")
        if not recomputed["gate"]["allowed"]:
            raise PolicyBlocked("; ".join(recomputed["gate"]["reasons"]))
        with self._lock():
            previous = self.current()
            model_sha = digest(candidate)
            evaluation_sha = digest(recomputed)
            if previous and previous["release"]["model_sha256"] == model_sha and previous["release"]["evaluation_sha256"] == evaluation_sha:
                return {"release_sha256": previous["release_sha256"], "changed": False}
            if previous and digest(baseline) != previous["release"]["model_sha256"]:
                raise PolicyBlocked("Baseline is not the active model; re-evaluate against the current release")
            self._store(candidate)
            self._store(recomputed)
            baseline_sha = self._store(baseline)
            release = {"schema": "modelgate.release.v1", "model_sha256": model_sha,
                       "baseline_sha256": baseline_sha, "evaluation_sha256": evaluation_sha,
                       "dataset_sha256": digest(data), "parent_release_sha256": previous["release_sha256"] if previous else None,
                       "operation": "promote"}
            release_sha = self._store(release)
            write_json(self.root / "current.json", {"release_sha256": release_sha})
            return {"release_sha256": release_sha, "changed": True}

    def rollback(self) -> dict[str, Any]:
        """Restore the last release's model while keeping an immutable audit event."""
        with self._lock():
            active = self.current()
            if not active or not active["release"]["parent_release_sha256"]:
                raise PolicyBlocked("No previous release is available")
            previous_sha = active["release"]["parent_release_sha256"]
            # Use exactly the same closure validation as current(), including the
            # target baseline, before publishing a pointer to that release.
            previous = self._load_release(previous_sha)["release"]
            release = {**previous, "operation": "rollback", "parent_release_sha256": active["release_sha256"],
                       "restored_release_sha256": previous_sha}
            release_sha = self._store(release)
            write_json(self.root / "current.json", {"release_sha256": release_sha})
            return {"release_sha256": release_sha, "restored_release_sha256": previous_sha, "changed": True}
