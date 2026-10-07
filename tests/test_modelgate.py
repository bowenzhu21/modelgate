from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from modelgate.artifacts import IntegrityError, canonical, digest, load_json, write_json
from modelgate.data import generate_dataset, training_records, validate_dataset
from modelgate.evaluation import POLICY, assess, evaluate
from modelgate.model import broken_candidate, features, train, validate_model
from modelgate.registry import PolicyBlocked, Registry


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset()
        cls.baseline = train(cls.data, "axis-v1")
        cls.candidate = train(cls.data, "invariant-v2")
        cls.report = evaluate(cls.data, cls.baseline, cls.candidate)
        cls.successor = train(cls.data, "invariant-v2", epochs=900)
        cls.successor_report = evaluate(cls.data, cls.candidate, cls.successor)

    def test_seed_is_reproducible(self):
        self.assertEqual(digest(self.data), digest(generate_dataset()))
        self.assertEqual(digest(self.candidate), digest(train(self.data, "invariant-v2")))

    def test_grouped_holdout_is_disjoint_and_complete(self):
        train_groups = {r["group"] for r in self.data["records"] if r["split"] == "train"}
        test_groups = {r["group"] for r in self.data["records"] if r["split"] == "test"}
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(len(test_groups), 54)
        for group in test_groups:
            self.assertEqual(len([r for r in self.data["records"] if r["group"] == group]), 4)

    def test_split_leakage_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["records"][0]["split"] = "test" if data["records"][0]["split"] == "train" else "train"
        with self.assertRaisesRegex(IntegrityError, "leakage"):
            validate_dataset(data)

    def test_training_does_not_consume_holdout(self):
        changed = copy.deepcopy(self.data)
        for record in changed["records"]:
            if record["split"] == "test":
                record["points"] = (np.asarray(record["points"]) * 1.4).tolist()
        retrained = train(changed, "invariant-v2")
        for key in ("weights", "bias", "normalizer"):
            self.assertEqual(self.candidate[key], retrained[key])
        self.assertNotEqual(self.candidate["training"]["dataset_sha256"], retrained["training"]["dataset_sha256"])
        self.assertEqual(self.candidate["training"]["fit_sha256"], retrained["training"]["fit_sha256"])

    def test_feature_invariance_is_mathematical_not_a_score_mock(self):
        sample = training_records(self.data)[0]["points"]
        rotation, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(3, 3)))
        transformed = np.asarray(sample) @ rotation * 3.1 + np.array([8.0, -9.0, 2.0])
        np.testing.assert_allclose(features(sample, "invariant-v2"), features(transformed, "invariant-v2"), atol=1e-10)

    def test_missing_slice_duplicate_id_and_nonnumeric_cloud_rejected(self):
        for mutate in [lambda d: d["records"].pop(),
                       lambda d: d["records"].append(d["records"][0]),
                       lambda d: d["records"][0]["points"][0].__setitem__(0, "1.0")]:
            data = copy.deepcopy(self.data)
            mutate(data)
            with self.assertRaises(IntegrityError):
                validate_dataset(data)

    def test_nan_and_degenerate_cloud_rejected(self):
        for points in [[[0.0] * 3] * 256, [[float("nan")] * 3] * 256]:
            data = copy.deepcopy(self.data)
            data["records"][0]["points"] = points
            with self.assertRaises(IntegrityError):
                validate_dataset(data)

    def test_model_schema_rejects_invalid_training_provenance(self):
        mutations = [("sample_count", True), ("epochs", -1), ("learning_rate", float("nan")),
                     ("l2", "0.1"), ("slices", ["test"]), ("slices", ["clean", "clean"])]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                model = copy.deepcopy(self.candidate)
                model["training"][key] = value
                with self.assertRaises(IntegrityError):
                    validate_model(model)

    def test_unhashable_feature_version_and_invalid_weights_are_rejected(self):
        for key, value in [("feature_version", []), ("weights", [[0.0]]), ("bias", [True, 0, 0])]:
            model = copy.deepcopy(self.candidate)
            model[key] = value
            with self.assertRaises(IntegrityError):
                validate_model(model)

    def test_dataset_and_fit_hash_mismatch_fail_closed(self):
        for key in ("dataset_sha256", "train_sha256", "fit_sha256"):
            model = copy.deepcopy(self.candidate)
            model["training"][key] = "0" * 64
            with self.assertRaisesRegex(IntegrityError, "hash"):
                evaluate(self.data, self.baseline, model)

    def test_real_candidate_passes_and_improves_baseline(self):
        self.assertTrue(self.report["gate"]["allowed"])
        evidence = self.report["evidence"]
        self.assertGreater(evidence["candidate"]["overall"]["accuracy"], .95)
        self.assertGreater(evidence["comparison"]["accuracy_delta"], .30)
        self.assertEqual(evidence["holdout_groups"], 54)

    def test_real_bad_candidate_is_rejected(self):
        report = evaluate(self.data, self.baseline, broken_candidate(self.candidate))
        self.assertFalse(report["gate"]["allowed"])
        self.assertAlmostEqual(report["evidence"]["candidate"]["overall"]["accuracy"], 1 / 3)

    def test_slice_class_failure_blocks_even_with_good_aggregate(self):
        model = copy.deepcopy(self.candidate)
        model["bias"][0] += 0.5
        report = evaluate(self.data, self.baseline, model)
        self.assertGreater(report["evidence"]["candidate"]["overall"]["accuracy"], POLICY["minimum_accuracy"])
        self.assertFalse(report["gate"]["allowed"])
        self.assertTrue(any("noisy cylinder recall" in reason for reason in report["gate"]["reasons"]))

    def test_missing_nan_inconsistent_and_boolean_metric_fail_closed(self):
        mutations = [lambda e: e["candidate"]["slices"].pop("scale"),
                     lambda e: e["candidate"]["overall"].__setitem__("accuracy", float("nan")),
                     lambda e: e["candidate"]["overall"].__setitem__("accuracy", True),
                     lambda e: e["candidate"]["overall"].__setitem__("log_loss", .01),
                     lambda e: e["comparison"].__setitem__("accuracy_delta", 1.0)]
        for mutate in mutations:
            evidence = copy.deepcopy(self.report["evidence"])
            mutate(evidence)
            with self.assertRaises(IntegrityError):
                assess(evidence)

    def test_report_policy_does_not_alias_gate_policy(self):
        report = evaluate(self.data, self.baseline, self.candidate)
        report["policy"]["minimum_accuracy"] = 0
        self.assertEqual(POLICY["minimum_accuracy"], .88)

    def test_strict_json_disallows_duplicate_keys_nan_and_infinity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ['{"a": 1, "a": 2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}']:
                path.write_text(text)
                with self.assertRaises(IntegrityError):
                    load_json(path)

    def test_registry_promotes_idempotently_and_validates_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            first = registry.promote(self.data, self.baseline, self.candidate, self.report)
            duplicate = registry.promote(self.data, self.baseline, self.candidate, self.report)
            self.assertTrue(first["changed"])
            self.assertFalse(duplicate["changed"])
            self.assertEqual(first["release_sha256"], duplicate["release_sha256"])
            self.assertEqual(digest(registry.current()["model"]), digest(self.candidate))

    def test_forged_report_and_mutated_model_cannot_move_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            registry.promote(self.data, self.baseline, self.candidate, self.report)
            pointer = (Path(directory) / "current.json").read_bytes()
            forged = copy.deepcopy(self.report)
            forged["evidence"]["candidate"]["overall"]["accuracy"] = 1.0
            with self.assertRaises(IntegrityError):
                registry.promote(self.data, self.baseline, self.candidate, forged)
            with self.assertRaises(IntegrityError):
                registry.promote(self.data, self.baseline, broken_candidate(self.candidate), self.report)
            self.assertEqual(pointer, (Path(directory) / "current.json").read_bytes())

    def test_corrupted_stored_object_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            registry.promote(self.data, self.baseline, self.candidate, self.report)
            path = Path(directory) / "objects" / f"{digest(self.candidate)}.json"
            os.chmod(path, 0o644)
            altered = copy.deepcopy(self.candidate)
            altered["bias"][0] += 1.0
            path.write_bytes(canonical(altered))
            with self.assertRaisesRegex(IntegrityError, "hash mismatch"):
                registry.current()

    def test_active_baseline_required_before_second_release(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            registry.promote(self.data, self.baseline, self.candidate, self.report)
            wrong_baseline_report = evaluate(self.data, self.baseline, self.successor)
            with self.assertRaisesRegex(PolicyBlocked, "active model"):
                registry.promote(self.data, self.baseline, self.successor, wrong_baseline_report)

    def test_rollback_restores_hash_and_preserves_immutable_history(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            first = registry.promote(self.data, self.baseline, self.candidate, self.report)
            with self.assertRaises(PolicyBlocked):
                registry.rollback()
            second = registry.promote(self.data, self.candidate, self.successor, self.successor_report)
            undone = registry.rollback()
            current = registry.current()
            self.assertEqual(digest(current["model"]), digest(self.candidate))
            self.assertEqual(undone["restored_release_sha256"], first["release_sha256"])
            self.assertEqual(current["release"]["parent_release_sha256"], second["release_sha256"])
            for event in (first, second, undone):
                self.assertTrue((Path(directory) / "objects" / f"{event['release_sha256']}.json").exists())

    def test_concurrent_same_promotion_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            def promote():
                return Registry(directory).promote(self.data, self.baseline, self.candidate, self.report)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: promote(), range(2)))
            self.assertEqual(sum(result["changed"] for result in results), 1)
            self.assertEqual(results[0]["release_sha256"], results[1]["release_sha256"])
            Registry(directory).current()

    def test_rollback_rejects_corrupt_target_closure_without_moving_pointer(self):
        # The target's original baseline is not referenced by the active release.
        # Its corruption previously allowed rollback to publish an unreadable pointer.
        for field in ("baseline_sha256", "model_sha256", "evaluation_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                registry = Registry(directory)
                first = registry.promote(self.data, self.baseline, self.candidate, self.report)
                registry.promote(self.data, self.candidate, self.successor, self.successor_report)
                target = registry._get(first["release_sha256"])
                path = Path(directory) / "objects" / f"{target[field]}.json"
                corrupt = registry._get(target[field])
                corrupt["unexpected_corruption"] = True
                os.chmod(path, 0o644)
                path.write_bytes(canonical(corrupt))
                pointer = (Path(directory) / "current.json").read_bytes()
                if field != "model_sha256":
                    self.assertIsNotNone(registry.current())
                with self.assertRaisesRegex(IntegrityError, "hash mismatch"):
                    registry.rollback()
                self.assertEqual(pointer, (Path(directory) / "current.json").read_bytes())

    def test_rollback_validates_all_target_bindings_and_policy_before_publish(self):
        mutations = {
            "candidate_binding": lambda r, e: r.__setitem__("model_sha256", digest(self.successor)),
            "baseline_binding": lambda r, e: r.__setitem__("baseline_sha256", digest(self.candidate)),
            "dataset_binding": lambda r, e: r.__setitem__("dataset_sha256", "0" * 64),
            "release_schema": lambda r, e: r.__setitem__("schema", "modelgate.release.v999"),
            "release_shape": lambda r, e: r.__setitem__("unexpected", "field"),
            "report_candidate": lambda r, e: e["models"].__setitem__("candidate_sha256", "0" * 64),
            "report_baseline": lambda r, e: e["models"].__setitem__("baseline_sha256", "0" * 64),
            "report_features": lambda r, e: e["models"].__setitem__("candidate_features", "axis-v1"),
            "report_dataset": lambda r, e: e["dataset"].__setitem__("sha256", "0" * 64),
            "report_train": lambda r, e: e["dataset"].__setitem__("train_sha256", "0" * 64),
            "policy": lambda r, e: e["policy"].__setitem__("minimum_accuracy", 0),
            "gate": lambda r, e: e["gate"].__setitem__("allowed", "true"),
            "evidence": lambda r, e: e["evidence"]["candidate"]["overall"].__setitem__("samples", 999),
        }
        # Build valid releases once, then independently inject internally hashed
        # malformed target objects. Checksums alone must not bypass schema/bindings.
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            first = registry.promote(self.data, self.baseline, self.candidate, self.report)
            second = registry.promote(self.data, self.candidate, self.successor, self.successor_report)
            original_target = registry._get(first["release_sha256"])
            original_active = registry._get(second["release_sha256"])
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    target = copy.deepcopy(original_target)
                    proof = copy.deepcopy(self.report)
                    mutate(target, proof)
                    with registry._lock():
                        target["evaluation_sha256"] = registry._store(proof)
                        target_sha = registry._store(target)
                        active = {**original_active, "parent_release_sha256": target_sha}
                        active_sha = registry._store(active)
                        write_json(Path(directory) / "current.json", {"release_sha256": active_sha})
                    self.assertIsNotNone(registry.current())
                    pointer = (Path(directory) / "current.json").read_bytes()
                    with self.assertRaises(IntegrityError):
                        registry.rollback()
                    self.assertEqual(pointer, (Path(directory) / "current.json").read_bytes())

    def test_crash_before_pointer_publish_keeps_old_release(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory)
            first = registry.promote(self.data, self.baseline, self.candidate, self.report)
            real_write = write_json
            def fail_pointer(path, value):
                if Path(path).name == "current.json":
                    raise OSError("Simulated crash before pointer publication")
                return real_write(path, value)
            with patch("modelgate.registry.write_json", side_effect=fail_pointer):
                with self.assertRaises(OSError):
                    registry.promote(self.data, self.candidate, self.successor, self.successor_report)
            self.assertEqual(registry.current()["release_sha256"], first["release_sha256"])
            # Orphan objects are harmless; retry completes the same content transaction.
            registry.promote(self.data, self.candidate, self.successor, self.successor_report)
            self.assertEqual(digest(registry.current()["model"]), digest(self.successor))

    def test_cli_invalid_json_has_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"a":NaN}')
            result = subprocess.run([sys.executable, "-m", "modelgate", "train", "--dataset", str(path),
                                     "--features", "invariant-v2", "--out", str(Path(directory) / "model.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stderr)["error"], "invalid_artifact")


if __name__ == "__main__":
    unittest.main()
