"""Reproducible release experiments from a small command-line interface."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import IntegrityError, digest, load_json, write_json
from .data import generate_dataset
from .evaluation import evaluate
from .model import broken_candidate, train
from .registry import PolicyBlocked, Registry
from .report import render_report


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "os": platform.system(), "architecture": platform.machine(), "modelgate": "1.0.0"}


def demo(output: Path) -> dict[str, Any]:
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    data = generate_dataset()
    baseline, candidate = train(data, "axis-v1"), train(data, "invariant-v2")
    broken = broken_candidate(candidate)
    regression = copy.deepcopy(candidate)
    regression["name"] = "injected-slice-regression"
    regression["bias"][0] += 0.5
    successor = train(data, "invariant-v2", epochs=900)
    successor["name"] = "candidate-900-epochs"
    for name, artifact in [("dataset", data), ("baseline", baseline), ("candidate", candidate),
                           ("broken", broken), ("slice-regression", regression), ("successor", successor)]:
        write_json(artifacts / f"{name}.json", artifact)
    report = evaluate(data, baseline, candidate)
    broken_report = evaluate(data, baseline, broken)
    regression_report = evaluate(data, baseline, regression)
    write_json(output / "evaluation.json", report)
    write_json(output / "broken-evaluation.json", broken_report)
    write_json(output / "slice-regression.json", regression_report)
    registry = Registry(artifacts / "registry")
    first = registry.promote(data, baseline, candidate, report)
    saved_pointer = (registry.root / "current.json").read_bytes()
    blocked: dict[str, bool] = {}
    for name, model, evidence in [("broken", broken, broken_report), ("slice_regression", regression, regression_report)]:
        try:
            registry.promote(data, baseline, model, evidence)
        except PolicyBlocked:
            blocked[name] = True
        else:
            raise RuntimeError(f"Failure drill unexpectedly promoted {name}")
    forged = copy.deepcopy(broken_report)
    forged["gate"] = {"allowed": True, "decision": "PASS", "reasons": []}
    try:
        registry.promote(data, baseline, broken, forged)
    except IntegrityError:
        tamper_blocked = (registry.root / "current.json").read_bytes() == saved_pointer
    else:
        raise RuntimeError("Forged evaluation report was accepted")
    successor_report = evaluate(data, candidate, successor)
    write_json(output / "successor-evaluation.json", successor_report)
    second = registry.promote(data, candidate, successor, successor_report)
    rollback = registry.rollback()
    restored = registry.current()
    rollback_verified = restored is not None and digest(restored["model"]) == digest(candidate)
    if not rollback_verified or not tamper_blocked:
        raise RuntimeError("Failure drills did not preserve expected registry state")
    summary = {"schema": "modelgate.demo.v1", "seed": 42, "environment": environment(),
               "decision": report["gate"]["decision"], "candidate_accuracy": report["evidence"]["candidate"]["overall"]["accuracy"],
               "baseline_accuracy": report["evidence"]["baseline"]["overall"]["accuracy"],
               "tamper_blocked": tamper_blocked, "broken_candidate_blocked": blocked["broken"],
               "slice_regression": {"accuracy": regression_report["evidence"]["candidate"]["overall"]["accuracy"],
                                    "reasons": regression_report["gate"]["reasons"], "blocked": blocked["slice_regression"],
                                    "fault": "Added +0.5 to cube-class bias in a copy of the trained candidate"},
               "rollback_verified": rollback_verified, "releases": {"initial": first, "successor": second, "rollback": rollback},
               "hashes": {"dataset": digest(data), "candidate": digest(candidate), "evaluation": digest(report)}}
    write_json(output / "summary.json", summary)
    render_report(report, data, summary, output / "report.html")
    return summary


def benchmark(output: Path, repeats: int = 3) -> dict[str, Any]:
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        data = generate_dataset()
        generated = time.perf_counter()
        baseline, candidate = train(data, "axis-v1"), train(data, "invariant-v2")
        trained = time.perf_counter()
        report = evaluate(data, baseline, candidate)
        completed = time.perf_counter()
        runs.append({"generate_seconds": generated - start, "train_both_seconds": trained - generated,
                     "evaluate_seconds": completed - trained, "total_seconds": completed - start})
    result = {"schema": "modelgate.benchmark.v1", "environment": environment(), "repeats": repeats,
              "scope": "Local CPU wall time including schema/hash checks; no network, no GPU, no registry writes",
              "runs": runs, "median_seconds": {key: statistics.median(run[key] for run in runs) for key in runs[0]},
              "dataset_records": len(data["records"]), "points_per_cloud": data["generator"]["points_per_cloud"],
              "dataset_sha256": digest(data), "evaluation_sha256": digest(report)}
    write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train, evaluate, and gate versioned geometry model releases")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate the seeded synthetic point-cloud dataset")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="Fit a real softmax classifier; persist inspectable JSON weights")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--features", choices=["axis-v1", "invariant-v2"], required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=700)
    for name in ("evaluate", "promote"):
        command = commands.add_parser(name)
        for flag in ("dataset", "baseline", "candidate"):
            command.add_argument(f"--{flag}", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--out", type=Path, required=True)
        else:
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--registry", type=Path, required=True)
    for name in ("rollback", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--registry", type=Path, required=True)
    demonstration = commands.add_parser("demo", help="Run training, gates, failure drills, promotion, and rollback")
    demonstration.add_argument("--out", type=Path, default=Path("runs/demo"))
    performance = commands.add_parser("benchmark", help="Measure the real local CPU pipeline")
    performance.add_argument("--out", type=Path, default=Path("runs/benchmark.json"))
    performance.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        code = 0
        if args.command == "generate":
            result = {"dataset_sha256": write_json(args.out, generate_dataset(seed=args.seed))}
        elif args.command == "train":
            result = {"model_sha256": write_json(args.out, train(load_json(args.dataset), args.features, args.epochs))}
        elif args.command in {"evaluate", "promote"}:
            data, baseline, candidate = (load_json(getattr(args, name)) for name in ("dataset", "baseline", "candidate"))
            if args.command == "evaluate":
                report = evaluate(data, baseline, candidate)
                write_json(args.out, report)
                result = {"report_sha256": digest(report), "gate": report["gate"]}
                code = 0 if report["gate"]["allowed"] else 3
            else:
                result = Registry(args.registry).promote(data, baseline, candidate, load_json(args.report))
        elif args.command == "rollback":
            result = Registry(args.registry).rollback()
        elif args.command == "inspect":
            current = Registry(args.registry).current()
            result = {"current": None if current is None else {k: v for k, v in current.items() if k != "model"}}
        elif args.command == "demo":
            result = demo(args.out)
        else:
            if args.repeats < 1 or args.repeats > 30:
                raise ValueError("Use 1–30 benchmark repeats")
            result = benchmark(args.out, args.repeats)
        print(json.dumps(result, indent=2, allow_nan=False))
        return code
    except PolicyBlocked as exc:
        print(json.dumps({"error": "policy_blocked", "message": str(exc)}), file=sys.stderr)
        return 3
    except (IntegrityError, ValueError, OSError) as exc:
        print(json.dumps({"error": "invalid_artifact", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
