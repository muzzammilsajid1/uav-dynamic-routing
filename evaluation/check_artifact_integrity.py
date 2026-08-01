"""Verify raw artifact completeness, uniqueness, and manifest traceability."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_file(
    path: Path,
    *,
    expected_manifest_hash: str,
    expected_run_ids: set[str],
) -> dict[str, object]:
    rows = _rows(path)
    run_ids = [row["run_id"] for row in rows]
    hashes = {row["manifest_sha256"] for row in rows}
    if len(rows) != len(expected_run_ids):
        raise RuntimeError(
            f"{path}: expected {len(expected_run_ids)} rows, found {len(rows)}"
        )
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError(f"{path}: duplicate run IDs")
    actual_run_ids = set(run_ids)
    if actual_run_ids != expected_run_ids:
        missing = sorted(expected_run_ids - actual_run_ids)[:5]
        unexpected = sorted(actual_run_ids - expected_run_ids)[:5]
        raise RuntimeError(
            f"{path}: incomplete Cartesian coverage; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if hashes != {expected_manifest_hash}:
        raise RuntimeError(f"{path}: manifest hash mismatch")
    return {
        "artifact_type": "route_runs",
        "path": str(path),
        "rows": len(rows),
        "unique_run_ids": len(set(run_ids)),
        "manifest_sha256": expected_manifest_hash,
    }


def _check_event_file(
    path: Path,
    *,
    route_rows: list[dict[str, str]],
    classical: bool,
) -> dict[str, object]:
    event_rows = _rows(path)
    expected_keys: set[tuple[str, int]] = set()
    for route in route_rows:
        if classical:
            events = json.loads(route["replan_events_json"])
            count = sum(
                str(event["reason"]).startswith("dynamic_change:")
                for event in events
            )
        else:
            count = int(route["dynamic_event_count"])
        expected_keys.update(
            (route["run_id"], index) for index in range(1, count + 1)
        )
    actual_keys = {
        (row["run_id"], int(row["event_index"])) for row in event_rows
    }
    if len(event_rows) != len(actual_keys):
        raise RuntimeError(f"{path}: duplicate event keys")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:5]
        unexpected = sorted(actual_keys - expected_keys)[:5]
        raise RuntimeError(
            f"{path}: event coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "artifact_type": "adaptability_events",
        "path": str(path),
        "rows": len(event_rows),
        "unique_event_keys": len(actual_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "manifests" / "benchmark_v2.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "research_experiments.json",
    )
    parser.add_argument(
        "--classical",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_suite_raw.csv",
    )
    parser.add_argument(
        "--classical-events",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / "classical_adaptability_events.csv"
        ),
    )
    parser.add_argument("--rl", nargs="*", type=Path, default=[])
    parser.add_argument("--rl-events", nargs="*", type=Path, default=[])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "integrity_report.json",
    )
    args = parser.parse_args()

    suite = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (
        suite.get("dynamics_timing") != "post_move_observed"
        or config.get("dynamics_timing") != "post_move_observed"
    ):
        raise RuntimeError(
            "Manifest and training configuration must both use the "
            "post_move_observed timing contract"
        )
    scenario_count = len(suite["scenarios"])
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    scenario_ids = {scenario["scenario_id"] for scenario in suite["scenarios"]}
    planners = {"astar", "dijkstra", "dstar_lite"}
    classical_run_ids = {
        f"{scenario_id}:{planner}:r{repetition:02d}"
        for scenario_id in scenario_ids
        for planner in planners
        for repetition in range(1, args.repetitions + 1)
    }
    reports = [
        _check_file(
            args.classical,
            expected_manifest_hash=manifest_hash,
            expected_run_ids=classical_run_ids,
        )
    ]
    classical_rows = _rows(args.classical)
    reports.append(
        _check_event_file(
            args.classical_events,
            route_rows=classical_rows,
            classical=True,
        )
    )
    snapshot_path = (
        PROJECT_ROOT / "evaluation" / "results" / "training_source_snapshot.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if args.rl_events and len(args.rl_events) != len(args.rl):
        raise RuntimeError("--rl and --rl-events must have the same length")
    rl_event_paths = args.rl_events or [
        path.with_name(path.name.replace("_raw.csv", "_adaptability_events.csv"))
        for path in args.rl
    ]
    for path, event_path in zip(args.rl, rl_event_paths):
        rows = _rows(path)
        if not rows:
            raise RuntimeError(f"{path}: no result rows")
        variants = {row["variant"] for row in rows}
        if len(variants) != 1:
            raise RuntimeError(f"{path}: expected one variant, found {variants}")
        variant = variants.pop()
        if variant not in config["variants"]:
            raise RuntimeError(f"{path}: unknown variant {variant}")
        seeds = {int(seed) for seed in config["policy_seeds"]}
        actual_seeds = {int(row["policy_seed"]) for row in rows}
        if actual_seeds != seeds:
            raise RuntimeError(
                f"{path}: expected policy seeds {sorted(seeds)}, "
                f"found {sorted(actual_seeds)}"
            )
        if config["variants"][variant]["observation_mode"] == "full":
            evaluated_scenarios = {
                scenario["scenario_id"]
                for scenario in suite["scenarios"]
                if int(scenario["grid_size"]) == int(config["grid_size"])
            }
        else:
            evaluated_scenarios = scenario_ids
        expected_run_ids = {
            f"{variant}:seed{seed:03d}:{scenario_id}:r{repetition:02d}"
            for seed in seeds
            for scenario_id in evaluated_scenarios
            for repetition in range(1, args.repetitions + 1)
        }
        reports.append(
            _check_file(
                path,
                expected_manifest_hash=manifest_hash,
                expected_run_ids=expected_run_ids,
            )
        )
        reports.append(
            _check_event_file(
                event_path,
                route_rows=rows,
                classical=False,
            )
        )
        for seed in seeds:
            metadata_path = (
                PROJECT_ROOT
                / "evaluation"
                / "results"
                / f"training_{variant}_seed_{seed:03d}.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            training = metadata["training"]
            if training.get("smoke_test") or not training.get("stages"):
                raise RuntimeError(f"{metadata_path}: incomplete training")
            if training["config"].get("dynamics_timing") != "post_move_observed":
                raise RuntimeError(
                    f"{metadata_path}: incorrect dynamics timing contract"
                )
            source = metadata.get("training_source_snapshot", {})
            if source.get("sha256") != snapshot.get("sha256"):
                raise RuntimeError(
                    f"{metadata_path}: training-source snapshot mismatch"
                )

    report = {
        "status": "passed",
        "suite_id": suite["suite_id"],
        "dynamics_timing": suite.get("dynamics_timing"),
        "training_source_sha256": snapshot["sha256"],
        "scenario_count": scenario_count,
        "repetitions": args.repetitions,
        "artifacts": reports,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Artifact integrity passed for {len(reports)} raw result files")


if __name__ == "__main__":
    main()
