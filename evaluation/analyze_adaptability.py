"""Summarize and statistically compare event-level adaptation outcomes."""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.analyze_research_results import _mean_ci
from evaluation.statistical_tests import _mcnemar_exact, _paired_numeric


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: object) -> float:
    if value in {None, "", "None", "NA", "N/A"}:
        return float("nan")
    return float(value)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _change_signature(row: dict[str, str]) -> str:
    if row.get("changed_cells_json"):
        cells = json.loads(row["changed_cells_json"])
    else:
        reason = row.get("change_reason", "")
        payload = reason.split("dynamic_change:", maxsplit=1)[-1]
        try:
            cells = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            cells = [payload]
    normalized = sorted(
        [list(cell) if isinstance(cell, (list, tuple)) else [str(cell)] for cell in cells]
    )
    return json.dumps(normalized, separators=(",", ":"))


def _method_rows(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        rows = _read(path)
        for row in rows:
            if row.get("planner"):
                method = row["planner"]
                seed = "classical"
            else:
                method = f"rl_{row['variant']}"
                seed = row["policy_seed"]
            delta = _optional_float(row.get("optimal_cost_delta"))
            extra = _optional_float(row.get("extra_optimal_cost"))
            if math.isnan(extra) and not math.isnan(delta):
                extra = max(0.0, delta)
            reaction = _optional_float(
                row.get("reaction_time_ms") or row.get("planning_time_ms")
            )
            recovery = _optional_float(row.get("recovery_steps"))
            records.append(
                {
                    "method": method,
                    "seed": str(seed),
                    "scenario_id": row["scenario_id"],
                    "split": row["split"],
                    "repetition": int(row["repetition"]),
                    "change_step": int(row.get("change_step") or 0),
                    "change_signature": _change_signature(row),
                    "post_change_success": _bool(row["post_change_success"]),
                    "recovered": not math.isnan(recovery),
                    "recovery_steps": recovery,
                    "optimal_cost_shock": delta,
                    "extra_optimal_cost": extra,
                    "finite_post_cost": math.isfinite(delta),
                    "reaction_time_ms": reaction,
                    "node_expansions": _optional_float(row.get("node_expansions")),
                }
            )
    return records


def _collapse_repetitions(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        key = (
            row["method"],
            row["seed"],
            row["scenario_id"],
            row["split"],
            row["change_step"],
            row["change_signature"],
        )
        grouped[key].append(row)

    collapsed: list[dict[str, object]] = []
    for key, group in grouped.items():
        first = group[0]
        output = {
            "method": key[0],
            "seed": key[1],
            "scenario_id": key[2],
            "split": key[3],
            "change_step": key[4],
            "change_signature": key[5],
            "post_change_success": first["post_change_success"],
            "recovered": first["recovered"],
            "finite_post_cost": first["finite_post_cost"],
        }
        for metric in (
            "recovery_steps",
            "optimal_cost_shock",
            "extra_optimal_cost",
            "reaction_time_ms",
            "node_expansions",
        ):
            values = [
                float(row[metric])
                for row in group
                if math.isfinite(float(row[metric]))
            ]
            output[metric] = statistics.mean(values) if values else float("nan")
        collapsed.append(output)
    return collapsed


def _metric_mean(group: list[dict[str, object]], metric: str) -> float:
    values = [
        float(row[metric])
        for row in group
        if math.isfinite(float(row[metric]))
    ]
    return statistics.mean(values) if values else float("nan")


def _summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for row in records:
        expanded.append(row)
        expanded.append({**row, "split": "all_dynamic"})

    by_unit: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in expanded:
        unit = (
            str(row["scenario_id"])
            if row["seed"] == "classical"
            else str(row["seed"])
        )
        by_unit[(str(row["method"]), str(row["split"]), unit)].append(row)

    unit_rows: list[dict[str, object]] = []
    for (method, split, unit), group in by_unit.items():
        unit_rows.append(
            {
                "method": method,
                "split": split,
                "unit": unit,
                "events": len(group),
                "post_change_success": statistics.mean(
                    bool(row["post_change_success"]) for row in group
                ),
                "recovery_rate": statistics.mean(
                    bool(row["recovered"]) for row in group
                ),
                "finite_post_cost_rate": statistics.mean(
                    bool(row["finite_post_cost"]) for row in group
                ),
                "recovery_steps": _metric_mean(group, "recovery_steps"),
                "optimal_cost_shock": _metric_mean(group, "optimal_cost_shock"),
                "extra_optimal_cost": _metric_mean(group, "extra_optimal_cost"),
                "reaction_time_ms": _metric_mean(group, "reaction_time_ms"),
                "node_expansions": _metric_mean(group, "node_expansions"),
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in unit_rows:
        grouped[(str(row["method"]), str(row["split"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (method, split), group in sorted(grouped.items()):
        output: dict[str, object] = {
            "method": method,
            "split": split,
            "independent_units": len(group),
            "events_per_unit_mean": statistics.mean(
                float(row["events"]) for row in group
            ),
        }
        for metric in (
            "post_change_success",
            "recovery_rate",
            "finite_post_cost_rate",
            "recovery_steps",
            "optimal_cost_shock",
            "extra_optimal_cost",
            "reaction_time_ms",
            "node_expansions",
        ):
            values = [
                float(row[metric])
                for row in group
                if math.isfinite(float(row[metric]))
            ]
            mean, ci = _mean_ci(values)
            output[f"{metric}_mean"] = mean
            output[f"{metric}_95ci"] = ci
        summaries.append(output)
    return summaries


def _event_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["split"],
        row["scenario_id"],
        row["change_step"],
        row["change_signature"],
    )


def _paired_tests(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method_seed: dict[
        str, dict[str, dict[tuple[object, ...], dict[str, object]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in records:
        by_method_seed[str(row["method"])][str(row["seed"])][_event_key(row)] = row

    results: list[dict[str, object]] = []
    for left_name, right_name in itertools.combinations(sorted(by_method_seed), 2):
        left = by_method_seed[left_name]
        right = by_method_seed[right_name]
        common_seeds = sorted(set(left) & set(right))
        seed_pairs = (
            [(seed, seed) for seed in common_seeds]
            or [
                (left_seed, right_seed)
                for left_seed in sorted(left)
                for right_seed in sorted(right)
                if left_seed == "classical" or right_seed == "classical"
            ]
        )
        for left_seed, right_seed in seed_pairs:
            common_keys = sorted(set(left[left_seed]) & set(right[right_seed]))
            left_rows = [left[left_seed][key] for key in common_keys]
            right_rows = [right[right_seed][key] for key in common_keys]
            if not common_keys:
                continue

            for metric in ("post_change_success", "recovered"):
                left_only, right_only, p_value = _mcnemar_exact(
                    [bool(row[metric]) for row in left_rows],
                    [bool(row[metric]) for row in right_rows],
                )
                results.append(
                    {
                        "method_left": left_name,
                        "method_right": right_name,
                        "seed_left": left_seed,
                        "seed_right": right_seed,
                        "metric": metric,
                        "n": len(common_keys),
                        "difference_right_minus_left": (
                            statistics.mean(bool(row[metric]) for row in right_rows)
                            - statistics.mean(bool(row[metric]) for row in left_rows)
                        ),
                        "statistic": f"{left_only}/{right_only}",
                        "p_value": p_value,
                        "effect_size": (
                            (right_only - left_only) / len(common_keys)
                        ),
                    }
                )

            for metric in (
                "recovery_steps",
                "optimal_cost_shock",
                "extra_optimal_cost",
                "reaction_time_ms",
                "node_expansions",
            ):
                pairs = [
                    (float(left_row[metric]), float(right_row[metric]))
                    for left_row, right_row in zip(left_rows, right_rows)
                    if math.isfinite(float(left_row[metric]))
                    and math.isfinite(float(right_row[metric]))
                ]
                if not pairs:
                    continue
                paired = _paired_numeric(
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                )
                results.append(
                    {
                        "method_left": left_name,
                        "method_right": right_name,
                        "seed_left": left_seed,
                        "seed_right": right_seed,
                        "metric": metric,
                        "n": paired["n"],
                        "difference_right_minus_left": paired["mean_difference"],
                        "statistic": paired["wilcoxon_statistic"],
                        "p_value": paired["p_value"],
                        "effect_size": paired["rank_biserial"],
                    }
                )
    return results


def _holm_adjust(rows: list[dict[str, object]]) -> None:
    finite = sorted(
        [
            (index, float(row["p_value"]))
            for index, row in enumerate(rows)
            if math.isfinite(float(row["p_value"]))
        ],
        key=lambda item: item[1],
    )
    adjusted: dict[int, float] = {}
    running_maximum = 0.0
    total = len(finite)
    for rank, (index, p_value) in enumerate(finite, start=1):
        running_maximum = max(
            running_maximum,
            min(1.0, (total - rank + 1) * p_value),
        )
        adjusted[index] = running_maximum
    for index, row in enumerate(rows):
        row["holm_adjusted_p_value"] = adjusted.get(index, float("nan"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / "classical_adaptability_events.csv"
        ),
    )
    parser.add_argument("--rl", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "adaptability_summary.csv",
    )
    parser.add_argument(
        "--tests-out",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / "adaptability_paired_tests.csv"
        ),
    )
    args = parser.parse_args()

    records = _collapse_repetitions(_method_rows([args.classical, *args.rl]))
    summaries = _summarize(records)
    tests = _paired_tests(records)
    _holm_adjust(tests)
    _write_csv(args.summary_out, summaries)
    _write_csv(args.tests_out, tests)
    print(
        f"Wrote {len(summaries)} adaptability summaries and "
        f"{len(tests)} paired tests"
    )


if __name__ == "__main__":
    main()
