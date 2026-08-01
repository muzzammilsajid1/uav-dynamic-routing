"""Aggregate scenario repetitions and policy seeds with uncertainty estimates."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _critical_value(n: int) -> float:
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except ImportError:
        return 1.96


def _mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, float("nan")
    half_width = _critical_value(len(values)) * statistics.stdev(values) / math.sqrt(
        len(values)
    )
    return mean, half_width


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collapse_repetitions(rows: list[dict[str, str]], method: str) -> list[dict]:
    grouped: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            method,
            row.get("policy_seed") or row["scenario_id"],
            row["scenario_id"],
        )
        grouped[key].append(row)

    collapsed: list[dict] = []
    for (_, seed, scenario_id), group in grouped.items():
        first = group[0]
        success = first["success"].lower() == "true"
        route_cost_value = first.get("route_path_cost", "NA")
        initial_value = first.get("initial_optimal_cost") or first.get(
            "initial_path_cost"
        )
        route_cost = (
            float(route_cost_value)
            if route_cost_value not in {"NA", "N/A", ""}
            else float("nan")
        )
        initial_cost = float(initial_value) if initial_value else float("nan")
        collapsed.append(
            {
                "method": method,
                "seed": seed,
                "scenario_id": scenario_id,
                "split": first["split"],
                "grid_size": int(first["grid_size"]),
                "success": success,
                "path_cost": route_cost,
                "path_cost_gap": (
                    route_cost - initial_cost if success else float("nan")
                ),
                "compute_time_ms": statistics.mean(
                    float(row["route_compute_time_ms"]) for row in group
                ),
                "decision_latency_ms": statistics.mean(
                    float(
                        row.get("mean_decision_latency_ms")
                        or float(row["route_compute_time_ms"])
                        / max(float(row["steps_taken"]), 1.0)
                    )
                    for row in group
                ),
                "steps": float(first["steps_taken"]),
                "replans": (
                    float(first["replans"])
                    if first.get("replans") not in {None, ""}
                    else float("nan")
                ),
                "node_expansions": (
                    float(first["node_expansions"])
                    if first.get("node_expansions")
                    else float("nan")
                ),
            }
        )
    return collapsed


def summarize(collapsed: list[dict]) -> list[dict[str, object]]:
    by_method_split_seed: dict[tuple, list[dict]] = defaultdict(list)
    for row in collapsed:
        by_method_split_seed[(row["method"], row["split"], row["seed"])].append(row)

    seed_summaries: list[dict] = []
    for (method, split, seed), group in by_method_split_seed.items():
        successful = [row for row in group if row["success"]]
        seed_summaries.append(
            {
                "method": method,
                "split": split,
                "seed": seed,
                "scenarios": len(group),
                "success_rate": len(successful) / len(group),
                "path_cost_gap": (
                    statistics.mean(row["path_cost_gap"] for row in successful)
                    if successful
                    else float("nan")
                ),
                "compute_time_ms": statistics.mean(
                    row["compute_time_ms"] for row in group
                ),
                "decision_latency_ms": statistics.mean(
                    row["decision_latency_ms"] for row in group
                ),
                "steps": statistics.mean(row["steps"] for row in group),
                "replans": (
                    statistics.mean(
                        row["replans"]
                        for row in group
                        if not math.isnan(row["replans"])
                    )
                    if any(not math.isnan(row["replans"]) for row in group)
                    else float("nan")
                ),
                "node_expansions": (
                    statistics.mean(
                        row["node_expansions"]
                        for row in group
                        if not math.isnan(row["node_expansions"])
                    )
                    if any(
                        not math.isnan(row["node_expansions"]) for row in group
                    )
                    else float("nan")
                ),
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in seed_summaries:
        grouped[(row["method"], row["split"])].append(row)

    output: list[dict[str, object]] = []
    for (method, split), group in sorted(grouped.items()):
        result: dict[str, object] = {
            "method": method,
            "split": split,
            "independent_units": len(group),
            "scenarios_per_unit": group[0]["scenarios"],
        }
        for metric in [
            "success_rate",
            "path_cost_gap",
            "compute_time_ms",
            "decision_latency_ms",
            "steps",
            "replans",
            "node_expansions",
        ]:
            values = [
                float(row[metric])
                for row in group
                if not math.isnan(float(row[metric]))
            ]
            mean, ci = _mean_ci(values)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_95ci"] = ci
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classical",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "classical_suite_raw.csv",
    )
    parser.add_argument("--rl", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "research_summary.csv",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "research_summary.md",
    )
    args = parser.parse_args()

    collapsed: list[dict] = []
    classical_rows = _read(args.classical)
    for planner in sorted({row["planner"] for row in classical_rows}):
        collapsed.extend(
            _collapse_repetitions(
                [row for row in classical_rows if row["planner"] == planner],
                planner,
            )
        )
    for path in args.rl:
        rows = _read(path)
        if rows:
            collapsed.extend(
                _collapse_repetitions(rows, f"rl_{rows[0]['variant']}")
            )

    summaries = summarize(collapsed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Research benchmark summary",
        "",
        "| Method | Split | Units | Success (95% CI) | Cost gap (95% CI) | "
        "Compute ms (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['split']} | {row['independent_units']} | "
            f"{row['success_rate_mean']:.3f} +/- {row['success_rate_95ci']:.3f} | "
            f"{row['path_cost_gap_mean']:.3f} +/- {row['path_cost_gap_95ci']:.3f} | "
            f"{row['compute_time_ms_mean']:.3f} +/- {row['compute_time_ms_95ci']:.3f} |"
        )
    args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(summaries)} summary rows to {args.out}")


if __name__ == "__main__":
    main()
