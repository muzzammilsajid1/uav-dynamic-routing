"""Run paired, seed-aware statistical comparisons from raw route results."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collapse(
    rows: list[dict[str, str]], method: str
) -> dict[tuple[str, str, str], dict[str, object]]:
    repetitions: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seed = row.get("policy_seed") or "classical"
        repetitions[(row["split"], seed, row["scenario_id"])].append(row)

    output: dict[tuple[str, str, str], dict[str, object]] = {}
    for (split, seed, scenario_id), group in repetitions.items():
        first = group[0]
        success = first["success"].lower() == "true"
        cost_text = first["route_path_cost"]
        output[(split, seed, scenario_id)] = {
            "method": method,
            "split": split,
            "seed": seed,
            "scenario_id": scenario_id,
            "success": success,
            "cost": (
                float(cost_text)
                if success and cost_text not in {"NA", "N/A", ""}
                else float("nan")
            ),
            "time": statistics.mean(
                float(row["route_compute_time_ms"]) for row in group
            ),
        }
    return output


def _mcnemar_exact(left: list[bool], right: list[bool]) -> tuple[int, int, float]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    p_value = float(
        stats.binomtest(
            min(left_only, right_only), discordant, p=0.5, alternative="two-sided"
        ).pvalue
    )
    return left_only, right_only, p_value


def _paired_numeric(left: list[float], right: list[float]) -> dict[str, float]:
    differences = [b - a for a, b in zip(left, right)]
    if not differences:
        return {
            "n": 0,
            "median_difference": float("nan"),
            "mean_difference": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "p_value": float("nan"),
            "rank_biserial": float("nan"),
        }
    nonzero = [difference for difference in differences if abs(difference) > 1e-12]
    if not nonzero:
        statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
    else:
        result = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            correction=True,
            alternative="two-sided",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
        ranks = stats.rankdata([abs(value) for value in nonzero])
        positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
        negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
        rank_biserial = (positive - negative) / (positive + negative)
    return {
        "n": len(differences),
        "median_difference": statistics.median(differences),
        "mean_difference": statistics.mean(differences),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
    }


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
        default=PROJECT_ROOT / "evaluation" / "results" / "paired_tests.csv",
    )
    args = parser.parse_args()

    datasets: dict[str, dict] = {}
    classical_rows = _read(args.classical)
    for planner in sorted({row["planner"] for row in classical_rows}):
        datasets[planner] = _collapse(
            [row for row in classical_rows if row["planner"] == planner],
            planner,
        )
    for path in args.rl:
        rows = _read(path)
        if rows:
            method = f"rl_{rows[0]['variant']}"
            datasets[method] = _collapse(rows, method)

    results: list[dict[str, object]] = []
    for left_name, right_name in itertools.combinations(sorted(datasets), 2):
        left = datasets[left_name]
        right = datasets[right_name]
        splits = sorted(
            {key[0] for key in left}.intersection(key[0] for key in right)
        )
        for split in splits:
            left_seeds = sorted({key[1] for key in left if key[0] == split})
            right_seeds = sorted({key[1] for key in right if key[0] == split})
            seed_pairs = (
                [(seed, seed) for seed in set(left_seeds) & set(right_seeds)]
                or [
                    (left_seed, right_seed)
                    for left_seed in left_seeds
                    for right_seed in right_seeds
                    if left_seed == "classical" or right_seed == "classical"
                ]
            )
            for left_seed, right_seed in seed_pairs:
                scenario_ids = sorted(
                    {
                        key[2]
                        for key in left
                        if key[0] == split and key[1] == left_seed
                    }
                    & {
                        key[2]
                        for key in right
                        if key[0] == split and key[1] == right_seed
                    }
                )
                left_rows = [
                    left[(split, left_seed, scenario_id)]
                    for scenario_id in scenario_ids
                ]
                right_rows = [
                    right[(split, right_seed, scenario_id)]
                    for scenario_id in scenario_ids
                ]
                left_only, right_only, success_p = _mcnemar_exact(
                    [bool(row["success"]) for row in left_rows],
                    [bool(row["success"]) for row in right_rows],
                )
                results.append(
                    {
                        "method_left": left_name,
                        "method_right": right_name,
                        "split": split,
                        "seed_left": left_seed,
                        "seed_right": right_seed,
                        "metric": "success",
                        "n": len(scenario_ids),
                        "difference_right_minus_left": (
                            statistics.mean(
                                bool(row["success"]) for row in right_rows
                            )
                            - statistics.mean(
                                bool(row["success"]) for row in left_rows
                            )
                        ),
                        "statistic": f"{left_only}/{right_only}",
                        "p_value": success_p,
                        "effect_size": (
                            (right_only - left_only) / len(scenario_ids)
                            if scenario_ids
                            else float("nan")
                        ),
                    }
                )

                both_success = [
                    (left_row, right_row)
                    for left_row, right_row in zip(left_rows, right_rows)
                    if left_row["success"] and right_row["success"]
                ]
                for metric in ["cost", "time"]:
                    metric_pairs = (
                        both_success
                        if metric == "cost"
                        else list(zip(left_rows, right_rows))
                    )
                    paired = _paired_numeric(
                        [float(pair[0][metric]) for pair in metric_pairs],
                        [float(pair[1][metric]) for pair in metric_pairs],
                    )
                    results.append(
                        {
                            "method_left": left_name,
                            "method_right": right_name,
                            "split": split,
                            "seed_left": left_seed,
                            "seed_right": right_seed,
                            "metric": metric,
                            "n": paired["n"],
                            "difference_right_minus_left": paired[
                                "mean_difference"
                            ],
                            "statistic": paired["wilcoxon_statistic"],
                            "p_value": paired["p_value"],
                            "effect_size": paired["rank_biserial"],
                        }
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        [
            item
            for item in enumerate(results)
            if math.isfinite(float(item[1]["p_value"]))
        ],
        key=lambda item: float(item[1]["p_value"]),
    )
    adjusted_by_index: dict[int, float] = {}
    running_maximum = 0.0
    total_tests = len(ordered)
    for rank, (original_index, result) in enumerate(ordered, start=1):
        adjusted = min(
            1.0, (total_tests - rank + 1) * float(result["p_value"])
        )
        running_maximum = max(running_maximum, adjusted)
        adjusted_by_index[original_index] = running_maximum
    for index, result in enumerate(results):
        result["holm_adjusted_p_value"] = adjusted_by_index.get(
            index, float("nan")
        )

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} paired statistical tests to {args.out}")


if __name__ == "__main__":
    main()
