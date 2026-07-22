"""Generate Week 4 result plots from the paired Week 3 evaluation CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIJKSTRA_CSV = PROJECT_ROOT / "evaluation" / "week3_dynamic_baseline_results.csv"
RL_CSV = PROJECT_ROOT / "evaluation" / "week3_rl_results.csv"
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"


def _read_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        return {int(row["scenario_id"]): row for row in csv.DictReader(f)}


def _paired_rows() -> list[dict[str, float | int | bool]]:
    dijkstra_rows = _read_csv(DIJKSTRA_CSV)
    rl_rows = _read_csv(RL_CSV)
    rows: list[dict[str, float | int | bool]] = []

    for scenario_id in sorted(set(dijkstra_rows) & set(rl_rows)):
        dijkstra = dijkstra_rows[scenario_id]
        rl = rl_rows[scenario_id]
        if (dijkstra["start"], dijkstra["goal"]) != (rl["start"], rl["goal"]):
            continue

        rl_success = rl["success"] == "True"
        dijkstra_success = dijkstra["success"] == "True"
        rows.append(
            {
                "scenario_id": scenario_id,
                "dijkstra_success": dijkstra_success,
                "rl_success": rl_success,
                "dijkstra_cost": float(dijkstra["dynamic_path_cost"]),
                "rl_cost": float(rl["rl_path_cost"]) if rl_success else float("nan"),
                "dijkstra_time_ms": float(dijkstra["total_planning_time_ms"]),
                "rl_time_ms": float(rl["compute_time_ms"]),
            }
        )

    return rows


def _successful_pairs(rows: list[dict[str, float | int | bool]]) -> list[dict[str, float | int | bool]]:
    return [row for row in rows if row["dijkstra_success"] and row["rl_success"]]


def _save_path_cost_plot(rows: list[dict[str, float | int | bool]]) -> None:
    pairs = _successful_pairs(rows)
    scenario_ids = [int(row["scenario_id"]) for row in pairs]
    dijkstra_costs = [float(row["dijkstra_cost"]) for row in pairs]
    rl_costs = [float(row["rl_cost"]) for row in pairs]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(scenario_ids, dijkstra_costs, marker="o", linewidth=1.4, markersize=3.5, label="Dijkstra replanning")
    ax.plot(scenario_ids, rl_costs, marker="s", linewidth=1.4, markersize=3.5, label="RL policy")
    ax.set_title("Week 3 dynamic path cost by matched scenario")
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Path cost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "week4_path_cost_comparison.png", dpi=180)
    plt.close(fig)


def _save_cost_difference_plot(rows: list[dict[str, float | int | bool]]) -> None:
    pairs = _successful_pairs(rows)
    scenario_ids = [int(row["scenario_id"]) for row in pairs]
    diffs = [float(row["rl_cost"]) - float(row["dijkstra_cost"]) for row in pairs]

    colors = ["#8aa2ff" if diff == 0 else "#d7816a" for diff in diffs]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(scenario_ids, diffs, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Week 3 path-cost gap (RL - Dijkstra)")
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Cost difference")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "week4_path_cost_gap.png", dpi=180)
    plt.close(fig)


def _save_compute_time_plot(rows: list[dict[str, float | int | bool]]) -> None:
    pairs = _successful_pairs(rows)
    scenario_ids = [int(row["scenario_id"]) for row in pairs]
    dijkstra_times = [float(row["dijkstra_time_ms"]) for row in pairs]
    rl_times = [float(row["rl_time_ms"]) for row in pairs]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(scenario_ids, dijkstra_times, marker="o", linewidth=1.4, markersize=3.5, label="Dijkstra cumulative planning")
    ax.plot(scenario_ids, rl_times, marker="s", linewidth=1.4, markersize=3.5, label="RL policy evaluation")
    ax.set_title("Week 3 compute time by matched scenario")
    ax.set_xlabel("Scenario ID")
    ax.set_ylabel("Compute time (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "week4_compute_time_comparison.png", dpi=180)
    plt.close(fig)


def _save_success_rate_plot(rows: list[dict[str, float | int | bool]]) -> None:
    dijkstra_successes = sum(1 for row in rows if row["dijkstra_success"])
    rl_successes = sum(1 for row in rows if row["rl_success"])
    total = len(rows)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    methods = ["Dijkstra", "RL"]
    success_rates = [dijkstra_successes / total * 100, rl_successes / total * 100]
    bars = ax.bar(methods, success_rates, color=["#8aa2ff", "#d7816a"])
    ax.set_title("Week 3 dynamic success rate")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, successes in zip(bars, [dijkstra_successes, rl_successes]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{successes}/{total}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "week4_success_rate.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = _paired_rows()
    _save_path_cost_plot(rows)
    _save_cost_difference_plot(rows)
    _save_compute_time_plot(rows)
    _save_success_rate_plot(rows)
    print(f"Generated Week 4 plots in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
