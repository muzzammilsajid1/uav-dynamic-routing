"""Generate paper figures exclusively from aggregate result CSVs."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["astar", "dijkstra", "dstar_lite", "rl_full"]
CORE_METHODS = set(METHOD_ORDER)
METHOD_LABELS = {
    "astar": "A*",
    "dijkstra": "Dijkstra",
    "dstar_lite": "D* Lite",
    "rl_full": "DDQN+HER",
}
METHOD_COLORS = {
    "astar": "#0072B2",
    "dijkstra": "#E69F00",
    "dstar_lite": "#009E73",
    "rl_full": "#CC79A7",
}
METHOD_MARKERS = {
    "astar": "o",
    "dijkstra": "s",
    "dstar_lite": "^",
    "rl_full": "D",
}
METHOD_LINESTYLES = {
    "astar": "-",
    "dijkstra": "--",
    "dstar_lite": "-.",
    "rl_full": ":",
}
METHOD_HATCHES = {
    "astar": "",
    "dijkstra": "///",
    "dstar_lite": "xx",
    "rl_full": "..",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "savefig.facecolor": "white",
    }
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    """Write a vector publication figure and a high-resolution fallback."""
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _error_bars(
    values: list[float], errors: list[float], *, log_scale: bool
) -> list[list[float]] | list[float]:
    if not log_scale:
        return errors
    lower = [min(error, max(value * 0.95, 0.0)) for value, error in zip(values, errors)]
    return [lower, errors]


def _scaling_plots(rows: list[dict[str, str]], figure_dir: Path) -> None:
    scaling = [
        row
        for row in rows
        if row["split"].startswith("scale_")
        and row["method"] in CORE_METHODS
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scaling:
        grouped[row["method"]].append(row)

    for metric, label, stem, log_scale in [
        ("compute_time_ms_mean", "Route compute time (ms)", "scaling_compute_time", True),
        ("success_rate_mean", "Success rate", "scaling_success_rate", False),
        ("path_cost_gap_mean", "Path-cost gap", "scaling_path_cost_gap", False),
        ("replans_mean", "Replanning calls", "scaling_replans", False),
        ("node_expansions_mean", "Search node expansions", "scaling_node_expansions", True),
    ]:
        fig, axis = plt.subplots(figsize=(3.45, 2.42))
        for method in METHOD_ORDER:
            method_rows = grouped.get(method, [])
            ordered = sorted(
                method_rows, key=lambda row: int(row["split"].split("_")[1])
            )
            valid = [
                row for row in ordered if math.isfinite(float(row[metric]))
            ]
            if not valid:
                continue
            sizes = [int(row["split"].split("_")[1]) for row in valid]
            values = [float(row[metric]) for row in valid]
            errors = [
                (
                    float(row[metric.replace("_mean", "_95ci")])
                    if math.isfinite(
                        float(row[metric.replace("_mean", "_95ci")])
                    )
                    else 0.0
                )
                for row in valid
            ]
            axis.errorbar(
                sizes,
                values,
                yerr=_error_bars(values, errors, log_scale=log_scale),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                capsize=2,
                elinewidth=0.8,
                label=METHOD_LABELS[method],
            )
        axis.set_xlabel(r"Grid width $N$ ($N\times N$)")
        axis.set_ylabel(label)
        axis.set_xticks([15, 30, 50, 100])
        if log_scale:
            axis.set_yscale("log")
        if metric == "success_rate_mean":
            axis.set_ylim(0, 1.05)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.legend(ncol=2, frameon=False, loc="best")
        fig.tight_layout()
        _save(fig, figure_dir, stem)

    overview_specs = [
        ("success_rate_mean", "Success rate", "(a) Reliability", False),
        (
            "compute_time_ms_mean",
            "Route compute time (ms)",
            "(b) Decision computation",
            True,
        ),
        (
            "node_expansions_mean",
            "Search node expansions",
            "(c) Classical search work",
            True,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.34))
    for axis, (metric, label, title, log_scale) in zip(axes, overview_specs):
        for method in METHOD_ORDER:
            method_rows = sorted(
                grouped.get(method, []),
                key=lambda row: int(row["split"].split("_")[1]),
            )
            valid = [
                row for row in method_rows if math.isfinite(float(row[metric]))
            ]
            if not valid:
                continue
            sizes = [int(row["split"].split("_")[1]) for row in valid]
            values = [float(row[metric]) for row in valid]
            errors = [
                (
                    float(row[metric.replace("_mean", "_95ci")])
                    if math.isfinite(
                        float(row[metric.replace("_mean", "_95ci")])
                    )
                    else 0.0
                )
                for row in valid
            ]
            axis.errorbar(
                sizes,
                values,
                yerr=_error_bars(values, errors, log_scale=log_scale),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                capsize=1.8,
                elinewidth=0.75,
                label=METHOD_LABELS[method],
            )
        axis.set_title(title, pad=4)
        axis.set_xlabel(r"Grid width $N$")
        axis.set_ylabel(label)
        axis.set_xticks([15, 30, 50, 100])
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0, 1.05)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.1)
    _save(fig, figure_dir, "scaling_overview")


def _split_success_plot(rows: list[dict[str, str]], figure_dir: Path) -> None:
    splits = [
        "seen_layout_unseen_pairs",
        "unseen_layout_same_density",
        "denser_unseen_layout",
        "new_dynamic_obstacle_locations",
        "changed_toggle_periods",
    ]
    methods = METHOD_ORDER
    width = 0.8 / max(len(methods), 1)
    fig, axis = plt.subplots(figsize=(7.16, 2.82))
    for method_index, method in enumerate(methods):
        values = []
        errors = []
        for split in splits:
            match = next(
                (
                    row
                    for row in rows
                    if row["method"] == method and row["split"] == split
                ),
                None,
            )
            values.append(float(match["success_rate_mean"]) if match else 0.0)
            errors.append(
                (
                    float(match["success_rate_95ci"])
                    if match
                    and math.isfinite(float(match["success_rate_95ci"]))
                    else 0.0
                )
            )
        positions = [
            index - 0.4 + width / 2 + method_index * width
            for index in range(len(splits))
        ]
        axis.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.45,
            hatch=METHOD_HATCHES[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(range(len(splits)))
    axis.set_xticklabels(
        [
            "Held-out\npairs",
            "Unseen\nlayout",
            "Denser\nlayout",
            "New obstacle\nsites",
            "New toggle\nperiods",
        ]
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Success rate")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.legend(
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, figure_dir, "generalization_success")


def _adaptability_plots(rows: list[dict[str, str]], figure_dir: Path) -> None:
    by_method = {
        row["method"]: row
        for row in rows
        if row["split"] == "all_dynamic" and row["method"] in CORE_METHODS
    }
    aggregate = [by_method[method] for method in METHOD_ORDER if method in by_method]
    if not aggregate:
        raise RuntimeError("No all_dynamic adaptability summary rows")

    methods = [row["method"] for row in aggregate]
    positions = list(range(len(methods)))

    fig, axis = plt.subplots(figsize=(3.45, 2.42))
    width = 0.38
    for index, (offset, metric, label, color, hatch) in enumerate((
        (-width / 2, "post_change_success", "Post-change success", "#56B4E9", ""),
        (width / 2, "recovery_rate", "Recovered in route", "#D55E00", "///"),
    )):
        values = [float(row[f"{metric}_mean"]) for row in aggregate]
        errors = [
            (
                float(row[f"{metric}_95ci"])
                if math.isfinite(float(row[f"{metric}_95ci"]))
                else 0.0
            )
            for row in aggregate
        ]
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            yerr=errors,
            capsize=2,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            hatch=hatch,
            label=label,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([METHOD_LABELS[method] for method in methods])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Event fraction")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.legend(
        ncol=1,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, figure_dir, "adaptability_recovery")

    for metric, label, filename, log_scale in (
        (
            "reaction_time_ms",
            "Reaction computation (ms)",
            "adaptability_reaction_time.png",
            True,
        ),
        (
            "extra_optimal_cost",
            "Positive optimal-cost shock",
            "adaptability_cost_shock.png",
            False,
        ),
    ):
        valid = [
            row
            for row in aggregate
            if math.isfinite(float(row[f"{metric}_mean"]))
            and (not log_scale or float(row[f"{metric}_mean"]) > 0)
        ]
        fig, axis = plt.subplots(figsize=(3.45, 2.42))
        values = [float(row[f"{metric}_mean"]) for row in valid]
        errors = [
            (
                float(row[f"{metric}_95ci"])
                if math.isfinite(float(row[f"{metric}_95ci"]))
                else 0.0
            )
            for row in valid
        ]
        for index, (row, value, error) in enumerate(zip(valid, values, errors)):
            method = row["method"]
            axis.bar(
                index,
                value,
                yerr=_error_bars([value], [error], log_scale=log_scale),
                capsize=2,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.45,
                hatch=METHOD_HATCHES[method],
            )
        axis.set_xticks(range(len(valid)))
        axis.set_xticklabels(
            [METHOD_LABELS[row["method"]] for row in valid],
        )
        axis.set_ylabel(label)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        _save(fig, figure_dir, filename.removesuffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results" / "research_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "paper_latex_v2" / "figures",
    )
    parser.add_argument(
        "--adaptability-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / "adaptability_summary.csv"
        ),
    )
    args = parser.parse_args()
    rows = _read(args.summary)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    _scaling_plots(rows, args.figure_dir)
    _split_success_plot(rows, args.figure_dir)
    _adaptability_plots(_read(args.adaptability_summary), args.figure_dir)
    print(f"Generated research figures in {args.figure_dir}")


if __name__ == "__main__":
    main()
