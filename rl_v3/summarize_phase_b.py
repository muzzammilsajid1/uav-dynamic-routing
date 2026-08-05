"""Create traceable Phase B comparison tables and figures from completed pilots."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "rl_v3" / "phase_b"
PILOTS = ("P1", "P2", "P3", "P4")


def summarize() -> dict:
    manifest = json.loads((ROOT / "evaluation/manifests/rl_v3_validation_v2.json").read_text())
    scenarios = {row["scenario_id"]: row for row in manifest["scenarios"]}
    output = {"validation_v2_hash": manifest["manifest_sha256"], "pilots": {}}
    episode_rows = {}
    for pilot in PILOTS:
        status = json.loads((RUN_ROOT / pilot / "status.json").read_text())
        aggregate = json.loads((RUN_ROOT / pilot / "evaluation/step_100000/aggregates.json").read_text())
        rows = list(csv.DictReader((RUN_ROOT / pilot / "evaluation/step_100000/episodes.csv").open()))
        episode_rows[pilot] = rows
        failures = Counter(row["failure_label"] for row in rows)
        output["pilots"][pilot] = {
            "configuration": status["pilot"],
            "parameter_count": status["parameter_count"],
            "inference_latency_ms": status["initial_inference_latency_ms"],
            "learning_curve": status["history"],
            "final_successes": failures.get("success", 0),
            "final_episodes": len(rows),
            "failure_taxonomy": dict(sorted(failures.items())),
            "by_scale": {key.split("/")[1]: value for key, value in aggregate.items() if key.startswith("scale/")},
            "by_family": {key.split("/")[1]: value for key, value in aggregate.items() if key.startswith("family/")},
            "by_route": {key.split("/")[1]: value for key, value in aggregate.items() if key.startswith("route/")},
            "overall": aggregate["all"],
        }
    output["decision"] = {
        "continue_to_250k": [],
        "reason": "No model passed the empty-route or scale credibility gates; no 250k continuation is authorized.",
        "largest_observed_improvement": "R2 at 25k (4/96), but it regressed to 1/96 at 100k and is not reliable.",
        "recency_retained": False,
        "R2_retained": False,
        "recurrence_evidence": "Loop-dominated failures justify a future controlled recurrent pilot, but these data do not establish recurrence as necessary because optimization, curriculum, and reward-scale alternatives remain confounded.",
    }
    figures = RUN_ROOT / "comparison_figures"
    figures.mkdir(parents=True, exist_ok=True)
    _learning_curve(output, figures / "phase_b_learning_curves.png")
    _failure_plot(output, figures / "phase_b_failure_taxonomy.png")
    _scale_plot(output, figures / "phase_b_success_by_scale.png")
    for scenario_id in ("VAL2-G050-EMPTY-LONG-02", "VAL2-G015-STRUCTURED-MEDIUM-02"):
        _trajectory_comparison(scenario_id, scenarios[scenario_id], figures / f"trajectories_{scenario_id}.png")
    (RUN_ROOT / "phase_b_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def _learning_curve(summary, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for pilot, data in summary["pilots"].items():
        curve = data["learning_curve"]
        ax.plot([x["interactions"] for x in curve], [x["validation_success_rate"] for x in curve], marker="o", label=pilot)
    ax.set(xlabel="Training interactions", ylabel="Validation success rate", ylim=(0, .08), title="Phase B fixed-suite learning curves")
    ax.grid(alpha=.25); ax.legend(ncol=4)
    fig.savefig(path, dpi=200); plt.close(fig)


def _failure_plot(summary, path):
    labels = ["two_cell_oscillation", "longer_repeated_loop", "excessive_detour", "success"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bottom = np.zeros(4)
    for label in labels:
        values = np.array([summary["pilots"][p]["failure_taxonomy"].get(label, 0) for p in PILOTS])
        ax.bar(PILOTS, values, bottom=bottom, label=label.replace("_", " "))
        bottom += values
    ax.set(ylabel="Validation episodes (n=96)", title="Failure taxonomy at 100k")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path, dpi=200); plt.close(fig)


def _scale_plot(summary, path):
    values = np.array([[summary["pilots"][p]["by_scale"][str(s)]["success_rate"] for s in (15,30,50,100)] for p in PILOTS])
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(.125, values.max()))
    ax.set_xticks(range(4), (15,30,50,100)); ax.set_yticks(range(4), PILOTS)
    ax.set(xlabel="Grid size", ylabel="Pilot", title="Success rate by scale at 100k")
    for r in range(4):
        for c in range(4): ax.text(c, r, f"{values[r,c]*100:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Success rate")
    fig.savefig(path, dpi=200); plt.close(fig)


def _trajectory_comparison(scenario_id, scenario, path):
    size = int(scenario["grid_size"])
    blocked = np.zeros((size, size))
    for row, col in scenario["blocked"]: blocked[row, col] = 1
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), constrained_layout=True)
    for ax, pilot in zip(axes, PILOTS):
        evidence = json.loads((RUN_ROOT / pilot / "evaluation/step_100000/trajectories" / f"{scenario_id}.json").read_text())
        trajectory = np.asarray(evidence["trajectory"])
        ax.imshow(blocked, cmap="Greys", origin="upper", interpolation="nearest", vmin=0, vmax=1)
        ax.plot(trajectory[:,1], trajectory[:,0], color="tab:orange", linewidth=1, alpha=.85)
        ax.scatter(scenario["start"][1], scenario["start"][0], c="tab:blue", s=25)
        ax.scatter(scenario["goal"][1], scenario["goal"][0], c="tab:green", marker="*", s=50)
        result = evidence["summary"]
        ax.set_title(f"{pilot}: {result['failure_label']}\n{result['decisions']} decisions", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(scenario_id, fontsize=10)
    fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    result = summarize()
    print(json.dumps(result["decision"], indent=2))
