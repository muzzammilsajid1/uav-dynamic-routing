"""Create a paired development comparison from corrected Phase C2 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCALES = (15, 30, 50, 100)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(run_dir: Path) -> dict:
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    evaluations = []
    for path in run_dir.glob("evaluation_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("action_masking_applied"):
            raise ValueError(f"{path}: action masking was not applied")
        if payload.get("collision_field") != "crashed":
            raise ValueError(f"{path}: non-authoritative collision field")
        if payload.get("invalid_action_count") != 0:
            raise ValueError(f"{path}: invalid actions were selected")
        evaluations.append((path, payload))
    evaluations.sort(key=lambda item: int(item[1]["completed_interactions"]))
    if not evaluations:
        raise FileNotFoundError(f"No evaluations in {run_dir}")
    final_path, final = evaluations[-1]
    final_interactions = int(final["completed_interactions"])
    model_path = run_dir / f"model_{final_interactions:06d}.zip"
    model = MaskablePPO.load(model_path, device="cpu")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad
    )
    return {
        "run_dir": run_dir,
        "provenance": provenance,
        "evaluations": evaluations,
        "final_path": final_path,
        "final": final,
        "model_path": model_path,
        "trainable_parameters": trainable_parameters,
    }


def route_signature(payload: dict) -> list[tuple]:
    return [
        (
            int(row["grid_size"]),
            row["distance_bin"],
            tuple(row["start"]),
            tuple(row["goal"]),
        )
        for row in payload["episodes"]
    ]


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    tail = sum(
        math.comb(total, value) * 0.5**total
        for value in range(min(discordant_a, discordant_b) + 1)
    )
    return min(1.0, 2.0 * tail)


def plot_comparison(m1: dict, m2: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    figure, (curve_axis, scale_axis) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    colors = {"M1": "#0072B2", "M2": "#D55E00"}
    for run in (m1, m2):
        label = run["provenance"]["model_type"]
        x = [payload["completed_interactions"] / 1000.0 for _, payload in run["evaluations"]]
        y = [100.0 * payload["aggregates"]["all"]["success_rate"] for _, payload in run["evaluations"]]
        curve_axis.plot(x, y, marker="o", linewidth=2.5, color=colors[label], label=label)
    curve_axis.set_title("Paired development learning curves")
    curve_axis.set_xlabel("Completed interactions (thousands)")
    curve_axis.set_ylabel("Success rate (%)")
    curve_axis.set_ylim(0, 103)
    curve_axis.grid(alpha=0.22)
    curve_axis.legend(frameon=False, loc="lower right")

    x = np.arange(len(SCALES))
    width = 0.36
    for offset, run in ((-width / 2, m1), (width / 2, m2)):
        label = run["provenance"]["model_type"]
        values = [
            100.0 * run["final"]["aggregates"][f"scale/{scale}"]["success_rate"]
            for scale in SCALES
        ]
        bars = scale_axis.bar(x + offset, values, width, label=label, color=colors[label])
        scale_axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=9)
    scale_axis.set_title("Final checkpoint by scale")
    scale_axis.set_xticks(x, [f"{scale}x{scale}" for scale in SCALES])
    scale_axis.set_xlabel("Grid scale")
    scale_axis.set_ylabel("Success rate (%)")
    scale_axis.set_ylim(0, 108)
    scale_axis.grid(axis="y", alpha=0.22)
    scale_axis.legend(frameon=False, loc="lower left")

    figure.suptitle("Phase C2 corrected M1 versus M2 (seed 42)", fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "Same 240 development routes; single training seed - training-seed uncertainty is not shown.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m1_run_dir", type=Path)
    parser.add_argument("m2_run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    m1 = load_run(args.m1_run_dir)
    m2 = load_run(args.m2_run_dir)
    if m1["provenance"]["model_type"] != "M1" or m2["provenance"]["model_type"] != "M2":
        raise ValueError("Expected M1 followed by M2")
    if route_signature(m1["final"]) != route_signature(m2["final"]):
        raise ValueError("M1 and M2 final evaluations do not use identical route order")
    m1_steps = [payload["completed_interactions"] for _, payload in m1["evaluations"]]
    m2_steps = [payload["completed_interactions"] for _, payload in m2["evaluations"]]
    if m1_steps != m2_steps:
        raise ValueError("M1 and M2 checkpoint schedules differ")

    paired = list(zip(m1["final"]["episodes"], m2["final"]["episodes"]))
    m1_only = sum(a["is_success"] and not b["is_success"] for a, b in paired)
    m2_only = sum(b["is_success"] and not a["is_success"] for a, b in paired)
    m1_all = m1["final"]["aggregates"]["all"]
    m2_all = m2["final"]["aggregates"]["all"]
    summary = {
        "schema_version": 1,
        "classification": "paired_development_validation_single_training_seed",
        "claim_limit": (
            "The paired route comparison and exact McNemar test do not capture "
            "training-seed variability; confirmatory multi-seed evidence is required."
        ),
        "identical_route_order": True,
        "completed_interactions": m1_steps,
        "models": {},
        "paired_final_comparison": {
            "m1_only_successes": m1_only,
            "m2_only_successes": m2_only,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(m1_only, m2_only),
            "m2_minus_m1_success_rate_percentage_points": 100.0 * (
                m2_all["success_rate"] - m1_all["success_rate"]
            ),
            "m1_to_m2_parameter_ratio": m1["trainable_parameters"] / m2["trainable_parameters"],
            "m1_to_m2_model_file_size_ratio": m1["model_path"].stat().st_size / m2["model_path"].stat().st_size,
            "m1_to_m2_inference_latency_ratio": (
                m1_all["mean_policy_inference_latency_ms"]
                / m2_all["mean_policy_inference_latency_ms"]
            ),
            "m2_relative_reduction_in_mean_success_path_ratio": (
                (m1_all["mean_success_path_cost_ratio"] - m2_all["mean_success_path_cost_ratio"])
                / m1_all["mean_success_path_cost_ratio"]
            ),
        },
    }
    for run in (m1, m2):
        label = run["provenance"]["model_type"]
        summary["models"][label] = {
            "git_commit": run["provenance"]["git_commit"],
            "seed": run["provenance"]["seed"],
            "trainable_parameters": run["trainable_parameters"],
            "final_model_bytes": run["model_path"].stat().st_size,
            "final_model_sha256": sha256(run["model_path"]),
            "final_evaluation_sha256": sha256(run["final_path"]),
            "latest_checkpoint_bundle_sha256": sha256(run["run_dir"] / "latest_checkpoint_bundle.zip"),
            "overall_curve": [
                {
                    "completed_interactions": payload["completed_interactions"],
                    "success_rate": payload["aggregates"]["all"]["success_rate"],
                }
                for _, payload in run["evaluations"]
            ],
            "final_all": run["final"]["aggregates"]["all"],
            "final_by_scale": {
                str(scale): run["final"]["aggregates"][f"scale/{scale}"] for scale in SCALES
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = "phase_c2_v12_m1_m2_seed42_comparison"
    summary_path = args.output_dir / f"{output_stem}.json"
    figure_path = args.output_dir / f"{output_stem}.png"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_comparison(m1, m2, figure_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
