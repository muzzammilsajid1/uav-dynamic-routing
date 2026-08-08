"""Summarize a Phase C2 run from raw checkpoint evaluations."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCALES = (15, 30, 50, 100)
DISTANCE_BINS = ("short", "medium", "long")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / denominator
    half_width = z * math.sqrt(
        p * (1.0 - p) / episodes + z * z / (4.0 * episodes * episodes)
    ) / denominator
    return [center - half_width, center + half_width]


def checkpoint_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("action_masking_applied"):
        raise ValueError(f"{path.name}: validation did not apply action masking")
    if payload.get("collision_field") != "crashed":
        raise ValueError(f"{path.name}: collision field is not authoritative 'crashed'")
    if int(payload.get("invalid_action_count", -1)) != 0:
        raise ValueError(f"{path.name}: validation contains invalid actions")
    overall = payload["aggregates"]["all"]
    record = {
        "source_file": path.name,
        "source_sha256": sha256(path),
        "validation_manifest_sha256": payload["validation_manifest_sha256"],
        "requested_interactions": int(payload["requested_interactions"]),
        "completed_interactions": int(payload["completed_interactions"]),
        "successes": int(overall["successes"]),
        "episodes": int(overall["episodes"]),
        "success_rate": float(overall["success_rate"]),
        "success_rate_wilson_95": wilson(
            int(overall["successes"]), int(overall["episodes"])
        ),
        "collisions": int(overall["collisions"]),
        "invalid_actions": int(payload["invalid_action_count"]),
        "by_scale": {},
        "by_distance": {},
    }
    for scale in SCALES:
        aggregate = payload["aggregates"][f"scale/{scale}"]
        record["by_scale"][str(scale)] = {
            "successes": int(aggregate["successes"]),
            "episodes": int(aggregate["episodes"]),
            "success_rate": float(aggregate["success_rate"]),
            "success_rate_wilson_95": wilson(
                int(aggregate["successes"]), int(aggregate["episodes"])
            ),
        }
    for distance_bin in DISTANCE_BINS:
        aggregate = payload["aggregates"][f"distance/{distance_bin}"]
        record["by_distance"][distance_bin] = {
            "successes": int(aggregate["successes"]),
            "episodes": int(aggregate["episodes"]),
            "success_rate": float(aggregate["success_rate"]),
            "success_rate_wilson_95": wilson(
                int(aggregate["successes"]), int(aggregate["episodes"])
            ),
        }
    return record


def plot_summary(
    records: list[dict],
    final_evaluation: dict,
    output: Path,
    model_type: str,
    seed: int,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    figure, (curve_axis, matrix_axis) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    x = np.asarray([record["completed_interactions"] for record in records]) / 1000.0
    curve_axis.plot(
        x,
        [100.0 * record["success_rate"] for record in records],
        color="#111827",
        marker="o",
        linewidth=2.8,
        label="Overall",
        zorder=5,
    )
    colors = {15: "#009E73", 30: "#0072B2", 50: "#E69F00", 100: "#CC79A7"}
    for scale in SCALES:
        curve_axis.plot(
            x,
            [100.0 * record["by_scale"][str(scale)]["success_rate"] for record in records],
            color=colors[scale],
            marker=".",
            linewidth=1.8,
            label=f"{scale}x{scale}",
        )
    curve_axis.axvline(151.552, color="#6B7280", linestyle="--", linewidth=1.0)
    curve_axis.text(154, 3, "original ceiling", color="#4B5563", fontsize=8)
    curve_axis.set_title(f"Corrected {model_type} validation learning curve")
    curve_axis.set_xlabel("Completed interactions (thousands)")
    curve_axis.set_ylabel("Success rate (%)")
    curve_axis.set_ylim(0, 103)
    curve_axis.grid(alpha=0.22)
    curve_axis.legend(ncol=2, frameon=False, loc="lower right")

    matrix = np.asarray([
        [
            100.0
            * final_evaluation["aggregates"][f"scale_distance/{scale}/{bucket}"]["success_rate"]
            for bucket in DISTANCE_BINS
        ]
        for scale in SCALES
    ])
    image = matrix_axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value >= 62 else "#111827"
            matrix_axis.text(
                column, row, f"{value:.0f}%", ha="center", va="center",
                color=text_color, fontweight="bold"
            )
    matrix_axis.set_xticks(range(len(DISTANCE_BINS)), [name.title() for name in DISTANCE_BINS])
    matrix_axis.set_yticks(range(len(SCALES)), [f"{scale}x{scale}" for scale in SCALES])
    matrix_axis.set_xlabel("Route-distance bin")
    matrix_axis.set_ylabel("Grid scale")
    matrix_axis.set_title("Final checkpoint success matrix")
    colorbar = figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Success rate (%)")
    figure.suptitle(
        f"Phase C2 corrected {model_type} seed-{seed} development run",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Fixed 240-route validation manifest; development seed only - training-seed uncertainty is not shown.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--output-stem",
        help="Output filename stem; defaults to phase_c2_<model>_seed<seed>",
    )
    args = parser.parse_args()
    evaluation_paths = sorted(args.run_dir.glob("evaluation_*.json"))
    if not evaluation_paths:
        raise FileNotFoundError(f"no evaluation files in {args.run_dir}")
    records = [checkpoint_record(path) for path in evaluation_paths]
    records.sort(key=lambda record: record["completed_interactions"])
    manifest_hashes = {record["validation_manifest_sha256"] for record in records}
    if len(manifest_hashes) != 1:
        raise ValueError(f"evaluation files use multiple validation manifests: {manifest_hashes}")
    final_path = args.run_dir / records[-1]["source_file"]
    final_evaluation = json.loads(final_path.read_text(encoding="utf-8"))
    provenance_path = args.run_dir / "provenance.json"
    status_path = args.run_dir / "status.json"
    model_path = args.run_dir / f"model_{records[-1]['completed_interactions']:06d}.zip"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_type = str(provenance["model_type"])
    seed = int(provenance["seed"])
    output_stem = args.output_stem or f"phase_c2_{model_type.lower()}_seed{seed}"
    summary = {
        "schema_version": 1,
        "classification": "development_validation_single_seed",
        "claim_limit": (
            "Route-level Wilson intervals do not capture training-seed variability; "
            "confirmatory multi-seed evidence is still required."
        ),
        "run_directory": str(args.run_dir.resolve()),
        "validation_manifest_sha256": next(iter(manifest_hashes)),
        "provenance": provenance,
        "checkpoints": records,
        "best_checkpoint_by_overall_validation": max(
            records, key=lambda record: record["success_rate"]
        )["completed_interactions"],
        "final_checkpoint": records[-1]["completed_interactions"],
        "final_checkpoint_metrics": final_evaluation["aggregates"]["all"],
        "final_failure_labels": dict(sorted(collections.Counter(
            episode["failure_label"]
            for episode in final_evaluation["episodes"]
            if not episode["is_success"]
        ).items())),
        "artifact_sha256": {
            "final_evaluation": sha256(final_path),
            "final_model": sha256(model_path),
            "provenance": sha256(provenance_path),
            "status": sha256(status_path),
            "latest_checkpoint_bundle": sha256(args.run_dir / "latest_checkpoint_bundle.zip"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{output_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_summary(
        records,
        final_evaluation,
        args.output_dir / f"{output_stem}_learning_curve.png",
        model_type,
        seed,
    )
    print(summary_path)


if __name__ == "__main__":
    main()
