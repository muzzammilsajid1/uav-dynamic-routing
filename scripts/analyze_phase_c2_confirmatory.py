"""Verify and summarize the frozen Phase C2 M2 confirmatory experiment."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


SEEDS = (11, 22, 33, 44, 55)
CHECKPOINTS = (26624, 51200, 75776, 100352, 151552)
SCALES = (15, 30, 50, 100)
DISTANCE_BINS = ("short", "medium", "long")
FINAL_CHECKPOINT = CHECKPOINTS[-1]
CORE_HASH_KEYS = (
    "config",
    "validation_manifest",
    "train_generator",
    "reward_wrapper",
    "observations",
    "phase_c2_env",
    "phase_c2_runner",
    "phase_c0_env",
    "action_masking",
    "kaggle_runner",
)
FORBIDDEN_CONTAINER_MARKERS = (
    "checkpoint_bundle_",
    "latest_checkpoint_bundle",
    "raw_artifacts",
    "_complete",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_interval(values: list[float], confidence: float = 0.95) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("at least two independent seeds are required")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    critical = float(t.ppf((1.0 + confidence) / 2.0, df=array.size - 1))
    half_width = critical * sd / math.sqrt(array.size)
    return {
        "n_training_seeds": int(array.size),
        "values": [float(value) for value in array],
        "mean": mean,
        "sample_sd": sd,
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean_t_95_ci": [mean - half_width, mean + half_width],
        "mean_t_95_ci_bounded_0_1": [
            max(0.0, mean - half_width), min(1.0, mean + half_width)
        ],
    }


def verify_zip(path: Path, expected_names: set[str]) -> dict:
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        names = archive.namelist()
    if corrupt_member is not None:
        raise ValueError(f"{path}: corrupt ZIP member {corrupt_member}")
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate ZIP member names")
    if set(names) != expected_names:
        missing = sorted(expected_names - set(names))
        extra = sorted(set(names) - expected_names)
        raise ValueError(f"{path}: archive mismatch; missing={missing}, extra={extra}")
    forbidden = [
        name for name in names
        if any(marker in name.lower() for marker in FORBIDDEN_CONTAINER_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"{path}: nested container archives found: {forbidden}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "entries": len(names),
        "zip_integrity": "passed",
        "nested_container_archives": [],
    }


def load_evaluation(path: Path, seed: int, checkpoint: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["completed_interactions"]) != checkpoint:
        raise ValueError(f"{path}: wrong completed interaction count")
    if payload["model_type"] != "M2":
        raise ValueError(f"{path}: expected M2")
    if payload.get("action_masking_applied") is not True:
        raise ValueError(f"{path}: action masking was not applied")
    if payload.get("collision_field") != "crashed":
        raise ValueError(f"{path}: collision field is not authoritative 'crashed'")
    if int(payload.get("invalid_action_count", -1)) != 0:
        raise ValueError(f"{path}: invalid actions detected")
    if int(payload["aggregates"]["all"]["episodes"]) != 240:
        raise ValueError(f"{path}: expected exactly 240 validation routes")
    if len(payload["episodes"]) != 240:
        raise ValueError(f"{path}: episode records are not Cartesian-complete")
    identities = {
        (
            int(episode["grid_size"]),
            str(episode["distance_bin"]),
            tuple(episode["start"]),
            tuple(episode["goal"]),
        )
        for episode in payload["episodes"]
    }
    if len(identities) != 240:
        raise ValueError(f"{path}: duplicate validation route identities")
    return payload


def expected_archive_names(include_final_inventory: bool) -> set[str]:
    names = {"inventory.txt", "provenance.json", "status.json"}
    for checkpoint in CHECKPOINTS:
        names.update({
            f"evaluation_{checkpoint:06d}.json",
            f"generator_{checkpoint:06d}.json",
            f"model_{checkpoint:06d}.zip",
            f"rng_{checkpoint:06d}.pt",
        })
    if include_final_inventory:
        names.add("final_inventory.txt")
    return names


def analyze(root: Path) -> dict:
    queue_path = root / "confirmatory_queue_status.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("status") != "complete" or queue.get("active_seed") is not None:
        raise ValueError("confirmatory queue is not complete")
    if tuple(queue.get("seeds", ())) != SEEDS:
        raise ValueError("confirmatory queue seed set/order differs from frozen protocol")
    if int(queue.get("completed_interactions_per_seed", -1)) != FINAL_CHECKPOINT:
        raise ValueError("confirmatory queue endpoint differs from frozen protocol")

    runs: list[dict] = []
    manifest_hashes: set[str] = set()
    core_hash_sets: list[dict[str, str]] = []
    route_orders: list[list[tuple]] = []
    bundle_names = expected_archive_names(False)
    final_names = expected_archive_names(True)

    for seed in SEEDS:
        seed_root = root / f"seed_{seed:03d}"
        artifacts = seed_root / "artifacts"
        provenance_path = artifacts / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if int(provenance["seed"]) != seed or provenance["model_type"] != "M2":
            raise ValueError(f"seed {seed}: provenance identity mismatch")
        if int(provenance["completed_interactions"]) != FINAL_CHECKPOINT:
            raise ValueError(f"seed {seed}: incomplete provenance")
        if provenance.get("source_hash_mode") != "lf_normalized_sha256_v1":
            raise ValueError(f"seed {seed}: noncanonical source hash mode")
        core_hashes = {key: provenance["hashes"][key] for key in CORE_HASH_KEYS}
        core_hash_sets.append(core_hashes)

        evaluations: list[dict] = []
        for checkpoint in CHECKPOINTS:
            path = artifacts / f"evaluation_{checkpoint:06d}.json"
            evaluation = load_evaluation(path, seed, checkpoint)
            manifest_hashes.add(evaluation["validation_manifest_sha256"])
            evaluations.append(evaluation)

        final = evaluations[-1]
        route_orders.append([
            (
                int(episode["grid_size"]),
                str(episode["distance_bin"]),
                tuple(episode["start"]),
                tuple(episode["goal"]),
            )
            for episode in final["episodes"]
        ])

        queue_run = queue["runs"][str(seed)]
        latest_path = artifacts / "latest_checkpoint_bundle.zip"
        latest = verify_zip(latest_path, bundle_names)
        if latest["sha256"] != queue_run["bundle_sha256"]:
            raise ValueError(f"seed {seed}: queue bundle SHA-256 mismatch")
        raw = verify_zip(artifacts / "rl_v3_phase_c2_M2_raw_artifacts.zip", final_names)
        complete = verify_zip(seed_root / "phase_c2_M2_COMPLETE.zip", final_names)
        if raw["sha256"] != complete["sha256"]:
            raise ValueError(f"seed {seed}: raw and complete archives differ")

        checkpoint_rates = [
            float(evaluation["aggregates"]["all"]["success_rate"])
            for evaluation in evaluations
        ]
        drops = [
            100.0 * (checkpoint_rates[index] - checkpoint_rates[index - 1])
            for index in range(1, len(checkpoint_rates))
        ]
        all_metrics = final["aggregates"]["all"]
        failures = collections.Counter(
            episode["failure_label"]
            for episode in final["episodes"]
            if not episode["is_success"]
        )
        runs.append({
            "seed": seed,
            "git_commit": provenance["git_commit"],
            "prior_queue_only_commit": bool(queue_run.get("prior_queue_only_commit")),
            "core_source_hashes": core_hashes,
            "checkpoint_success_rates": dict(zip(map(str, CHECKPOINTS), checkpoint_rates)),
            "largest_checkpoint_drop_percentage_points": float(min(drops)),
            "transition_75776_to_100352_percentage_points": 100.0 * (
                checkpoint_rates[3] - checkpoint_rates[2]
            ),
            "final": {
                "successes": int(all_metrics["successes"]),
                "episodes": int(all_metrics["episodes"]),
                "success_rate": float(all_metrics["success_rate"]),
                "collisions": int(all_metrics["collisions"]),
                "timeouts": int(all_metrics["timeouts"]),
                "invalid_actions": int(final["invalid_action_count"]),
                "mean_success_path_cost_ratio": float(all_metrics["mean_success_path_cost_ratio"]),
                "mean_policy_inference_latency_ms": float(all_metrics["mean_policy_inference_latency_ms"]),
                "failure_labels": dict(sorted(failures.items())),
            },
            "by_scale": {
                str(scale): final["aggregates"][f"scale/{scale}"] for scale in SCALES
            },
            "by_distance": {
                bucket: final["aggregates"][f"distance/{bucket}"]
                for bucket in DISTANCE_BINS
            },
            "by_scale_distance": {
                f"{scale}/{bucket}": final["aggregates"][f"scale_distance/{scale}/{bucket}"]
                for scale in SCALES for bucket in DISTANCE_BINS
            },
            "artifact_sha256": {
                "final_evaluation": sha256(artifacts / f"evaluation_{FINAL_CHECKPOINT:06d}.json"),
                "final_model": sha256(artifacts / f"model_{FINAL_CHECKPOINT:06d}.zip"),
                "provenance": sha256(provenance_path),
            },
            "archives": {"latest_bundle": latest, "raw": raw, "complete": complete},
        })

    if len(manifest_hashes) != 1:
        raise ValueError(f"multiple validation manifests used: {sorted(manifest_hashes)}")
    if any(hashes != core_hash_sets[0] for hashes in core_hash_sets[1:]):
        raise ValueError("core training/evaluation source hashes differ across seeds")
    if any(order != route_orders[0] for order in route_orders[1:]):
        raise ValueError("final validation route order differs across seeds")

    overall_rates = [run["final"]["success_rate"] for run in runs]
    path_ratios = [run["final"]["mean_success_path_cost_ratio"] for run in runs]
    latencies = [run["final"]["mean_policy_inference_latency_ms"] for run in runs]
    total_successes = sum(run["final"]["successes"] for run in runs)
    total_episodes = sum(run["final"]["episodes"] for run in runs)
    total_failures = collections.Counter()
    for run in runs:
        total_failures.update(run["final"]["failure_labels"])

    checkpoint_statistics = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_statistics[str(checkpoint)] = mean_interval([
            run["checkpoint_success_rates"][str(checkpoint)] for run in runs
        ])
    by_scale = {
        str(scale): mean_interval([
            float(run["by_scale"][str(scale)]["success_rate"]) for run in runs
        ]) for scale in SCALES
    }
    by_distance = {
        bucket: mean_interval([
            float(run["by_distance"][bucket]["success_rate"]) for run in runs
        ]) for bucket in DISTANCE_BINS
    }
    by_scale_distance = {
        f"{scale}/{bucket}": mean_interval([
            float(run["by_scale_distance"][f"{scale}/{bucket}"]["success_rate"])
            for run in runs
        ]) for scale in SCALES for bucket in DISTANCE_BINS
    }

    return {
        "schema_version": 1,
        "classification": "confirmatory_validation_multi_training_seed",
        "claim_limit": (
            "The five independent training seeds are the inferential units. The 1,200 "
            "route outcomes are reported descriptively and are not treated as independent "
            "replicates of training. The final test set remains sealed."
        ),
        "protocol": {
            "model_type": "M2",
            "development_seed_excluded": 42,
            "confirmatory_seeds": list(SEEDS),
            "requested_interactions_per_seed": 150000,
            "completed_interactions_per_seed": FINAL_CHECKPOINT,
            "checkpoint_selection": "fixed preregistered endpoint; no per-seed selection",
            "validation_routes_per_seed": 240,
            "validation_manifest_sha256": next(iter(manifest_hashes)),
            "final_test_status": "sealed_not_accessed",
        },
        "integrity": {
            "status": "passed",
            "queue_status_sha256": sha256(queue_path),
            "all_core_source_hashes_identical": True,
            "all_final_route_orders_identical": True,
            "all_archives_pass_zip_integrity": True,
            "all_archives_exclude_nested_container_archives": True,
            "core_source_hashes": core_hash_sets[0],
            "seed_11_queue_only_commit_difference": True,
        },
        "primary_success_rate": mean_interval(overall_rates),
        "descriptive_route_total": {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes,
            "inferential_unit_warning": "descriptive only; do not use n=1200 as the training-seed sample size",
        },
        "secondary_seed_level_metrics": {
            "mean_success_path_cost_ratio": mean_interval(path_ratios),
            "mean_policy_inference_latency_ms": mean_interval(latencies),
        },
        "final_failures": {
            "collisions": sum(run["final"]["collisions"] for run in runs),
            "timeouts": sum(run["final"]["timeouts"] for run in runs),
            "invalid_actions": sum(run["final"]["invalid_actions"] for run in runs),
            "failure_labels": dict(sorted(total_failures.items())),
        },
        "checkpoint_statistics": checkpoint_statistics,
        "curriculum_transition_change_75776_to_100352_percentage_points": mean_interval([
            run["transition_75776_to_100352_percentage_points"] for run in runs
        ]),
        "by_scale": by_scale,
        "by_distance": by_distance,
        "by_scale_distance": by_scale_distance,
        "runs": runs,
    }


def write_seed_table(summary: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "seed", "successes", "episodes", "success_rate_percent", "collisions",
            "timeouts", "invalid_actions", "mean_success_path_cost_ratio",
            "mean_policy_inference_latency_ms", "transition_75k_to_100k_pp",
            "git_commit", "latest_bundle_sha256", "complete_archive_sha256",
        ])
        for run in summary["runs"]:
            writer.writerow([
                run["seed"], run["final"]["successes"], run["final"]["episodes"],
                100.0 * run["final"]["success_rate"], run["final"]["collisions"],
                run["final"]["timeouts"], run["final"]["invalid_actions"],
                run["final"]["mean_success_path_cost_ratio"],
                run["final"]["mean_policy_inference_latency_ms"],
                run["transition_75776_to_100352_percentage_points"], run["git_commit"],
                run["archives"]["latest_bundle"]["sha256"],
                run["archives"]["complete"]["sha256"],
            ])


def plot_summary(summary: dict, path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "figure.dpi": 120,
    })
    colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), gridspec_kw={"width_ratios": [1.6, 0.9, 1.2]})

    x = np.asarray(CHECKPOINTS) / 1000.0
    rate_matrix = np.asarray([
        [100.0 * run["checkpoint_success_rates"][str(checkpoint)] for checkpoint in CHECKPOINTS]
        for run in summary["runs"]
    ])
    for row, (run, color) in enumerate(zip(summary["runs"], colors)):
        axes[0].plot(x, rate_matrix[row], marker="o", linewidth=1.5, alpha=0.82,
                     color=color, label=f"Seed {run['seed']}")
    axes[0].plot(x, rate_matrix.mean(axis=0), marker="D", linewidth=3.0,
                 color="#111827", label="Seed mean", zorder=8)
    axes[0].axvspan(75.776, 100.352, color="#6B7280", alpha=0.10)
    axes[0].set(title="Learning curves reveal curriculum instability",
                xlabel="Completed interactions (thousands)", ylabel="Validation success (%)",
                ylim=(0, 103))
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, ncol=2, loc="lower right")

    final_rates = rate_matrix[:, -1]
    seed_positions = np.linspace(-0.10, 0.10, len(SEEDS))
    axes[1].scatter(seed_positions, final_rates, s=55, c=colors, edgecolor="white", zorder=4)
    for x_position, value, seed in zip(seed_positions, final_rates, SEEDS):
        axes[1].annotate(str(seed), (x_position, value), xytext=(0, -11),
                         textcoords="offset points", ha="center", fontsize=7.5,
                         color="#374151")
    primary = summary["primary_success_rate"]
    mean = 100.0 * primary["mean"]
    lower, upper = [100.0 * value for value in primary["mean_t_95_ci_bounded_0_1"]]
    axes[1].errorbar([0.34], [mean], yerr=[[mean - lower], [upper - mean]], fmt="D",
                     color="#111827", capsize=6, linewidth=2.2, label="Mean and 95% t CI")
    axes[1].set_xticks([0, 0.34], ["Seeds", "Mean"])
    axes[1].set_xlim(-0.22, 0.56)
    axes[1].set_ylim(90, 101)
    axes[1].set_ylabel("Final success (%)")
    axes[1].set_title("Fixed 151,552-step endpoint")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].text(0.5, 0.03, f"{mean:.2f}%  (95% CI {lower:.2f}–{upper:.2f})",
                 transform=axes[1].transAxes, ha="center", fontsize=8.5)

    scale_values = np.asarray([
        [100.0 * run["by_scale"][str(scale)]["success_rate"] for scale in SCALES]
        for run in summary["runs"]
    ])
    positions = np.arange(len(SCALES))
    for row, color in enumerate(colors):
        axes[2].plot(positions, scale_values[row], color=color, alpha=0.38, linewidth=1.0)
        axes[2].scatter(positions, scale_values[row], color=color, s=25, zorder=3)
    axes[2].plot(positions, scale_values.mean(axis=0), color="#111827", marker="D",
                 linewidth=2.8, label="Seed mean", zorder=5)
    axes[2].set_xticks(positions, [f"{scale}×{scale}" for scale in SCALES])
    axes[2].set(title="Final performance by native grid scale", xlabel="Grid scale",
                ylabel="Success (%)", ylim=(70, 102))
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False, loc="lower left")

    figure.suptitle("Phase C2 M2 confirmatory validation — five independent training seeds",
                    fontsize=14, fontweight="bold")
    figure.text(0.5, 0.012,
                "Fixed 240-route validation manifest per seed; final test remains sealed. Confidence interval uses n=5 training seeds.",
                ha="center", fontsize=8.5, color="#4B5563")
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_matrix(summary: dict, path: Path) -> None:
    matrix = np.asarray([
        [100.0 * summary["by_scale_distance"][f"{scale}/{bucket}"]["mean"]
         for bucket in DISTANCE_BINS]
        for scale in SCALES
    ])
    ranges = [[
        summary["by_scale_distance"][f"{scale}/{bucket}"]
        for bucket in DISTANCE_BINS
    ] for scale in SCALES]
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=70, vmax=100, aspect="auto")
    for row in range(len(SCALES)):
        for column in range(len(DISTANCE_BINS)):
            item = ranges[row][column]
            value = matrix[row, column]
            color = "white" if value >= 91 else "#111827"
            axis.text(column, row, f"{value:.1f}%\n[{100*item['minimum']:.0f}, {100*item['maximum']:.0f}]",
                      ha="center", va="center", color=color, fontweight="bold", fontsize=10)
    axis.set_xticks(range(len(DISTANCE_BINS)), [bucket.title() for bucket in DISTANCE_BINS])
    axis.set_yticks(range(len(SCALES)), [f"{scale}×{scale}" for scale in SCALES])
    axis.set_xlabel("Route-distance bin")
    axis.set_ylabel("Native grid scale")
    axis.set_title("Final success by scale and distance\nseed mean [minimum, maximum], n=5")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean success (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = analyze(args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "phase_c2_v15_m2_confirmatory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_seed_table(summary, args.output_dir / "phase_c2_v15_m2_confirmatory_seed_table.csv")
    plot_summary(summary, args.output_dir / "phase_c2_v15_m2_confirmatory_overview.png")
    plot_matrix(summary, args.output_dir / "phase_c2_v15_m2_confirmatory_scale_distance.png")
    print(summary_path)


if __name__ == "__main__":
    main()
