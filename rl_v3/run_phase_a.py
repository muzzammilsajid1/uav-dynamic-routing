"""Generate Phase A assets and diagnostics without substantive PPO training."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_v3.checkpointing import save_smoke_checkpoint
from rl_v3.diagnostics import (
    empty_map_scenarios,
    evaluate_episode,
    load_frozen_ddqn,
    write_csv,
)
from rl_v3.scenario_generation import (
    build_training_generator_asset,
    generate_training_episode,
    generate_validation_manifest,
    sha256_json,
    verify_manifest_separation,
    write_stable_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "rl_v3_phase_a.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "rl_v3" / "phase_a",
    )
    parser.add_argument(
        "--limit-validation-per-grid",
        type=int,
        default=6,
        help="Small diagnostic cap per grid size; does not affect the manifest.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    training_asset = build_training_generator_asset(config)
    generator_asset_path = PROJECT_ROOT / "evaluation" / "manifests" / "rl_v3_training_generator.json"
    generator_hash = write_stable_json(generator_asset_path, training_asset)

    validation = generate_validation_manifest(config)
    validation_path = PROJECT_ROOT / "evaluation" / "manifests" / "rl_v3_validation.json"
    validation_hash = write_stable_json(validation_path, validation)

    separation = verify_manifest_separation(config, validation)
    (args.out_dir / "manifest_separation.json").write_text(
        json.dumps(separation, indent=2) + "\n", encoding="utf-8"
    )

    training_preview = [generate_training_episode(config, index) for index in range(9)]
    (args.out_dir / "training_generator_preview.json").write_text(
        json.dumps(training_preview, indent=2) + "\n", encoding="utf-8"
    )

    ddqn_cfg = config["ddqn_diagnostic"]
    research_config = json.loads((PROJECT_ROOT / "configs" / "research_experiments.json").read_text(encoding="utf-8"))
    variant_config = research_config["variants"][ddqn_cfg["variant"]]
    model = load_frozen_ddqn(
        ddqn_cfg["variant"],
        int(ddqn_cfg["seeds"][0]),
        ddqn_cfg["checkpoint_stage"],
        variant_config,
    )

    validation_rows = _run_validation_diagnostics(
        model,
        validation["scenarios"],
        variant_config,
        ddqn_cfg,
        ddqn_cfg["modes"],
        args.out_dir,
        args.limit_validation_per_grid,
    )
    validation_csv = args.out_dir / "ddqn_original_vs_posthoc_masked.csv"
    write_csv(validation_csv, validation_rows)

    empty_rows = _run_empty_map_diagnostics(model, config, variant_config, ddqn_cfg, ddqn_cfg["modes"], args.out_dir)
    empty_csv = args.out_dir / "empty_map_diagnostics.csv"
    write_csv(empty_csv, empty_rows)

    _write_counts(args.out_dir / "failure_taxonomy_counts.csv", validation_rows + empty_rows)
    _write_summary(args.out_dir / "phase_a_summary.json", config, generator_hash, validation_hash, separation, validation_rows, empty_rows)
    _plot_empty_results(args.out_dir, empty_rows)
    _plot_representative_trajectories(args.out_dir)

    checkpoint_status = save_smoke_checkpoint(
        run_dir=args.out_dir / "checkpoint_smoke",
        config=config,
        policy_seed=int(config["checkpoint_smoke"]["seed"]),
        generator_state={"previewed_training_episodes": len(training_preview)},
        manifest_hashes={"validation": validation_hash},
        generator_hash=generator_hash,
        timesteps=int(config["checkpoint_smoke"]["timesteps"]),
    )
    print(f"training_generator_hash={generator_hash}")
    print(f"validation_manifest_hash={validation_hash}")
    print(f"manifest_separation_passed={separation['passed']}")
    print(f"validation_rows={len(validation_rows)}")
    print(f"empty_map_rows={len(empty_rows)}")
    print(f"checkpoint_resume_contract={checkpoint_status['resume_verified']}")


def _run_validation_diagnostics(model, scenarios, variant_config, ddqn_cfg, modes, out_dir, limit_per_grid):
    rows = []
    per_grid = defaultdict(int)
    for scenario in scenarios:
        grid_size = int(scenario["grid_size"])
        if per_grid[grid_size] >= limit_per_grid:
            continue
        per_grid[grid_size] += 1
        for mode in modes:
            trajectory_path = out_dir / "trajectories" / f"{scenario['scenario_id']}_{mode}.json"
            rows.append(
                evaluate_episode(
                    model,
                    scenario,
                    mode=mode,
                    variant_config=variant_config,
                    diagnostic_budget_to_astar_cost=float(ddqn_cfg["diagnostic_budget_to_astar_cost"]),
                    minimum_diagnostic_budget=int(ddqn_cfg["minimum_diagnostic_budget"]),
                    save_trajectory_path=trajectory_path,
                )
            )
    return rows


def _run_empty_map_diagnostics(model, config, variant_config, ddqn_cfg, modes, out_dir):
    rows = []
    for scenario in empty_map_scenarios(config):
        for mode in modes:
            trajectory_path = out_dir / "trajectories" / f"{scenario['scenario_id']}_{mode}.json"
            rows.append(
                evaluate_episode(
                    model,
                    scenario,
                    mode=mode,
                    variant_config=variant_config,
                    diagnostic_budget_to_astar_cost=float(ddqn_cfg["diagnostic_budget_to_astar_cost"]),
                    minimum_diagnostic_budget=int(ddqn_cfg["minimum_diagnostic_budget"]),
                    save_trajectory_path=trajectory_path,
                )
            )
    return rows


def _write_counts(path: Path, rows: list[dict]) -> None:
    counts = Counter((row["grid_size"], row["mode"], row["failure_label"]) for row in rows)
    table = [
        {
            "grid_size": grid_size,
            "mode": mode,
            "failure_label": label,
            "count": count,
        }
        for (grid_size, mode, label), count in sorted(counts.items())
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grid_size", "mode", "failure_label", "count"])
        writer.writeheader()
        writer.writerows(table)


def _write_summary(path, config, generator_hash, validation_hash, separation, validation_rows, empty_rows):
    def summarize(rows):
        by_mode = defaultdict(list)
        for row in rows:
            by_mode[row["mode"]].append(row)
        return {
            mode: {
                "episodes": len(items),
                "success_rate": sum(1 for item in items if item["success"]) / max(1, len(items)),
                "failure_counts": dict(Counter(item["failure_label"] for item in items)),
            }
            for mode, items in by_mode.items()
        }
    summary = {
        "schema_version": 1,
        "generator_hash": generator_hash,
        "validation_hash": validation_hash,
        "manifest_separation": separation,
        "validation_diagnostics": summarize(validation_rows),
        "empty_map_diagnostics": summarize(empty_rows),
        "final_test_exposed": False,
        "substantive_ppo_training_launched": False,
        "dependency_resolution": {
            "stable_baselines3": "2.9.0",
            "sb3_contrib": "2.9.0",
            "compatibility": "sb3-contrib 2.9.0 declares stable-baselines3>=2.9.0,<3.0",
        },
        "config_sha256": sha256_json(config),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _plot_empty_results(out_dir: Path, rows: list[dict]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for mode in sorted({row["mode"] for row in rows}):
        grouped = defaultdict(list)
        for row in rows:
            if row["mode"] == mode:
                grouped[int(row["grid_size"])].append(1.0 if row["success"] else 0.0)
        xs = sorted(grouped)
        ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Grid size")
        plt.ylabel("Success rate")
        plt.title(f"Empty-map success ({mode})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"empty_success_by_grid_{mode}.png", dpi=160)
        plt.close()


def _plot_representative_trajectories(out_dir: Path) -> None:
    plot_dir = out_dir / "plots" / "trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    trajectory_files = sorted((out_dir / "trajectories").glob("*.json"))
    chosen = []
    seen_labels = set()
    for path in trajectory_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        route = data["route"]
        key = (route["mode"], route["failure_label"])
        if key in seen_labels:
            continue
        chosen.append((path, data))
        seen_labels.add(key)
        if len(chosen) >= 8:
            break
    for path, data in chosen:
        route = data["route"]
        trajectory = data["trajectory"]
        astar_path = data.get("initial_astar_path", [])
        size = int(route["grid_size"])
        plt.figure(figsize=(5, 5))
        if astar_path:
            plt.plot(
                [cell[1] for cell in astar_path],
                [cell[0] for cell in astar_path],
                color="#4C78A8",
                linewidth=1.5,
                label="Initial A*",
            )
        plt.plot(
            [cell[1] for cell in trajectory],
            [cell[0] for cell in trajectory],
            color="#F58518",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label="DDQN trajectory",
        )
        plt.scatter([trajectory[0][1]], [trajectory[0][0]], color="#54A24B", s=35, label="Start")
        plt.scatter([trajectory[-1][1]], [trajectory[-1][0]], color="#E45756", s=35, label="End")
        plt.gca().invert_yaxis()
        plt.xlim(-0.5, size - 0.5)
        plt.ylim(size - 0.5, -0.5)
        plt.grid(True, alpha=0.25)
        plt.title(f"{route['mode']} / {route['failure_label']} / {route['scenario_id']}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        safe_name = path.stem.replace(":", "_")
        plt.savefig(plot_dir / f"{safe_name}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
