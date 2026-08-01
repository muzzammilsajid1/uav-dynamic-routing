"""Local, resumable multi-seed DQN+HER curriculum and ablation runner."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from envs.grid_environment import DynamicObstacle, default_dynamic_obstacles
from evaluation.experiment_metadata import collect_environment_metadata
from rl_agent.double_dqn import DoubleDQN
from rl_agent.safe_her_buffer import SafeHerReplayBuffer
from rl_agent.uav_env import UAVRoutingEnv


def _dynamic_obstacles(kind: str) -> list[DynamicObstacle]:
    if kind == "none":
        return []
    if kind == "mild":
        return [DynamicObstacle((8, 8), period=10, initial_state="blocked")]
    if kind == "full":
        return default_dynamic_obstacles()
    raise ValueError(f"Unknown dynamic-obstacle stage: {kind}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_env(
    config: dict,
    stage: dict,
    variant: dict,
    layout_seed: int,
) -> Monitor:
    dynamic = _dynamic_obstacles(stage["dynamic"])
    curriculum = bool(stage["curriculum_enabled"] and variant["curriculum"])
    env = UAVRoutingEnv(
        grid_size=int(config["grid_size"]),
        obstacle_density=float(stage["obstacle_density"]),
        no_fly_density=0.0,
        dynamic_obstacles_enabled=bool(dynamic),
        dynamic_obstacles=dynamic,
        fixed_grid=True,
        seed=layout_seed,
        curriculum_enabled=curriculum,
        potential_shaping_enabled=bool(variant["potential_shaping"]),
        observation_mode=str(variant["observation_mode"]),
        dynamics_timing=str(config["dynamics_timing"]),
    )
    env.reset(seed=layout_seed)
    env.action_space.seed(layout_seed)
    return Monitor(env)


def _new_model(
    model_class: type[DQN],
    env: Monitor,
    config: dict,
    variant: dict,
    seed: int,
    stage_steps: int,
) -> DQN:
    model_config = config["model"]
    learning_starts = min(
        int(model_config["learning_starts"]), max(50, stage_steps // 4)
    )
    kwargs: dict[str, object] = {}
    if variant["her"]:
        kwargs.update(
            replay_buffer_class=SafeHerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": int(model_config["her_sampled_goals"]),
                "goal_selection_strategy": "future",
            },
        )
    return model_class(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=float(model_config["learning_rate"]),
        buffer_size=int(model_config["buffer_size"]),
        learning_starts=learning_starts,
        batch_size=int(model_config["batch_size"]),
        tau=1.0,
        gamma=float(model_config["gamma"]),
        train_freq=int(model_config["train_freq"]),
        gradient_steps=int(model_config["gradient_steps"]),
        target_update_interval=int(model_config["target_update_interval"]),
        exploration_fraction=float(model_config["exploration_fraction"]),
        exploration_initial_eps=float(model_config["exploration_initial_eps"]),
        exploration_final_eps=float(model_config["exploration_final_eps"]),
        policy_kwargs={"net_arch": list(model_config["network"])},
        verbose=0,
        seed=seed,
        device="auto",
        **kwargs,
    )


def train_seed(
    *,
    config_path: Path,
    config: dict,
    variant_name: str,
    seed: int,
    smoke_steps: int | None,
    requested_stage: str | None,
) -> None:
    variant = config["variants"][variant_name]
    model_class = DoubleDQN if variant["double_dqn"] else DQN
    model_dir = PROJECT_ROOT / "models" / "research" / variant_name / f"seed_{seed:03d}"
    run_dir = PROJECT_ROOT / "runs" / "research" / variant_name / f"seed_{seed:03d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_records: list[dict[str, object]] = []
    prior_checkpoint: Path | None = None
    stages = config["stages"]
    for stage_index, stage in enumerate(stages):
        if requested_stage and stage["name"] != requested_stage:
            prior_candidate = model_dir / f"{stage_index:02d}_{stage['name']}_final.zip"
            if prior_candidate.exists():
                prior_checkpoint = prior_candidate
            continue

        _seed_everything(seed)
        configured_steps = int(
            variant.get("stage_timesteps", {}).get(
                stage["name"], stage["timesteps"]
            )
        )
        steps = smoke_steps if smoke_steps is not None else configured_steps
        env = _make_env(config, stage, variant, int(config["layout_seed"]))
        from_scratch = (
            stage["name"] == "static"
            or (
                variant["dynamic_from_scratch"]
                and stage["name"] == "dynamic_full"
            )
        )
        if from_scratch:
            model = _new_model(model_class, env, config, variant, seed, steps)
            source = "new"
        else:
            if prior_checkpoint is None or not prior_checkpoint.exists():
                raise FileNotFoundError(
                    f"Stage {stage['name']} requires its prior checkpoint"
                )
            model = model_class.load(prior_checkpoint, env=env, device="auto")
            warmup = min(int(config["model"]["learning_starts"]), max(50, steps // 4))
            model.learning_starts = model.num_timesteps + warmup
            source = str(prior_checkpoint)

        stage_run_dir = run_dir / stage["name"]
        stage_run_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(stage_run_dir), ["csv"]))
        checkpoint = CheckpointCallback(
            save_freq=max(1, steps // 2),
            save_path=str(stage_run_dir),
            name_prefix=f"{variant_name}_seed{seed}_{stage['name']}",
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=from_scratch,
            callback=checkpoint,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - started

        final_path = model_dir / f"{stage_index:02d}_{stage['name']}_final"
        model.save(final_path)
        prior_checkpoint = final_path.with_suffix(".zip")
        env.close()
        stage_records.append(
            {
                "stage": stage["name"],
                "configured_timesteps": configured_steps,
                "executed_timesteps": steps,
                "source_checkpoint": source,
                "output_checkpoint": str(prior_checkpoint),
                "elapsed_seconds": elapsed,
                "steps_per_second": steps / elapsed,
            }
        )
        print(
            f"seed={seed} variant={variant_name} stage={stage['name']} "
            f"steps={steps} elapsed={elapsed:.1f}s"
        )

    metadata = collect_environment_metadata(PROJECT_ROOT)
    metadata["training"] = {
        "config_path": str(config_path),
        "config": config,
        "variant": variant_name,
        "variant_config": variant,
        "policy_seed": seed,
        "layout_seed": config["layout_seed"],
        "smoke_test": smoke_steps is not None,
        "stages": stage_records,
    }
    output = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / f"training_{variant_name}_seed_{seed:03d}.json"
    )
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "research_experiments.json",
    )
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--stage", choices=["static", "dynamic_mild", "dynamic_full"])
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Override each selected stage with a short validation run.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported training configuration schema")
    if args.variant not in config["variants"]:
        raise ValueError(f"Unknown variant: {args.variant}")
    seeds = args.seeds or [int(seed) for seed in config["policy_seeds"]]
    for seed in seeds:
        train_seed(
            config_path=args.config,
            config=config,
            variant_name=args.variant,
            seed=seed,
            smoke_steps=args.smoke_steps,
            requested_stage=args.stage,
        )


if __name__ == "__main__":
    main()
