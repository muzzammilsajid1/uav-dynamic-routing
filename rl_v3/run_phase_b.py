"""Generate, train, resume, evaluate, and summarize the controlled Phase B pilots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sb3_contrib import MaskablePPO

from rl_v3.phase_b_env import PhaseBCurriculumEnv
from rl_v3.phase_b_evaluation import evaluate_manifest
from rl_v3.phase_b_policy import PhaseBFeatureExtractor, inference_latency_ms, model_parameter_count
from rl_v3.phase_b_scenarios import generate_validation_v2, manifest_balance
from rl_v3.phase_b_visuals import render_coarse_contract_gallery
from rl_v3.scenario_generation import write_stable_json

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "rl_v3" / "phase_b"
CONFIG_PATH = ROOT / "configs" / "rl_v3_phase_b.json"
MANIFEST_PATH = ROOT / "evaluation" / "manifests" / "rl_v3_validation_v2.json"
SOURCE_FILES = (
    "configs/rl_v3_phase_b.json", "requirements.txt", "rl_agent/uav_env.py",
    "rl_v3/action_masking.py", "rl_v3/env_builders.py", "rl_v3/observations.py",
    "rl_v3/wrappers.py", "rl_v3/phase_b_env.py", "rl_v3/phase_b_policy.py",
    "rl_v3/phase_b_scenarios.py", "rl_v3/phase_b_evaluation.py",
    "rl_v3/run_phase_b.py",
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def setup() -> dict:
    config = load_config()
    existing_phase_a = ROOT / "evaluation" / "manifests" / "rl_v3_validation.json"
    phase_a_hash_before = hashlib.sha256(existing_phase_a.read_bytes()).hexdigest()
    manifest = generate_validation_v2(config)
    write_stable_json(MANIFEST_PATH, manifest)
    phase_a_hash_after = hashlib.sha256(existing_phase_a.read_bytes()).hexdigest()
    if phase_a_hash_before != phase_a_hash_after:
        raise RuntimeError("Phase A validation manifest changed")
    seeds = {int(row["episode_seed"]) for row in manifest["scenarios"]}
    final_low, final_high = config["private_final_seed_range"]
    if any(final_low <= seed <= final_high for seed in seeds):
        raise RuntimeError("validation_v2 overlaps private final-test seed range")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    render_coarse_contract_gallery(RUN_ROOT / "coarse_debug")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_v2_hash": manifest["manifest_sha256"],
        "balance": manifest_balance(manifest),
        "phase_a_manifest_hash_unchanged": phase_a_hash_before,
        "private_final_test_exposed": False,
        "source_provenance": source_provenance(),
        "hardware": hardware_manifest(),
    }
    write_stable_json(RUN_ROOT / "setup.json", result)
    return result


def build_model(config: dict, pilot: dict, env: PhaseBCurriculumEnv) -> MaskablePPO:
    training = config["training"]
    return MaskablePPO(
        "MultiInputPolicy", env,
        learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]), batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]), gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]), ent_coef=float(training["ent_coef"]),
        seed=int(config["seed"]), device="cpu", verbose=1,
        policy_kwargs={
            "features_extractor_class": PhaseBFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            "activation_fn": torch.nn.ReLU,
        },
    )


def run_pilot(pilot_id: str) -> dict:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = load_config()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pilot = next(item for item in config["pilots"] if item["id"] == pilot_id)
    run_dir = RUN_ROOT / pilot_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if source_provenance()["aggregate_sha256"] != json.loads((RUN_ROOT / "setup.json").read_text())["source_provenance"]["aggregate_sha256"]:
        raise RuntimeError("Phase B training sources changed after setup; regenerate setup intentionally")
    env = PhaseBCurriculumEnv(config, pilot["observation"], pilot["reward"])
    status_path = run_dir / "status.json"
    completed = 0
    history: list[dict] = []
    checkpoint = None
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = int(status.get("completed_interactions", 0))
        history = list(status.get("history", []))
        candidate = run_dir / f"model_{completed:06d}.zip"
        if completed and candidate.exists():
            env.set_generator_state(status["generator_state"])
            checkpoint = candidate
    if checkpoint:
        model = MaskablePPO.load(checkpoint, env=env, device="cpu")
    else:
        model = build_model(config, pilot, env)
    parameters = model_parameter_count(model.policy)
    observation, _ = env.reset(seed=int(config["seed"]))
    latency = inference_latency_ms(model, observation, env.action_masks(), repeats=100)
    process = psutil.Process()
    checkpoints = [int(value) for value in config["training"]["checkpoints"]]
    for target in checkpoints:
        if target <= completed:
            continue
        delta = target - completed
        cpu_start = process.cpu_times()
        wall_start = time.perf_counter()
        model.learn(total_timesteps=delta, reset_num_timesteps=False, progress_bar=False)
        wall = time.perf_counter() - wall_start
        cpu_end = process.cpu_times()
        cpu_seconds = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
        completed = target
        model_path = run_dir / f"model_{target:06d}"
        model.save(model_path)
        evaluation_dir = run_dir / "evaluation" / f"step_{target:06d}"
        rows = evaluate_manifest(model, manifest, pilot, config, evaluation_dir)
        success_rate = sum(bool(row["success"]) for row in rows) / len(rows)
        item = {
            "interactions": target,
            "training_wall_seconds": wall,
            "environment_steps_per_second": delta / wall,
            "average_process_cpu_utilization_percent": 100.0 * cpu_seconds / max(wall, 1e-9) / max(1, psutil.cpu_count()),
            "gpu_utilization_percent": None,
            "gpu_note": "PyTorch CPU build; CUDA unavailable",
            "validation_success_rate": success_rate,
            "validation_episodes": len(rows),
        }
        history.append(item)
        status = {
            "pilot": pilot, "seed": config["seed"],
            "completed_interactions": completed, "target_interactions": 100000,
            "parameter_count": parameters, "initial_inference_latency_ms": latency,
            "generator_state": env.get_generator_state(), "history": history,
            "source_provenance": source_provenance(),
            "source_provenance_history": _provenance_history(
                status if status_path.exists() else None, source_provenance()
            ),
            "validation_v2_hash": manifest["manifest_sha256"],
            "resume_semantics": "statistically_equivalent",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_stable_json(status_path, status)
    env.close()
    plot_learning_curve(run_dir, history)
    return json.loads(status_path.read_text(encoding="utf-8"))


def verify_resume() -> dict:
    """Short save/load proof; conservative classification avoids false determinism."""
    config = load_config()
    pilot = config["pilots"][1]
    proof_dir = RUN_ROOT / "resume_verification"
    proof_dir.mkdir(parents=True, exist_ok=True)
    env = PhaseBCurriculumEnv(config, pilot["observation"], pilot["reward"])
    model = build_model(config, pilot, env)
    model.learn(total_timesteps=1000, progress_bar=False)
    model.save(proof_dir / "model_001000")
    state = env.get_generator_state()
    write_stable_json(proof_dir / "generator_state.json", state)
    reloaded_env = PhaseBCurriculumEnv(config, pilot["observation"], pilot["reward"])
    reloaded_env.set_generator_state(state)
    reloaded = MaskablePPO.load(proof_dir / "model_001000.zip", env=reloaded_env, device="cpu")
    obs, _ = reloaded_env.reset()
    action, _ = reloaded.predict(obs, deterministic=True, action_masks=reloaded_env.action_masks())
    result = {
        "checkpoint_loads": True,
        "optimizer_and_model_state_restored": True,
        "generator_state_restored": reloaded_env.get_generator_state(),
        "post_resume_action_is_legal": bool(reloaded_env.action_masks()[int(action)]),
        "classification": "statistically_equivalent",
        "reason": "SB3 save/load does not serialize a partially active environment transition; bit-identical or transition-identical continuation is therefore not claimed.",
    }
    write_stable_json(proof_dir / "result.json", result)
    env.close(); reloaded_env.close()
    return result


def source_provenance() -> dict:
    files = {}
    aggregate = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        files[relative] = digest
        aggregate.update(relative.encode()); aggregate.update(digest.encode())
    return {"aggregate_sha256": aggregate.hexdigest(), "files": files}


def _provenance_history(previous_status: dict | None, current: dict) -> list[dict]:
    history = [] if previous_status is None else list(previous_status.get("source_provenance_history", []))
    previous = None if previous_status is None else previous_status.get("source_provenance")
    for item in (previous, current):
        if item and all(existing.get("aggregate_sha256") != item["aggregate_sha256"] for existing in history):
            history.append(item)
    return history


def hardware_manifest() -> dict:
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    }


def plot_learning_curve(run_dir: Path, history: list[dict]) -> None:
    if not history: return
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.plot([x["interactions"] for x in history], [x["validation_success_rate"] for x in history], marker="o")
    ax.set(xlabel="Training interactions", ylabel="Validation success rate", ylim=(0, 1), title=run_dir.name)
    ax.grid(alpha=0.25)
    fig.savefig(run_dir / "learning_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("setup", "verify-resume", "pilot", "all"))
    parser.add_argument("--pilot", choices=("P1", "P2", "P3", "P4"))
    args = parser.parse_args()
    if args.command == "setup": print(json.dumps(setup(), indent=2))
    elif args.command == "verify-resume": print(json.dumps(verify_resume(), indent=2))
    elif args.command == "pilot":
        if not args.pilot: parser.error("--pilot is required")
        print(json.dumps(run_pilot(args.pilot), indent=2))
    else:
        for pilot in ("P1", "P2", "P3", "P4"): run_pilot(pilot)


if __name__ == "__main__":
    main()
