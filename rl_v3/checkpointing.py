"""RL V3 checkpoint contract and smoke demonstration."""
from __future__ import annotations

import base64
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from evaluation.experiment_metadata import collect_environment_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def save_smoke_checkpoint(
    *,
    run_dir: Path,
    config: dict,
    policy_seed: int,
    generator_state: dict,
    manifest_hashes: dict[str, str],
    generator_hash: str,
    timesteps: int,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(timesteps):
        x = torch.randn(4)
        target = torch.tensor([0.25, -0.25])
        loss = torch.nn.functional.mse_loss(model(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestep": timesteps,
            "configuration": config,
            "policy_seed": policy_seed,
            "environment_generator_state": generator_state,
            "python_rng_state": repr(random.getstate()),
            "numpy_rng_state": _encode_numpy_state(np.random.get_state()),
            "torch_rng_state_b64": base64.b64encode(torch.random.get_rng_state().numpy().tobytes()).decode("ascii"),
            "normalization_statistics": {},
            "source_revision": source_revision(),
            "manifest_hashes": manifest_hashes,
            "generator_hash": generator_hash,
            "software_hardware_manifest": collect_environment_metadata(PROJECT_ROOT),
            "resume_semantics": (
                "single-process smoke checkpoint; bit-identical resume is not "
                "claimed for future vectorized workers until separately verified"
            ),
            "run_status": "interrupted_smoke_checkpoint",
        },
        checkpoint_path,
    )
    status = {
        "schema_version": 1,
        "checkpoint_path": str(checkpoint_path),
        "timestep": timesteps,
        "policy_seed": policy_seed,
        "losses": losses,
        "source_revision": source_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "resume_verified": verify_smoke_checkpoint(checkpoint_path),
    }
    (run_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def verify_smoke_checkpoint(path: Path) -> bool:
    data = torch.load(path, map_location="cpu")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "timestep",
        "configuration",
        "policy_seed",
        "environment_generator_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state_b64",
        "normalization_statistics",
        "source_revision",
        "manifest_hashes",
        "generator_hash",
        "software_hardware_manifest",
        "run_status",
    }
    return required.issubset(data)


def _encode_numpy_state(state: tuple) -> dict:
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }
