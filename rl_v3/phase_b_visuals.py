"""Visual checks for coarse-map semantics and Phase B results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_agent.uav_env import CELL_OBSTACLE, UAVRoutingEnv
from rl_v3.observations import coarse_global_map


def render_coarse_contract_gallery(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _debug_cases()
    paths = []
    for name, env in cases:
        coarse = coarse_global_map(env, output_size=32, visited={(1, 1): 3})
        display = coarse[0] + 0.5 * (coarse[0] * coarse[6])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        axes[0].imshow(env.grid, cmap="gray_r", interpolation="nearest")
        axes[0].scatter(env.uav_pos[1], env.uav_pos[0], c="tab:blue", marker="o", label="UAV")
        axes[0].scatter(env.goal_pos[1], env.goal_pos[0], c="tab:green", marker="*", label="Goal")
        axes[0].set_title(f"Original: {name}")
        axes[0].legend(fontsize=7)
        axes[1].imshow(display, cmap="magma", interpolation="nearest", vmin=0, vmax=1.5)
        axes[1].set_title("32x32 blocked + mixed/free signal")
        for axis in axes:
            axis.set_xticks([]); axis.set_yticks([])
        path = output_dir / f"coarse_{name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _debug_cases() -> list[tuple[str, UAVRoutingEnv]]:
    outputs = []
    for name, size in [
        ("one_cell_wall", 50), ("narrow_corridor", 50), ("u_shape", 50),
        ("isolated_obstacle", 50), ("non_divisible", 47), ("same_bin_agent_goal", 100),
    ]:
        env = UAVRoutingEnv(grid_size=size, obstacle_density=0.0, fixed_grid=True, seed=1)
        env.reset(seed=1)
        env.grid = np.zeros((size, size), dtype=np.int32)
        env.uav_pos = np.array([1, 1], dtype=np.int32)
        env.goal_pos = np.array([size - 2, size - 2], dtype=np.int32)
        mid = size // 2
        if name == "one_cell_wall": env.grid[3:-3, mid] = CELL_OBSTACLE
        elif name == "narrow_corridor":
            env.grid[3:-3, mid - 1] = CELL_OBSTACLE; env.grid[3:-3, mid + 1] = CELL_OBSTACLE
        elif name == "u_shape":
            env.grid[mid:, mid - 8] = CELL_OBSTACLE; env.grid[mid:, mid + 8] = CELL_OBSTACLE; env.grid[-4, mid - 8:mid + 9] = CELL_OBSTACLE
        elif name == "isolated_obstacle": env.grid[mid, mid] = CELL_OBSTACLE
        elif name == "same_bin_agent_goal": env.goal_pos = np.array([2, 2], dtype=np.int32)
        outputs.append((name, env))
    return outputs
