"""
Phase 3 curriculum training: fine-tune dqn_her_phase2_100k.zip on the FULL
default_dynamic_obstacles() config (all 3 cells, period=5) for 100k steps.
"""
from __future__ import annotations

import sys
import os
import math
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from envs.grid_environment import default_dynamic_obstacles
import rl_agent.safe_her_buffer as safe_her_buffer_module
from rl_agent.uav_env import UAVRoutingEnv

sys.modules["safe_her_buffer"] = safe_her_buffer_module

CHECKPOINT_IN  = str(PROJECT_ROOT / "models" / "dqn_her_phase2_100k.zip")
CHECKPOINT_OUT = str(PROJECT_ROOT / "models" / "dqn_her_phase3_100k")
TOTAL_TIMESTEPS = 100_000
SEED = 42

PHASE3_OBSTACLES = default_dynamic_obstacles()

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with torch.no_grad():
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def main() -> None:
    assert os.path.exists(CHECKPOINT_IN), f"Checkpoint not found: {CHECKPOINT_IN}"

    train_env = Monitor(UAVRoutingEnv(
        grid_size=15, obstacle_density=0.0, no_fly_density=0.0,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=PHASE3_OBSTACLES,
        fixed_grid=True, seed=SEED,
    ))

    model = DoubleDQN.load(
        CHECKPOINT_IN, env=train_env,
        custom_objects={
            "exploration_fraction": 0.20,
            "exploration_initial_eps": 0.30,
            "exploration_final_eps": 0.05,
        },
    )

    loaded_steps = model.num_timesteps
    model.learning_starts = loaded_steps + 5_000

    print(f"Loaded from: {CHECKPOINT_IN}")
    print(f"Checkpoint timesteps: {loaded_steps:,}")
    print(f"Obstacles: {[(o.cell, o.period, o.initial_state) for o in PHASE3_OBSTACLES]}")

    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, progress_bar=False)
    elapsed = time.perf_counter() - t0

    model.save(CHECKPOINT_OUT)
    print(f"Training complete in {elapsed/60:.1f} min")
    print(f"Saved: {CHECKPOINT_OUT}.zip")
    train_env.close()


if __name__ == "__main__":
    main()
