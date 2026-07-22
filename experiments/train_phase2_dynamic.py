"""
Phase 2 curriculum training: fine-tune dqn_her_300k_final.zip on a mild
dynamic obstacle config (one obstacle, period=10) for 100k additional steps.

Rules:
  - Loads from checkpoint, NOT from scratch
  - obstacle_density=0.0  (same as eval — no static obstacle divergence)
  - Only the environment's dynamic obstacle config differs from the static run
  - Reward shaping, network arch, and hyperparameters are UNCHANGED
  - Saves to models/dqn_her_phase2_100k.zip
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

from envs.grid_environment import DynamicObstacle
import rl_agent.safe_her_buffer as safe_her_buffer_module
from rl_agent.uav_env import UAVRoutingEnv

# ---- Phase 2 config ----------------------------------------------------------
CHECKPOINT_IN  = str(PROJECT_ROOT / "models" / "dqn_her_300k_final.zip")
CHECKPOINT_OUT = str(PROJECT_ROOT / "models" / "dqn_her_phase2_100k")
TOTAL_TIMESTEPS = 100_000
N_EVAL_EPISODES = 20
SEED = 42

# Mild Phase 2 obstacle: just the (8,8) "blocked" cell, slower toggle (period=10)
PHASE2_OBSTACLES = [
    DynamicObstacle(cell=(8, 8), period=10, initial_state="blocked"),
]

# ---- Double DQN (identical to training run) ----------------------------------
class DoubleDQN(DQN):
    """Double DQN — matches the implementation in train_her_colab_v2.py exactly."""
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


def make_env(seed=None):
    env = UAVRoutingEnv(
        grid_size=15,
        obstacle_density=0.0,
        no_fly_density=0.0,
        dynamic_obstacles_enabled=True,
        dynamic_obstacles=PHASE2_OBSTACLES,
        fixed_grid=True,
        seed=seed,
    )
    return Monitor(env)


def main() -> None:
    # ---- Load checkpoint (NOT from scratch) ----------------------------------
    assert os.path.exists(CHECKPOINT_IN), f"Checkpoint not found: {CHECKPOINT_IN}"

    train_env = make_env(seed=SEED)

    # Patch safe_her_buffer into sys.modules so SB3 can unpickle it from the zip
    sys.modules["safe_her_buffer"] = safe_her_buffer_module

    model = DoubleDQN.load(
        CHECKPOINT_IN,
        env=train_env,
        # Override exploration for fine-tuning: shorter warm-up, lower final eps
        custom_objects={
            "exploration_fraction": 0.20,
            "exploration_initial_eps": 0.30,
            "exploration_final_eps": 0.05,
        },
    )

    loaded_steps = model.num_timesteps
    print(f"Loaded from checkpoint: {CHECKPOINT_IN}")
    print(f"Checkpoint timesteps: {loaded_steps:,}")
    print(f"Fine-tuning for {TOTAL_TIMESTEPS:,} additional steps")
    print(f"Phase 2 obstacles: {[(obs.cell, obs.period, obs.initial_state) for obs in PHASE2_OBSTACLES]}")
    print()

    # With reset_num_timesteps=False, num_timesteps starts at loaded_steps (300k).
    # SB3 checks `num_timesteps > learning_starts` to decide whether to sample.
    # Since 300k > any small value, it would immediately sample from the empty
    # post-load buffer — causing the HER RuntimeError.
    # Fix: set learning_starts = loaded_steps + 5_000 so the check doesn't pass
    # until 5k fresh transitions have been collected into the new buffer.
    model.learning_starts = loaded_steps + 5_000
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, progress_bar=False)
    elapsed = time.perf_counter() - t0

    # ---- Save ----------------------------------------------------------------
    model.save(CHECKPOINT_OUT)
    saved_file = CHECKPOINT_OUT + ".zip"
    assert os.path.exists(saved_file), f"Save failed — {saved_file} not found"
    print(f"Training complete in {elapsed/60:.1f} min")
    print(f"Saved: {saved_file}")
    print()

    # ---- Evaluate ------------------------------------------------------------
    train_env.close()
    eval_env = make_env(seed=SEED + 1)

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
        total_steps += ep_steps
        total_reward += ep_reward
        if info.get("is_success"):
            successes += 1

    eval_env.close()

    print(f"Checkpoint loaded from:  {CHECKPOINT_IN}")
    print(f"Checkpoint saved to:     {saved_file}")
    print(f"Success rate:            {successes}/{N_EVAL_EPISODES} ({100*successes/N_EVAL_EPISODES:.0f}%)")
    print(f"Avg steps/episode:       {total_steps / N_EVAL_EPISODES:.1f}")
    print(f"Avg reward/episode:      {total_reward / N_EVAL_EPISODES:.3f}")


if __name__ == "__main__":
    main()
