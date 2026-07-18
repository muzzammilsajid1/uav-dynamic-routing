"""
train_static.py — Train a DQN agent on the static 15×15 UAV routing grid.

This script:
  1. Instantiates the custom UAVRoutingEnv.
  2. Wraps it with a Gymnasium Monitor for episode logging.
  3. Configures a DQN agent (MlpPolicy) via Stable-Baselines3.
  4. Trains for a user-specified number of timesteps.
  5. Saves the trained policy network to disk.
  6. Runs a short evaluation loop and reports per-episode cumulative reward.

Usage
-----
    python -m rl_agent.train_static          # from the project root
    python rl_agent/train_static.py          # or directly

Notes
-----
* Hyperparameters below are deliberately conservative (small buffer, moderate
  exploration) because the 15×15 static grid is a low-dimensional problem.
  For dynamic or larger grids, increase buffer_size, learning_starts, and
  total_timesteps accordingly.
* The trained model is saved to  models/dqn_static_uav.zip  (SB3 appends the
  .zip extension automatically).

Author: Research Team
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `rl_agent.uav_env` resolves
# regardless of how this script is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN                      # noqa: E402
from stable_baselines3.her import HerReplayBuffer       # noqa: E402
from stable_baselines3.common.monitor import Monitor    # noqa: E402
from stable_baselines3.common.callbacks import (        # noqa: E402
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from rl_agent.uav_env import UAVRoutingEnv              # noqa: E402


# ============================================================================
#  Configuration
# ============================================================================

# ---- Environment -----------------------------------------------------------
GRID_SIZE: int = 15               # N×N operating area
OBSTACLE_DENSITY: float = 0.20    # fraction of cells with physical obstacles
NO_FLY_DENSITY: float = 0.05     # fraction of cells in regulatory no-fly zones

# ---- DQN hyperparameters --------------------------------------------------
# These are tuned for a small, static grid.  Key considerations:
#
#   • learning_rate:  1e-3 is aggressive but converges fast on a tiny state
#     space.  Drop to 3e-4 or use a schedule for larger / dynamic grids.
#
#   • buffer_size:  50 000 transitions is ample for a 15×15 grid where max
#     episode length is 225 steps.
#
#   • exploration_fraction:  We let ε decay from 1.0 → exploration_final_eps
#     over the first 40 % of training, then exploit.  On a static grid the
#     agent can learn the optimal path relatively quickly.
#
#   • target_update_interval:  Sync the target Q-network every 1 000 gradient
#     steps to stabilise learning (standard practice).
#
#   • batch_size:  64 is a reasonable default; 128 may smooth gradients for
#     noisier dynamic environments.
#
#   • train_freq:  Update the network every 4 environment steps to balance
#     sample efficiency and wall-clock time.

HYPERPARAMS = dict(
    policy="MultiInputPolicy",
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=5_000,         # collect 1 k random transitions before learning
    batch_size=256,                # larger batch size for stable gradient updates
    tau=1.0,                       # hard target-network update
    gamma=0.99,                    # discount factor (long-horizon planning)
    train_freq=2,                  # gradient steps per N env steps
    gradient_steps=1,              # 1 gradient step per train_freq call
    target_update_interval=1_000,  # sync target network every 1 k updates
    exploration_fraction=0.50,     # fraction of total_timesteps for ε decay (shorter exploration)
    exploration_initial_eps=1.0,   # start fully random
    exploration_final_eps=0.05,    # floor ε — keep a sliver of exploration
    policy_kwargs=dict(net_arch=[128, 128]), # custom network architecture [128, 128]
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    verbose=1,
)

TOTAL_TIMESTEPS: int = 150_000  # total training budget (environment steps)
EVAL_EPISODES: int = 20           # evaluation episodes after training

# ---- Paths -----------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "dqn_static_uav"
LOG_DIR = PROJECT_ROOT / "logs" / "dqn_static"


# ============================================================================
#  Helper: build & wrap the environment
# ============================================================================

def make_env(seed: int | None = None) -> Monitor:
    """
    Create a UAVRoutingEnv instance and wrap it in a SB3 Monitor for
    automatic episode-level logging (reward, length, wall-time).
    """
    env = UAVRoutingEnv(
        grid_size=GRID_SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        no_fly_density=NO_FLY_DENSITY,
        fixed_grid=True,
        seed=seed,
    )
    # Monitor writes per-episode CSVs into LOG_DIR — useful for plotting
    # learning curves later.
    os.makedirs(LOG_DIR, exist_ok=True)
    return Monitor(env, filename=str(LOG_DIR / "monitor"))


# ============================================================================
#  Training
# ============================================================================

def train() -> DQN:
    """
    Instantiate the environment and DQN agent, run training, and save the
    learned policy to disk.

    Returns
    -------
    model : DQN
        The trained Stable-Baselines3 DQN model.
    """
    print("=" * 60)
    print("  UAV Static-Grid DQN Training")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}  |  Timesteps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60)

    # ---- Environment ---------------------------------------------------------
    train_env = make_env(seed=42)
    eval_env = make_env(seed=42)       # separate env for periodic evaluation

    # ---- DQN model -----------------------------------------------------------
    model = DQN(
        env=train_env,
        **HYPERPARAMS,
        seed=42,
        device="auto",                  # use GPU if available, else CPU
    )

    # ---- Optional: early-stopping callback ----------------------------------
    # Stop if evaluation reward hasn't improved in 10 consecutive evaluations.
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR),
        eval_freq=50_000,              # evaluate every 50 k steps
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    # ---- Train ---------------------------------------------------------------
    print("\nStarting training ...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_cb,
        log_interval=10,               # print stats every 10 episodes
        progress_bar=True,             # tqdm progress bar
    )

    # ---- Save ----------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = MODEL_DIR / MODEL_NAME
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}.zip")

    train_env.close()
    eval_env.close()

    return model


# ============================================================================
#  Evaluation
# ============================================================================

def evaluate(model: DQN, n_episodes: int = EVAL_EPISODES) -> None:
    """
    Run *n_episodes* deterministic roll-outs of the trained policy and
    report per-episode cumulative reward and episode length.

    Parameters
    ----------
    model : DQN
        A trained Stable-Baselines3 DQN model.
    n_episodes : int
        Number of evaluation episodes to run.
    """
    env = make_env(seed=42)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    print("\n" + "=" * 60)
    print("  Evaluation")
    print("=" * 60)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Deterministic action selection (no ε-greedy noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        outcome = "GOAL" if total_reward > 0 else ("CRASH" if terminated else "TIMEOUT")
        print(
            f"  Episode {ep:>3d}/{n_episodes}  |  "
            f"Reward: {total_reward:>8.1f}  |  "
            f"Steps: {steps:>4d}  |  "
            f"{outcome}"
        )

    env.close()

    # ---- Summary statistics --------------------------------------------------
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    print("\n" + "-" * 60)
    print(f"  Mean reward:   {rewards.mean():>8.2f}  ±  {rewards.std():.2f}")
    print(f"  Mean length:   {lengths.mean():>8.2f}  ±  {lengths.std():.2f}")
    print(f"  Success rate:  {(rewards > 0).sum()}/{n_episodes}")
    print("-" * 60)


# ============================================================================
#  Entry point
# ============================================================================

if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)
