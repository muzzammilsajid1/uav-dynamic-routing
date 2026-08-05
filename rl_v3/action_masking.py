"""One-step legality masks for shielded RL diagnostics.

The mask deliberately calls only the V2 environment's one-step neighbor
contract. It does not call A*, Dijkstra, D* Lite, route-distance logic, or any
planner. This preserves the legacy occupied-cell escape semantics: if the UAV
currently occupies a blocked cell, legal exits are still those returned by the
V2 neighbor function for the current position.
"""
from __future__ import annotations

import numpy as np

from rl_agent.uav_env import UAVRoutingEnv


def legal_action_mask(env: UAVRoutingEnv) -> np.ndarray:
    """Return a bool mask over the V2 action space for immediately legal moves."""
    current = env.uav_pos
    if current is None:
        raise RuntimeError("environment must be reset before masking actions")
    legal_destinations = {
        tuple(int(value) for value in neighbor)
        for neighbor, _ in env.get_neighbors(current)
    }
    mask = np.zeros(env.action_space.n, dtype=bool)
    for action, delta in enumerate(env.ACTION_DELTAS):
        destination = current + delta
        mask[action] = tuple(int(value) for value in destination) in legal_destinations
    return mask


def highest_q_legal_action(q_values: np.ndarray, mask: np.ndarray) -> int:
    """Choose the highest-Q legal action for post-hoc shielded diagnostics."""
    if q_values.shape[0] != mask.shape[0]:
        raise ValueError("q_values and mask must have the same action dimension")
    if not mask.any():
        return int(np.argmax(q_values))
    masked = np.full_like(q_values, -np.inf, dtype=float)
    masked[mask] = q_values[mask]
    return int(np.argmax(masked))
