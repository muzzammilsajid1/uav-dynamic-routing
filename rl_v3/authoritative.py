"""Document the V2 functions that remain authoritative for RL V3."""
from __future__ import annotations

AUTHORITATIVE_V2_FUNCTIONS = {
    "rl_agent.uav_env.UAVRoutingEnv.step": "transition, reward, termination, dynamics timing",
    "rl_agent.uav_env.UAVRoutingEnv.reset": "episode initialization contract",
    "rl_agent.uav_env.UAVRoutingEnv.get_neighbors": "one-step legal movement and edge costs",
    "rl_agent.uav_env.UAVRoutingEnv._toggle_dynamic_obstacles": "RL dynamic-obstacle updates",
    "rl_agent.uav_env.UAVRoutingEnv.ACTION_DELTAS": "action-to-move mapping",
    "envs.grid_environment.GridEnvironment.get_neighbors": "classical graph-neighbor contract",
    "envs.grid_environment.GridEnvironment.step_dynamics": "classical dynamic-obstacle updates",
}

