# RL V3 Phase A Protocol

This branch starts RL V3 without modifying the completed V2 evidence. Phase A
adds scenario separation, V2/V3 transition parity checks, one-step action
masking, richer observation construction, failure diagnostics, and a checkpoint
contract smoke test. It does not launch substantive PPO training.

## Authoritative V2 Semantics

V3 must reuse these V2 functions and constants as the source of truth:

- `rl_agent.uav_env.UAVRoutingEnv.step`
- `rl_agent.uav_env.UAVRoutingEnv.reset`
- `rl_agent.uav_env.UAVRoutingEnv.get_neighbors`
- `rl_agent.uav_env.UAVRoutingEnv._toggle_dynamic_obstacles`
- `rl_agent.uav_env.UAVRoutingEnv.ACTION_DELTAS`
- `rl_agent.uav_env.CELL_FREE`
- `rl_agent.uav_env.CELL_OBSTACLE`
- `rl_agent.uav_env.CELL_NO_FLY`
- `envs.grid_environment.GridEnvironment.get_neighbors`
- `envs.grid_environment.GridEnvironment.step_dynamics`

The V3 wrapper may add observations, masks, diagnostics, and run bookkeeping,
but it must not duplicate movement, collision, traversal-cost, dynamic-obstacle,
or termination semantics.

## Final-Test Isolation

No readable final-test scenario manifest is stored in the development
repository. The final-test protocol is frozen here and the private final-test
seed or external manifest will only be supplied after the V3 model family,
hyperparameters, seed list, and analysis protocol are frozen.

Final-test generation algorithm:

1. Use the same `rl-v3-generator-v1` algorithm as training and validation.
2. Use a private seed in the reserved range `910000..919999`, or a separately
   supplied external manifest with the same schema.
3. Generate exactly 120 scenarios:
   - grid sizes: 15, 30, 50, 100;
   - route buckets: short, medium, long;
   - 10 scenarios for each grid-size/bucket combination.
4. Obstacle density is sampled uniformly from `[0.05, 0.30]`.
5. Dynamic obstacles are sampled with count `0..4` and periods from
   `{3, 4, 5, 7, 9, 11}`.
6. Short, medium, and long route buckets are based on initial A* cost divided
   by grid size:
   - short: `[0.20, 0.40)`;
   - medium: `[0.40, 0.70)`;
   - long: `[0.70, 1.10]`.
7. Every scenario must be solvable under the initial state by fresh A*.
8. Scenario IDs, parameters, and manifest SHA-256 are recorded before final
   evaluation begins.

Primary metrics:

- success rate;
- path-cost gap against fresh A*;
- collision and invalid-action crash rate;
- timeout rate;
- two-cell oscillation rate;
- longer-loop rate;
- excessive-detour rate;
- decisions per attempted route;
- per-decision inference latency;
- attempted-episode compute time;
- successful-route compute time;
- post-change completion;
- event regret;
- deterministic primary failure label.

Statistical protocol:

- freeze policy seeds before final-test exposure;
- aggregate repeated timing samples within seed/scenario before paired tests;
- report seed-aware means and confidence intervals;
- compare original and shielded diagnostics separately from trained masked RL;
- never use final-test scenarios for model selection, hyperparameter tuning, or
  early stopping.

## Phase A Outputs

Phase A writes under `runs/rl_v3/phase_a/` and `evaluation/manifests/rl_v3_*`.
It does not overwrite V2 raw results, figures, paper artifacts, or benchmark
manifests.
