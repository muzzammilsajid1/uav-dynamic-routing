# RL Formulation

This is the clean static RL formulation used for the first replacement agent.

## Environment

The RL agent uses the same shared grid as the Dijkstra baseline:

```python
GridEnvironment(size=15, obstacle_density=0.2, seed=42, diagonal=True)
```

The RL code must not define separate obstacle or movement rules. Valid movement is determined by:

```python
GridEnvironment.get_neighbors(node)
```

## State Space

For the tabular static agent, the state is the UAV's current cell:

```text
(row, col)
```

The goal is fixed at `(14, 14)` for the Week 1 static setup, so it does not need to be included separately in the tabular state.

## Action Space

There are 8 discrete actions:

```text
0: N   -> (-1,  0)
1: S   -> ( 1,  0)
2: W   -> ( 0, -1)
3: E   -> ( 0,  1)
4: NW  -> (-1, -1)
5: NE  -> (-1,  1)
6: SW  -> ( 1, -1)
7: SE  -> ( 1,  1)
```

This exactly matches the handoff contract.

## Reward Function

The static tabular agent uses:

- `+100` for reaching the goal
- `-10` for invalid moves into obstacles or out of bounds
- a small step penalty
- movement cost penalty
- a small progress reward for reducing Euclidean distance to the goal

This reward is intentionally simple and inspectable. The goal is to get a reliable static RL baseline before adding dynamic obstacles.

## Why Tabular Q-Learning First

The DQN/HER path was stuck around 10% success and showed Q-value overestimation/divergence. Tabular Q-learning avoids neural-network value divergence and gives a stable sanity-check agent on the 15x15 static grid.

This is not the final dynamic RL method yet. It is the cleanest way to prove the environment, actions, rewards, and evaluation loop are correct before moving back to DQN or dynamic obstacles.

