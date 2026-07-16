# Handoff For Muzzammil

Please adapt the RL environment to this exact contract so the comparison is fair.

## Use These Environment Rules

- Coordinates are `(row, col)`.
- Grid is `15x15` for Week 1.
- Default start is `(0, 0)`.
- Default goal is `(14, 14)`.
- Obstacles are hard-blocked cells.
- Movement is 8-directional.
- Straight move cost is `1.0`.
- Diagonal move cost is `sqrt(2)`.
- Dynamic obstacle updates are disabled in Week 1.

## Most Important Integration Rule

Do not reimplement movement rules separately in the RL wrapper. Use:

```python
neighbors = env.get_neighbors(current_node)
```

This returns:

```python
[(neighbor_node, move_cost), ...]
```

The RL action logic can map actions to candidate moves, but final validity should match the environment contract.

## Suggested RL Action Mapping

Use this action order if you need a stable mapping:

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

If an action leads outside the grid or into a blocked cell, treat it as invalid/collision according to the RL reward design.

## Reproducibility

Use the same seed when comparing Dijkstra and RL:

```python
env = GridEnvironment(size=15, obstacle_density=0.2, seed=42, diagonal=True)
```

Do not compare methods on different random obstacle maps.

