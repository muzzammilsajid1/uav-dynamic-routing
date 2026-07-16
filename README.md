# UAV Dynamic Routing

Dijkstra-baseline vs reinforcement-learning UAV routing in a dynamic grid environment.

## Week 1 Environment Contract

- Coordinates: `(row, col)` everywhere.
- Grid size: start with `15x15`.
- Movement: 8-direction movement.
- Move cost: straight = `1.0`, diagonal = `sqrt(2)`.
- Obstacles: hard-blocked cells, not traversable.
- Start and goal: never blocked.
- Dynamic obstacles: not active in Week 1.
- Shared rule: Dijkstra and RL must both use `GridEnvironment.get_neighbors()`.

## Run Tests

```bash
python -m unittest discover -s tests
```

## Run Static Baseline

```bash
python experiments/run_static_baseline.py
```

