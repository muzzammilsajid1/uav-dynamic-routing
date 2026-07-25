# UAV Dynamic Routing

Dijkstra-baseline vs reinforcement-learning UAV routing in a dynamic grid environment.

## Paper

The research paper lives in two versions:

- [`paper_latex_v2/`](paper_latex_v2) — **current version.** Compares naive Dijkstra, A\*, and DQN+HER on static and dynamic benchmarks, with a full hyperparameter table and re-measured compute times.
- [`paper_latex_v1/`](paper_latex_v1) — archived original version (Dijkstra-only classical baseline). Kept for reference; see `paper_latex_v2/README.md` for what changed and why.

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
python -m pytest -q
```

## Run Static Baseline

```bash
python experiments/run_static_baseline.py
```

## Run Static Q-Learning

```bash
python experiments/train_q_learning_static.py
```

## Colab Training

`train_her_colab_v2.py` is intended to be run in Google Colab. It contains
Colab notebook magic for installing dependencies and is therefore not a
standalone Python script.
