# UAV Dynamic Routing

Reproducible comparison of classical and reinforcement-learning UAV routing in
dynamic grid environments.

## Paper

- [`paper_latex_v2/`](paper_latex_v2) - current expanded source. It compares
  Dijkstra, A*, D* Lite, and multi-seed DQN+HER across generalization, scaling,
  realism, and ablation benchmarks.
- [`paper_latex_v1/`](paper_latex_v1) - archived original single-baseline paper.

The current release is integrity-gated. The final status is summarized in
[`docs/RESEARCH_EXECUTION_STATUS.md`](docs/RESEARCH_EXECUTION_STATUS.md), and
the known loopholes, claim boundaries, and future fixes are tracked in
[`docs/LOOPHOLE_REGISTER.md`](docs/LOOPHOLE_REGISTER.md).

## Environment contract

- Coordinates: `(row, col)`.
- Movement: eight-connected.
- Straight cost: `1.0`.
- Diagonal cost: `sqrt(2)`.
- Diagonal corner cutting is allowed.
- Dynamics advance after a move and the changed state is observed before the
  next decision (`post_move_observed`) for every method.
- Classical methods use `GridEnvironment.get_neighbors()`.
- RL evaluation injects the exact persisted grid and dynamics for each
  scenario.

## Test

```bash
python -m pytest -q
```

On Windows, if a local `venv` was created with a Python installation that later
disappeared, `venv\Scripts\python.exe` may fail before tests start. Recreate
the environment from an installed Python, reinstall `requirements.txt`, and
rerun the command above.

## Reproduce the classical pilot

```bash
python scripts/reproduce_classical.py
```

## Run the expanded research suite

Evaluate every completed baseline and ablation checkpoint, regenerate raw and
aggregate evidence, rebuild every paper table and figure, and compile the
integrity-gated PDF:

```bash
python scripts/run_full_research.py
```

From a clean checkout, include checkpointed five-seed training for the complete
method and every controlled ablation:

```bash
python scripts/run_full_research.py --train
```

This is intentionally a long research run. Use `--variants full` for a
baseline-only replication. All variants are defined in
[`configs/research_experiments.json`](configs/research_experiments.json).
See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for timing,
statistics, adaptability definitions, and reproducibility boundaries.

## Legacy Colab script

`train_her_colab_v2.py` contains Colab notebook magic and is retained only as a
historical artifact. New research training uses
`experiments/train_multiseed.py`.
