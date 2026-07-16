# Environment Spec

This spec is the contract between Simra's Dijkstra baseline and Muzzammil's RL agent.

## Grid

- The airspace is a square `N x N` grid.
- Each free cell is a traversable state/node.
- Each blocked cell is an obstacle/no-fly hard block.
- Week 1 size: `15x15`.

## Coordinates

Use `(row, col)` everywhere.

- Top-left cell is `(0, 0)`.
- Bottom-right cell is `(size - 1, size - 1)`.
- `row` increases downward.
- `col` increases rightward.

## Start And Goal

- Default start: `(0, 0)`.
- Default goal: `(size - 1, size - 1)`.
- Start and goal must never be blocked.

## Movement

Use 8-direction movement:

```text
N, S, E, W, NE, NW, SE, SW
```

Movement is valid only when the destination cell:

- is inside the grid
- is not blocked

For Week 1, diagonal corner cutting is allowed if the destination cell is free. This keeps the first implementation simple and reproducible.

## Costs

- Orthogonal movement cost: `1.0`
- Diagonal movement cost: `sqrt(2)`

All edge weights are non-negative, so Dijkstra is valid.

## Shared API

Both Dijkstra and RL should rely on this method for movement:

```python
GridEnvironment.get_neighbors(node)
```

It returns:

```python
[(neighbor_node, move_cost), ...]
```

Do not duplicate movement rules separately in RL code.

## Week 1 Scope

Included:

- static obstacles
- seeded grid generation
- Dijkstra baseline
- basic tests

Not included yet:

- dynamic obstacles
- no-fly high-penalty zones
- reward function tuning
- RL training

