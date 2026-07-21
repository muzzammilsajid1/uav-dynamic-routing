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

## Week 3: Dynamic Obstacles

Agreed with Muzzammil before either side builds against this:

- **No-fly zones are dropped from scope entirely.** Only hard-blocked
  obstacles exist — no soft/penalty cells.
- **Obstacles are pre-positioned, not randomly generated.** A dynamic
  obstacle is a fixed `(row, col)` cell with a toggle `period` and an
  `initial_state` (`"blocked"` or `"passable"`). Both sides must read
  the exact same list — `envs.grid_environment.default_dynamic_obstacles()`
  is the single source of truth for the default 15x15/seed=42 env, so
  obstacle placement can't silently diverge between the Dijkstra and RL
  sides again.
- **No advance visibility.** Neither the Dijkstra replanner nor the RL
  agent gets any signal about when a cell will next toggle. Both only
  ever observe the *current* grid state (`is_blocked()` / the RL local
  view), updated the instant a toggle happens, with no lookahead. This
  keeps Dijkstra's replanning genuinely "naive" (plan Section 4.1) and
  keeps the adaptability metric (plan Section 6) meaningful — it only
  measures something if the change is a surprise to both sides.

### API

```python
from envs.grid_environment import GridEnvironment, DynamicObstacle, default_dynamic_obstacles

env = GridEnvironment(dynamic_obstacles=default_dynamic_obstacles())

changed = env.step_dynamics()   # advance one timestep, returns cells that toggled THIS step
env.reset_dynamics()            # restore initial states, reset the toggle clock to 0
env.elapsed_steps               # current timestep counter
```

`step_dynamics()` never looks ahead — it only reports what changed on
the step just taken. Build any replanning/reactive logic off its return
value, not off inspecting `dynamic_obstacles` directly (that would be
equivalent to peeking at the schedule in advance).

