# Methodology Draft

## Environment Model

The routing environment is represented as a square 15x15 grid. Each grid cell corresponds to a possible UAV location, and movement is allowed in eight directions: north, south, east, west, and the four diagonals. Orthogonal moves have cost 1.0 and diagonal moves have cost sqrt(2). These non-negative edge weights make Dijkstra's algorithm valid for shortest-path computation.

Coordinates are represented consistently as `(row, col)`, with `(0,0)` at the top-left of the grid. The default goal for the full-grid setup is `(14,14)`, although Week 3 evaluation uses 50 matched start-goal pairs rather than a single fixed pair. Start and goal cells are never allowed to be blocked.

Both methods use the same movement contract: a move is valid only if the destination cell is inside the grid and not blocked. This rule is exposed through `GridEnvironment.get_neighbors(node)`, which returns valid neighboring cells and their movement costs. Keeping this interface shared prevents the Dijkstra and RL implementations from silently using different transition rules.

## Dynamic Obstacle Model

The dynamic environment uses fixed-position obstacles that toggle between blocked and passable states. For the Week 3 paired evaluation, static obstacle density is set to 0.0 on both sides. This removes seeded static-layout mismatch between the classical environment and the RL environment, ensuring that the only changing cells are the three shared dynamic obstacles: `(4,4)`, `(8,8)`, and `(12,11)`.

The full Week 3 dynamic configuration uses the shared `default_dynamic_obstacles()` source of truth. Dynamic changes occur on a fixed timer, and neither method receives advance visibility into future toggles. The planner and agent only react to the current grid state after a change occurs.

## Dijkstra Replanning Baseline

The classical baseline applies Dijkstra's shortest-path algorithm to the current grid graph. In the static case, Dijkstra is optimal because all edge weights are non-negative. In the dynamic case, the previous shortest path may become invalid or suboptimal after an obstacle toggles.

The Week 3 baseline therefore uses naive replanning: whenever the dynamic environment changes, Dijkstra is recomputed from the UAV's current position to the current goal. This strategy is intentionally simple. It is correct and easy to interpret, but it may pay the full graph-search cost multiple times during a single route.

For a grid graph with `V` free cells and `E` valid movement edges, Dijkstra with a binary heap has time complexity O((V + E) log V). Since each cell has at most eight neighbors, E grows approximately linearly with V, so the grid case is commonly summarized as O(V log V). Space complexity is O(V).

## Reinforcement Learning Agent

The RL side is evaluated as a policy-based navigation method. The UAV chooses among the same eight movement actions used by the Dijkstra environment. The dynamic RL model is trained through curriculum phases: first on a static environment, then on a milder dynamic setup, and finally on the full three-obstacle dynamic configuration.

The RL formulation uses local grid observations around the UAV together with positional information about the goal. The reward function encourages reaching the goal, penalizes invalid moves and collisions, and includes step or movement penalties so shorter routes are preferred. Unlike Dijkstra, the RL policy does not explicitly search the graph at evaluation time; it selects actions from the trained policy.

## Evaluation Protocol

Both methods are evaluated on the same 50 scenario IDs. Each scenario fixes the start and goal cells. Pairing is essential: comparing unrelated random episodes would mix method performance with scenario difficulty. The paired setup allows row-by-row comparison of path cost and compute time.

The evaluation validates that each shared `scenario_id` has the same start and goal for both methods. In the final Week 3 result, all 50 scenario IDs matched and there were zero start-goal mismatches.

## Metrics

The evaluation reports success rate, dynamic path cost, path-cost gap (`RL - Dijkstra`), compute time, and statistical significance using a Wilcoxon signed-rank test on paired differences.

Path-cost statistics are computed on the 49 scenarios where both Dijkstra and RL succeeded. The one RL failure is included in success-rate reporting but excluded from paired path-cost testing because no valid RL path cost exists for that scenario.
