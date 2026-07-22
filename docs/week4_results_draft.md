# Week 4 Results Draft

## Dynamic Evaluation Setup

The Week 3 dynamic evaluation used 50 matched start-goal scenarios on the same 15x15 grid configuration for both methods. Static obstacle density was set to `0.0` on both sides to remove seeded obstacle-layout mismatch between the Dijkstra environment and the RL environment. The only changing cells were the three shared dynamic obstacles at `(4,4)`, `(8,8)`, and `(12,11)`.

The Dijkstra baseline used naive replanning: whenever the dynamic environment changed, the planner recomputed a full shortest path from the UAV's current position to the goal. The RL policy was evaluated on the same `scenario_id`, start, and goal pairs using the Phase 3 dynamic model.

## Success Rate

Dijkstra successfully completed all 50 dynamic scenarios. The RL policy completed 49 of the 50 scenarios, with one failure. This gives a success rate of 100% for Dijkstra and 98% for RL on the matched Week 3 scenario set.

This result should be reported directly rather than hidden: the RL policy is highly reliable on the tested dynamic setup, but it is not perfectly reliable in the same way as Dijkstra when a valid path exists.

## Path Cost

Path-cost statistics were computed on the 49 scenarios where both methods succeeded. Dijkstra was never more expensive than RL: RL matched Dijkstra in 16 scenarios and produced a higher path cost in 33 scenarios.

The mean path-cost difference, measured as `RL - Dijkstra`, was `+1.369441`, with a median difference of `+0.828427`. A Wilcoxon signed-rank test on the paired path-cost differences gave `W = 0` and `p = 5.644724e-07`.

This is statistically significant at conventional thresholds. The correct interpretation is that the Dijkstra replanning baseline produced significantly shorter or equal paths than the RL policy in this Week 3 dynamic evaluation.

## Compute Time

Compute-time statistics were also computed on the 49 paired successful scenarios. Mean cumulative Dijkstra planning time was `4.627753 ms`, while mean RL evaluation time was `5.334265 ms`. The mean difference, measured as `RL - Dijkstra`, was `+0.706512 ms`.

The Wilcoxon p-value for compute time was `0.062865`, which is not statistically significant at `p < 0.05`. On this small 15x15 grid, the RL policy did not show a statistically significant compute-time advantage over naive Dijkstra replanning.

## Interpretation For The Paper

These results support a trade-off framing rather than a claim that RL outperforms Dijkstra. In the tested dynamic grid, Dijkstra remains stronger on path optimality and reliability: it succeeds on every scenario and produces significantly shorter paths. The RL policy adapts to dynamic obstacles and succeeds in most cases, but it incurs extra path cost and has one failed scenario.

The expected RL advantage in this experiment would be computational scalability rather than raw path optimality. However, on the current 15x15 grid, measured compute time does not yet show a statistically significant RL advantage. This should be presented as an honest limitation of the current experiment and a motivation for future evaluation on larger grids or more frequent environmental changes, where repeated full-graph replanning may become more expensive.

## Figures To Include

- `docs/figures/week4_path_cost_comparison.png`
- `docs/figures/week4_path_cost_gap.png`
- `docs/figures/week4_compute_time_comparison.png`
- `docs/figures/week4_success_rate.png`
