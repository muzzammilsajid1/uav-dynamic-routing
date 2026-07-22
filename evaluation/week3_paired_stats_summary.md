# Week 3 Paired Dynamic Evaluation Summary

Inputs:

- Dijkstra CSV: `evaluation/week3_dynamic_baseline_results.csv`
- RL CSV: `evaluation/week3_rl_results.csv`
- Scenario setup: 50 shared `scenario_id` rows, `obstacle_density=0.0`, dynamic cells `(4,4)`, `(8,8)`, `(12,11)`.

Validation:

- Shared scenario IDs: 50
- Start/goal mismatches: 0
- Dijkstra success rate: 50/50
- RL success rate: 49/50
- Paired successful rows used for path-cost statistics: 49

Path cost, measured as `RL - Dijkstra` on paired successful scenarios:

- Mean difference: +1.369441
- Median difference: +0.828427
- RL lower/equal/higher than Dijkstra: 0 / 16 / 33
- Wilcoxon signed-rank statistic: W = 0
- Wilcoxon p-value: 5.644724e-07

Interpretation: Dijkstra produced significantly shorter or equal paths than the RL policy in the Week 3 dynamic evaluation. This supports the paper's trade-off framing: the RL agent adapts and succeeds on most scenarios, but it does not outperform Dijkstra on path optimality.

Compute time, measured as `RL - Dijkstra` in milliseconds on paired successful scenarios:

- Mean Dijkstra time: 4.627753 ms
- Mean RL time: 5.334265 ms
- Mean difference: +0.706512 ms
- Wilcoxon p-value: 0.062865

Interpretation: The compute-time difference is not statistically significant at p < 0.05 on this 15x15 setup.
