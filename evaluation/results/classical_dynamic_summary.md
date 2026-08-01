# Repeated classical dynamic benchmark

| Planner | Runs | Success | Path cost mean | Route time mean (ms) | Median (ms) | SD (ms) | 95% CI half-width (ms) | Expansions mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| astar | 500 | 1.000 | 10.5637 | 0.5982 | 0.4505 | 0.4729 | 0.0415 | 32.6 |
| dijkstra | 500 | 1.000 | 10.5637 | 5.1742 | 4.4935 | 3.4049 | 0.2985 | 251.1 |

The confidence interval summarizes raw route-run timings. Rows are retained
in `classical_dynamic_raw.csv`; environment details are in
`classical_dynamic_environment.json`.
