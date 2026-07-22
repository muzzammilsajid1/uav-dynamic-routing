# Discussion And Limitations Draft

## Discussion

The Week 3 dynamic evaluation shows that naive Dijkstra replanning remains the stronger method on the current controlled benchmark. Dijkstra succeeded on all 50 scenarios, while the RL policy succeeded on 49 out of 50. On the 49 paired successful scenarios, RL never produced a lower-cost path than Dijkstra. It matched Dijkstra in 16 cases and produced a higher-cost path in 33 cases.

This does not make the RL result useless. The RL policy still demonstrates substantial adaptability: it completes nearly all dynamic scenarios under the same obstacle schedule and scenario pairing as the classical baseline. However, the result does not support a claim that RL outperforms Dijkstra in this setup. The correct research claim is more careful: the trained RL agent adapts to dynamic obstacles with high success, but the classical replanning baseline remains superior in path optimality and reliability on the tested 15x15 grid.

The compute-time result is also important. A common motivation for RL is that a trained policy may avoid repeated graph search during deployment. In this experiment, however, the measured compute-time difference was not statistically significant at p < 0.05. This likely reflects the small grid size: a 15x15 graph is small enough that repeated Dijkstra replanning remains cheap. The expected computational advantage of RL may only become visible on larger grids, denser graphs, more frequent changes, or environments with more expensive transition models.

Overall, the study supports an honest trade-off framing. Dijkstra provides optimality and reliability when the current graph is known, but it must replan after changes. RL provides policy-based adaptation, but its learned decisions can be suboptimal and can fail. The current results favor Dijkstra for the tested scale while leaving open the possibility that RL becomes more attractive as environment size and complexity increase.

## Limitations

First, the grid is small. A 15x15 environment is useful for controlled development and reproducibility, but it is not large enough to expose the full computational burden of repeated graph search.

Second, Week 3 evaluation sets static obstacle density to 0.0. This was a deliberate fairness decision because the Dijkstra environment and RL environment originally generated static obstacles differently. Removing static obstacles ensures that both sides compare against the same three dynamic cells, but it also makes the environment simpler than a cluttered UAV routing domain.

Third, the dynamic setup uses only three fixed toggle cells. This is enough to test whether both systems react to environmental change, but it does not represent dense moving obstacles, stochastic obstacles, weather changes, wind fields, or multi-agent airspace constraints.

Fourth, the RL policy had one failed scenario. That failure should be reported directly because success rate is part of the comparison. Dijkstra's 50/50 success rate remains an advantage in the current benchmark.

Fifth, compute-time measurements at millisecond scale can be sensitive to implementation details, hardware, Python overhead, and measurement method. The compute-time result should be interpreted as a small-scale empirical measurement, not as a universal statement about Dijkstra and RL inference speed.

Finally, the current paper evaluates one classical baseline and one trained RL policy. Future work should compare additional classical methods such as A*, D* Lite, or Lifelong Planning A*, and additional RL variants or larger-scale deep RL models.

## Future Work

Future experiments should increase grid size, reintroduce matched static obstacles, vary dynamic obstacle frequency, and evaluate more complex UAV constraints such as energy budgets, no-fly zones, and partial observability. A larger benchmark would better test the central hypothesis that RL can reduce deployment-time computation when full replanning becomes expensive.
