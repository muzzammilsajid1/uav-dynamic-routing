# Paper Outline

Working title: Dynamic UAV Routing: A Controlled Comparison of Naive Dijkstra Replanning and Reinforcement Learning

## Abstract

Summarize the problem, methods, dynamic-grid setup, paired evaluation, and main finding. The abstract should avoid claiming that RL outperforms Dijkstra. The honest result is that Dijkstra remained superior in path optimality and reliability on the tested 15x15 dynamic grid, while RL achieved high but imperfect adaptability.

## 1. Introduction

Motivate UAV routing in dynamic environments where obstacles can change during flight. Classical shortest-path methods are reliable and optimal on known graphs, but may require repeated replanning when the environment changes. Reinforcement learning is introduced as an alternative policy-based approach that can react through learned action selection rather than explicit graph search.

Research question:

Can a dynamically trained RL policy match or outperform naive Dijkstra replanning on path cost, success rate, and compute time in a controlled UAV grid-routing environment with changing obstacles?

Contributions:

- A shared 15x15 UAV grid environment with identical movement, cost, and dynamic-obstacle rules for both methods.
- A naive Dijkstra replanning baseline that recomputes after each dynamic obstacle change.
- A dynamically trained RL evaluation on the same 50 scenario IDs and start-goal pairs.
- A paired statistical comparison using Wilcoxon signed-rank testing.

## 2. Related Work

Use the existing related-work draft as the source. Organize it around classical graph and sampling-based UAV path planners, reinforcement learning and deep reinforcement learning for UAV path planning, and the gap addressed by this project: controlled same-scenario comparison with paired statistics.

## 3. Methodology

Include the grid environment, dynamic obstacles, Dijkstra replanning baseline, RL formulation and training phases, evaluation protocol, and metrics.

## 4. Experimental Setup

Describe the 15x15 grid, 8-connected movement, movement costs of 1.0 and sqrt(2), static obstacle density of 0.0 for Week 3 paired evaluation, dynamic cells at (4,4), (8,8), and (12,11), default toggle period 5, and 50 matched scenarios.

## 5. Results

Use the Week 4 figures and the paired statistical summary. Required figures are path cost comparison, path-cost gap, compute-time comparison, and success-rate comparison.

## 6. Discussion

Emphasize that this is not an RL-wins paper. The stronger claim is a measured trade-off: RL adapts in most scenarios but does not beat Dijkstra on path optimality or compute time at this scale.

## 7. Limitations

Include the small grid size, only three dynamic obstacles, static obstacle density set to zero for fair comparison, one RL failure, compute timing sensitivity on small problems, and need for larger-scale evaluation before claiming RL compute advantages.

## 8. Conclusion

Conclude that Dijkstra replanning remains the stronger method under the current controlled setup, while RL is a promising adaptive approach whose advantages may appear in larger, more complex, or partially observed environments.

## References

Convert the source list in the related-work draft into BibTeX entries in the later LaTeX phase.
