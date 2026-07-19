# Related Work

Literature review for the Dijkstra-baseline vs RL-agent UAV routing project.
This maps to Section 3 ("Related Work") of the paper outline and closes out
Simra's Week 2 deliverable (complexity analysis + literature review).

---

## 1. Classical and Graph-Based Path Planning

Classical planners remain the reference point for UAV path planning because
they come with provable guarantees on optimality and completeness, and many
learning-based planners still lean on them as components or comparison
points<cite index="4-1">, since well-understood classical methods provide guarantees on optimality, completeness, and computational behavior that many modern planners still rely on</cite>.
Grid- and graph-search methods (Dijkstra, A*), sampling-based methods
(RRT, PRM), and potential-field methods are the three families most
commonly used in this category. Sampling-based approaches like
rapidly-exploring random trees can search large spaces cheaply but are
randomized, so they are not guaranteed to return the best path and can be
slow to converge in cluttered environments<cite index="8-1">, since they rely on randomness and can fail to find a path at all in very complex environments</cite>. This is exactly why the project uses Dijkstra rather than RRT as the classical baseline: it's deterministic, optimal on a static graph, and its complexity is easy to characterize and defend in a paper.

The recurring weakness identified across this literature is computational
cost under scale and under change: classical planners generally assume a
known, mostly static environment and must be recomputed when that
environment changes<cite index="9-1">, which becomes a central challenge in dynamic environments with limited or evolving environmental knowledge</cite>. This is the exact framing our project's core research question is built on — Dijkstra is optimal but expensive to rerun after every change, and that cost is precisely what we're quantifying.

## 2. Reinforcement Learning for UAV Path Planning

A growing body of work replaces the explicit search step with a learned
policy that maps observed state directly to a movement action, trained
through trial-and-error interaction rather than an explicit graph
search<cite index="4-1">, where RL forms a complementary paradigm to classical planning by learning navigation policies directly through environment interaction</cite>. Q-learning and its deep variants (DQN, Double DQN) are the most common entry point for grid-world UAV navigation, and several papers report that vanilla tabular or single-layer Q-learning struggles once the state space grows, motivating either deep function approximation or algorithmic tweaks to the reward/update rule<cite index="2-1">, where an improved Q-learning variant with revised reward and Q-value update rules outperformed a standard SARSA baseline on the same UAV path-planning task</cite>. This matches our own DQN-side experience directly — the tabular agent was a deliberate simplification used to sanity-check the environment before trusting the deep network, and it's reassuring to see other UAV-RL work independently converge on similar Q-learning stabilization concerns.

A systematic review comparing classical and RL-based coverage/path
planning methods for UAV networks concludes that RL-based methods trade
some of the strict guarantees of classical planning for greater autonomy
and adaptability to conditions the designer didn't fully anticipate in
advance<cite index="3-1">, with RL methods found to allow more autonomy than the classical alternatives surveyed</cite>. Some deep-RL papers report outperforming classical baselines (A*, RRT) and even single-layer DQN on dynamic scenes by combining global and local information in a layered policy<cite index="6-1">, where a two-layer RL planner combining global and local information reportedly outperformed A*, RRT, and single-layer DQN in dynamic path-planning effectiveness</cite>, while other applied work (e.g. search-and-rescue UAV base stations locating survivors by signal strength) reports that both traditional RL and traditional path planning tend to degrade once the environment gets more complex or the state space grows, which is part of the motivation for moving to deep RL formulations<cite index="7-1">, since traditional RL and traditional path-planning algorithms tend to perform well only in simple scenarios with small state dimensions, degrading as environments become more complex</cite>. Taken together, this literature supports our project's framing: RL is not being pitched as strictly better than Dijkstra, but as a different point on the optimality/adaptability trade-off curve — which is exactly the comparison our evaluation metrics (Section 6 of the plan) are designed to measure.

## 3. Where This Project Fits

Most of the RL-vs-classical UAV literature above compares methods across
different metrics, environments, and hardware, which makes head-to-head
comparison across papers difficult. Few of them run both approaches on
the *exact same* graph, obstacle layout, and start/goal pairs, and report
a paired statistical test on the difference — most report policy
performance and success rate but not a rigorous like-for-like comparison
against a shared classical baseline with matched inputs. That is the
concrete gap this project fills: a controlled, same-environment,
same-scenario comparison of Dijkstra vs. a Q-learning/DQN agent, evaluated
on path length, per-decision compute time, cumulative compute time, and a
custom adaptability metric, with a paired statistical test (t-test or
Wilcoxon) across 30–50 shared test episodes rather than an anecdotal
before/after comparison. This is a narrower, more tightly controlled
claim than most of the surveyed papers attempt, which is exactly what
makes it a defensible, publishable contribution at this project's scale.

---

## Sources

1. Mannan et al. (2023), *Classical versus reinforcement learning
   algorithms for unmanned aerial vehicle network communication and
   coverage path planning: A systematic literature review*, International
   Journal of Communication Systems.
2. *UAV Path Planning and Obstacle Avoidance Based on Reinforcement
   Learning in 3D Environments* (2023), Drones (MDPI).
3. *A Survey of Risk-Calibrated Certifiably Safe and Resource-Aware (RCSR)
   Path Planning for Unmanned Aerial Vehicles* (2026), Drones (MDPI).
4. *UAV path planning techniques: a survey* (2024), RAIRO Operations
   Research.
5. *UAV Path Planning Based on Deep Reinforcement Learning* (2024),
   ResearchGate.
6. *Path Planning Research of a UAV Base Station Searching for Disaster
   Victims' Location Information Based on Deep Reinforcement Learning*
   (2022), PMC.
7. *Dynamic Q-planning for Online UAV Path Planning in Unknown and
   Complex Environments* (2024), arXiv:2402.06297.
8. *A comprehensive review of path planning algorithms for autonomous
   navigation* (2025), ScienceDirect.

*(Full citation details/DOIs to be formatted in BibTeX once the paper
moves to Overleaf — see plan Section 11.)*
