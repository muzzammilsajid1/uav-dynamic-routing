# Autonomous UAV Routing: Dijkstra Baseline vs Reinforcement Learning Agent

A full technical breakdown for a one-month remote research project.

---

## 1. The Core Research Question

**Can a reinforcement learning agent outperform classical shortest-path algorithms (Dijkstra) for UAV routing in environments where conditions change during flight?**

This framing matters because it gives you a clean, defensible contribution: Dijkstra is provably optimal for *static* graphs, but has to be fully recomputed whenever the graph changes. An RL agent, once trained, can react to changes without recomputation. The paper's job is to quantify that trade-off: how much does RL cost you in optimality, and how much does it save you in adaptability and computation time?

---

## 2. System Architecture

```
┌─────────────────────────────────────────┐
│           Environment Layer              │
│  Graph/grid representation of airspace   │
│  Static obstacles, dynamic obstacles,    │
│  no-fly zones, cost weights              │
└─────────────────┬─────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼────────┐
│  Dijkstra       │   │  RL Agent       │
│  Baseline       │   │  (DQN or        │
│                 │   │  Q-Learning)    │
└───────┬────────┘   └────────┬────────┘
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Evaluation Layer    │
        │  Path length,        │
        │  compute time,       │
        │  success rate,       │
        │  adaptability score  │
        └──────────────────────┘
```

---

## 3. Environment Design (build this first, everything depends on it)

### 3.1 Representation choice
Use a **grid-based graph**, not a free-form graph, for your first version. It's far easier to reason about, visualize, and debug, and it's still perfectly valid for a UAV routing paper (grid-based airspace discretization is standard in the literature).

- Represent the airspace as an `N x N` grid (start with something small like 15x15 or 20x20, scale up later if time allows)
- Each cell is a node; edges connect adjacent cells (4-directional or 8-directional movement, 8-directional is more realistic for UAVs)
- Each edge has a cost (distance, or later, energy cost)
- Obstacles = cells marked as impassable
- No-fly zones = cells with a very high traversal penalty rather than a hard block (more realistic, gives the RL agent something more interesting to learn)

### 3.2 Static vs dynamic version
Build **two versions of the same environment**:

1. **Static**: obstacles fixed at the start, never change. Use this to validate both methods work correctly before adding complexity.
2. **Dynamic**: obstacles can appear, disappear, or move on a timer (e.g. every 5 timesteps, some cells flip between passable/blocked). This is where the paper's actual contribution lives.

### 3.3 Implementation
```python
# Rough skeleton, not final code
class GridEnvironment:
    def __init__(self, size, obstacle_density, dynamic=False):
        self.size = size
        self.grid = self._generate_grid(obstacle_density)
        self.dynamic = dynamic
        self.start = ...
        self.goal = ...

    def step_dynamics(self):
        # only called if self.dynamic
        # randomly toggles some cells between passable/blocked
        ...

    def get_neighbors(self, node):
        # returns valid neighboring nodes and edge costs
        ...

    def is_blocked(self, node):
        ...
```

Use **NetworkX** to wrap this once it's working, it gives you free graph utilities (shortest path validation, visualization, centrality metrics if you want extra analysis later).

---

## 4. Dijkstra Baseline

This is the easier half, get it done fast so more time goes to the RL side.

### 4.1 What to implement
- Standard Dijkstra's algorithm on the grid graph
- For the dynamic environment: **naive replanning strategy**, recompute the full shortest path from the UAV's current position every time the environment changes. This is intentionally "dumb" because that's the point of comparison, it's correct but expensive.

### 4.2 What to measure
- Path length (optimal, by definition, in the static case)
- Computation time per replan
- Number of replans needed over a full dynamic run
- Total cumulative computation time over a full route in the dynamic environment

### 4.3 Effort-saving tip
Python's `heapq` module gives you a working Dijkstra in under 40 lines. Don't hand-roll a priority queue. NetworkX also has `nx.dijkstra_path()` built in, if the paper doesn't require you to show algorithmic novelty here (it doesn't, Dijkstra is the *known* baseline), just use the library function and cite it. Save your implementation effort for the RL side, which is where your actual contribution is.

---

## 5. Reinforcement Learning Agent

This is the heart of the paper and where most of your month's effort should go.

### 5.1 Choice of algorithm
**Use Deep Q-Network (DQN)**, not tabular Q-learning, if your grid is larger than ~10x10 (state space gets too big for a table). For a 15x15 or 20x20 grid, DQN is the right call. Use **Stable-Baselines3** (free, well-documented, handles most of the boilerplate) rather than writing DQN from scratch, unless you specifically want the "we implemented it ourselves" angle for the paper (this adds credibility but also adds a week of debugging, weigh this against your timeline).

**Recommendation: use Stable-Baselines3.** Minimizing effort matters more than building from scratch here, a working, well-tuned DQN using a library is a stronger paper than a half-working custom implementation.

### 5.2 Defining the RL problem

**State space**: 
- UAV's current position (x, y)
- Goal position (x, y)
- Local view of surrounding cells (e.g. a 5x5 or 7x7 window around the UAV showing obstacles), this gives the agent enough local information to react to nearby changes without needing the full grid state (which would explode the state space)

**Action space**:
- 8 discrete actions (move N, S, E, W, NE, NW, SE, SW), or 4 if you simplify to orthogonal movement only

**Reward function** (this is the part reviewers will scrutinize most, get it right):
- Small negative reward per step (e.g. -1), encourages shorter paths
- Large positive reward on reaching the goal (e.g. +100)
- Large negative reward for hitting an obstacle or going out of bounds (e.g. -50), episode ends
- Optional: small negative reward proportional to proximity to no-fly zones, encourages the agent to keep a safety margin rather than skirting obstacles exactly

```python
def reward(self, action_result):
    if action_result == "goal_reached":
        return 100
    elif action_result == "collision":
        return -50
    elif action_result == "no_fly_zone":
        return -20
    else:
        return -1  # step penalty
```

### 5.3 Training procedure
- **Phase 1**: train on the static environment first. This is your sanity check, if the agent can't learn to solve a static maze, it definitely won't handle dynamic obstacles.
- **Phase 2**: once static performance is solid (agent reaches goal >90% of the time, path length close to Dijkstra's optimal), move to dynamic training. Retrain (don't just fine-tune) on the dynamic version, since the state distribution changes meaningfully.
- **Curriculum tip**: start dynamic training with infrequent, small changes, then gradually increase change frequency/magnitude. This tends to converge faster than throwing the agent into full chaos immediately.

### 5.4 Compute requirements
- Static environment training: runs fine on a laptop CPU, maybe 20 to 40 minutes for a 15x15 grid
- Dynamic environment training: will take longer, possibly a few hours. Use **Google Colab free tier** (GPU access at no cost) if your laptop is slow. Save checkpoints regularly in case the session times out.

---

## 6. Evaluation Metrics (this is what fills your Results section)

Run both methods on the **same set of test scenarios** (same start/goal pairs, same obstacle patterns) so the comparison is fair. Generate at least 30 to 50 test episodes for statistical validity.

| Metric | Dijkstra | RL Agent | Why it matters |
|---|---|---|---|
| Path length (static) | Optimal by definition | Compare % above optimal | Shows RL's cost in raw efficiency |
| Path length (dynamic) | Recomputed each change | Adaptive | Shows RL's advantage in changing conditions |
| Computation time per decision | Fast per-call, but recomputes fully | Near-instant (forward pass only) | Key selling point for RL |
| Total compute time over full dynamic route | Sum of all replans | Sum of all forward passes | Direct efficiency comparison |
| Success rate | ~100% (always finds a path if one exists) | Track this, may fail sometimes | Honesty about RL's limitations |
| Adaptability (define this yourself) | e.g. how much extra path length is incurred right after an environment change | Compare | This is your paper's novel metric |

Run a simple statistical test (paired t-test or Wilcoxon signed-rank) comparing path lengths/times across the test set, this makes your comparison rigorous rather than anecdotal.

---

## 7. Task Split (Simra vs You)

| Simra (DSA/algorithms) | You (RL/UAV domain) |
|---|---|
| Graph/grid environment construction | RL state/action/reward design |
| Dijkstra implementation + replanning logic | DQN training pipeline (Stable-Baselines3 setup) |
| Complexity analysis (Big-O comparison of Dijkstra recompute vs DQN inference) | UAV-specific framing (energy cost interpretation, no-fly zone realism, literature on UAV path planning) |
| Statistical testing of results | Curriculum design for dynamic training |
| Shared: evaluation script, plotting, write-up | Shared: evaluation script, plotting, write-up |

Work in a shared GitHub repo from day one. Agree on the environment interface (the `GridEnvironment` class methods) together first, in one call, so you're not building incompatible pieces independently.

---

## 8. Week-by-Week Timeline

### Week 1: Foundations
- Day 1 to 2: Agree on environment spec together (grid size, obstacle rules, reward structure). Set up shared GitHub repo.
- Day 3 to 5: Simra builds the grid environment + Dijkstra baseline (static version). You start reading up on Stable-Baselines3 and drafting the RL environment wrapper (needs to follow the Gym/Gymnasium interface for compatibility).
- Day 6 to 7: Merge and test, confirm Dijkstra runs correctly on the shared environment class.

### Week 2: RL agent, static environment
- Train DQN on the static environment. Iterate on reward function and hyperparameters until the agent reliably reaches the goal with near-optimal paths.
- Simra starts the complexity analysis write-up and begins literature review in parallel.

### Week 3: Dynamic environment + full comparison
- Add dynamic obstacle logic to the environment.
- Retrain RL agent on dynamic version (this is the most time/compute-intensive part, start early in the week).
- Implement Dijkstra's naive replanning for the dynamic case.
- Run both methods on the full test scenario set, collect all metrics.

### Week 4: Analysis and write-up
- Statistical tests, generate all plots (path length comparison, compute time comparison, success rate, adaptability metric)
- Draft paper sections in parallel: Simra writes Methodology (Dijkstra side) and Related Work, you write Methodology (RL side) and Discussion
- Combine, edit, format for target venue

---

## 9. How to Minimize Effort Without Cutting Corners

- **Use libraries wherever the library isn't the point of the paper.** NetworkX for graph handling, Stable-Baselines3 for DQN, matplotlib/seaborn for plots. Your contribution is the *comparison and framing*, not reimplementing well-known algorithms.
- **Start small, scale only if time allows.** A working 15x15 grid study beats a broken 50x50 one. If Week 3 goes smoothly, consider scaling up grid size or adding multi-UAV as a bonus section, don't attempt it from the start.
- **Automate the evaluation loop early.** Write one script that runs both methods on the same test set and dumps results to a CSV. Once this exists, every future change (bigger grid, retrained model) is just a re-run, not a rebuild.
- **Version control everything, including trained model checkpoints.** Losing a half-trained DQN to a crashed Colab session is the single biggest time-waster in RL projects. Save checkpoints every N episodes.
- **Don't chase perfect RL performance.** A paper showing "RL trades 8% path efficiency for 60% less total computation time in dynamic environments" is a complete, publishable finding. You don't need the RL agent to beat Dijkstra outright, the trade-off itself is the contribution.

---

## 10. Rough Paper Structure (for when you write it up)

1. **Abstract**
2. **Introduction** (motivate why dynamic UAV routing matters, cite existing UAV path-planning literature)
3. **Related Work** (classical algorithms vs learning-based approaches, briefly)
4. **Methodology**
   - Environment design
   - Dijkstra baseline + replanning strategy
   - RL formulation (state, action, reward) + training procedure
5. **Experimental Setup** (grid size, test scenarios, hardware used)
6. **Results** (tables/plots from Section 6 above)
7. **Discussion** (trade-offs, limitations, when RL is/isn't worth it)
8. **Conclusion + Future Work** (mention multi-UAV extension as future work if you don't get to build it)

---

## 11. Tools Checklist (all free)

- Python 3.10+
- NetworkX (graph handling)
- Stable-Baselines3 + Gymnasium (RL)
- PyTorch (installed automatically as SB3 dependency)
- Matplotlib / Seaborn (plots)
- Google Colab free tier (training compute, if needed)
- GitHub (shared repo, version control)
- Overleaf (free LaTeX writing, good for later when drafting the actual paper)
