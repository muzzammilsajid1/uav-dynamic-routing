# RL V3 Phase C0 Report: Single-Scenario Overfitting Test

## Objective
The goal of Phase C0 was to determine whether the RL V3 MaskablePPO pipeline could deliberately master one extremely simple fixed navigation task before generalizing. This isolates basic reinforcement learning capability—observations, masking, rewards, and environment resets—from the complexities of generalization, dynamic obstacles, and varying grid sizes.

If PPO could not master this task, it would point to a fundamental defect in the environment interface, reward structure, or observation encoding.

## Experimental Setup

### Scenario Design
We created `PhaseC0Env`, a deterministic, fixed-scenario environment wrapper around the V2 `UAVRoutingEnv`:
- **Grid Size**: 15x15
- **Obstacles/No-Fly**: None (0% density)
- **Start Position**: `(2, 2)`
- **Goal Position**: `(12, 7)`
- **Optimal A* Route**: Requires 10 steps (a mix of straight South and diagonal South-East moves)
- **Max Steps**: 60
- **Randomization**: Disabled. Every reset returns the exact same start, goal, and empty grid.

### Architecture
The training configuration matched the Phase B P2 Pilot precisely, using the `global_local` architecture:
- **Observation Family**: `global_local` (11x11 local view + 32x32 global map + scalars)
- **Reward Family**: `R1` (sparse +1.0 for goal, -1.0 for crash, -0.2 total step penalty across budget)
- **Feature Extractor**: `PhaseBFeatureExtractor` with 256 dimensions
- **Policy Network**: `[128, 64]` for actor and critic, ReLU activation
- **Hyperparameters**: `n_steps=256`, `batch_size=64`, `n_epochs=10`, `learning_rate=3e-4`
- **Curriculum**: Disabled. Training was conducted exclusively on the 15x15 fixed scenario.

### Evaluation Criteria
Training checked for early stopping against a threshold of **>=99/100 success** on two consecutive checkpoints. Checkpoints were evaluated at 1,000, 5,000, 10,000, 25,000, 50,000, and 100,000 interactions. A preflight verification suite mathematically and behaviorally validated the scenario and observation state before allowing training to start.

## Results

**Preflight Verification:** PASS.
The preflight script confirmed:
- A* cost of ~12.071 for the selected start/goal.
- Consistent, deterministic resets.
- Correct local and global observation channel formatting.
- Correct action masking allowing all in-bounds moves.

**Training Outcomes:**
Training successfully reached the mastery condition and early-stopped after evaluating the 25,000-interaction checkpoint.

| Interactions | Success Rate | Crashes | Timeouts | Route Steps | Path Cost Gap | Mean Return |
|--------------|--------------|---------|----------|-------------|---------------|-------------|
| 1,000        | 0.0%         | 0       | 100      | N/A         | N/A           | -0.200      |
| 5,000        | 0.0%         | 0       | 100      | N/A         | N/A           | -0.200      |
| 10,000       | 100.0%       | 0       | 0        | 10.0        | 0.000         | 0.970       |
| 25,000       | 100.0%       | 0       | 0        | 10.0        | 0.000         | 0.970       |

- **Earliest Mastery:** Mastery was reached by the 10,000 interaction checkpoint, where the agent demonstrated 100/100 success.
- **Mastery Confirmed:** Mastery was confirmed by the second consecutive 100/100 checkpoint at 25,000 interactions, which triggered early stopping.
- **Optimality:** The agent solved the task in exactly 10 route steps. The path cost gap evaluates to exactly 0 (realized path cost approximately 12.071 vs A* path cost 12.071), yielding a mean return of ~0.97 factoring in the step penalty.
- **Failures at Initialization:** At 1,000 and 5,000 interactions, the agent timed out on 100% of evaluation episodes by entering localized 2-cell oscillations or wandering until the 60-step limit.

## Conclusion
The RL V3 Maskable PPO pipeline can learn and execute one fixed deterministic navigation task. This rules out a completely non-functional training integration, but does not establish generalization, curriculum adequacy, global-map usefulness, or large-scale routing competence.
