# RL V3 Phase C1 Diagnostic Report: Endpoint Generalization Failure

## 1. Executive Summary

Phase C1 evaluated dynamic goal conditioning in an empty grid using the Sparse R1 reward. Training failed to generalize to unseen endpoints, achieving a 5.8% validation success rate. This diagnostic suite isolated the root cause by evaluating environmental solvability, representation expressivity, and credit assignment scaling. 

Results indicate that the failure is attributable to the sparse reward structure (R1). Under multi-seed evaluation, substituting R1 with a potential-based shaping reward (R2-PB) yielded mean validation success rates of 95.5%, demonstrating that the existing architecture and features are sufficient for endpoint generalization when provided dense credit assignment.

## 2. Root Cause Analysis & Evidence

We evaluated three hypotheses for the failure:

### Hypothesis 1: Environment Misconfiguration
*Are the validation routes mathematically solvable?*
- **Diagnostic:** Geometric Oracle Audit.
- **Evidence:** `PhaseC1Oracle`, implementing deterministic A* shortest-path logic with octile distance tie-breaking, was evaluated on all 50,400 possible empty-grid pairs. It recorded zero path-cost discrepancies compared to standard A*.
- **Conclusion:** The task is geometrically solvable; endpoint placement and grid logic do not prevent success.

### Hypothesis 2: Representation Expressivity Failure
*Is the model capable of distinguishing between optimal actions using the given observation features?*
- **Diagnostic:** Representation Expressivity Audit (Supervised Classifiers).
- **Evidence:** We generated a dataset of 5,000 transitions labeled by the Oracle. We trained two supervised classifiers using cross-entropy loss:
  - **E1 (Scalar Only):** Achieved 100% rollout success across the 120-pair validation manifest.
  - **E2 (CNN + Scalars):** Achieved 100% rollout success across the 120-pair validation manifest.
- **Conclusion:** Both representation pipelines contain sufficient information to deduce optimal routing. Representation expressivity is not the bottleneck.

### Hypothesis 3: Credit Assignment Failure (Root Cause)
*Does the sparse R1 terminal reward provide sufficient gradient signals as endpoint diversity scales?*
- **Diagnostic:** Endpoint-Cardinality Ladder & R2-PB Multi-Seed Confirmation.
- **Evidence 1 (Ladder):** We constrained training to subsets of 4, 16, 64, and 256 endpoints without modifying the reward. As diversity increased, training success declined (D1: 100%, D2: 87.5%, D3: 59.4%, D4: 38.3%). Terminal goal-reward frequencies dropped concurrently.
- **Evidence 2 (Multi-Seed R2-PB):** We implemented a dense, potential-based reward using normalized octile distance (R2-PB). We trained standard global-local policies across seeds 42, 43, and 44, comparing R1 against R2-PB:
  - **R1:** Mean validation success: 28.9% (max 40.0% at 50k interactions).
  - **R2-PB:** Mean validation success: 95.5% (min 91.7%, max 100.0% at 50k interactions).
- **Conclusion:** The sparse terminal reward (R1) does not support robust learning over diverse endpoint spaces. The agent fails to discover the goal reliably enough to maintain positive policy gradients during exploration.

## 3. Fast Diagnostic Execution

A fast, reproducible diagnostic test suite executes without requiring a GPU. It verifies the Oracle, tests E1 Expressivity on a miniature dataset, verifies R2-PB truncation mathematics, and evaluates the D1 ladder.

**Command:**
```bash
pytest -q --basetemp=tmp\pytest-phase-c1-diagnostic-final tests/test_phase_c1_diagnostics.py
```

## 4. Recommendations for Phase C2

1. **Adopt Dense Reward (R2-PB):** The sparse reward R1 should be replaced with the R2-PB potential-based reward for multi-endpoint training.
2. **Transition to Phase C2:** Since the endpoint generalization issue is diagnosed, Phase C2 (adding static obstacles) can proceed using R2-PB.
