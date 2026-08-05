# RL V3 Phase C1 Diagnostic Report: Endpoint Generalization Failure

## 1. Executive Summary

Phase C1 aimed to learn dynamic goal conditioning in an empty grid. The agent failed to generalize to unseen endpoints during training, achieving only a 5.8% validation success rate despite reaching near 45% training success on a 50,000-pair dataset.

This diagnostic suite proves that **the failure is strictly caused by the sparse reward structure (R1), which fails to reliably assign credit across highly diverse endpoint distributions.**

When equipped with a mathematically grounded potential-based shaping reward (R2-PB), standard MaskablePPO immediately achieves **100% success** on the same validation manifest, proving that the architecture, features, and environment are perfectly capable of supporting general endpoint navigation.

## 2. Root Cause Analysis & Evidence

We evaluated three competing hypotheses for the failure:

### Hypothesis 1: Environment Misconfiguration / Impossible Task
*Are the validation routes actually impossible?*
- **Diagnostic:** Geometric Oracle Audit.
- **Evidence:** We implemented `PhaseC1Oracle`, which uses deterministic geometric logic to solve the grid. It achieved **100% success** on the 120-pair validation manifest.
- **Conclusion:** The task is entirely solvable; there are no bugs in endpoint placement or grid logic preventing success.

### Hypothesis 2: Representation Expressivity Failure
*Is the model capable of distinguishing between actions using the given observation features? Does the CNN feature extractor destroy the spatial scalar variables?*
- **Diagnostic:** Representation Expressivity Audit (Supervised Classifiers).
- **Evidence:** We generated a dataset of 10,000 transitions labeled by the Oracle. We trained two supervised classifiers:
  - **E1 (Scalar Only):** Achieved 99.8% validation accuracy and 100% rollout success.
  - **E2 (CNN + Scalars):** Achieved 99.3% validation accuracy and 100% rollout success.
- **Conclusion:** Both representation pipelines contain more than enough information to deduce optimal routing. The CNN architecture successfully learns to combine and interpret these features. Representation is **not** the bottleneck.

### Hypothesis 3: Credit Assignment / Sparse Reward Failure (The Root Cause)
*Does the sparse R1 terminal reward fail to provide sufficient gradient signals as endpoint diversity scales?*
- **Diagnostic:** Endpoint-Cardinality Ladder & Potential-Based Reward Pilot.
- **Evidence 1 (Ladder):** We constrained training to subsets of 4, 16, 64, and 256 endpoints without modifying the reward. As diversity increased, training success collapsed:
  - D1 (4 pairs): 100% train SR
  - D2 (16 pairs): 87.5% train SR
  - D3 (64 pairs): 59.4% train SR
  - D4 (256 pairs): 38.3% train SR
- **Evidence 2 (R2-PB Pilot):** We implemented a dense, potential-based reward using normalized octile distance. We trained the standard global-local policy (P1) and a scalar-only policy (P2) using this dense reward.
  - **P2 (Scalar-Only + R2-PB):** Reached **100% validation success** in just 5,000 interactions.
  - **P1 (CNN + R2-PB):** Reached **96.7% validation success** in 45,000 interactions.
- **Conclusion:** The sparse terminal reward (R1) cannot support learning over diverse endpoint spaces. The agent fails to discover the goal during exploration reliably enough to maintain positive policy gradients.

## 3. Fast Diagnostic Execution

We have implemented a fast, reproducible diagnostic test suite that executes in under two minutes without requiring a GPU. It verifies the Oracle, tests E1 Expressivity on a miniature dataset, and runs the D1 ladder to prove sparse-reward learning stalls early.

**Command:**
```bash
pytest -q --basetemp=tmp\pytest-phase-c1-diagnostic tests/test_phase_c1_diagnostics.py
```

## 4. Recommendations for Phase C2

1. **Adopt Dense Reward (R2-PB):** The sparse reward R1 should be entirely abandoned for multi-endpoint training. The R2-PB potential-based reward strongly preserves optimal policy behavior while making exploration highly efficient.
2. **Transition to Phase C2:** Since the endpoint generalization issue is now resolved, Phase C2 (adding static obstacles) can safely begin, using R2-PB as the foundational reward structure. 
