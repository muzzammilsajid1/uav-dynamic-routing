# Phase C1 Representation Expressivity Audit

## Overview
To determine if the state observation vector provides sufficient information to solve the empty-grid dynamic routing task, two supervised classifiers were trained to mimic the optimal geometric oracle on a randomly generated dataset of empty-grid states.

- **E1:** A small MLP using only the 4-dimensional scalar feature vector.
- **E2:** The exact `PhaseBFeatureExtractor` CNN pipeline used by PPO (which processes global and local spatial maps alongside scalars).

The datasets contained completely disjoint training (3,434 states) and validation (634 states) endpoint sets.

## Results

| Metric | E1 (Scalar-Only) | E2 (Global-Local CNN) |
|--------|------------------|-----------------------|
| **Train Accuracy** | 99.82% | 99.50% |
| **Validation Accuracy** | 99.84% | 99.36% |
| **Illegal Prediction Rate (Val)** | 0.0% | 0.0% |
| **Rollout Success Rate (120 Val Pairs)** | 100.0% | 100.0% |

### Stratified Validation Accuracy (E1)
- **Short:** 98.38%
- **Medium:** 100.0%
- **Long:** 100.0%
- **Vertical:** 99.43%
- **Horizontal:** 100.0%
- **Diagonal:** 100.0%
- **Mixed:** 100.0%

### Stratified Validation Accuracy (E2)
- **Short:** 98.38%
- **Medium:** 99.55%
- **Long:** 99.42%
- **Vertical:** 99.43%
- **Horizontal:** 98.29%
- **Diagonal:** 100.0%
- **Mixed:** 100.0%

## Conclusion
Both the E1 scalar-only representation and the E2 global-local feature pipeline contain more than enough information to deduce the optimal step. Furthermore, the CNN feature extractor does *not* obscure the scalar information; the supervised action head easily learns to extract the correct behavior from the fused representation.

According to the decision logic: **Because both classifiers succeed in supervised rollout but PPO failed to learn during RL training, the primary problem is heavily localized to reward credit assignment, state exploration, or PPO optimization mechanics, not representation expressivity.**
