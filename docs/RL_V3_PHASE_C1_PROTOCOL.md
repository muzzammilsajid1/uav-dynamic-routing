# RL V3 Phase C1 Protocol: Endpoint Generalization on Empty 15x15 Grid

## Objective
Determine whether the RL V3 global-local Maskable PPO policy can generalize across unseen start-goal pairs on one empty 15x15 grid. This isolates basic endpoint generalization from obstacles, dynamic environments, and scale.

## Setup
- **Grid**: 15x15, completely empty.
- **Architecture**: MaskablePPO with `global_local` observations (11x11 local, 32x32 global, 4 scalars).
- **Reward**: R1 (+1.0 goal, -1.0 crash, -0.2 step penalty).
- **Episode Budget**: Dynamic, `max(10, int(astar_cost * 2.0))` to allow exploration while preventing endless loops.

## Endpoint Generation
Endpoints are categorized into 3 distance bins based on A* path cost:
- **Short**: cost < 6.2 (32.3% of pairs)
- **Medium**: 6.2 <= cost < 10.1 (35.3% of pairs)
- **Long**: cost >= 10.1 (32.4% of pairs)

A fixed validation manifest of 120 pairs (40 per bin) is generated deterministically. These pairs, and their inverses, are strictly excluded from the training distribution. Training pairs are sampled uniformly from the remaining population across bins.

## Early Stopping Criteria
Stop early if 2 consecutive checkpoints satisfy:
- >= 95% overall validation success
- >= 90% validation success in every bin
- 0 collisions
- < 5% combined oscillation/longer loop failure rate
