# RL V3 Phase B Protocol

Phase B is a development-only Maskable PPO pilot. It preserves Version 2 and
Phase A evidence, uses the authoritative Version 2 transition environment, and
does not access or create a private final-test manifest.

The fixed development suite contains 96 scenarios: 24 at each of 15, 30, 50,
and 100 cells; 32 in each short/medium/long initial A* route bin; and 24 in each
empty, random-static, structured, and dynamic family. Every scale-family-route
cell contains exactly two scenarios. Dynamic scenarios use
`post_move_observed` timing.

## Policy inputs

All policies receive an 11x11 eight-channel local crop and four scalars:
normalized relative goal row, normalized relative goal column, normalized grid
size, and previous action. `global_local` adds a fixed 32x32 eight-channel
global map. `global_local_recency` activates the visitation channel in both
maps. No planner output, A* distance, oracle direction, future dynamic state,
or optimal path is a policy input.

Global bins use channel-specific maximum aggregation for blocked cells, hard
no-fly cells, penalties, recently changed cells, and visitation. Separate
blocked and free-presence channels distinguish fully free, fully blocked, and
mixed bins, preserving one-cell barriers and narrow corridors. Agent and goal
channels remain separate even when both occupy the same coarse bin.

The feature extractor uses separate moderate CNN branches for local and global
maps, a two-layer scalar MLP, concatenation and a 256-unit fused layer, followed
by separate `[128, 64]` policy and value heads. The local-only policy has
159,929 parameters; global policies have 470,025.

## Rewards

Let `B` be the fixed episode budget, `G=1`, `C=-1`, and `s=-0.20/B`.

R1 returns `G` at the goal, `C` on collision, and `s` otherwise.

R2 adds `0.10 * (0.99 Phi(x') - Phi(x))` to non-terminal R1 transitions, where
`Phi(x) = -d_octile(x, goal) / ((N-1)*sqrt(2))`. The potential depends only on
agent and goal coordinates. Its coefficient is small relative to the terminal
objective and allows temporary movement away from the goal.

## Curriculum and selection

The four logged stages progressively add 50x50, 100x100, structured barriers,
and dynamics while retaining earlier sizes. Each pilot uses seed 314159 and is
evaluated at 25k, 50k, and 100k interactions. Selection uses fixed-suite
success, failure taxonomy, scale/family performance, path cost, and computation,
not training return. No 250k stage may start until the 100k comparison is
reported.

Checkpoint restore is classified as statistically equivalent. Model,
optimizer, generator counters, and legal action behavior restore, but a
partially active environment transition is not serialized, so bit-identical or
transition-identical continuation is not claimed.
