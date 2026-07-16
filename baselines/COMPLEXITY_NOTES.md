# Dijkstra Complexity Notes

## What Dijkstra Does

Dijkstra finds the minimum-cost path from a start node to a goal node when all edge weights are non-negative.

It keeps:

- a distance table storing the best known cost to each node
- a parent table for reconstructing the final path
- a min-priority queue for expanding the cheapest currently known node
- a visited set for finalized nodes

## Why It Is Correct Here

The UAV grid uses only non-negative movement costs:

- straight move: `1.0`
- diagonal move: `sqrt(2)`

Because there are no negative weights, once Dijkstra removes a node from the priority queue with the smallest cost, that node's shortest distance is finalized.

## Time Complexity

With a binary heap priority queue:

```text
O((V + E) log V)
```

Where:

- `V` = number of free grid cells/nodes
- `E` = number of valid movement edges

In a grid, each cell has at most 8 neighbors, so `E` grows roughly proportional to `V`.

So for this project, it is reasonable to explain the grid case as:

```text
O(V log V)
```

## Space Complexity

```text
O(V)
```

The algorithm stores distances, parents, queue entries, and visited nodes.

## Why Dynamic Environments Hurt Dijkstra

In the static case, Dijkstra is optimal and reliable.

In the dynamic case, if obstacles or costs change during flight, the previous shortest path may become invalid or suboptimal. The baseline strategy is to recompute Dijkstra from the UAV's current position after every environment change.

That means the recomputation cost can be paid many times during one route.

This is the comparison point against RL:

- Dijkstra: optimal for known static graph, but recomputes under changes.
- RL agent: may be less optimal, but can choose actions from its trained policy without full graph recomputation.

