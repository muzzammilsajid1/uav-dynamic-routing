
"""
uav_env.py -- Gymnasium wrapper for UAV grid-world routing.

Models a single fixed-wing or multi-rotor UAV navigating a 2-D grid from a
start waypoint to a goal waypoint while avoiding obstacles (physical airspace
constraints) and no-fly zones (regulatory / geofenced regions).

Observation:  Dict{ "observation" (61-d), "achieved_goal" (2-d), "desired_goal" (2-d) }
Action:       Discrete(8) -- 8-connected heading commands (N, S, W, E, NW, NE, SW, SE)
Reward:       Potential-based shaping (Ng et al. 1999):
              sparse_reward + gamma * Phi(next_state, goal) - Phi(state, goal)
              where Phi(s, g) = -Dijkstra_distance(s, g)

Author: Research Team
"""

from __future__ import annotations

import math
from collections import deque
import heapq
import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE: int = 15
LOCAL_VIEW_RADIUS: int = 3
LOCAL_VIEW_SIZE: int = 2 * LOCAL_VIEW_RADIUS + 1   # 7

CELL_FREE: int = 0
CELL_OBSTACLE: int = 1
CELL_NO_FLY: int = 2

# Legacy constants kept for reference (no longer used in step() reward)
REWARD_STEP: float = -0.1
REWARD_GOAL: float = 1000.0
REWARD_COLLISION: float = -50.0
REWARD_NO_FLY: float = -20.0

MAX_STEPS: int = GRID_SIZE * GRID_SIZE  # 225


class UAVRoutingEnv(gym.Env):
    """
    2-D grid-world environment for RL-based UAV autopilot.
    Conforms to Gymnasium API and GoalEnv contract for HER compatibility.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    #   Index   Heading     (dr, dc)
    #     0      N           (-1,  0)
    #     1      S           ( 1,  0)
    #     2      W           ( 0, -1)
    #     3      E           ( 0,  1)
    #     4      NW          (-1, -1)
    #     5      NE          (-1,  1)
    #     6      SW          ( 1, -1)
    #     7      SE          ( 1,  1)
    ACTION_DELTAS = np.array(
        [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)],
        dtype=np.int32,
    )

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        obstacle_density: float = 0.20,
        no_fly_density: float = 0.05,
        render_mode: str | None = None,
        seed: int | None = None,
        curriculum_enabled: bool = False,
        curriculum_start_dist: int = 3,
        curriculum_step_episodes: int = 5000,
        curriculum_step_dist: int = 1,
        fixed_grid: bool = True,
    ) -> None:
        super().__init__()

        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.no_fly_density = no_fly_density
        self.render_mode = render_mode
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_start_dist = curriculum_start_dist
        self.curriculum_step_episodes = curriculum_step_episodes
        self.curriculum_step_dist = curriculum_step_dist
        self.episode_count = 0
        self.fixed_grid = fixed_grid
        self._initial_grid = None

        # ---- Action space ----------------------------------------------------
        self.action_space = spaces.Discrete(8)
        self.n_actions = self.action_space.n
        self.last_action = None
        self.visited_cells = set()

        # ---- Observation space (Dict for HER) --------------------------------
        obs_dim = 4 + LOCAL_VIEW_SIZE * LOCAL_VIEW_SIZE + 8   # 4 + 49 + 8 = 61
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-1.0,
                high=max(CELL_NO_FLY, 1.0),
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                low=0.0, high=float(grid_size - 1), shape=(2,), dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=0.0, high=float(grid_size - 1), shape=(2,), dtype=np.float32
            ),
        })

        # ---- Internal state --------------------------------------------------
        self.grid: np.ndarray | None = None
        self.uav_pos: np.ndarray | None = None
        self.goal_pos: np.ndarray | None = None
        self.current_step: int = 0

        # ---- All-pairs distance table (fixed_grid=True only) -----------------
        # When fixed_grid=True the obstacle layout never changes between
        # episodes, so we precompute Dijkstra distances from every free cell
        # to every other reachable cell once and cache the result.
        #
        # This is critical for performance: HER\'s compute_reward() is called
        # once per relabeled sample per gradient update, and it needs distances
        # between arbitrary (achieved_goal, desired_goal) pairs -- not just
        # distances to the one fixed goal. Without precomputation, every
        # compute_reward() call triggers a live Dijkstra search (~4 fps).
        # With the table, each call is an O(1) dict lookup (~128 fps).
        self._distance_table: dict | None = None

        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------
    #  Gymnasium API: reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        if self.fixed_grid and self._initial_grid is not None:
            self.grid = self._initial_grid.copy()
        else:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            total_cells = self.grid_size * self.grid_size

            n_obstacles = int(total_cells * self.obstacle_density)
            obstacle_indices = rng.choice(total_cells, size=n_obstacles, replace=False)
            rows_o, cols_o = np.unravel_index(obstacle_indices, self.grid.shape)
            self.grid[rows_o, cols_o] = CELL_OBSTACLE

            free_mask = (self.grid == CELL_FREE)
            free_indices = np.flatnonzero(free_mask.ravel())
            n_no_fly = min(int(total_cells * self.no_fly_density), len(free_indices))
            if n_no_fly > 0:
                nfz_indices = rng.choice(free_indices, size=n_no_fly, replace=False)
                rows_n, cols_n = np.unravel_index(nfz_indices, self.grid.shape)
                self.grid[rows_n, cols_n] = CELL_NO_FLY

            if self.fixed_grid:
                self._initial_grid = self.grid.copy()
                # Build the all-pairs distance table once when the fixed grid
                # is first generated. All subsequent resets reuse this table.
                self._distance_table = self._build_distance_table()

        free_cells = np.argwhere(self.grid == CELL_FREE)
        assert len(free_cells) >= 2

        goal_idx = rng.choice(len(free_cells))
        self.goal_pos = free_cells[goal_idx].copy()

        if self.curriculum_enabled:
            current_max_dist = (
                self.curriculum_start_dist
                + (self.episode_count // self.curriculum_step_episodes)
                * self.curriculum_step_dist
            )
        else:
            current_max_dist = float("inf")

        valid_uav_cells = [
            cell for cell in free_cells
            if not np.array_equal(cell, self.goal_pos)
            and int(np.sum(np.abs(cell - self.goal_pos))) <= current_max_dist
        ]
        if not valid_uav_cells:
            valid_uav_cells = [c for c in free_cells if not np.array_equal(c, self.goal_pos)]

        uav_idx = rng.choice(len(valid_uav_cells))
        self.uav_pos = valid_uav_cells[uav_idx].copy()

        self.current_step = 0
        self.episode_count += 1
        self.previous_distance = self._bfs_distance(self.uav_pos, self.goal_pos)
        self.last_action = None
        self.visited_cells.clear()
        self.visited_cells.add(tuple(self.uav_pos))

        return self._build_observation(), self._build_info()

    # ------------------------------------------------------------------
    #  Gymnasium API: step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.current_step += 1

        delta = self.ACTION_DELTAS[action]
        next_pos = self.uav_pos + delta

        terminated = False
        truncated = False
        sparse_reward = 0.0
        self._last_crashed = False
        prev_pos = self.uav_pos.copy()

        neighbors = self.get_neighbors(prev_pos)
        valid_next_positions = [tuple(n[0]) for n in neighbors]

        if tuple(next_pos) not in valid_next_positions:
            sparse_reward = -1.0
            terminated = True
            self._last_crashed = True
            next_pos = prev_pos.copy()
        else:
            self.uav_pos = next_pos.copy()
            if np.array_equal(self.uav_pos, self.goal_pos):
                sparse_reward = 1.0
                terminated = True

        if not terminated and self.current_step >= MAX_STEPS:
            truncated = True

        # Potential-based shaping (Ng, Harada, Russell 1999)
        gamma = 0.99
        reward = (
            sparse_reward
            + gamma * self._potential(next_pos, self.goal_pos)
            - self._potential(prev_pos, self.goal_pos)
        )

        self.last_action = action
        self._last_prev_pos = prev_pos

        return self._build_observation(), reward, terminated, truncated, self._build_info()

    # ------------------------------------------------------------------
    #  Observation / info helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> dict:
        norm = float(self.grid_size - 1)
        row_disp_norm = (self.goal_pos[0] - self.uav_pos[0]) / norm
        col_disp_norm = (self.goal_pos[1] - self.uav_pos[1]) / norm

        pos_obs = np.array(
            [self.uav_pos[0] / norm, self.uav_pos[1] / norm,
             row_disp_norm, col_disp_norm],
            dtype=np.float32,
        )
        local_grid = self._get_local_observation()
        last_action_onehot = [0.0] * self.n_actions
        if self.last_action is not None:
            last_action_onehot[self.last_action] = 1.0

        obs_array = np.concatenate([pos_obs, local_grid, last_action_onehot], dtype=np.float32)
        return {
            "observation": obs_array,
            "achieved_goal": self.uav_pos.astype(np.float32),
            "desired_goal": self.goal_pos.astype(np.float32),
        }

    def _get_local_observation(self) -> np.ndarray:
        r, c = self.uav_pos
        local = np.full((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), CELL_OBSTACLE, dtype=np.int32)
        for dr in range(-LOCAL_VIEW_RADIUS, LOCAL_VIEW_RADIUS + 1):
            for dc in range(-LOCAL_VIEW_RADIUS, LOCAL_VIEW_RADIUS + 1):
                gr, gc = r + dr, c + dc
                if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                    local[dr + LOCAL_VIEW_RADIUS, dc + LOCAL_VIEW_RADIUS] = self.grid[gr, gc]
        return local.flatten().astype(np.float32)

    def _build_info(self) -> dict:
        manhattan = int(np.sum(np.abs(self.uav_pos - self.goal_pos)))
        prev_pos = self._last_prev_pos if hasattr(self, "_last_prev_pos") else self.uav_pos
        crashed = getattr(self, "_last_crashed", False)
        return {
            "uav_pos": self.uav_pos.tolist(),
            "previous_uav_pos": prev_pos.tolist(),
            "crashed": crashed,
            "goal_pos": self.goal_pos.tolist(),
            "manhattan_distance": manhattan,
            "step": self.current_step,
        }

    # ------------------------------------------------------------------
    #  Neighbors / distance helpers
    # ------------------------------------------------------------------
    def get_neighbors(self, node: np.ndarray) -> list:
        neighbors = []
        r, c = node[0], node[1]
        for dr, dc in self.ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if self.grid[nr, nc] != CELL_OBSTACLE:
                    cost = 1.0 if dr == 0 or dc == 0 else math.sqrt(2)
                    neighbors.append((np.array([nr, nc]), cost))
        return neighbors

    def _bfs_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])
        if sr == gr and sc == gc:
            return 0.0

        # O(1) fast path via precomputed all-pairs table (fixed_grid=True)
        if self._distance_table is not None:
            row = self._distance_table.get((sr, sc))
            if row is not None:
                return row.get((gr, gc), float(abs(sr - gr) + abs(sc - gc)))

        # Live Dijkstra fallback (fixed_grid=False or table not yet built)
        pq = [(0.0, sr, sc)]
        visited = set()
        while pq:
            dist, r, c = heapq.heappop(pq)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if r == gr and c == gc:
                return float(dist)
            for neighbor_pos, cost in self.get_neighbors(np.array([r, c])):
                nr, nc = neighbor_pos[0], neighbor_pos[1]
                if (nr, nc) not in visited:
                    heapq.heappush(pq, (dist + cost, nr, nc))
        return float(abs(sr - gr) + abs(sc - gc))

    def _build_distance_table(self) -> dict:
        """
        Precompute all-pairs shortest distances on the current (fixed) grid.
        Called ONCE when fixed_grid=True and the grid is first established.
        NOT called on every reset() -- subsequent resets reuse this table.

        This is needed because HER\'s compute_reward() must evaluate distances
        between arbitrary relabeled (achieved_goal, desired_goal) pairs, not
        just distances to the single episode goal, so we cannot precompute
        only from one source.
        """
        table: dict = {}
        for sr in range(self.grid_size):
            for sc in range(self.grid_size):
                if self.grid[sr, sc] == CELL_OBSTACLE:
                    continue
                dist_from_src: dict = {(sr, sc): 0.0}
                pq = [(0.0, sr, sc)]
                visited: set = set()
                while pq:
                    dist, r, c = heapq.heappop(pq)
                    if (r, c) in visited:
                        continue
                    visited.add((r, c))
                    for neighbor_pos, cost in self.get_neighbors(np.array([r, c])):
                        nr, nc = int(neighbor_pos[0]), int(neighbor_pos[1])
                        new_dist = dist + cost
                        if (nr, nc) not in visited and new_dist < dist_from_src.get((nr, nc), float("inf")):
                            dist_from_src[(nr, nc)] = new_dist
                            heapq.heappush(pq, (new_dist, nr, nc))
                table[(sr, sc)] = dist_from_src
        return table

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return float(math.dist(pos1, pos2))

    def _potential(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """Phi(state, goal) = -distance(state, goal)  [Ng et al. 1999]"""
        return -self._bfs_distance(pos, goal)

    def _in_bounds(self, pos: np.ndarray) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    # ------------------------------------------------------------------
    #  HER reward function (GoalEnv contract)
    # ------------------------------------------------------------------
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict | list
    ) -> float | np.ndarray:
        """
        Compute the reward for Hindsight Experience Replay (HER).
        Uses potential-based reward shaping (Ng et al. 1999) identically to step().
        Vectorized to support batches of shape (batch_size, 2).
        """
        is_single = False
        if len(achieved_goal.shape) == 1:
            achieved_goal = np.expand_dims(achieved_goal, axis=0)
            desired_goal = np.expand_dims(desired_goal, axis=0)
            info = [info]
            is_single = True

        batch_size = achieved_goal.shape[0]
        rewards = np.zeros(batch_size, dtype=np.float32)
        gamma = 0.99

        for i in range(batch_size):
            ag = np.round(achieved_goal[i]).astype(np.int32)
            dg = np.round(desired_goal[i]).astype(np.int32)

            prev_pos = ag
            crashed = False

            if isinstance(info, list) and len(info) > i:
                inf = info[i]
                if "previous_uav_pos" in inf:
                    prev_pos = np.round(inf["previous_uav_pos"]).astype(np.int32)
                if "crashed" in inf:
                    crashed = inf["crashed"]
            elif isinstance(info, dict):
                if "previous_uav_pos" in info:
                    prev_pos = np.round(info["previous_uav_pos"][i]).astype(np.int32)
                if "crashed" in info:
                    crashed = info["crashed"][i]

            # Check crashed FIRST (matches step() precedence exactly)
            if crashed:
                sparse_reward = -1.0
            elif np.array_equal(ag, dg):
                sparse_reward = 1.0
            else:
                sparse_reward = 0.0

            rewards[i] = sparse_reward + gamma * self._potential(ag, dg) - self._potential(prev_pos, dg)

        if is_single:
            return float(rewards[0])
        return rewards

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------
    def _render_human(self) -> None:
        symbols = {CELL_FREE: ".", CELL_OBSTACLE: "#", CELL_NO_FLY: "~"}
        print(f"\n--- Step {self.current_step} ---")
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                if np.array_equal([r, c], self.uav_pos):
                    row_chars.append("U")
                elif np.array_equal([r, c], self.goal_pos):
                    row_chars.append("G")
                else:
                    row_chars.append(symbols.get(self.grid[r, c], "?"))
            print(" ".join(row_chars))
        print()
