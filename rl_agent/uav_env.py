"""
uav_env.py — Gymnasium wrapper for UAV grid-world routing.

Models a single fixed-wing or multi-rotor UAV navigating a 2-D grid from a
start waypoint to a goal waypoint while avoiding obstacles (physical airspace
constraints) and no-fly zones (regulatory / geofenced regions).

Observation:  [x_uav, y_uav, x_goal, y_goal, <flattened 7x7 local grid>]
Action:       Discrete(8) — 8-connected heading commands (N, NE, E, SE, S, SW, W, NW)
Reward:       See _compute_reward() docstring.

Design notes
------------
* The environment is intentionally *decoupled* from the underlying grid data-
  structure so that a collaborator can later drop in a full `GridEnvironment`
  backend (dynamic obstacles, wind fields, etc.) without touching the RL
  interface.
* All spatial quantities are in *grid units*; mapping to real-world metres or
  NED coordinates is deferred to the integration layer.

Author: Research Team
"""

from __future__ import annotations

import math
from collections import deque
import heapq
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.grid_environment import DynamicObstacle, default_dynamic_obstacles


# ---------------------------------------------------------------------------
# Constants — keep these module-level so they are easy to tune from notebooks
# ---------------------------------------------------------------------------
GRID_SIZE: int = 15              # N×N grid cells
LOCAL_VIEW_RADIUS: int = 3       # radius of the local sensor window (7x7)
LOCAL_VIEW_SIZE: int = 2 * LOCAL_VIEW_RADIUS + 1   # 7

# Cell type encoding (matches what a future GridEnvironment will emit)
CELL_FREE: int = 0               # traversable airspace
CELL_OBSTACLE: int = 1           # physical obstacle (building, terrain, etc.)
CELL_NO_FLY: int = 2             # regulatory no-fly zone (TFR / geofence)

# Reward shaping (energy-budget framing)
REWARD_STEP: float = -0.1         # continuous energy/battery consumption
REWARD_GOAL: float = 1.0         # successful mission payload delivery/waypoint reached
REWARD_COLLISION: float = -1.0   # catastrophic airframe loss
REWARD_NO_FLY: float = 0.0       # regulated airspace violation

# Maximum episode length — prevents infinite wandering
MAX_STEPS: int = GRID_SIZE * GRID_SIZE  # 225 steps should be generous


class UAVRoutingEnv(gym.Env):
    """
    A 2-D grid-world environment for training an RL-based UAV autopilot.

    The UAV must navigate from a random (or fixed) start cell to a goal cell
    while minimising energy expenditure (cumulative step penalty) and avoiding
    collisions with obstacles or prolonged incursions into no-fly zones.

    This wrapper conforms to the Gymnasium API so it can be used directly with
    Stable-Baselines3 algorithms (DQN, PPO, A2C, etc.).
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    # 8-directional heading commands -------------------------------------------
    # Ordered as compass headings starting from North, proceeding clockwise.
    # Each entry is (Δrow, Δcol) in a matrix-style frame where row 0 is the
    # top (North) edge.
    #
    #   Index   Heading     (Δrow, Δcol)
    #   ─────   ───────     ────────────
    #     0      N           (-1,  0)
    #     1      S           ( 1,  0)
    #     2      W           ( 0, -1)
    #     3      E           ( 0,  1)
    #     4      NW          (-1, -1)
    #     5      NE          (-1,  1)
    #     6      SW          ( 1, -1)
    #     7      SE          ( 1,  1)
    #
    # WARNING: This action mapping is a BREAKING CHANGE from earlier versions.
    # Any previously trained model checkpoints are invalid and must be retrained.
    ACTION_DELTAS = np.array(
        [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)],
        dtype=np.int32,
    )

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        obstacle_density: float = 0.20,
        no_fly_density: float = 0.0,
        render_mode: str | None = None,
        seed: int | None = None,
        curriculum_enabled: bool = False,
        curriculum_start_dist: int = 3,
        curriculum_step_episodes: int = 5000,
        curriculum_step_dist: int = 1,
        fixed_grid: bool = True,
        dynamic_obstacles_enabled: bool = False,
        dynamic_obstacles: list[DynamicObstacle] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        grid_size : int
            Side length of the square operating area (default 15).
        obstacle_density : float ∈ (0, 1)
            Fraction of cells that are physical obstacles at reset.
        no_fly_density : float ∈ (0, 1)
            Fraction of cells designated as no-fly zones at reset.
        render_mode : str or None
            'human' for console rendering, None to disable.
        seed : int or None
            RNG seed for reproducibility.
        curriculum_enabled : bool
            If True, starts the agent close to the goal and expands starting radius.
        """
        super().__init__()

        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.no_fly_density = no_fly_density
        self.render_mode = render_mode
        self.curriculum_enabled = curriculum_enabled
        self.curriculum_start_dist = curriculum_start_dist
        self.curriculum_step_episodes = curriculum_step_episodes
        self.curriculum_step_dist = curriculum_step_dist
        self.dynamic_obstacles_enabled = dynamic_obstacles_enabled
        if self.dynamic_obstacles_enabled and dynamic_obstacles is None:
            self.dynamic_obstacles = default_dynamic_obstacles()
        else:
            self.dynamic_obstacles = dynamic_obstacles or []
        self.episode_count = 0
        self.fixed_grid = fixed_grid
        self._initial_grid = None
        self.max_dist = math.sqrt((self.grid_size - 1)**2 + (self.grid_size - 1)**2)

        # ---- Action space: 8 discrete heading commands -----------------------
        self.action_space = spaces.Discrete(8)
        self.n_actions = self.action_space.n
        self.last_action = None
        self.visited_cells = set()

        # ---- Observation space -----------------------------------------------
        # Positional components are normalised to [0, 1]. Local grid is {0, 1, 2}.
        obs_dim = 4 + LOCAL_VIEW_SIZE * LOCAL_VIEW_SIZE + 8   # 4 + 49 + 8 = 61
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-1.0,
                high=max(CELL_NO_FLY, 1.0),
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                low=0.0, high=float(grid_size - 1), shape=(4,), dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=0.0, high=float(grid_size - 1), shape=(4,), dtype=np.float32
            ),
        })

        # ---- Internal state (populated in reset()) ---------------------------
        self.grid: np.ndarray | None = None          # (grid_size, grid_size)
        self.uav_pos: np.ndarray | None = None       # [row, col]
        self.goal_pos: np.ndarray | None = None       # [row, col]
        self.current_step: int = 0
        self._elapsed_steps: int = 0

        # ---- All-pairs distance table (fixed_grid=True only) -----------------
        # When fixed_grid=True, the obstacle layout never changes between
        # episodes, so we precompute Dijkstra distances from every free cell
        # to every other reachable cell once and cache the result.
        #
        # This is critical for performance: HER's compute_reward() is called
        # once per relabeled sample per gradient update, and it needs distances
        # between arbitrary (achieved_goal, desired_goal) pairs — not just
        # distances to the one fixed goal. Without precomputation, every
        # compute_reward() call triggers a live Dijkstra search, making a
        # 300k-step HER run prohibitively slow (~13 hours). With the table,
        # each call is an O(1) dict lookup.
        self._distance_table: dict | None = None

        # Seed the RNG
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------
    #  Gymnasium API: reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        1. Re-generate the obstacle / no-fly grid.
        2. Place the UAV and goal on random *free* cells.
        3. Return the initial observation and an info dict.
        """
        super().reset(seed=seed)
        rng = self.np_random   # seeded RNG from gymnasium

        # ---- Build the grid --------------------------------------------------
        if self.fixed_grid and self._initial_grid is not None:
            self.grid = self._initial_grid.copy()
        else:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            total_cells = self.grid_size * self.grid_size
    
            # Scatter obstacles (physical airspace constraints)
            n_obstacles = int(total_cells * self.obstacle_density)
            obstacle_indices = rng.choice(total_cells, size=n_obstacles, replace=False)
            rows_o, cols_o = np.unravel_index(obstacle_indices, self.grid.shape)
            self.grid[rows_o, cols_o] = CELL_OBSTACLE
    
            # Scatter no-fly zones (regulatory geofences)
            free_mask = (self.grid == CELL_FREE)
            free_indices = np.flatnonzero(free_mask.ravel())
            n_no_fly = min(int(total_cells * self.no_fly_density), len(free_indices))
            if n_no_fly > 0:
                nfz_indices = rng.choice(free_indices, size=n_no_fly, replace=False)
                rows_n, cols_n = np.unravel_index(nfz_indices, self.grid.shape)
                self.grid[rows_n, cols_n] = CELL_NO_FLY
            
            # Ensure dynamic cells are set to their initial state
            if self.dynamic_obstacles_enabled:
                for obs in self.dynamic_obstacles:
                    if obs.initial_state == "blocked":
                        self.grid[obs.cell] = CELL_OBSTACLE
                    else:
                        self.grid[obs.cell] = CELL_FREE

            if self.fixed_grid:
                self._initial_grid = self.grid.copy()
                # Precompute all-pairs distances once when the fixed grid is
                # first generated. Subsequent resets reuse _initial_grid and
                # _distance_table without re-running Dijkstra.
                self._distance_table = self._build_distance_table()

        # ---- Place UAV and goal on distinct free cells -----------------------
        free_cells = np.argwhere(self.grid == CELL_FREE)
        candidate_cells = []
        if self.dynamic_obstacles_enabled:
            dyn_cells = {obs.cell for obs in self.dynamic_obstacles}
        else:
            dyn_cells = set()
            
        for c in free_cells:
            if tuple(c) not in dyn_cells:
                candidate_cells.append(c)
                
        assert len(candidate_cells) >= 2, (
            "Grid generation left fewer than 2 valid free cells for start/goal."
        )
        
        # Determine maximum starting distance for the UAV
        if self.curriculum_enabled:
            current_max_dist = self.curriculum_start_dist + (self.episode_count // self.curriculum_step_episodes) * self.curriculum_step_dist
        else:
            current_max_dist = float('inf')

        MAX_RETRIES = 50
        for _ in range(MAX_RETRIES):
            # Place goal first
            goal_idx = rng.choice(len(candidate_cells))
            self.goal_pos = candidate_cells[goal_idx].copy()
            
            # Find valid free cells within the maximum distance
            valid_uav_cells = []
            for cell in candidate_cells:
                if not np.array_equal(cell, self.goal_pos):
                    dist = int(np.sum(np.abs(cell - self.goal_pos))) # Manhattan distance
                    if dist <= current_max_dist:
                        valid_uav_cells.append(cell)
                        
            # Fallback to any free cell if the local radius is completely boxed in
            if not valid_uav_cells:
                valid_uav_cells = [c for c in candidate_cells if not np.array_equal(c, self.goal_pos)]
                
            uav_idx = rng.choice(len(valid_uav_cells))
            self.uav_pos = valid_uav_cells[uav_idx].copy()

            # Validate that the goal is reachable even if all dynamic obstacles are active
            if self.dynamic_obstacles_enabled and self.dynamic_obstacles:
                original_states = {}
                for obs in self.dynamic_obstacles:
                    original_states[obs.cell] = self.grid[obs.cell]
                    self.grid[obs.cell] = CELL_OBSTACLE
                
                is_reachable = self._is_reachable(self.uav_pos, self.goal_pos)
                
                for cell, state in original_states.items():
                    self.grid[cell] = state
                    
                if is_reachable:
                    break
            else:
                break
        else:
            raise RuntimeError("Could not find a start/goal pair not sealed off by dynamic obstacles.")

        self.current_step = 0
        self._elapsed_steps = 0
        self.episode_count += 1
        # Use BFS distance for reward shaping so the agent is rewarded for
        # obstacle-aware progress, not straight-line distance.
        self.previous_distance = self._bfs_distance(self.uav_pos, self.goal_pos)

        self.last_action = None
        self.visited_cells.clear()
        self.visited_cells.add(tuple(self.uav_pos))

        observation = self._build_observation()
        info = self._build_info()

        if self.render_mode == "human":
            self._render_human()

        return observation, info

    # ------------------------------------------------------------------
    #  Gymnasium API: step
    # ------------------------------------------------------------------
    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one heading command and advance the simulation by one time-step.

        Parameters
        ----------
        action : int ∈ {0, …, 7}
            Index into ACTION_DELTAS (compass heading).

        Returns
        -------
        observation : np.ndarray   — new state vector
        reward      : float        — scalar reward
        terminated  : bool         — True if episode ended (goal or crash)
        truncated   : bool         — True if max steps exceeded
        info        : dict         — auxiliary diagnostics
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.current_step += 1

        self._toggle_dynamic_obstacles()

        # ---- Compute candidate next position --------------------------------
        delta = self.ACTION_DELTAS[action]
        next_pos = self.uav_pos + delta

        # ---- Evaluate transition and determine reward / termination ----------
        terminated = False
        truncated = False
        sparse_reward = REWARD_STEP
        self._last_crashed = False

        prev_pos = self.uav_pos.copy()

        # 1. Move validation via get_neighbors
        neighbors = self.get_neighbors(prev_pos)
        valid_next_positions = [tuple(n[0]) for n in neighbors]

        if tuple(next_pos) not in valid_next_positions:
            sparse_reward = REWARD_COLLISION
            terminated = True
            self._last_crashed = True
            # UAV crashed, physical position does not update
            next_pos = prev_pos.copy()
        else:
            # ---- Valid move — update UAV position ----------------------------
            self.uav_pos = next_pos.copy()

            # 4. Goal reached (mission success)
            if np.array_equal(self.uav_pos, self.goal_pos):
                sparse_reward = REWARD_GOAL
                terminated = True

        # 5. Episode truncation: battery / flight-time budget exceeded
        if not terminated and self.current_step >= MAX_STEPS:
            truncated = True

        # Potential-based shaping (Ng, Harada, Russell 1999)
        gamma = 0.99
        reward = sparse_reward + gamma * self._potential(next_pos, self.goal_pos) - self._potential(prev_pos, self.goal_pos)

        self.last_action = action
        self._last_prev_pos = prev_pos

        observation = self._build_observation()
        info = self._build_info()

        if self.render_mode == "human":
            self._render_human()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    #  Observation helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> dict:
        """
        Assemble the full observation dict.
        """
        norm = float(self.grid_size - 1)

        # NOTE: Model checkpoints trained on the old 53-dim observation 
        # will no longer be compatible due to the addition of relative 
        # displacement features. A fresh retrain is required. Do NOT 
        # attempt to load or reuse old checkpoints.
        row_disp_norm = (self.goal_pos[0] - self.uav_pos[0]) / norm
        col_disp_norm = (self.goal_pos[1] - self.uav_pos[1]) / norm

        pos_obs = np.array(
            [
                self.uav_pos[0] / norm,   # row (≈ North offset)
                self.uav_pos[1] / norm,   # col (≈ East offset)
                row_disp_norm,
                col_disp_norm,
            ],
            dtype=np.float32,
        )

        local_grid = self._get_local_observation()
        
        last_action_onehot = [0.0] * self.n_actions
        if self.last_action is not None:
            last_action_onehot[self.last_action] = 1.0
            
        obs_array = np.concatenate([pos_obs, local_grid, last_action_onehot], dtype=np.float32)

        prev_pos = self._last_prev_pos if hasattr(self, "_last_prev_pos") else self.uav_pos
        return {
            "observation": obs_array,
            "achieved_goal": np.concatenate([self.uav_pos, prev_pos]).astype(np.float32),
            "desired_goal": np.concatenate([self.goal_pos, self.goal_pos]).astype(np.float32),
        }

    def _get_local_observation(self) -> np.ndarray:
        """
        Extract a 7x7 local grid patch centred on the UAV's current position
        and return it as a flattened float32 vector.

        Cells that fall outside the grid boundary are filled with
        CELL_OBSTACLE (the UAV should treat the boundary as impassable
        terrain, consistent with the out-of-bounds penalty).

        Returns
        -------
        np.ndarray, shape (49,), dtype float32
            Flattened 7x7 local sensor reading.

        Notes
        -----
        This method is deliberately isolated as a helper so a future
        `GridEnvironment` backend can override or augment the local
        sensing model (e.g., add LiDAR range noise, altitude layers,
        or dynamic obstacle velocities).
        """
        r, c = self.uav_pos

        # Initialise with OBSTACLE so out-of-bounds cells look impassable
        local = np.full(
            (LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), CELL_OBSTACLE, dtype=np.int32
        )

        for dr in range(-LOCAL_VIEW_RADIUS, LOCAL_VIEW_RADIUS + 1):
            for dc in range(-LOCAL_VIEW_RADIUS, LOCAL_VIEW_RADIUS + 1):
                gr, gc = r + dr, c + dc
                if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                    local[dr + LOCAL_VIEW_RADIUS, dc + LOCAL_VIEW_RADIUS] = (
                        self.grid[gr, gc]
                    )

        return local.flatten().astype(np.float32)

    def get_neighbors(self, node: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """
        Return valid neighboring cells reachable from `node`, paired with move cost.
        Respects grid bounds and CELL_OBSTACLE as impassable. CELL_NO_FLY is passable.
        """
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
        """
        Compute the shortest-path distance from *start* to *goal* via Dijkstra
        on the current grid, using 8-directional movement.

        Cells with CELL_OBSTACLE are impassable; CELL_NO_FLY cells ARE
        traversable (the UAV can fly through them at a penalty), so this
        reflects realistic routing rather than pretending no-fly zones
        are walls.

        This is used for dense reward shaping instead of Euclidean /
        Manhattan distance so that detours around obstacles are correctly
        recognised as progress toward the goal.

        Falls back to Manhattan distance if no path exists (should not
        happen on a well-formed grid).
        """
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        if sr == gr and sc == gc:
            return 0.0

        # O(1) fast path: use precomputed all-pairs table when available.
        # The table is built once in reset() when fixed_grid=True and the
        # grid is first generated. This avoids live Dijkstra calls inside
        # compute_reward() during HER relabeling.
        if self._distance_table is not None:
            row = self._distance_table.get((sr, sc))
            if row is not None:
                return row.get((gr, gc), float(abs(sr - gr) + abs(sc - gc)))
            # Source is an obstacle cell (unreachable); fall through to live search

        # Live Dijkstra fallback (fixed_grid=False, or table not yet built)
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

        # Fallback: no path found — use Manhattan distance as safe default
        return float(abs(sr - gr) + abs(sc - gc))

    def _is_reachable(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """Check if goal is reachable from start via BFS."""
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])
        if sr == gr and sc == gc:
            return True
            
        pq = [(sr, sc)]
        visited = {(sr, sc)}
        while pq:
            r, c = pq.pop(0)
            if r == gr and c == gc:
                return True
            for neighbor_pos, _ in self.get_neighbors(np.array([r, c])):
                nr, nc = int(neighbor_pos[0]), int(neighbor_pos[1])
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    pq.append((nr, nc))
        return False

    def _build_distance_table(self) -> dict:
        """
        Precompute all-pairs shortest distances on the current (fixed) grid.

        Runs one Dijkstra search from every non-obstacle cell, storing the
        result as a nested dict:
            self._distance_table[(sr, sc)][(tr, tc)] = float distance

        Uses get_neighbors() for move validity and edge costs (1.0 straight,
        sqrt(2) diagonal), matching _bfs_distance() exactly.

        Called ONCE when fixed_grid=True and the grid is first established.
        NOT called on every reset() — subsequent resets reuse this table.

        This is needed because HER's compute_reward() must evaluate distances
        between arbitrary relabeled (achieved_goal, desired_goal) pairs, not
        just distances to the single episode goal, so we cannot precompute
        only from one source.
        """
        table: dict = {}

        # Run Dijkstra from every traversable (non-obstacle) cell
        for sr in range(self.grid_size):
            for sc in range(self.grid_size):
                if self.grid[sr, sc] == CELL_OBSTACLE:
                    continue  # Impassable; no distances emanate from here

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
                        if (nr, nc) not in visited and new_dist < dist_from_src.get((nr, nc), float('inf')):
                            dist_from_src[(nr, nc)] = new_dist
                            heapq.heappush(pq, (new_dist, nr, nc))

                table[(sr, sc)] = dist_from_src

        return table

    def _toggle_dynamic_obstacles(self) -> None:
        """Toggle designated dynamic obstacles on a fixed timer."""
        if self.dynamic_obstacles_enabled and self.dynamic_obstacles:
            self._elapsed_steps += 1
            for obs in self.dynamic_obstacles:
                if self._elapsed_steps % obs.period == 0:
                    r, c = obs.cell
                    if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                        current_val = self.grid[r, c]
                        self.grid[r, c] = CELL_FREE if current_val == CELL_OBSTACLE else CELL_OBSTACLE

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two 2-D coordinates."""
        return float(math.dist(pos1, pos2))

    def _potential(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """
        Potential function for reward shaping (Ng et al. 1999).
        Phi(state, goal) = -distance(state, goal)
        """
        phi = -self._bfs_distance(pos, goal) / self.max_dist
        return phi

    # ------------------------------------------------------------------
    #  Internal utilities
    # ------------------------------------------------------------------
    def _in_bounds(self, pos: np.ndarray) -> bool:
        """Return True if *pos* is inside the grid boundaries."""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _build_info(self) -> dict:
        """
        Compile an auxiliary info dict for logging / debugging and HER.
        """
        manhattan = int(np.sum(np.abs(self.uav_pos - self.goal_pos)))
        
        # Fallbacks for initial reset() step where a prior transition hasn't occurred
        prev_pos = self._last_prev_pos if hasattr(self, "_last_prev_pos") else self.uav_pos
        crashed = getattr(self, "_last_crashed", False)
        
        is_success = bool(np.array_equal(self.uav_pos, self.goal_pos))
        
        return {
            "uav_pos": self.uav_pos.tolist(),
            "previous_uav_pos": prev_pos.tolist(),
            "crashed": crashed,
            "goal_pos": self.goal_pos.tolist(),
            "manhattan_distance": manhattan,
            "step": self.current_step,
            "is_success": is_success,
        }

    # ------------------------------------------------------------------
    #  HER Reward Function
    # ------------------------------------------------------------------
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict | list[dict]
    ) -> float | np.ndarray:
        """
        Compute the reward for Hindsight Experience Replay (HER).
        Uses potential-based reward shaping (Ng et al. 1999) identically to step().
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
            # Extract and immediately cast to integers to prevent float precision issues
            ag = np.round(achieved_goal[i]).astype(np.int32)
            dg = np.round(desired_goal[i]).astype(np.int32)
            
            if len(ag) == 4:
                current_pos = ag[0:2]
                prev_pos = ag[2:4]
                goal_pos = dg[0:2]
            else:
                current_pos = ag
                prev_pos = ag
                goal_pos = dg
                
            crashed = False
            if isinstance(info, list) and len(info) > i:
                inf = info[i]
                if "crashed" in inf:
                    crashed = inf["crashed"]

            # 1. Sparse reward term
            if crashed:
                sparse_reward = REWARD_COLLISION
            elif np.array_equal(current_pos, goal_pos):
                sparse_reward = REWARD_GOAL
            else:
                sparse_reward = REWARD_STEP

            # 2. Shaping term: gamma * Phi(next_state, goal) - Phi(state, goal)
            rewards[i] = sparse_reward + gamma * self._potential(current_pos, goal_pos) - self._potential(prev_pos, goal_pos)

        if is_single:
            return float(rewards[0])
        return rewards

    # ------------------------------------------------------------------
    #  Rendering (lightweight ASCII — good enough for SSH / notebooks)
    # ------------------------------------------------------------------
    def _render_human(self) -> None:
        """Print an ASCII map of the current grid state to stdout."""
        symbols = {
            CELL_FREE: "·",
            CELL_OBSTACLE: "█",
            CELL_NO_FLY: "▒",
        }

        header = f"\n--- Step {self.current_step} ---"
        print(header)

        for r in range(self.grid_size):
            row_chars: list[str] = []
            for c in range(self.grid_size):
                if np.array_equal([r, c], self.uav_pos):
                    row_chars.append("U")
                elif np.array_equal([r, c], self.goal_pos):
                    row_chars.append("G")
                else:
                    row_chars.append(symbols.get(self.grid[r, c], "?"))
            print(" ".join(row_chars))

        print()
