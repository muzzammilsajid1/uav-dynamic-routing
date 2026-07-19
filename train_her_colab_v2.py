# =============================================================================
# train_her_colab_v2.py
# DQN + HER training for UAV routing — Colab-ready, potential-based shaping,
# fixed_grid=True, all-pairs distance table speed fix.
#
# Paste each cell block into a separate Colab code cell.
# =============================================================================


# %% ---------------------------------------------------------------------------
# CELL 1: Install dependencies & mount Google Drive
# ------------------------------------------------------------------------------

!pip install -q "stable-baselines3[extra]>=2.0" gymnasium numpy

from google.colab import drive
drive.mount("/content/drive")

import os
DRIVE_SAVE_DIR = "/content/drive/MyDrive/uav_her_training"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
print(f"Drive save directory: {DRIVE_SAVE_DIR}")


# %% ---------------------------------------------------------------------------
# CELL 2: Write uav_env.py to /content/uav_env.py
#
# HOW TO USE IN COLAB:
#   Paste this entire block as a single Colab code cell.
#   The %%writefile magic on the FIRST LINE causes Colab to write
#   the rest of the cell verbatim to disk with zero string-escaping.
# ------------------------------------------------------------------------------

%%writefile /content/uav_env.py
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
            
            if self.fixed_grid:
                self._initial_grid = self.grid.copy()
                # Precompute all-pairs distances once when the fixed grid is
                # first generated. Subsequent resets reuse _initial_grid and
                # _distance_table without re-running Dijkstra.
                self._distance_table = self._build_distance_table()

        # ---- Place UAV and goal on distinct free cells -----------------------
        free_cells = np.argwhere(self.grid == CELL_FREE)
        assert len(free_cells) >= 2, (
            "Grid generation left fewer than 2 free cells — "
            "reduce obstacle_density or no_fly_density."
        )
        
        # Place goal first
        goal_idx = rng.choice(len(free_cells))
        self.goal_pos = free_cells[goal_idx].copy()
        
        # Determine maximum starting distance for the UAV
        if self.curriculum_enabled:
            current_max_dist = self.curriculum_start_dist + (self.episode_count // self.curriculum_step_episodes) * self.curriculum_step_dist
        else:
            current_max_dist = float('inf')
            
        # Find valid free cells within the maximum distance
        valid_uav_cells = []
        for cell in free_cells:
            if not np.array_equal(cell, self.goal_pos):
                dist = int(np.sum(np.abs(cell - self.goal_pos))) # Manhattan distance
                if dist <= current_max_dist:
                    valid_uav_cells.append(cell)
                    
        # Fallback to any free cell if the local radius is completely boxed in
        if not valid_uav_cells:
            valid_uav_cells = [c for c in free_cells if not np.array_equal(c, self.goal_pos)]
            
        uav_idx = rng.choice(len(valid_uav_cells))
        self.uav_pos = valid_uav_cells[uav_idx].copy()

        self.current_step = 0
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

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two 2-D coordinates."""
        return float(math.dist(pos1, pos2))

    def _potential(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """
        Potential function for reward shaping (Ng et al. 1999).
        Phi(state, goal) = -distance(state, goal)
        """
        phi = -self._bfs_distance(pos, goal) / self.max_dist
        if not hasattr(self, '_printed_phi'):
            print(f"[DEBUG uav_env] Raw Phi value at first potential call: {phi:.3f}")
            self._printed_phi = True
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


# %% ---------------------------------------------------------------------------
# CELL 2.5: Write safe_her_buffer.py to /content/safe_her_buffer.py
# ------------------------------------------------------------------------------

# %%writefile /content/safe_her_buffer.py
import torch
import numpy as np
from stable_baselines3.her import HerReplayBuffer

class SafeHerReplayBuffer(HerReplayBuffer):
    def sample(self, batch_size, env=None):
        samples = super().sample(batch_size, env)
        
        # Force dones=1.0 for transitions where the achieved goal matches the desired goal
        ag = np.round(samples.next_observations["achieved_goal"].cpu().numpy()).astype(np.int32)
        dg = np.round(samples.observations["desired_goal"].cpu().numpy()).astype(np.int32)
        
        # Check spatial equivalence (first 2 elements)
        goal_reached = np.all(ag[:, 0:2] == dg[:, 0:2], axis=1)
        
        # Update dones inplace
        if np.any(goal_reached):
            # Since samples.dones is a PyTorch tensor, we modify it in-place
            samples.dones[goal_reached] = 1.0
            
        return samples

# %% ---------------------------------------------------------------------------
# CELL 3: Train DQN + HER (with checkpointing every 50k steps)
# ------------------------------------------------------------------------------

import os
import time
import numpy as np
from stable_baselines3 import DQN
from safe_her_buffer import SafeHerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from uav_env import UAVRoutingEnv

# ---- Config ------------------------------------------------------------------
GRID_SIZE        = 15
OBSTACLE_DENSITY = 0.20
NO_FLY_DENSITY   = 0.0
TOTAL_TIMESTEPS  = 300_000
SEED             = 42

DRIVE_SAVE_DIR   = "/content/drive/MyDrive/uav_her_training"
LOCAL_CKPT_DIR   = "/content/checkpoints"
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

# ---- FPS logging callback ----------------------------------------------------
class FPSCallback(BaseCallback):
    """Prints fps and a projected finish time at the first logging interval."""
    def __init__(self, total_timesteps: int, log_every: int = 10_000):
        super().__init__(verbose=0)
        self._total   = total_timesteps
        self._every   = log_every
        self._t0      = None
        self._printed_projection = False

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n > 0 and n % self._every == 0:
            elapsed = time.perf_counter() - self._t0
            fps = n / elapsed
            remaining_steps = self._total - n
            eta_min = (remaining_steps / fps) / 60
            print(f"  [Step {n:>7,}]  fps={fps:.1f}  ETA={eta_min:.0f} min")
            if not self._printed_projection and n >= self._every:
                proj_total_min = (self._total / fps) / 60
                print(f"  >> Projected total time at current fps: {proj_total_min:.0f} min ({proj_total_min/60:.2f} h)")
                self._printed_projection = True
        return True

# ---- Drive checkpoint callback -----------------------------------------------
class DriveCheckpointCallback(BaseCallback):
    """Copies latest checkpoint from local /content/checkpoints/ to Drive."""
    def __init__(self, save_freq: int, drive_dir: str):
        super().__init__(verbose=0)
        self._freq     = save_freq
        self._drive    = drive_dir

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n > 0 and n % self._freq == 0:
            # SB3 CheckpointCallback names files as rl_model_{n}_steps.zip
            src = f"/content/checkpoints/rl_model_{n}_steps.zip"
            dst = os.path.join(self._drive, f"rl_model_{n}_steps.zip")
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)
                print(f"  [Checkpoint] Saved to Drive: {dst}")
        return True

# ---- Environment -------------------------------------------------------------
from stable_baselines3.common.monitor import Monitor

env = UAVRoutingEnv(
    grid_size=GRID_SIZE,
    obstacle_density=OBSTACLE_DENSITY,
    no_fly_density=NO_FLY_DENSITY,
    fixed_grid=True,
    seed=SEED,
)
env = Monitor(env)



import torch
import torch.nn.functional as F

class DoubleDQN(DQN):
    """Subclass of SB3\'s DQN that implements Double DQN."""
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            
            with torch.no_grad():
                # 1. Select action with highest value from ONLINE network
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                
                # 2. Evaluate that action\'s value using TARGET network
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

# ---- Model -------------------------------------------------------------------
model = DoubleDQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=256,
    tau=1.0,
    gamma=0.99,           # Must match gamma in _potential shaping formula
    train_freq=2,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.50,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
    replay_buffer_class=SafeHerReplayBuffer,
    replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
    verbose=1,
    seed=SEED,
    device="auto",
    tensorboard_log=os.path.join(DRIVE_SAVE_DIR, "tensorboard_logs"),
)

# ---- Callbacks ---------------------------------------------------------------
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=LOCAL_CKPT_DIR,
    name_prefix="rl_model",
    verbose=1,
)
drive_cb   = DriveCheckpointCallback(save_freq=50_000, drive_dir=DRIVE_SAVE_DIR)
fps_cb     = FPSCallback(total_timesteps=TOTAL_TIMESTEPS, log_every=10_000)

# ---- Train -------------------------------------------------------------------
print("=" * 60)
print(f"  DQN + HER Training  |  {TOTAL_TIMESTEPS:,} steps")
print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}  |  seed={SEED}")
print(f"  Reward: potential-based (Ng 1999) + sparse +/-1")
print(f"  Distance table: precomputed (O(1) lookups)")
print("=" * 60)

from stable_baselines3.common.logger import configure
# Configure logger to save to Drive (CSV + Stdout + TensorBoard)
drive_logger = configure(os.path.join(DRIVE_SAVE_DIR, "sb3_logs"), ["stdout", "csv", "tensorboard"])
model.set_logger(drive_logger)

t_start = time.perf_counter()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, drive_cb, fps_cb],
    log_interval=10,
    progress_bar=False,    # Disable tqdm in Colab (clutters output)
)
t_total = time.perf_counter() - t_start

# ---- Save final model to both local and Drive --------------------------------
final_local = "/content/dqn_her_300k_final"
final_drive = os.path.join(DRIVE_SAVE_DIR, "dqn_her_300k_final.zip")
model.save(final_local)

import shutil
shutil.copy2(final_local + ".zip", final_drive)

print(f"\nTraining complete in {t_total/60:.1f} min ({t_total:.0f}s)")
print(f"Model saved locally:  {final_local}.zip")
print(f"Model saved to Drive: {final_drive}")
env.close()


# %% ---------------------------------------------------------------------------
# CELL 4: Evaluate the trained model (20 episodes, deterministic)
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from uav_env import UAVRoutingEnv

class DoubleDQN(DQN):
    """Subclass of SB3's DQN that implements Double DQN."""
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            
            with torch.no_grad():
                # 1. Select action with highest value from ONLINE network
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                
                # 2. Evaluate that action's value using TARGET network
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

# Load from local (or Drive path if session restarted)
MODEL_PATH = "/content/dqn_her_300k_final.zip"
# MODEL_PATH = "/content/drive/MyDrive/uav_her_training/dqn_her_300k_final.zip"

dummy_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.0, fixed_grid=True
)
model = DoubleDQN.load(MODEL_PATH, env=dummy_env)

eval_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.0,
    fixed_grid=True, seed=42
)

N_EPISODES = 20
goals, crashes, timeouts = 0, 0, 0

print("=" * 60)
print("  Evaluation (20 episodes, deterministic)")
print("=" * 60)

for ep in range(1, N_EPISODES + 1):
    obs, info = eval_env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action))
        total_reward += reward
        steps += 1
        done = terminated or truncated

    crashed = info.get("crashed", False)
    ag = np.round(obs["achieved_goal"]).astype(np.int32)
    dg = np.round(obs["desired_goal"]).astype(np.int32)

    if crashed:
        outcome = "CRASH"
        crashes += 1
    elif np.array_equal(ag[:2], dg[:2]):
        outcome = "GOAL"
        goals += 1
    else:
        outcome = "TIMEOUT"
        timeouts += 1

    print(f"  Ep {ep:>2d} | {outcome:<7} | Steps: {steps:>4d} | Reward: {total_reward:>8.3f}")

print()
print("=" * 60)
print(f"  GOAL:    {goals}/{N_EPISODES}")
print(f"  CRASH:   {crashes}/{N_EPISODES}")
print(f"  TIMEOUT: {timeouts}/{N_EPISODES}")
print("=" * 60)

eval_env.close()
dummy_env.close()


# %% ---------------------------------------------------------------------------
# CELL 5: Trace 3 TIMEOUT Episodes
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from uav_env import UAVRoutingEnv

class DoubleDQN(DQN):
    """Subclass of SB3's DQN that implements Double DQN."""
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            
            with torch.no_grad():
                # 1. Select action with highest value from ONLINE network
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                
                # 2. Evaluate that action's value using TARGET network
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

# Load from local (or Drive path if session restarted)
MODEL_PATH = "/content/dqn_her_300k_final.zip"

dummy_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.0, fixed_grid=True
)
model = DoubleDQN.load(MODEL_PATH, env=dummy_env)

eval_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.0,
    fixed_grid=True, seed=42
)

TIMEOUTS_TO_TRACE = 3
timeouts_traced = 0
ep = 0

print("=" * 60)
print(f"  Tracing {TIMEOUTS_TO_TRACE} TIMEOUT episodes step-by-step")
print("=" * 60)

while timeouts_traced < TIMEOUTS_TO_TRACE:
    ep += 1
    obs, info = eval_env.reset()
    done = False
    
    # Buffer the trace so we only print it if the episode is a TIMEOUT
    trace_buffer = []
    trace_buffer.append(f"\n--- Episode {ep} Trace ---")
    
    ag = np.round(obs["achieved_goal"]).astype(np.int32)
    dg = np.round(obs["desired_goal"]).astype(np.int32)
    # Using the underlying unwrapped environment to compute exact BFS distance
    dist = eval_env.unwrapped._bfs_distance(ag, dg)
    trace_buffer.append(f"Start pos: {ag.tolist()} | Goal pos: {dg.tolist()} | Initial Dist: {dist:.3f}")
    
    steps = 0
    total_reward = 0.0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(int(action))
        
        steps += 1
        total_reward += float(reward)
        ag = np.round(obs["achieved_goal"]).astype(np.int32)
        dg = np.round(obs["desired_goal"]).astype(np.int32)
        
        current_dist = eval_env.unwrapped._bfs_distance(ag, dg)
        
        trace_buffer.append(f"  Step {steps:>3d} | Pos: {ag.tolist()} | Action: {int(action)} | Dist: {current_dist:>6.3f} | Reward: {reward:>6.3f}")
        
        done = terminated or truncated

    crashed = info.get("crashed", False)
    ag = np.round(obs["achieved_goal"]).astype(np.int32)
    dg = np.round(obs["desired_goal"]).astype(np.int32)
    
    if crashed:
        outcome = "CRASH"
    elif np.array_equal(ag[:2], dg[:2]):
        outcome = "GOAL"
    else:
        outcome = "TIMEOUT"

    if outcome == "TIMEOUT":
        timeouts_traced += 1
        final_dist = eval_env.unwrapped._bfs_distance(ag, dg)
        trace_buffer.append(f"--- Episode ended in TIMEOUT after {steps} steps ---")
        trace_buffer.append(f"Final UAV pos: {ag.tolist()} | Goal pos: {dg.tolist()}")
        trace_buffer.append(f"Final BFS distance to goal: {final_dist:.3f}")
        
        print("\n".join(trace_buffer))
        
    # Safety break in case the model is too perfect and we don't hit enough timeouts
    if ep >= 100:
        print(f"\nStopped early: Reached 100 episodes but only found {timeouts_traced} timeouts.")
        break

eval_env.close()
dummy_env.close()

# %% ---------------------------------------------------------------------------
# CELL 6: Q-Value Inspection for specific state
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
from uav_env import UAVRoutingEnv

class DoubleDQN(DQN):
    """Subclass of SB3's DQN that implements Double DQN."""
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            
            with torch.no_grad():
                # 1. Select action with highest value from ONLINE network
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                
                # 2. Evaluate that action's value using TARGET network
                next_q_values_target = self.q_net_target(replay_data.next_observations)
                next_q_values = torch.gather(next_q_values_target, dim=1, index=next_actions)
                
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

# Load from local (or Drive path if session restarted)
MODEL_PATH = "/content/dqn_her_300k_final.zip"

dummy_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.0, fixed_grid=True
)
model = DoubleDQN.load(MODEL_PATH, env=dummy_env)

eval_env = UAVRoutingEnv(
    grid_size=15, obstacle_density=0.20, no_fly_density=0.0,
    fixed_grid=True, seed=42
)
eval_env.reset()

# Override positions
eval_env.unwrapped.uav_pos = np.array([7, 4])
eval_env.unwrapped.goal_pos = np.array([7, 5])

def print_q_values_for_history(last_action_val, label):
    eval_env.unwrapped.last_action = last_action_val
    obs = eval_env.unwrapped._build_observation()
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
    
    q_values_np = q_values.cpu().numpy()[0]
    action_names = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
    
    print("=" * 60)
    print(f"  Q-Values for UAV at [7, 4] with Goal at [7, 5]")
    print(f"  {label}")
    print("=" * 60)
    
    for i, (name, q_val) in enumerate(zip(action_names, q_values_np)):
        print(f"  Action {i} ({name:<2}) : {q_val:>8.3f}")
    
    best_action = int(np.argmax(q_values_np))
    print("-" * 60)
    print(f"  Greedy Choice : Action {best_action} ({action_names[best_action]})")
    print("=" * 60)
    print()

print_q_values_for_history(0, "History: Arrived via Action 0 (N)")
print_q_values_for_history(None, "History: None (Default / No History)")

eval_env.close()
dummy_env.close()
