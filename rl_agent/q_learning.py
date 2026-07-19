from __future__ import annotations

import random
from dataclasses import dataclass

from envs.grid_environment import GridEnvironment, Node

Action = int

ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),   # 0: N
    (1, 0),    # 1: S
    (0, -1),   # 2: W
    (0, 1),    # 3: E
    (-1, -1),  # 4: NW
    (-1, 1),   # 5: NE
    (1, -1),   # 6: SW
    (1, 1),    # 7: SE
)

ACTION_NAMES: tuple[str, ...] = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    crashed: bool
    timed_out: bool
    steps: int
    total_reward: float
    path: list[Node]


class GridRoutingEnv:
    def __init__(
        self,
        grid: GridEnvironment,
        max_steps: int = 100,
        step_penalty: float = -0.1,
        goal_reward: float = 100.0,
        invalid_move_penalty: float = -10.0,
        distance_reward_scale: float = 1.0,
    ) -> None:
        self.grid = grid
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.invalid_move_penalty = invalid_move_penalty
        self.distance_reward_scale = distance_reward_scale
        self.position = grid.start
        self.steps = 0

    @property
    def action_count(self) -> int:
        return len(ACTION_DELTAS)

    def reset(self) -> Node:
        self.position = self.grid.start
        self.steps = 0
        return self.position

    def step(self, action: Action) -> tuple[Node, float, bool, dict[str, bool]]:
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count - 1}]")

        previous = self.position
        d_row, d_col = ACTION_DELTAS[action]
        candidate = (previous[0] + d_row, previous[1] + d_col)
        valid_moves = {neighbor: cost for neighbor, cost in self.grid.get_neighbors(previous)}
        self.steps += 1

        if candidate not in valid_moves:
            done = True
            info = {"success": False, "crashed": True, "timed_out": False}
            return previous, self.invalid_move_penalty, done, info

        self.position = candidate

        if self.position == self.grid.goal:
            info = {"success": True, "crashed": False, "timed_out": False}
            return self.position, self.goal_reward, True, info

        timed_out = self.steps >= self.max_steps
        reward = self._shaped_step_reward(previous, self.position, valid_moves[candidate])
        info = {"success": False, "crashed": False, "timed_out": timed_out}
        return self.position, reward, timed_out, info

    def _shaped_step_reward(self, previous: Node, current: Node, move_cost: float) -> float:
        before = euclidean_distance(previous, self.grid.goal)
        after = euclidean_distance(current, self.grid.goal)
        progress = before - after
        return self.step_penalty - move_cost + self.distance_reward_scale * progress


class QLearningAgent:
    def __init__(
        self,
        actions: int,
        learning_rate: float = 0.2,
        discount: float = 0.95,
        epsilon: float = 1.0,
        min_epsilon: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: int | None = 42,
    ) -> None:
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q_values: dict[Node, list[float]] = {}

    def select_action(self, state: Node, explore: bool = True) -> Action:
        self._ensure_state(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        return self.best_action(state)

    def best_action(self, state: Node) -> Action:
        self._ensure_state(state)
        values = self.q_values[state]
        best_value = max(values)
        best_actions = [action for action, value in enumerate(values) if value == best_value]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: Node,
        action: Action,
        reward: float,
        next_state: Node,
        done: bool,
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_value = self.q_values[state][action]
        future_value = 0.0 if done else max(self.q_values[next_state])
        target = reward + self.discount * future_value
        self.q_values[state][action] = old_value + self.learning_rate * (target - old_value)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _ensure_state(self, state: Node) -> None:
        if state not in self.q_values:
            self.q_values[state] = [0.0 for _ in range(self.actions)]


def train_q_learning(
    env: GridRoutingEnv,
    episodes: int = 5000,
    seed: int | None = 42,
) -> tuple[QLearningAgent, list[EpisodeResult]]:
    agent = QLearningAgent(actions=env.action_count, seed=seed)
    history: list[EpisodeResult] = []

    for _ in range(episodes):
        state = env.reset()
        path = [state]
        total_reward = 0.0
        final_info = {"success": False, "crashed": False, "timed_out": False}

        for _step in range(env.max_steps):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            path.append(state)
            final_info = info

            if done:
                break

        agent.decay_epsilon()
        history.append(
            EpisodeResult(
                success=final_info["success"],
                crashed=final_info["crashed"],
                timed_out=final_info["timed_out"],
                steps=len(path) - 1,
                total_reward=total_reward,
                path=path,
            )
        )

    return agent, history


def evaluate_agent(env: GridRoutingEnv, agent: QLearningAgent, episodes: int = 20) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []

    for _ in range(episodes):
        state = env.reset()
        path = [state]
        total_reward = 0.0
        final_info = {"success": False, "crashed": False, "timed_out": False}

        for _step in range(env.max_steps):
            action = agent.select_action(state, explore=False)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            path.append(state)
            final_info = info

            if done:
                break

        results.append(
            EpisodeResult(
                success=final_info["success"],
                crashed=final_info["crashed"],
                timed_out=final_info["timed_out"],
                steps=len(path) - 1,
                total_reward=total_reward,
                path=path,
            )
        )

    return results


def euclidean_distance(node: Node, goal: Node) -> float:
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

