"""Gymnasium 栅格路径环境（局部窗口 + 相对目标）。"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pathplan.common import octile_heuristic
from pathplan.env.grid import GridEnvironment

_ACTION_DELTAS: list[tuple[int, int]] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


class GridPathfindingGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env: GridEnvironment,
        window: int = 21,
        max_episode_steps: int = 700,
        step_penalty: float = 0.015,
        goal_reward: float = 150.0,
        hit_wall_penalty: float = 0.35,
        distance_shaping_scale: float = 0.8,
    ):
        super().__init__()
        self._m = env
        self._window = int(window)
        if self._window % 2 == 0:
            raise ValueError("window 必须为奇数")
        self._max_episode_steps = max_episode_steps
        self._step_penalty = step_penalty
        self._goal_reward = goal_reward
        self._hit_wall_penalty = hit_wall_penalty
        self._distance_shaping_scale = distance_shaping_scale

        patch_sz = self._window * self._window
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(patch_sz + 4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(8)
        self._cur = self._m.start
        self._steps = 0

    def _obs(self) -> np.ndarray:
        r, c = self._cur
        gr, gc = self._m.goal
        half = self._window // 2
        patch = np.zeros((self._window, self._window), dtype=np.float32)
        for i in range(self._window):
            for j in range(self._window):
                rr, cc = r + i - half, c + j - half
                if self._m.in_bounds(rr, cc):
                    patch[i, j] = 1.0 if self._m.is_obstacle(rr, cc) else 0.0
        rel = np.array(
            [
                (gr - r) / max(self._m.rows - 1, 1),
                (gc - c) / max(self._m.cols - 1, 1),
                r / max(self._m.rows - 1, 1),
                c / max(self._m.cols - 1, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([patch.ravel(), rel], axis=0)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._cur = self._m.start
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        self._steps += 1
        dr, dc = _ACTION_DELTAS[int(action)]
        r, c = self._cur
        nr, nc = r + dr, c + dc
        gr, gc = self._m.goal
        old_h = octile_heuristic(r, c, gr, gc)

        if self._m.edge_cost(r, c, nr, nc) is None:
            reward = -self._hit_wall_penalty - self._step_penalty
            terminated = False
        else:
            self._cur = (nr, nc)
            new_h = octile_heuristic(self._cur[0], self._cur[1], gr, gc)
            reward = (
                self._distance_shaping_scale * (old_h - new_h) - self._step_penalty
            )
            terminated = False
            if self._cur == self._m.goal:
                reward += self._goal_reward
                terminated = True

        truncated = self._steps >= self._max_episode_steps
        if truncated:
            reward -= 2.0
        return self._obs(), reward, terminated, truncated, {}
