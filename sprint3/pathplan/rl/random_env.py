"""每回合从地图池随机抽样的船舶管道 Gym 环境。"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from pathplan.env.grid import GridEnvironment
from pathplan.rl.gym_env import GridPathfindingGymEnv


class PooledShipPipeGymEnv(gym.Env):
    """
    从预生成的 GridEnvironment 池中抽样；每次 reset 换一张图。
    观测 / 动作空间与 GridPathfindingGymEnv 一致。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_pool: list[GridEnvironment],
        *,
        window: int = 21,
        max_episode_steps: int = 700,
        seed: int | None = None,
        **gym_kwargs,
    ):
        if not map_pool:
            raise ValueError("map_pool 不能为空")
        super().__init__()
        self._pool = list(map_pool)
        self._rng = np.random.default_rng(seed)
        self._window = int(window)
        self._max_episode_steps = int(max_episode_steps)
        self._gym_kwargs = gym_kwargs
        self._inner: GridPathfindingGymEnv | None = None
        self._attach(self._pool[0])

    def _attach(self, grid: GridEnvironment) -> None:
        self._inner = GridPathfindingGymEnv(
            grid,
            window=self._window,
            max_episode_steps=self._max_episode_steps,
            **self._gym_kwargs,
        )
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    @property
    def current_grid(self) -> GridEnvironment:
        assert self._inner is not None
        return self._inner._m

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(0, len(self._pool)))
        self._attach(self._pool[idx])
        assert self._inner is not None
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        assert self._inner is not None
        return self._inner.step(action)
