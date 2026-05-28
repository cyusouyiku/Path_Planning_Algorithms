"""在固定栅格上评测 /  rollout PPO 路径。"""

from __future__ import annotations

import time
from pathlib import Path

from pathplan.common import PlanResult, octile_heuristic
from pathplan.env.grid import GridEnvironment
from pathplan.rl.gym_env import GridPathfindingGymEnv


def default_max_episode_steps(env: GridEnvironment, *, slack: float = 3.5) -> int:
    """按最优路径长度留足步数预算，避免未到达终点即被截断。"""
    sr, sc = env.start
    gr, gc = env.goal
    ideal = octile_heuristic(sr, sc, gr, gc)
    return int(max(ideal * slack + 60, max(env.rows, env.cols) * 1.5))


def plan_with_ppo(
    env: GridEnvironment,
    model_path: Path,
    window: int = 21,
    max_episode_steps: int | None = None,
) -> PlanResult:
    from stable_baselines3 import PPO

    if max_episode_steps is None:
        max_episode_steps = default_max_episode_steps(env)

    gym_env = GridPathfindingGymEnv(
        env,
        window=window,
        max_episode_steps=max_episode_steps,
    )
    model = PPO.load(str(model_path))
    t0 = time.perf_counter()
    obs, _ = gym_env.reset()
    path = [env.start]
    steps = 0
    success = False

    while steps < max_episode_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _ = gym_env.step(int(action))
        steps += 1
        path.append(gym_env._cur)
        if term and gym_env._cur == env.goal:
            success = True
            break
        if trunc:
            break

    wall = time.perf_counter() - t0
    if success and env.path_is_valid(path):
        cost = env.compute_path_cost(path)
        return PlanResult("PPO", True, path, cost, steps, wall)
    return PlanResult(
        "PPO",
        False,
        path if len(path) > 1 else None,
        float("inf"),
        steps,
        wall,
        extra={"reached_goal": success},
    )


def evaluate_on_grid(
    env: GridEnvironment,
    model_path: Path,
    n_episodes: int = 20,
    window: int = 21,
    max_episode_steps: int | None = None,
) -> dict:
    if max_episode_steps is None:
        max_episode_steps = default_max_episode_steps(env)

    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError("请安装 stable-baselines3") from e

    model = PPO.load(str(model_path))
    successes = 0
    costs: list[float] = []
    steps_ok: list[int] = []

    for ep in range(n_episodes):
        gym_env = GridPathfindingGymEnv(
            env,
            window=window,
            max_episode_steps=max_episode_steps,
        )
        obs, _ = gym_env.reset(seed=ep + 11)
        path = [env.start]
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _ = gym_env.step(int(action))
            steps += 1
            path.append(gym_env._cur)
            done = term or trunc

        if term and gym_env._cur == env.goal and env.path_is_valid(path):
            successes += 1
            costs.append(env.compute_path_cost(path))
            steps_ok.append(steps)

    return {
        "eval_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "mean_path_cost_on_success": float(sum(costs) / len(costs)) if costs else None,
        "mean_steps_on_success": float(sum(steps_ok) / len(steps_ok))
        if steps_ok
        else None,
    }
