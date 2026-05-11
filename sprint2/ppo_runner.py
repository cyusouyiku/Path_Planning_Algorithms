"""PPO 训练与评测（Stable-Baselines3）。"""

from __future__ import annotations

import os
from pathlib import Path

from env import GridEnvironment
from rl_env import GridPathfindingGymEnv

def train_and_eval(
    total_timesteps: int = 200_000,
    model_path: str | Path | None = None,
    seed: int = 0,
) -> dict:
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError(
            "需要安装 stable-baselines3 与 torch，请执行: pip install -r requirements.txt"
        ) from e

    base = Path(__file__).resolve().parent
    out = model_path or (base / "ppo_model.zip")
    out = Path(out)

    env_f = lambda: GridPathfindingGymEnv(GridEnvironment(seed=seed))
    vec = None
    try:
        from stable_baselines3.common.env_util import make_vec_env

        vec = make_vec_env(env_f, n_envs=1, seed=seed)
        model = PPO(
            "MlpPolicy",
            vec,
            verbose=0,
            seed=seed,
            tensorboard_log=None,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
        )
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model.save(str(out))
    finally:
        if vec is not None:
            vec.close()

    # 评测若干回合
    eval_env = GridPathfindingGymEnv(GridEnvironment(seed=seed))
    model = PPO.load(str(out))
    n_ep = 30
    successes = 0
    steps_ok: list[int] = []

    for ep in range(n_ep):
        obs, _ = eval_env.reset(seed=seed + ep + 7)
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _ = eval_env.step(int(action))
            steps += 1
            done = term or trunc
        if term and eval_env._cur == eval_env._m.goal:
            successes += 1
            steps_ok.append(steps)

    return {
        "model_path": str(out),
        "eval_episodes": n_ep,
        "success_rate": successes / n_ep,
        "mean_steps_on_success": float(sum(steps_ok) / len(steps_ok))
        if steps_ok
        else float("nan"),
        "total_timesteps": total_timesteps,
    }


if __name__ == "__main__":
    r = train_and_eval()
    print(r)
