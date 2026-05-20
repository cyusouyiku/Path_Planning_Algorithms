"""在随机船舶管道地图池上训练 PPO，并定期在留出地图上评测。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pathplan.common import octile_heuristic
from pathplan.env.grid import EnvBuildConfig, GridEnvironment
from pathplan.env.presets import _ship_params
from pathplan.rl.random_env import PooledShipPipeGymEnv


def _grid_from_stage(stage: dict[str, Any], seed: int) -> GridEnvironment:
    side = int(stage["rows"])
    cfg = EnvBuildConfig(
        rows=side,
        cols=int(stage.get("cols", side)),
        seed=int(seed),
        layout="ship_pipe",
        safe_radius=int(stage.get("safe_radius", max(3, side // 25))),
        ship_pipe=_ship_params(stage.get("ship_pipe")),
    )
    return GridEnvironment(cfg)


def build_map_pool(
    stage: dict[str, Any],
    *,
    pool_size: int,
    base_seed: int,
    seed_stride: int = 9973,
) -> list[GridEnvironment]:
    out: list[GridEnvironment] = []
    for i in range(pool_size):
        g = _grid_from_stage(stage, base_seed + i * seed_stride)
        out.append(g)
    return out


def _max_steps_for_grid(grid: GridEnvironment, stage: dict[str, Any]) -> int:
    if "max_episode_steps" in stage:
        return int(stage["max_episode_steps"])
    sr, sc = grid.start
    gr, gc = grid.goal
    ideal = octile_heuristic(sr, sc, gr, gc)
    return int(max(ideal * 2.8 + 80, max(grid.rows, grid.cols) * 2))


def eval_on_grids(
    model_path: Path,
    grids: list[GridEnvironment],
    *,
    window: int,
    max_episode_steps: int | None = None,
    episodes_per_map: int = 3,
) -> dict[str, Any]:
    """在若干固定图上评测（每图多回合）。"""
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path))
    total = len(grids) * episodes_per_map
    successes = 0
    costs: list[float] = []
    steps_ok: list[int] = []

    for gi, grid in enumerate(grids):
        cap = max_episode_steps or _max_steps_for_grid(grid, {})
        for ep in range(episodes_per_map):
            from pathplan.rl.gym_env import GridPathfindingGymEnv

            gym_env = GridPathfindingGymEnv(
                grid, window=window, max_episode_steps=cap
            )
            obs, _ = gym_env.reset(seed=gi * 1000 + ep + 7)
            path = [grid.start]
            term = trunc = False
            steps = 0
            while not (term or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, term, trunc, _ = gym_env.step(int(action))
                steps += 1
                path.append(gym_env._cur)

            if term and gym_env._cur == grid.goal and grid.path_is_valid(path):
                successes += 1
                costs.append(grid.compute_path_cost(path))
                steps_ok.append(steps)

    return {
        "maps": len(grids),
        "episodes_per_map": episodes_per_map,
        "total_episodes": total,
        "success_rate": successes / total if total else 0.0,
        "mean_path_cost_on_success": float(np.mean(costs)) if costs else None,
        "mean_steps_on_success": float(np.mean(steps_ok)) if steps_ok else None,
        "successes": successes,
    }


def train_random_curriculum(
    model_path: Path,
    log_path: Path | None = None,
    *,
    curriculum_path: Path,
    resume: bool = False,
    append_log: bool = False,
) -> dict[str, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise ImportError("请安装: pip install -r requirements.txt") from e

    with Path(curriculum_path).open(encoding="utf-8") as f:
        cfg_doc = yaml.safe_load(f)
    stages = list(cfg_doc["stages"])
    holdout_base = int(cfg_doc.get("holdout_base_seed", 9_000_000))
    holdout_stride = int(cfg_doc.get("holdout_seed_stride", 49999))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    vec = None
    t_all = time.perf_counter()
    stage_logs: list[dict[str, Any]] = []
    eval_logs: list[dict[str, Any]] = []

    if resume and model_path.is_file():
        print(f"[PPO-Random] 继续训练: {model_path.resolve()}", flush=True)

    try:
        for i, stage in enumerate(stages):
            name = stage.get("name", f"stage_{i}")
            window = int(stage.get("window", 21))
            timesteps = int(stage["timesteps"])
            pool_size = int(stage.get("map_pool_size", 20))
            train_base = int(stage.get("pool_base_seed", 1000 + i * 50000))
            max_steps = int(
                stage.get(
                    "max_episode_steps",
                    _max_steps_for_grid(
                        _grid_from_stage(stage, train_base), stage
                    ),
                )
            )

            print(f"\n[PPO-Random] 生成训练地图池 {pool_size} 张 …", flush=True)
            t_pool = time.perf_counter()
            train_pool = build_map_pool(
                stage, pool_size=pool_size, base_seed=train_base
            )
            print(
                f"  完成，耗时 {time.perf_counter() - t_pool:.1f}s | "
                f"示例障碍占比 {train_pool[0].obstacle_ratio()*100:.1f}%",
                flush=True,
            )

            holdout_size = int(stage.get("holdout_maps", 8))
            holdout = build_map_pool(
                stage,
                pool_size=holdout_size,
                base_seed=holdout_base + i * holdout_stride,
            )

            def _make(
                _pool=train_pool,
                _w=window,
                _m=max_steps,
                _si=i,
            ):
                return PooledShipPipeGymEnv(
                    _pool, window=_w, max_episode_steps=_m, seed=_si
                )

            if vec is not None:
                vec.close()
            vec = make_vec_env(_make, n_envs=1, seed=stage.get("seed", i))

            side = int(stage["rows"])
            print(
                f"\n{'=' * 60}\n"
                f"[PPO-Random] 阶段 {i + 1}/{len(stages)} | {name}\n"
                f"  规模: {side}×{side} | 训练池: {pool_size} | 留出评测: {holdout_size}\n"
                f"  train steps: {timesteps:,} | max_episode_steps: {max_steps}\n"
                f"{'=' * 60}",
                flush=True,
            )

            t0 = time.perf_counter()
            if model is None and resume and model_path.is_file():
                model = PPO.load(str(model_path), env=vec)
                model.verbose = 1
            elif model is None:
                model = PPO(
                    "MlpPolicy",
                    vec,
                    verbose=1,
                    seed=int(stage.get("seed", 0)),
                    n_steps=2048,
                    batch_size=256,
                    learning_rate=3e-4,
                )
            else:
                model.set_env(vec)

            eval_every = int(stage.get("eval_every", 0))
            if eval_every > 0 and timesteps >= eval_every:
                n_check = timesteps // eval_every
                for k in range(n_check):
                    chunk = eval_every if k < n_check - 1 else (
                        timesteps - eval_every * (n_check - 1)
                    )
                    try:
                        model.learn(
                            total_timesteps=chunk,
                            progress_bar=True,
                            reset_num_timesteps=False,
                        )
                    except ImportError:
                        model.learn(
                            total_timesteps=chunk,
                            progress_bar=False,
                            reset_num_timesteps=False,
                        )
                    model.save(str(model_path))
                    ev = eval_on_grids(
                        model_path,
                        holdout,
                        window=window,
                        max_episode_steps=max_steps,
                        episodes_per_map=int(stage.get("eval_episodes_per_map", 2)),
                    )
                    ev_entry = {
                        "stage": name,
                        "checkpoint_timesteps": (k + 1) * eval_every,
                        **ev,
                    }
                    eval_logs.append(ev_entry)
                    print(
                        f"  [eval @ {(k+1)*eval_every:,}] "
                        f"success={ev['success_rate']*100:.1f}% "
                        f"({ev['successes']}/{ev['total_episodes']})",
                        flush=True,
                    )
            else:
                try:
                    model.learn(
                        total_timesteps=timesteps,
                        progress_bar=True,
                        reset_num_timesteps=False,
                    )
                except ImportError:
                    model.learn(
                        total_timesteps=timesteps,
                        progress_bar=False,
                        reset_num_timesteps=False,
                    )

            dt = time.perf_counter() - t0
            model.save(str(model_path))
            final_ev = eval_on_grids(
                model_path,
                holdout,
                window=window,
                max_episode_steps=max_steps,
                episodes_per_map=int(stage.get("eval_episodes_per_map", 3)),
            )
            eval_logs.append({"stage": name, "checkpoint": "final", **final_ev})
            print(
                f"  [eval final] success={final_ev['success_rate']*100:.1f}%",
                flush=True,
            )

            stage_logs.append(
                {
                    "stage": name,
                    "grid": f"{side}×{side}",
                    "map_pool_size": pool_size,
                    "holdout_maps": holdout_size,
                    "mean_pool_obstacle_ratio": float(
                        np.mean([g.obstacle_ratio() for g in train_pool])
                    ),
                    "timesteps": timesteps,
                    "max_episode_steps": max_steps,
                    "wall_time_sec": dt,
                    "holdout_eval": final_ev,
                }
            )
            print(f"[PPO-Random] 阶段 {name} 完成，耗时 {dt:.1f}s", flush=True)

    finally:
        if vec is not None:
            vec.close()

    model.save(str(model_path))

    summary = {
        "model_path": str(model_path),
        "curriculum": str(curriculum_path),
        "mode": "random_map_pool",
        "total_wall_time_sec": time.perf_counter() - t_all,
        "total_timesteps": sum(s["timesteps"] for s in stage_logs),
        "stages": stage_logs,
        "eval_checkpoints": eval_logs,
    }
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if append_log and log_path.is_file():
            try:
                prev = json.loads(log_path.read_text(encoding="utf-8"))
                prev_stages = list(prev.get("stages", []))
                prev_evals = list(prev.get("eval_checkpoints", []))
                summary["stages"] = prev_stages + summary["stages"]
                summary["eval_checkpoints"] = prev_evals + summary["eval_checkpoints"]
                summary["total_timesteps"] = sum(
                    s["timesteps"] for s in summary["stages"]
                )
                summary["continued_from"] = prev.get("model_path")
            except (json.JSONDecodeError, OSError):
                pass
        log_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary
