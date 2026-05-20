"""PPO 课程学习训练。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from pathplan.env.grid import EnvBuildConfig, GridEnvironment
from pathplan.env.presets import build_preset, _ship_params
from pathplan.rl.gym_env import GridPathfindingGymEnv

_CURRICULUM_PATH = Path(__file__).resolve().parents[2] / "configs" / "curriculum.yaml"


def _load_curriculum() -> list[dict[str, Any]]:
    with _CURRICULUM_PATH.open(encoding="utf-8") as f:
        return list(yaml.safe_load(f)["stages"])


def _filter_stages(
    stages: list[dict[str, Any]],
    *,
    from_stage: str | None = None,
    only_stages: list[str] | None = None,
) -> list[dict[str, Any]]:
    if only_stages:
        want = set(only_stages)
        out = [s for s in stages if s.get("name") in want]
        missing = want - {s.get("name") for s in out}
        if missing:
            raise KeyError(f"课程中无阶段: {sorted(missing)}")
        return out
    if from_stage:
        for i, s in enumerate(stages):
            if s.get("name") == from_stage:
                return stages[i:]
        raise KeyError(f"未找到起始阶段: {from_stage}")
    return stages


def _grid_from_stage(stage: dict[str, Any]) -> GridEnvironment:
    if "preset" in stage:
        return build_preset(stage["preset"])
    layout = stage.get("layout", "ship_pipe")
    if layout == "ship_pipe":
        cfg = EnvBuildConfig(
            rows=int(stage["rows"]),
            cols=int(stage["cols"]),
            seed=int(stage.get("seed", 0)),
            layout="ship_pipe",
            force_no_obstacles=not stage.get("obstacles", True),
            ship_pipe=_ship_params(stage.get("ship_pipe")),
        )
    else:
        ratio = stage.get("obstacle_target_ratio")
        cfg = EnvBuildConfig(
            rows=int(stage["rows"]),
            cols=int(stage["cols"]),
            seed=int(stage.get("seed", 0)),
            layout="scatter",
            force_no_obstacles=not stage.get("obstacles", True),
            obstacle_target_ratio=(ratio[0], ratio[1]) if ratio else (0.0, 0.0),
        )
    return GridEnvironment(cfg)


def train_curriculum(
    model_path: Path,
    log_path: Path | None = None,
    *,
    curriculum_path: Path | None = None,
    from_stage: str | None = None,
    only_stages: list[str] | None = None,
    resume: bool = False,
    append_log: bool = False,
) -> dict[str, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise ImportError("请安装: pip install -r requirements.txt") from e

    cfg_path = curriculum_path or _CURRICULUM_PATH
    with cfg_path.open(encoding="utf-8") as f:
        all_stages = list(yaml.safe_load(f)["stages"])
    stages = _filter_stages(
        all_stages, from_stage=from_stage, only_stages=only_stages
    )
    if not stages:
        raise ValueError("过滤后无训练阶段")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = []
    if append_log and log_path and Path(log_path).is_file():
        try:
            prev = json.loads(Path(log_path).read_text(encoding="utf-8"))
            log = list(prev.get("stages", []))
        except (json.JSONDecodeError, OSError):
            pass

    model = None
    vec = None
    t_all = time.perf_counter()
    n_stages = len(stages)

    if resume and model_path.is_file():
        print(f"[PPO] 从已有权重继续: {model_path.resolve()}", flush=True)

    try:
        for i, stage in enumerate(stages):
            grid = _grid_from_stage(stage)
            window = int(stage.get("window", 21))
            max_steps = int(stage.get("max_episode_steps", 700))
            timesteps = int(stage["timesteps"])

            def _make(_g=grid, _w=window, _m=max_steps):
                return GridPathfindingGymEnv(_g, window=_w, max_episode_steps=_m)

            if vec is not None:
                vec.close()
            vec = make_vec_env(_make, n_envs=1, seed=stage.get("seed", i))

            name = stage.get("name", f"stage_{i}")
            preset = stage.get("preset", "")
            print(
                f"\n{'=' * 60}\n"
                f"[PPO] 阶段 {i + 1}/{n_stages} | {name}"
                f"{f' ({preset})' if preset else ''}\n"
                f"  地图: {grid.rows}×{grid.cols} | 障碍占比: {grid.obstacle_ratio()*100:.1f}%\n"
                f"  本阶段步数: {timesteps:,} | max_episode_steps: {max_steps}\n"
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
            entry = {
                "stage": stage.get("name", f"stage_{i}"),
                "preset": preset or None,
                "grid": f"{grid.rows}×{grid.cols}",
                "obstacle_ratio": grid.obstacle_ratio(),
                "timesteps": timesteps,
                "wall_time_sec": dt,
            }
            # 同阶段重跑时覆盖旧记录
            log = [e for e in log if e.get("stage") != entry["stage"]]
            log.append(entry)
            print(
                f"[PPO] 阶段 {name} 完成 | 耗时 {dt:.1f}s | "
                f"累计 {sum(e['timesteps'] for e in log):,} train steps",
                flush=True,
            )
    finally:
        if vec is not None:
            vec.close()

    model.save(str(model_path))
    summary = {
        "model_path": str(model_path),
        "total_wall_time_sec": time.perf_counter() - t_all,
        "stages": log,
        "total_timesteps": sum(s["timesteps"] for s in log),
    }
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary
