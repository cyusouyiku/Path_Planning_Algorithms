"""在预设环境上运行五种算法并汇总 JSON。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pathplan.env.grid import GridEnvironment
from pathplan.env.presets import PRESET_NAMES, build_preset
from pathplan.planners.astar import plan_timed as astar_timed
from pathplan.planners.dijkstra import plan_timed as dijkstra_timed
from pathplan.planners.dstar_lite import plan_timed as dstar_timed
from pathplan.planners.rrt_star import plan_timed as rrt_timed
from pathplan.rl.eval import evaluate_on_grid, plan_with_ppo


def _result_row(name: str, res, env: GridEnvironment) -> dict[str, Any]:
    pvalid = env.path_is_valid(res.path) if res.success else False
    pc = env.compute_path_cost(res.path) if res.success and res.path else float("inf")
    return {
        "algorithm": name,
        "success": res.success,
        "path_cost": float(pc) if res.success else None,
        "path_length": len(res.path) if res.path else None,
        "path_valid": pvalid,
        "wall_time_sec": float(res.wall_time_sec),
        "expanded": int(res.expanded),
        "extra": res.extra,
    }


def run_classical_on_env(
    env: GridEnvironment,
    *,
    skip_dijkstra_if_large: bool = True,
    skip_dstar_if_large: bool = False,
    rrt_runs: int = 5,
    rrt_seed_base: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    side = max(env.rows, env.cols)
    cells = env.rows * env.cols

    print(f"  → Dijkstra …", flush=True)
    if skip_dijkstra_if_large and side > 600:
        rows.append(
            {
                "algorithm": "Dijkstra",
                "success": None,
                "skipped": True,
                "note": f"地图边长 {side}>600，为控制耗时跳过",
            }
        )
    else:
        rows.append(_result_row("Dijkstra", dijkstra_timed(env), env))

    print(f"  → A* …", flush=True)
    rows.append(_result_row("A*", astar_timed(env), env))

    print(f"  → D* Lite …", flush=True)
    if skip_dstar_if_large and (side > 500 or cells > 300_000):
        rows.append(
            {
                "algorithm": "D* Lite",
                "success": None,
                "skipped": True,
                "note": f"规模 {env.rows}×{env.cols} 下 Python 实现过慢，静态图建议用 A*",
            }
        )
    else:
        rows.append(_result_row("D* Lite", dstar_timed(env), env))

    print(f"  → RRT* …", flush=True)
    rrt_n = 2 if side >= 900 else (3 if side >= 450 else rrt_runs)
    times: list[float] = []
    best: dict | None = None
    for s in range(rrt_n):
        t0 = time.perf_counter()
        res = rrt_timed(env, seed=rrt_seed_base + s)
        times.append(time.perf_counter() - t0)
        if res.success and res.path:
            cst = float(env.compute_path_cost(res.path))
            if best is None or cst < best["path_cost"]:
                best = {
                    "path_cost": cst,
                    "path_length": len(res.path),
                    "path_valid": env.path_is_valid(res.path),
                    "wall_time_sec": res.wall_time_sec,
                    "expanded": res.expanded,
                    "extra": res.extra,
                }

    rows.append(
        {
            "algorithm": "RRT*",
            "success": best is not None,
            "path_cost": best["path_cost"] if best else None,
            "path_length": best["path_length"] if best else None,
            "path_valid": best["path_valid"] if best else False,
            "wall_time_sec_mean": sum(times) / len(times) if times else None,
            "expanded": best["expanded"] if best else None,
            "note": f"{rrt_n} 次运行取最优代价",
            "extra": best["extra"] if best else {},
        }
    )
    return rows


def run_ppo_on_env(
    env: GridEnvironment,
    model_path: Path,
    *,
    window: int = 21,
    eval_episodes: int = 15,
) -> dict[str, Any]:
    model_path = Path(model_path)
    if not model_path.is_file():
        return {"algorithm": "PPO", "success": False, "error": "model not found"}

    plan_res = plan_with_ppo(env, model_path, window=window)
    eval_stats = evaluate_on_grid(
        env, model_path, n_episodes=eval_episodes, window=window
    )
    row = _result_row("PPO", plan_res, env)
    row["eval"] = eval_stats
    row["note"] = "单次 rollout 路径 + 多回合成功率统计"
    return row


def run_all_presets(
    output_dir: Path,
    model_path: Path | None = None,
    *,
    run_ppo: bool = True,
    preset_names: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path or (output_dir.parent / "models" / "ppo_curriculum.zip")

    names = list(preset_names) if preset_names else list(PRESET_NAMES)
    for n in names:
        if n not in PRESET_NAMES:
            raise KeyError(f"未知 preset: {n}，可选 {list(PRESET_NAMES)}")

    report: dict[str, Any] = {
        "presets": names,
        "ppo_model": str(model_path) if run_ppo else None,
        "environments": {},
    }

    for name in names:
        print(f"\n[benchmark] 环境 {name}", flush=True)
        env = build_preset(name)
        env_info = {
            "rows": env.rows,
            "cols": env.cols,
            "obstacle_count": env.obstacle_count(),
            "obstacle_ratio": round(env.obstacle_ratio(), 4),
            "start": env.start,
            "goal": env.goal,
            "description": getattr(env, "preset_description", ""),
        }
        classical = run_classical_on_env(env)
        ppo_row = None
        if run_ppo and Path(model_path).is_file():
            ppo_row = run_ppo_on_env(env, model_path)

        report["environments"][name] = {
            "info": env_info,
            "results": classical + ([ppo_row] if ppo_row else []),
        }

    out_json = output_dir / "benchmark_results.json"
    if preset_names and out_json.is_file():
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8"))
            merged_envs = dict(prev.get("environments", {}))
            merged_envs.update(report["environments"])
            report["environments"] = merged_envs
            report["presets"] = list(merged_envs.keys())
        except (json.JSONDecodeError, OSError):
            pass
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
