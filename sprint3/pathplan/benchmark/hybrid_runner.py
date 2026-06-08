"""Sprint4：A* / PPO / PPO→A* 对比 benchmark。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pathplan.env.grid import GridEnvironment
from pathplan.env.presets import PRESET_NAMES, build_preset
from pathplan.hybrid.ppo_astar_refine import plan_ppo_astar_refine
from pathplan.planners.astar import plan_timed as astar_timed
from pathplan.rl.eval import (
    default_max_episode_steps,
    evaluate_on_grid,
    plan_with_ppo,
)


def _row(name: str, res, env: GridEnvironment) -> dict[str, Any]:
    pvalid = env.path_is_valid(res.path) if res.success else False
    pc = env.compute_path_cost(res.path) if res.success and res.path else float("inf")
    row: dict[str, Any] = {
        "algorithm": name,
        "success": res.success,
        "path_cost": float(pc) if res.success else None,
        "path_length": len(res.path) if res.path else None,
        "path_valid": pvalid,
        "wall_time_sec": float(res.wall_time_sec),
        "expanded": int(res.expanded),
        "extra": res.extra,
    }
    return row


def run_comparison_on_env(
    env: GridEnvironment,
    model_path: Path,
    *,
    window: int = 21,
    eval_episodes: int = 15,
) -> list[dict[str, Any]]:
    model_path = Path(model_path)
    max_steps = default_max_episode_steps(env)
    rows: list[dict[str, Any]] = []

    print("  → A* …", flush=True)
    astar_res = astar_timed(env)
    rows.append(_row("A*", astar_res, env))
    opt_cost = astar_res.path_cost if astar_res.success else None

    if not model_path.is_file():
        rows.append({"algorithm": "PPO", "success": False, "error": "model not found"})
        rows.append({"algorithm": "PPO→A*", "success": False, "error": "model not found"})
        return rows

    print("  → PPO …", flush=True)
    ppo_res = plan_with_ppo(
        env, model_path, window=window, max_episode_steps=max_steps
    )
    ppo_row = _row("PPO", ppo_res, env)
    ppo_row["eval"] = evaluate_on_grid(
        env,
        model_path,
        n_episodes=eval_episodes,
        window=window,
        max_episode_steps=max_steps,
    )
    rows.append(ppo_row)

    print("  → PPO→A* …", flush=True)
    hybrid_res = plan_ppo_astar_refine(
        env,
        model_path,
        window=window,
        max_episode_steps=max_steps,
    )
    hybrid_row = _row("PPO→A*", hybrid_res, env)
    if opt_cost and hybrid_res.success:
        hybrid_row["extra"] = dict(hybrid_row.get("extra", {}))
        hybrid_row["extra"]["cost_vs_optimal_pct"] = (
            (hybrid_res.path_cost / opt_cost - 1.0) * 100.0
        )
    if opt_cost and ppo_res.success:
        ppo_row["extra"] = dict(ppo_row.get("extra", {}))
        ppo_row["extra"]["cost_vs_optimal_pct"] = (
            (ppo_res.path_cost / opt_cost - 1.0) * 100.0
        )
    rows.append(hybrid_row)

    return rows


def run_all_presets(
    output_dir: Path,
    model_path: Path,
    *,
    preset_names: list[str] | None = None,
    eval_episodes: int = 15,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    names = list(preset_names) if preset_names else list(PRESET_NAMES)
    report: dict[str, Any] = {
        "experiment": "Sprint4 PPO→A* Refine (PA-RPP)",
        "presets": names,
        "ppo_model": str(model_path),
        "environments": {},
    }

    for name in names:
        print(f"\n[benchmark] {name}", flush=True)
        env = build_preset(name)
        t0 = time.perf_counter()
        results = run_comparison_on_env(
            env, model_path, eval_episodes=eval_episodes
        )
        elapsed = time.perf_counter() - t0
        report["environments"][name] = {
            "info": {
                "rows": env.rows,
                "cols": env.cols,
                "obstacle_count": env.obstacle_count(),
                "obstacle_ratio": round(env.obstacle_ratio(), 4),
                "start": env.start,
                "goal": env.goal,
                "description": getattr(env, "preset_description", ""),
            },
            "results": results,
            "benchmark_wall_sec": round(elapsed, 3),
        }

    out_json = output_dir / "hybrid_benchmark_results.json"
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
