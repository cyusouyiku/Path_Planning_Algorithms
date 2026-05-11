#!/usr/bin/env python3
"""
在默认 GridEnvironment 上对比 Dijkstra / A* / RRT* / D* Lite / PPO 的路径代价、耗时与扩展量。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from env import GridEnvironment

ROOT = Path(__file__).resolve().parent
RESULT_JSON = ROOT / "experiment_results.json"


def _run_classical(seed: int = 0) -> list[dict]:
    from astar import plan_timed as astar_t
    from dijkstra import plan_timed as dijk_t
    from dstar_lite import plan_timed as dstar_t
    from rrt_star import plan_timed as rrt_t

    env = GridEnvironment(seed=seed)
    rows: list[dict] = []

    def add(name: str, res):
        pvalid = env.path_is_valid(res.path) if res.success else False
        pc = env.compute_path_cost(res.path) if res.success else float("inf")
        rows.append(
            {
                "algorithm": name,
                "success": res.success,
                "path_cost": float(pc) if res.success else None,
                "path_length": len(res.path) if res.path else None,
                "path_valid": pvalid,
                "wall_time_sec": float(res.wall_time_sec),
                "expanded": int(res.expanded),
                "extra": res.extra,
            }
        )

    e = env
    r = dijk_t(e)
    add("Dijkstra", r)

    e = GridEnvironment(seed=seed)
    r = astar_t(e)
    add("A*", r)

    e = GridEnvironment(seed=seed)
    r = dstar_t(e)
    add("D* Lite", r)

    # RRT*：多次随机采样，报告最优代价与平均运行时间
    times: list[float] = []
    best: dict | None = None
    for s in range(8):
        e = GridEnvironment(seed=seed)
        t0 = time.perf_counter()
        res = rrt_t(e, max_iter=60_000, seed=seed * 100 + s)
        times.append(time.perf_counter() - t0)
        if res.success and res.path:
            cst = float(e.compute_path_cost(res.path))
            if best is None or cst < best["path_cost"]:
                best = {
                    "path_cost": cst,
                    "path_length": len(res.path),
                    "path_valid": e.path_is_valid(res.path),
                    "extra": res.extra,
                }

    rrt_row: dict = {
        "algorithm": "RRT*",
        "success": best is not None,
        "path_cost": best["path_cost"] if best else None,
        "path_length": best["path_length"] if best else None,
        "path_valid": best["path_valid"] if best else False,
        "wall_time_sec_mean_8seeds": sum(times) / len(times),
        "note": "代价为 8 次运行中的最优值；时间为 8 次均值",
        "best_of_8": best,
    }
    if best and "extra" in best:
        rrt_row["extra"] = best["extra"]
    rows.append(rrt_row)

    return rows


def _run_ppo(seed: int = 0) -> dict | None:
    try:
        from ppo_runner import train_and_eval
    except ImportError:
        return None
    t0 = time.perf_counter()
    stats = train_and_eval(total_timesteps=260_000, seed=seed)
    stats["train_wall_sec"] = time.perf_counter() - t0
    return stats


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    classical = _run_classical(seed=seed)
    ppo_stats = _run_ppo(seed=seed)

    out = {
        "seed": seed,
        "grid": "100x100 默认障碍",
        "classical": classical,
        "ppo": ppo_stats,
    }
    RESULT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
