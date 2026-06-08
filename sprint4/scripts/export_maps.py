#!/usr/bin/env python3
"""导出 A* / PPO / PPO→A* 路径对比图。"""

from __future__ import annotations

import sys
from pathlib import Path

S4 = Path(__file__).resolve().parents[1]
S3 = S4.parent / "sprint3"
if str(S3) not in sys.path:
    sys.path.insert(0, str(S3))

from pathplan.env.presets import build_preset
from pathplan.hybrid.ppo_astar_refine import plan_ppo_astar_refine
from pathplan.planners.astar import plan as astar_plan
from pathplan.rl.eval import plan_with_ppo

MODEL = S3 / "outputs" / "models" / "ppo_random.zip"
OUT = S4 / "outputs" / "maps"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for name in ("ship_pipe_100", "ship_pipe_500", "ship_pipe_1000"):
        env = build_preset(name)
        a = astar_plan(env)
        p = plan_with_ppo(env, MODEL)
        h = plan_ppo_astar_refine(env, MODEL)

        env.render(
            a.path,
            title=f"{name} — A* (cost={a.path_cost:.2f})",
            save_path=str(OUT / f"{name}_astar.png"),
        )
        if p.path:
            env.render(
                p.path,
                title=f"{name} — PPO (cost={p.path_cost:.2f}, ok={p.success})",
                save_path=str(OUT / f"{name}_ppo.png"),
            )
        env.render(
            h.path,
            title=f"{name} — PPO→A* (cost={h.path_cost:.2f})",
            save_path=str(OUT / f"{name}_hybrid.png"),
        )
        print(f"已导出 {name} 对比图 → {OUT}")


if __name__ == "__main__":
    main()
