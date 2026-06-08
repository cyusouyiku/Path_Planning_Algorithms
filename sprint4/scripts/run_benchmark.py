#!/usr/bin/env python3
"""Sprint4：在 Sprint3 三张 preset 上对比 A* / PPO / PPO→A*。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

S4 = Path(__file__).resolve().parents[1]
S3 = S4.parent / "sprint3"
if str(S3) not in sys.path:
    sys.path.insert(0, str(S3))

from pathplan.benchmark.hybrid_runner import run_all_presets  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Sprint4 PA-RPP benchmark")
    p.add_argument(
        "--output",
        type=Path,
        default=S4 / "outputs" / "results",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=S3 / "outputs" / "models" / "ppo_random.zip",
    )
    p.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help="如 ship_pipe_100 ship_pipe_500 ship_pipe_1000",
    )
    p.add_argument("--eval-episodes", type=int, default=15)
    args = p.parse_args()

    report = run_all_presets(
        args.output,
        args.model,
        preset_names=args.presets,
        eval_episodes=args.eval_episodes,
    )
    out = args.output / "hybrid_benchmark_results.json"
    print(f"\n已写入 {out}")
    for name, block in report["environments"].items():
        print(f"\n=== {name} ===")
        for r in block["results"]:
            alg = r["algorithm"]
            if r.get("error"):
                print(f"  {alg}: ERROR {r['error']}")
                continue
            ok = r.get("success")
            cost = r.get("path_cost")
            t = r.get("wall_time_sec")
            extra = r.get("extra") or {}
            vs = extra.get("cost_vs_optimal_pct")
            vs_s = f" vs_opt={vs:+.2f}%" if vs is not None else ""
            print(f"  {alg}: success={ok} cost={cost} time={t:.4f}s{vs_s}")


if __name__ == "__main__":
    main()
