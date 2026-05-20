#!/usr/bin/env python3
"""在三种预设环境上运行经典算法 +（若已有模型）PPO。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathplan.benchmark.runner import run_all_presets


def main() -> None:
    p = argparse.ArgumentParser(description="Sprint3 五种算法横向对比")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "results",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs" / "models" / "ppo_curriculum.zip",
    )
    p.add_argument("--no-ppo", action="store_true", help="仅跑四种经典算法")
    p.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help="仅跑指定 preset，如 ship_pipe_100",
    )
    args = p.parse_args()

    report = run_all_presets(
        args.output,
        model_path=args.model,
        run_ppo=not args.no_ppo,
        preset_names=args.presets,
    )
    print(f"已写入 {args.output / 'benchmark_results.json'}")
    for name, block in report["environments"].items():
        print(f"\n=== {name} ===")
        for r in block["results"]:
            alg = r["algorithm"]
            if r.get("skipped"):
                print(f"  {alg}: skipped")
                continue
            ok = r.get("success")
            cost = r.get("path_cost")
            t = r.get("wall_time_sec") or r.get("wall_time_sec_mean")
            print(f"  {alg}: success={ok} cost={cost} time={t}")


if __name__ == "__main__":
    main()
