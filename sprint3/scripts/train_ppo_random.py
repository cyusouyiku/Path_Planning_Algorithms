#!/usr/bin/env python3
"""在随机船舶管道地图池上训练 PPO（独立于主 benchmark 三张固定图）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathplan.rl.train_random import train_random_curriculum


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs" / "models" / "ppo_random.zip",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=ROOT / "outputs" / "results" / "ppo_random_train.json",
    )
    p.add_argument(
        "--curriculum",
        type=Path,
        default=ROOT / "configs" / "ppo_random_curriculum.yaml",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--append-log",
        action="store_true",
        help="训练日志 JSON 与上一轮合并",
    )
    args = p.parse_args()

    print(f"[PPO-Random] 课程: {args.curriculum.name}", flush=True)
    print(f"[PPO-Random] 模型: {args.model.resolve()}", flush=True)
    print(f"[PPO-Random] 日志: {args.log.resolve()}", flush=True)

    summary = train_random_curriculum(
        args.model,
        log_path=args.log,
        curriculum_path=args.curriculum,
        resume=args.resume,
        append_log=args.append_log,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
