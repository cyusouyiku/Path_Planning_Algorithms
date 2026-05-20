#!/usr/bin/env python3
"""PPO 课程学习训练（无障碍小图 → 500×500 → 1000×1000）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathplan.rl.train import train_curriculum


def main() -> None:
    p = argparse.ArgumentParser(description="PPO 课程训练，终端 verbose=1 显示进度")
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs" / "models" / "ppo_curriculum.zip",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=ROOT / "outputs" / "results" / "ppo_curriculum_log.json",
    )
    p.add_argument(
        "--curriculum",
        type=Path,
        default=None,
        help="课程 YAML（默认 curriculum.yaml）",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="使用 curriculum_fast.yaml（含 500/1000 快速阶段）",
    )
    p.add_argument(
        "--only-500",
        action="store_true",
        help="仅训练 ship_pipe_500（configs/curriculum_500_only.yaml）",
    )
    p.add_argument(
        "--only-1000",
        action="store_true",
        help="仅训练 ship_pipe_1000（configs/curriculum_1000_only.yaml）",
    )
    p.add_argument(
        "--from-stage",
        type=str,
        default=None,
        help="从指定阶段名开始（含该阶段），如 target_500",
    )
    p.add_argument(
        "--only-stages",
        type=str,
        default=None,
        help="逗号分隔，仅跑这些阶段，如 target_500,target_1000",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="若 --model 已存在则加载后继续训练",
    )
    p.add_argument(
        "--append-log",
        action="store_true",
        help="将本 run 的阶段写入 --log 时与已有 JSON 合并",
    )
    args = p.parse_args()

    if args.only_500 and args.only_1000:
        p.error("不能同时指定 --only-500 与 --only-1000")

    if args.only_500:
        curriculum = ROOT / "configs" / "curriculum_500_only.yaml"
        if args.log == ROOT / "outputs" / "results" / "ppo_curriculum_log.json":
            args.log = ROOT / "outputs" / "results" / "ppo_train_500.json"
    elif args.only_1000:
        curriculum = ROOT / "configs" / "curriculum_1000_only.yaml"
        if args.log == ROOT / "outputs" / "results" / "ppo_curriculum_log.json":
            args.log = ROOT / "outputs" / "results" / "ppo_train_1000.json"
    elif args.curriculum:
        curriculum = args.curriculum
    elif args.fast:
        curriculum = ROOT / "configs" / "curriculum_fast.yaml"
    else:
        curriculum = ROOT / "configs" / "curriculum.yaml"

    only_stages = None
    if args.only_stages:
        only_stages = [s.strip() for s in args.only_stages.split(",") if s.strip()]

    print(f"[PPO] 课程文件: {curriculum.name}", flush=True)
    print(f"[PPO] 模型输出: {args.model.resolve()}", flush=True)
    print(f"[PPO] 训练日志: {args.log.resolve()}", flush=True)

    summary = train_curriculum(
        args.model,
        log_path=args.log,
        curriculum_path=curriculum,
        from_stage=args.from_stage,
        only_stages=only_stages,
        resume=args.resume,
        append_log=args.append_log,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
