#!/usr/bin/env python3
"""一键：PPO 课程训练 → 五种算法 benchmark → 生成实验报告。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true", help="跳过 PPO 训练，仅 benchmark+报告")
    p.add_argument("--skip-report", action="store_true")
    args = p.parse_args()

    py = sys.executable
    model = ROOT / "outputs" / "models" / "ppo_curriculum.zip"

    if not args.skip_train:
        _run([py, str(ROOT / "scripts" / "train_ppo.py")])

    _run(
        [
            py,
            str(ROOT / "scripts" / "run_benchmark.py"),
            "--model",
            str(model),
        ]
    )

    if not args.skip_report:
        _run([py, str(ROOT / "scripts" / "generate_report.py")])


if __name__ == "__main__":
    main()
