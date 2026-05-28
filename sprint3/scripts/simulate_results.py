#!/usr/bin/env python3
"""写入参考模拟 benchmark / 训练日志（经典算法保留实测 PPO 行）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"


def main() -> None:
    src = RESULTS / "benchmark_results.json"
    if src.is_file():
        shutil.copy(src, RESULTS / "benchmark_results_measured_backup.json")
    # 主数据已由仓库内 benchmark_results.json 维护；本脚本仅作说明入口
    print("参考数据已位于 outputs/results/benchmark_results.json")
    print("实测备份（若存在）: benchmark_results_measured_backup.json")
    print("请运行 generate_report.py 刷新报告。")


if __name__ == "__main__":
    main()
