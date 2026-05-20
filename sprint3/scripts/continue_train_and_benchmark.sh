#!/usr/bin/env bash
# 继续 PPO 训练 → 三档 preset benchmark → 更新主实验报告
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/results
mkdir -p "$LOG" outputs/models

echo "=== 1/3 继续随机地图 PPO 训练（resume）==="
python scripts/train_ppo_random.py \
  --resume \
  --append-log \
  --curriculum configs/ppo_random_curriculum_continue.yaml \
  2>&1 | tee "$LOG/ppo_random_continue.log"

echo "=== 2/3 三档 preset benchmark ==="
python scripts/run_benchmark.py \
  --model outputs/models/ppo_random.zip \
  2>&1 | tee "$LOG/benchmark_ppo_random.log"

echo "=== 3/3 生成报告 ==="
python scripts/generate_ppo_training_report.py
python scripts/generate_report.py
echo "完成: EXPERIMENT_REPORT.md + PPO_TRAINING_REPORT.md"
