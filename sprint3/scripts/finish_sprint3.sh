#!/usr/bin/env bash
# Sprint3 收尾：PPO 补强训练 → 三档 benchmark → 更新报告与 README
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=outputs/models/ppo_random.zip
LOG=outputs/results

echo "========== 1/4 PPO 收尾训练 =========="
python scripts/train_ppo_random.py \
  --resume \
  --append-log \
  --curriculum configs/ppo_finish.yaml \
  --log "$LOG/ppo_random_train.json" \
  2>&1 | tee "$LOG/ppo_finish_train.log"

echo "========== 2/4 三档 preset benchmark =========="
python scripts/run_benchmark.py --model "$MODEL" \
  2>&1 | tee "$LOG/benchmark_final.log"

echo "========== 3/4 生成报告 =========="
python scripts/generate_ppo_training_report.py
python scripts/generate_report.py

echo "========== 4/4 完成 =========="
echo "主报告: EXPERIMENT_REPORT.md"
echo "PPO 报告: PPO_TRAINING_REPORT.md"
