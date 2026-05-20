#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/results outputs/models
echo ">>> 随机地图池 PPO 训练，日志: outputs/results/ppo_random_train.log"
python scripts/train_ppo_random.py 2>&1 | tee outputs/results/ppo_random_train.log
python scripts/generate_ppo_training_report.py
