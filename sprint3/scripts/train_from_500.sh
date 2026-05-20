#!/usr/bin/env bash
# 从 target_500 起跑完整课程后半段（500 + 1000），适合已跑完小图阶段
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=outputs/results
mkdir -p "$LOG_DIR" outputs/models

echo ">>> 从 target_500 起训练 500+1000，日志: $LOG_DIR/ppo_train_500_1000.log"
python scripts/train_ppo.py \
  --from-stage target_500 \
  --resume \
  --append-log \
  2>&1 | tee "$LOG_DIR/ppo_train_500_1000.log"
