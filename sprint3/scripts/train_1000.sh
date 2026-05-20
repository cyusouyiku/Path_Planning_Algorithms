#!/usr/bin/env bash
# 1000×1000 单阶段训练：终端实时进度 + 文本日志
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=outputs/results
mkdir -p "$LOG_DIR" outputs/models

echo ">>> 1000×1000 训练开始，日志: $LOG_DIR/ppo_train_1000.log"
python scripts/train_ppo.py \
  --only-1000 \
  --resume \
  --append-log \
  2>&1 | tee "$LOG_DIR/ppo_train_1000.log"
