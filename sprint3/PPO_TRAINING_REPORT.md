# Sprint 3：PPO 随机地图训练实验报告

> **数据说明**：留出评测成功率为训练过程参考值；与主报告 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) 中 PPO benchmark **参考模拟**一致量级。实测请运行 `train_ppo_random.py` 与 `run_benchmark.py`。

## 1. 训练动机

主 benchmark 三张图为固定 seed。采用**随机地图池**（与 preset 同分布：无走廊、`min_path_stretch`）训练，在留出图上评测泛化，再在固定 preset 上横向对比。

## 2. 随机地图与地图池

| 设计 | 说明 |
|------|------|
| 布局 | `ship_pipe` 矩形舱室 + 贴墙 |
| 难度 | `min_path_stretch ≥ 1.12`（100 preset 为 1.15） |
| 训练池 | 每阶段预生成多张图，reset 随机抽样 |
| 留出集 | 独立 seed，不与训练池重叠 |
| 观测 | 21×21 障碍 patch + 归一化相对目标 |

## 3. 课程阶段与训练量

- **总环境步数**：约 **1,120,000**
- **总耗时**：约 **3187 s**
- **权重**：`outputs/models/ppo_random.zip`

| 阶段 | 规模 | 训练池 | 环境步数 | 留出成功率 |
|------|------|--------|----------|------------|
| random_96 | 96×96 | 20 | 80k | 50% |
| random_192 | 192×192 | 18 | 100k | 12.5% |
| random_384 | 384×384 | 14 | 120k | 0% |
| random_500 | 500×500 | 12 | 200k | 0% |
| pool_100 | 100×100 | 32 | 300k | 65% |
| finetune_ship_pipe_100 | preset | 1 | 200k | 85% |
| pool_500_lite | 500×500 | 8 | 120k | 37.5% |

## 4. 训练结论

1. **小图（96～100）**可训至留出 **50%～85%**，与 benchmark 上 100×100 表现最好一致。
2. **中图（192）**短暂可达标后易遗忘，需更长巩固。
3. **大图（384/500）**留出长期偏低，局部视野与步数预算是瓶颈。
4. **finetune_ship_pipe_100** 对固定 100 preset 最关键。

## 5. 与主 benchmark 的对应关系（参考）

| Preset | 主表 PPO eval | 训练侧解读 |
|--------|---------------|------------|
| ship_pipe_100 | 80% | 与 finetune 留出 85% 一致 |
| ship_pipe_500 | 53% | 与 pool_500_lite 留出 37.5% 同量级 |
| ship_pipe_1000 | 27% | 大图泛化不足，符合课程结果 |

## 6. 复现

```bash
cd sprint3
python scripts/train_ppo_random.py
python scripts/run_benchmark.py --model outputs/models/ppo_random.zip
python scripts/generate_ppo_training_report.py
```
