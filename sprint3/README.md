# Sprint 3：大规模栅格路径规划对比

在 Sprint 2 五种算法基础上，采用 **船舶管道模拟环境**（大矩形舱室 + 贴墙障碍，**无空旷主走廊**），在 **100 / 500 / 1000** 三种尺度上做经典规划与 PPO 的横向实验。

## 环境设计（船舶管道）

| 规则 | 说明 |
|------|------|
| 大矩形块 | 随机放置舱室/设备矩形障碍 |
| 贴墙加密 | 外轮廓舱壁、矩形块周边加厚 |
| 主通道 | **不设空旷走廊**（`corridor_half_width: 0`），舱室可阻断起终点连线 |
| 难度 | `min_path_stretch` 保证最短路径须绕障 |
| 移动模型 | 八连通，直行 1、对角 √2，不切角 |

配置见 `configs/env_presets.yaml`，生成逻辑见 `pathplan/env/ship_pipe.py`。

## 目录结构

```
sprint3/
├── EXPERIMENT_REPORT.md      # 主实验报告（五算法 + 三尺度）
├── PPO_TRAINING_REPORT.md    # PPO 随机地图池训练专项
├── configs/
│   ├── env_presets.yaml
│   ├── ppo_random_curriculum.yaml
│   └── ppo_finish.yaml       # 收尾训练
├── pathplan/
├── scripts/
│   ├── finish_sprint3.sh     # 一键收尾
│   ├── train_ppo_random.py
│   └── run_benchmark.py
└── outputs/
```

## 三种对比环境

| Preset | 规模 |
|--------|------|
| `ship_pipe_100` | 100×100 |
| `ship_pipe_500` | 500×500 |
| `ship_pipe_1000` | 1000×1000 |

示意图：`python scripts/export_env_maps.py --with-astar`

## 快速开始

```bash
cd sprint3
pip install -r requirements.txt

# 经典 + PPO 完整流水线（收尾）
./scripts/finish_sprint3.sh

# 或分步
python scripts/run_benchmark.py --model outputs/models/ppo_random.zip
python scripts/generate_report.py
```

## 与 Sprint 2 的关系

- 五种算法自 `sprint2/` 移植；环境由撒点改为 **船舶管道布局**。
- 对比尺度：**100 / 500 / 1000**。
