# Sprint 3：PPO 随机地图训练实验报告

> 本报告**单独**记录强化学习训练过程与留出地图上的评测结果。
> 训练完成后，再将同一权重接入主实验 `EXPERIMENT_REPORT.md` 的三张 **固定 preset** 地图做横向对比。

## 1. 训练动机

主 benchmark 的三张图（`ship_pipe_100/500/1000`）为固定 seed，
若在单张图上过拟合，横向对比中的 PPO 成功率会失真。
本实验在**与 preset 同分布**的船舶管道布局上，
用**随机地图池**（每回合 reset 换图）训练策略，并在**未参与训练的留出图**上评测泛化。

## 2. 随机地图与地图池

| 设计 | 说明 |
|------|------|
| 布局 | `ship_pipe`：矩形舱室 + 贴墙，**无主走廊**（`corridor_half_width: 0`） |
| 难度 | `min_path_stretch ≥ 1.12`，拒绝起终点直线可穿 |
| 训练池 | 每阶段预生成 `map_pool_size` 张图，reset 时均匀抽样 |
| 留出集 | 独立 `holdout_base_seed`，种子与训练池不重叠 |
| 观测 | 局部 21×21 障碍 + 归一化相对目标（与主实验一致） |

课程配置：`configs/ppo_random_curriculum.yaml`。

## 3. 课程阶段与训练量

- **总环境步数**：500,000
- **总耗时**：2424.7 s
- **权重**：`/Users/zhangzongyu/Desktop/RL/sprint3/outputs/models/ppo_random.zip`

| 阶段 | 规模 | 训练池 | 留出图 | 环境步数 | 单局步上限 | 耗时 (s) | 池均障碍% |
|------|------|--------|--------|----------|------------|----------|-----------|
| random_96 | 96×96 | 20 | 10 | 80,000 | 280 | 52.3 | 24.6 |
| random_192 | 192×192 | 18 | 8 | 100,000 | 520 | 66.9 | 20.3 |
| random_384 | 384×384 | 14 | 6 | 120,000 | 1100 | 86.8 | 22.0 |
| random_500 | 500×500 | 12 | 6 | 200,000 | 2800 | 155.2 | 25.8 |

## 4. 留出地图评测（训练集外）

下表为各阶段结束（或中间 checkpoint）在**留出地图**上的成功率 
（`eval_episodes_per_map` 回合/图）。

| 阶段 | checkpoint | 留出图数 | 总回合 | 成功率 | 成功时均步数 | 成功时均代价 |
|------|------------|----------|--------|--------|--------------|--------------|
| random_96 | 40000 | 10 | 20 | 60.0% | 143.83333333333334 | 155.77649104842422 |
| random_96 | 80000 | 10 | 20 | 50.0% | 123.8 | 142.9366665816369 |
| random_96 | final | 10 | 20 | 50.0% | 123.8 | 142.9366665816369 |
| random_192 | 50000 | 8 | 16 | 12.5% | 232.0 | 285.0193359837567 |
| random_192 | 100000 | 8 | 16 | 0.0% | — | — |
| random_192 | final | 8 | 16 | 0.0% | — | — |
| random_384 | 60000 | 6 | 12 | 0.0% | — | — |
| random_384 | 120000 | 6 | 12 | 0.0% | — | — |
| random_384 | final | 6 | 12 | 0.0% | — | — |
| random_500 | 100000 | 6 | 12 | 0.0% | — | — |
| random_500 | 200000 | 6 | 12 | 0.0% | — | — |
| random_500 | final | 6 | 12 | 0.0% | — | — |

**最终阶段（random_500）留出成功率：0.0%**
**全程最佳**：random_96 留出成功率 50.0%。

## 5. 结果解读（本批次）

1. **random_96** 留出成功率约 **50%**，说明在相近尺度、同分布随机图上策略已具一定导航能力。
2. 放大到 **192 / 384 / 500** 后留出成功率为 **0%**，常见原因包括：
   - 课程跨度过大，策略未在中间尺度充分巩固即进入大图；
   - 局部 21×21 视野在大图上不足以规划长绕障路径；
   - 单阶段环境步数相对 500×500 最优路径长度仍偏少。
3. **不宜**在未提升成功率前，将当前 `ppo_random.zip` 直接并入主 benchmark 五算法表；
   可先以本报告为 PPO 训练章节，达标后再跑 `run_benchmark.py --model outputs/models/ppo_random.zip`。

**建议下一轮**：延长 random_192/384 步数、在 500 阶段增至 40 万以上步，
或增大 `window` / 减小 `min_path_stretch` 做消融；亦可从 `random_96` 最佳 checkpoint 微调。

## 6. 与主 benchmark 的关系

1. 本训练**未**在 `ship_pipe_100/500/1000` 三张固定 preset 上拟合；
   留出评测仅反映**同分布随机图**上的能力。
2. 当留出成功率稳定（建议 final ≥ 70%）后，再执行：

```bash
python scripts/run_benchmark.py --model outputs/models/ppo_random.zip
python scripts/generate_report.py
```

将 PPO 行并入主实验报告的五算法对比表。

## 6. 复现

```bash
cd sprint3
pip install -r requirements.txt
python scripts/train_ppo_random.py 2>&1 | tee outputs/results/ppo_random_train.log
python scripts/generate_ppo_training_report.py
```

结构化日志：`outputs/results/ppo_random_train.json`

