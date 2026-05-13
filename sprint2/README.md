# Sprint 2 项目说明

本目录是在统一 **100×100 栅格路径规划环境**上，对比多种经典规划算法与 **PPO 强化学习** 的实验代码与结果。

---

## 目录结构（逻辑视图）

```text
sprint2/
├── README.md                 # 本文件：结构与文件职责说明
├── requirements.txt          # Python 依赖（numpy、matplotlib、gymnasium、torch、SB3 等）
│
├── env.py                    # 栅格环境：障碍、八邻域、起终点、可视化
├── common.py                 # 公共类型与启发函数（如八距离）
│
├── dijkstra.py               # Dijkstra 最短路
├── astar.py                  # A* 最短路
├── dstar_lite.py             # D* Lite（静态首次规划实现）
├── rrt_star.py               # 离散栅格上的 RRT* 近似实现
│
├── rl_env.py                 # Gymnasium 封装：局部观测 + 八离散动作（供 PPO）
├── ppo_runner.py             # Stable-Baselines3 PPO 训练与简单评测
│
├── run_experiments.py        # 一键跑对比实验，写出 experiment_results.json
├── map_environment.png       # 导出的环境示意图（障碍 + 起终点）
├── map_with_astar.png        # 同上 + 一条 A* 最优路径（便于报告插图）
│
├── EXPERIMENT_REPORT.md      # 实验报告（结论、表格、复现方式）
│
├── ppo_model.zip             # 训练得到的 PPO 策略权重（运行 ppo 后生成，可删后重训）
└── __pycache__/              # Python 字节码缓存（可忽略、勿提交版本库亦可）
```

---

## 各文件职责

| 文件 | 作用 |
|------|------|
| **env.py** | 定义 `GridEnvironment`：**第一版撒点障碍**（边界池 / 中心带 / 其余内部，整格；`seed` 可复现）、八邻域、`render`、路径校验与代价等。 |
| **common.py** | `PlanResult` 数据结构；`octile_heuristic`（八连通下的可采纳启发下界）。 |
| **dijkstra.py** | 在 `GridEnvironment` 上跑 Dijkstra，返回路径、代价、扩展次数等。 |
| **astar.py** | 同上，A*（f = g + 八距离启发）。 |
| **dstar_lite.py** | D* Lite 首次规划；与 Dijkstra/A* 在静态同权下应得到相同最优代价（实现细节见代码注释）。 |
| **rrt_star.py** | 随机采样 + 向样本扩展 + 尝试接终点；适合作为采样类基线，非保证全局最优。 |
| **rl_env.py** | `GridPathfindingGymEnv`：把同一地图变成 Gymnasium 环境，供 RL 训练（局部窗口 + 相对目标等观测）。 |
| **ppo_runner.py** | 用 SB3 的 `PPO` 训练上述环境，保存 `ppo_model.zip`，并做若干回合成功率/步长/路径代价统计。 |
| **run_experiments.py** | 依次调用上述算法与 PPO，汇总为 JSON；命令行可传随机种子，例如 `python run_experiments.py 0`。 |
| **requirements.txt** | 安装依赖：`pip install -r requirements.txt`。 |
| **experiment_results.json** | `run_experiments.py` 的输出，便于画图或写报告引用。 |
| **EXPERIMENT_REPORT.md** | 面向读者的实验报告：目的、设置、结果表、讨论与复现命令。 |
| **ppo_model.zip** | PPO 训练产物；删除后再次运行 `run_experiments.py` 或 `ppo_runner.py` 会重新训练生成。 |

---

## 典型用法

```bash
cd sprint2
pip install -r requirements.txt
python run_experiments.py 0
```

单独预览地图（弹窗）或导出 PNG：

```bash
python env.py                    # 弹窗预览（默认 seed=0）
python env.py --seed 0 --out map_environment.png
python env.py --seed 1 --out map.png --show   # 保存并弹窗
```

---

## 与仓库其它部分的关系

- Sprint 2 **自包含**在本目录：算法与实验脚本默认 `import` 同目录下的 `env` 等模块；在 `sprint2` 下执行脚本即可。
- 若将本目录加入更大项目，可把 `sprint2` 当作子包或把路径加入 `PYTHONPATH`。
