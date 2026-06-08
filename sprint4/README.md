# Sprint 4：PPO→A* 精修混合路径规划（PA-RPP）

在 Sprint 3 三张船舶管道 preset 上，对比 **A***、**PPO** 与 **PPO→A*** 精修算法。

## 快速开始

```bash
cd sprint4
pip install -r requirements.txt
python scripts/run_benchmark.py
python scripts/generate_report.py
```

- 结果 JSON：`outputs/results/hybrid_benchmark_results.json`
- 实验报告：`EXPERIMENT_REPORT.md`
- 算法实现：`../sprint3/pathplan/hybrid/ppo_astar_refine.py`

## PA-RPP 流程

1. PPO rollout 粗路径  
2. 路标提取（拐点 + 抽样）  
3. 路标间贪心直连 / 局部 A*  
4. 未达终点 → 终端 A* 修复  
5. 捷径合并 → 最优路径  
