#!/usr/bin/env python3
"""由 benchmark_results.json 与 PPO 训练日志生成 EXPERIMENT_REPORT.md（对齐 Sprint 2 结构）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results" / "benchmark_results.json"
PPO_RANDOM_LOG = ROOT / "outputs" / "results" / "ppo_random_train.json"
PPO_CURRICULUM_LOG = ROOT / "outputs" / "results" / "ppo_curriculum_log.json"
REPORT = ROOT / "EXPERIMENT_REPORT.md"
ALL_PRESETS = ("ship_pipe_100", "ship_pipe_500", "ship_pipe_1000")


def _fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        return f"{v:.{digits}f}"
    return str(v)


def _row(r: dict) -> str:
    alg = r["algorithm"]
    if r.get("skipped"):
        note = r.get("note", "跳过")
        return f"| {alg} | — | — | — | — | {note} |"
    ok = "是" if r.get("success") else "否"
    cost = _fmt(r.get("path_cost"), 3)
    plen = _fmt(r.get("path_length"), 0)
    exp = _fmt(r.get("expanded"), 0)
    t = r.get("wall_time_sec")
    if t is None:
        t = r.get("wall_time_sec_mean")
    ts = _fmt(t, 3)
    if alg == "PPO" and "eval" in r:
        ev = r["eval"]
        ok += f" ({ev['success_rate']*100:.0f}% eval)"
    return f"| {alg} | {ok} | {cost} | {plen} | {exp} | {ts} |"


def _best_classical(block: dict) -> tuple[str | None, float | None]:
    best_alg, best_cost = None, None
    for r in block["results"]:
        if r.get("skipped") or r["algorithm"] in ("RRT*", "PPO"):
            continue
        c = r.get("path_cost")
        if r.get("success") and c is not None:
            if best_cost is None or c < best_cost:
                best_cost, best_alg = c, r["algorithm"]
    return best_alg, best_cost


def _load_ppo_train_log() -> dict:
    if PPO_RANDOM_LOG.is_file():
        return json.loads(PPO_RANDOM_LOG.read_text(encoding="utf-8"))
    if PPO_CURRICULUM_LOG.is_file():
        return json.loads(PPO_CURRICULUM_LOG.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    if not RESULTS.is_file():
        print(f"缺少 {RESULTS}，请先运行 run_benchmark.py", file=sys.stderr)
        sys.exit(1)

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    data_note = data.get("data_note", "")
    done = list(data.get("environments", {}).keys())
    pending = [p for p in ALL_PRESETS if p not in done]
    ppo_log = _load_ppo_train_log()
    ppo_model = data.get("ppo_model", "outputs/models/ppo_random.zip")
    is_random_train = PPO_RANDOM_LOG.is_file() and ppo_log.get("mode") == "random_map_pool"

    parts: list[str] = [
        "# Sprint 3：大规模栅格路径规划对比实验报告",
        "",
    ]
    if data_note:
        parts += [f"> **数据说明**：{data_note}", ""]
    parts += [
        "## 1. 实验目的",
        "",
        "在 Sprint 2 五种算法（**Dijkstra、A\\*、RRT\\*、D\\* Lite、PPO**）基础上，",
        "将环境改为 **船舶管道模拟布局**（大矩形舱室块 + 外板/贴墙障碍），",
        "在 **100×100 / 500×500 / 1000×1000** 三张固定 preset 上对比路径代价、扩展量与运行时间。",
        "",
        "PPO 先在**随机地图池**上训练（详见 [PPO_TRAINING_REPORT.md](PPO_TRAINING_REPORT.md)），",
        "再以权重 `" + str(ppo_model).replace(str(ROOT) + "/", "") + "` 在三张 benchmark 图上评测。",
        "",
    ]
    if pending:
        parts += [
            f"> **进度**：已包含 **{', '.join(done)}**；待补：**{', '.join(pending)}**。",
            "",
        ]

    parts += [
        "## 2. 环境与障碍说明",
        "",
        "基础环境为 `pathplan/env/grid.py` 的 `GridEnvironment`，布局逻辑见 `pathplan/env/ship_pipe.py`。",
        "",
        "| 规则 | 说明 |",
        "|------|------|",
        "| 大矩形块 | 随机放置舱室/设备矩形障碍 |",
        "| 贴墙加密 | 外轮廓舱壁、障碍周边加厚 |",
        "| 主通道 | **不设空旷走廊**（`corridor_half_width: 0`），舱室可阻断起终点连线 |",
        "| 难度约束 | `min_path_stretch`：最短路径须明显绕障 |",
        "| 安全区 | 起终点 `safe_radius` 邻域内不置障 |",
        "",
        "移动模型：**八连通**（直行 1，对角 √2，不切角）。配置见 `configs/env_presets.yaml`。",
        "",
        "### 2.1 三种预设环境",
        "",
    ]

    for name, block in data["environments"].items():
        info = block["info"]
        parts += [
            f"**{name}**（{info.get('description', '')}）",
            "",
            f"- 规模：**{info['rows']}×{info['cols']}**",
            f"- 障碍格：**{info['obstacle_count']}**（占比 **{info['obstacle_ratio']*100:.1f}%**）",
            f"- 起终点：`{info['start']}` → `{info['goal']}`",
            "",
        ]

    parts += [
        "## 3. 对比算法简述",
        "",
        "| 算法 | 类型 | 说明 |",
        "|------|------|------|",
        "| Dijkstra | 图最短路 | 全局最优；1000×1000 为控时跳过。 |",
        "| A* | 启发式搜索 | 八距离启发，扩展量通常远小于 Dijkstra。 |",
        "| RRT* | 随机采样 | 多次采样取最优代价。 |",
        "| D* Lite | 增量最短路 | 静态图与最短路同价；大规模可能跳过。 |",
        "| PPO | 强化学习 | 21×21 局部视野 + 势函数塑形；**随机地图池训练**后固定图评测。 |",
        "",
        "## 4. 实验设置",
        "",
    ]

    if ppo_log:
        train_label = (
            "随机地图池课程（`ppo_random_curriculum.yaml`）"
            if is_random_train
            else "固定课程（`curriculum_fast.yaml` / `curriculum.yaml`）"
        )
        parts += [
            f"- **PPO 训练**：{train_label}，总步数 **{ppo_log.get('total_timesteps', '—'):,}**，"
            f"耗时 **{ppo_log.get('total_wall_time_sec', 0):.1f} s**。训练细节见 [PPO_TRAINING_REPORT.md](PPO_TRAINING_REPORT.md)。",
            f"- **PPO 评测权重**：`{ppo_model}`",
            "",
            "| 阶段 | 地图 | 障碍占比 | 步数 | 耗时 (s) |",
            "|------|------|----------|------|----------|",
        ]
        for s in ppo_log.get("stages", []):
            obs = s.get("mean_pool_obstacle_ratio", s.get("obstacle_ratio", 0))
            parts.append(
                f"| {s['stage']} | {s['grid']} | {obs*100:.1f}% | "
                f"{s['timesteps']} | {s['wall_time_sec']:.1f} |"
            )
        parts.append("")
    else:
        parts.append("- **PPO**：未找到训练日志。\n")

    parts += [
        "- **经典算法**：各 preset 运行一次；RRT\\* 多次采样取最优。",
        "- **PPO**：单次 rollout + 15 回合 eval 成功率。",
        "- **原始数据**：`outputs/results/benchmark_results.json`。",
        "",
        "## 5. 实验结果",
        "",
        "下表来自本次 benchmark（固定三张 preset + 训练后的 PPO）。",
        "",
    ]

    for name, block in data["environments"].items():
        info = block["info"]
        best_alg, best_cost = _best_classical(block)
        idx = list(data["environments"].keys()).index(name) + 1
        parts += [
            f"### 5.{idx} {name}（{info['rows']}×{info['cols']}）",
            "",
            "| 方法 | 成功 | 路径代价 | 路径长度（格子数） | 扩展量 / 采样 | 时间 (s) |",
            "|------|------|----------|-------------------|---------------|----------|",
        ]
        for r in block["results"]:
            parts.append(_row(r))
        parts.append("")
        if best_alg:
            parts.append(f"- **经典最优代价**（{best_alg}）：**{_fmt(best_cost, 3)}**。")
        rrt = next((r for r in block["results"] if r["algorithm"] == "RRT*"), None)
        if rrt and rrt.get("success"):
            parts.append(
                f"- **RRT\\***：{rrt.get('note', '')}，代价 {_fmt(rrt.get('path_cost'), 3)}。"
            )
        ppo = next((r for r in block["results"] if r["algorithm"] == "PPO"), None)
        if ppo:
            ev = ppo.get("eval", {})
            sr = ev.get("success_rate", 0) * 100
            sim = "（参考模拟）" if ppo.get("extra", {}).get("simulated") else ""
            if ppo.get("success"):
                opt = _fmt(best_cost, 3)
                pc = ppo.get("path_cost")
                pct = ""
                if pc and best_cost:
                    pct = f"，约高 **{(pc/best_cost-1)*100:.1f}%**"
                parts.append(
                    f"- **PPO**{sim}：单次代价 {_fmt(pc, 3)}{pct}；"
                    f"eval **{sr:.0f}%**。"
                )
            else:
                parts.append(
                    f"- **PPO**{sim}：单次未达终点；eval **{sr:.0f}%**。"
                )
        parts.append("")

    if len(done) > 1:
        parts += [
            "### 5.4 规模与耗时趋势（汇总）",
            "",
            "| Preset | 障碍占比 | A* 扩展 | A* 时间 (s) | PPO eval | 备注 |",
            "|--------|----------|---------|-------------|----------|------|",
        ]
        for name, block in data["environments"].items():
            info = block["info"]
            astar = next((r for r in block["results"] if r["algorithm"] == "A*"), {})
            ppo = next((r for r in block["results"] if r["algorithm"] == "PPO"), {})
            ev = ppo.get("eval", {})
            ppo_col = f"{ev.get('success_rate', 0)*100:.0f}%" if ppo else "—"
            notes = [
                f"{r['algorithm']} 跳过"
                for r in block["results"]
                if r.get("skipped")
            ]
            parts.append(
                f"| {name} | {info['obstacle_ratio']*100:.1f}% | "
                f"{_fmt(astar.get('expanded'), 0)} | {_fmt(astar.get('wall_time_sec'), 3)} | "
                f"{ppo_col} | {'; '.join(notes) if notes else '—'} |"
            )
        parts.append("")

    parts += [
        "## 6. 小结与讨论",
        "",
        "1. **经典算法**：Dijkstra / A\\* / D\\* Lite 在静态八连通权下最优代价一致；A\\* 扩展量与耗时通常最优。",
        "2. **规模**：1000×1000 上 Dijkstra 跳过；A\\* 仍可作大规模基线。",
        "3. **RRT\\***：离散栅格上多次采样后代价常高于最短路。",
        "4. **PPO**：在随机地图池上训练后于**固定 preset** 评测，反映跨图泛化；",
        "   与全局最优搜索比的是「学会导航」而非毫秒级查询。",
        "5. 早期未训好的 `ppo_curriculum` 结果已由本次 **`ppo_random`** 权重复测替换。",
        "",
        "## 7. 复现实验",
        "",
        "```bash",
        "cd sprint3",
        "pip install -r requirements.txt",
        "python scripts/train_ppo_random.py",
        "python scripts/generate_ppo_training_report.py",
        "python scripts/run_benchmark.py --model outputs/models/ppo_random.zip",
        "python scripts/generate_report.py",
        "```",
        "",
        "分尺度 benchmark：`python scripts/run_benchmark.py --presets ship_pipe_100 --model outputs/models/ppo_random.zip`",
        "",
        "## 8. 实验总结",
        "",
        "本实验在 Sprint 2 五种算法框架下，将障碍布局升级为**船舶管道模拟**（矩形舱室 + 贴墙加密、主通道可阻断），",
        "并在 **100 / 500 / 1000** 三种栅格尺度上完成横向对比。环境与算法实现均基于八连通、不切角代价模型，",
        "三张 benchmark 图由固定 seed 生成，保证经典算法与 PPO 评测可复现。",
        "",
        "**经典规划算法**方面，结论清晰且与理论一致：在可达静态图上，Dijkstra、A\\*、D\\* Lite 给出**相同最优路径代价**；",
        "A\\* 凭借启发式在扩展节点数与 wall time 上整体最优。RRT\\* 在离散栅格上虽能求可行路，",
        "但代价高于最短路，更适合作为连续空间参考。规模放大至 1000×1000 时，Dijkstra 因耗时被跳过，**A\\*** 仍是实用的大规模单次查询方案。",
        "",
        "**强化学习（PPO）**方面，本批次采用**随机地图池课程**后在三张**固定 preset** 上评测。",
        "训练留出集在小尺度随机图上具备一定成功率（见 [PPO_TRAINING_REPORT.md](PPO_TRAINING_REPORT.md)），",
        "但固定 benchmark 上三档均为 **0%**，说明策略尚未泛化到本实验指定难图。",
        "不宜将当前 PPO 与 A\\* 在最优代价上直接类比；应先训稳再并入第五节主表。",
        "",
        "**综合建议**：（1）静态已知地图优先 **A\\***；（2）RL 需先在同分布地图池上达标再横向对比；",
        "（3）船舶管道 + `min_path_stretch` 使对比更能反映绕障难度，达到 Sprint 3 环境设计目标。",
        "",
        "完整数据见 `outputs/results/benchmark_results.json`；PPO 训练见 `ppo_random_train.json` 与专项报告。",
        "",
    ]

    REPORT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"已生成 {REPORT}")


if __name__ == "__main__":
    main()
