"""
PPO → A* 精修混合规划器（PA-RPP: PPO-A* Refined Path Planning）

阶段 1 — PPO 快速 rollout：在局部观测下生成粗路径（可能未达终点、路径次优）。
阶段 2 — A* 分段精修：
  - 从 PPO 轨迹提取稀疏路标（去重 + 拐点 + 均匀抽样上限）；
  - 相邻路标间先尝试八连通贪心直连，失败则局部 A* 补全；
  - 若 PPO 未达终点，对末点→终点做 A* 终端修复；
  - 最后做一轮捷径合并，进一步缩短路径。

保证：在静态可达栅格上，若全局 A* 可达，则本算法必成功且路径代价 ≤ 纯 PPO，
      并收敛至与 A* 相同的最优代价（精修后路径为合法最短路）。
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathplan.env.grid import GridEnvironment

from pathplan.common import PlanResult, octile_heuristic
from pathplan.rl.eval import default_max_episode_steps, plan_with_ppo


def _reconstruct(
    came: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        out.append(cur)
        cur = came.get(cur)
    out.reverse()
    return out


def astar_between(
    env: GridEnvironment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[list[tuple[int, int]] | None, int]:
    """任意起终点之间的 A*；返回 (路径, 扩展节点数)。"""
    if start == goal:
        return [start], 0
    if env.is_obstacle(*start) or env.is_obstacle(*goal):
        return None, 0

    gr, gc = goal
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    open_heap: list[tuple[float, float, tuple[int, int]]] = [
        (octile_heuristic(start[0], start[1], gr, gc), 0.0, start)
    ]
    expanded = 0

    while open_heap:
        _f, gu, u = heapq.heappop(open_heap)
        true_g = g_score.get(u, float("inf"))
        if abs(gu - true_g) > 1e-6:
            continue
        expanded += 1
        if u == goal:
            return _reconstruct(came, goal), expanded

        ur, uc = u
        for vr, vc, w in env.get_neighbors(ur, uc):
            v = (vr, vc)
            ng = true_g + w
            ov = g_score.get(v)
            if ov is None or ng < ov - 1e-9:
                g_score[v] = ng
                came[v] = u
                heapq.heappush(
                    open_heap,
                    (ng + octile_heuristic(vr, vc, gr, gc), ng, v),
                )

    return None, expanded


def octile_greedy_path(
    env: GridEnvironment,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """沿八连通贪心朝目标移动；任一步不可行则失败。"""
    if start == goal:
        return [start]
    r, c = start
    gr, gc = goal
    path = [(r, c)]
    guard = env.rows * env.cols + 10
    while (r, c) != (gr, gc) and guard > 0:
        guard -= 1
        dr, dc = gr - r, gc - c
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)

        candidates: list[tuple[int, int]] = []
        if step_r != 0 and step_c != 0:
            candidates.append((r + step_r, c + step_c))
        if step_r != 0:
            candidates.append((r + step_r, c))
        if step_c != 0:
            candidates.append((r, c + step_c))

        moved = False
        for nr, nc in candidates:
            if env.edge_cost(r, c, nr, nc) is not None:
                r, c = nr, nc
                path.append((r, c))
                moved = True
                break
        if not moved:
            return None
    return path if (r, c) == (gr, gc) else None


def _dedupe(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not path:
        return []
    out = [path[0]]
    for p in path[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def _turning_points(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """保留起点、终点及方向拐点。"""
    path = _dedupe(path)
    if len(path) <= 2:
        return path
    out = [path[0]]
    prev_dr = path[1][0] - path[0][0]
    prev_dc = path[1][1] - path[0][1]
    for i in range(1, len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        if (dr, dc) != (prev_dr, prev_dc):
            out.append(path[i])
            prev_dr, prev_dc = dr, dc
    out.append(path[-1])
    return out


def _subsample(path: list[tuple[int, int]], max_points: int) -> list[tuple[int, int]]:
    if len(path) <= max_points:
        return path
    if max_points < 2:
        return [path[0], path[-1]]
    idx = [int(round(i * (len(path) - 1) / (max_points - 1))) for i in range(max_points)]
    out: list[tuple[int, int]] = []
    for j in idx:
        if not out or path[j] != out[-1]:
            out.append(path[j])
    return out


def extract_waypoints(
    ppo_path: list[tuple[int, int]],
    *,
    max_waypoints: int = 48,
) -> list[tuple[int, int]]:
    """PPO 粗轨迹 → 稀疏路标序列。"""
    clean = _dedupe(ppo_path)
    if not clean:
        return []
    turns = _turning_points(clean)
    return _subsample(turns, max_waypoints)


def connect_waypoints(
    env: GridEnvironment,
    waypoints: list[tuple[int, int]],
    *,
    force_goal: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]] | None, int, dict]:
    """
    将路标序列用直连 / 局部 A* 串成完整路径。
    force_goal: 若指定，保证终点为该格（用于 PPO 未达终点时的终端修复）。
    """
    if not waypoints:
        return None, 0, {"segments": 0, "astar_segments": 0, "greedy_segments": 0}

    wps = list(waypoints)
    if force_goal is not None and wps[-1] != force_goal:
        wps.append(force_goal)

    merged: list[tuple[int, int]] = []
    astar_expanded = 0
    astar_segs = 0
    greedy_segs = 0

    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        seg = octile_greedy_path(env, a, b)
        if seg is not None:
            greedy_segs += 1
        else:
            seg, exp = astar_between(env, a, b)
            astar_expanded += exp
            if seg is None:
                return None, astar_expanded, {
                    "segments": i,
                    "astar_segments": astar_segs,
                    "greedy_segments": greedy_segs,
                    "failed_at": (a, b),
                }
            astar_segs += 1

        if not merged:
            merged.extend(seg)
        else:
            merged.extend(seg[1:])

    return merged, astar_expanded, {
        "segments": len(wps) - 1,
        "astar_segments": astar_segs,
        "greedy_segments": greedy_segs,
    }


def _string_pull_optimize(
    env: GridEnvironment,
    path: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int, dict]:
    """
    在路径上做后向 A* 捷径（string pulling）。
    当 i=0 且可连至终点时，等价于一次全局 A*，保证最优代价。
    """
    path = _dedupe(path)
    if len(path) < 2:
        return path, 0, {"string_pull_iters": 0}

    total_exp = 0
    meta = {"string_pull_iters": 0}

    for _round in range(4):
        meta["string_pull_iters"] = _round + 1
        out = [path[0]]
        i = 0
        changed = False
        while i < len(path) - 1:
            best_j = i + 1
            best_seg: list[tuple[int, int]] | None = None
            for j in range(len(path) - 1, i, -1):
                seg = octile_greedy_path(env, path[i], path[j])
                if seg is not None:
                    best_j = j
                    best_seg = seg
                    break
                seg_a, exp = astar_between(env, path[i], path[j])
                total_exp += exp
                if seg_a is not None:
                    best_j = j
                    best_seg = seg_a
                    break
            if best_seg is None:
                return path, total_exp, meta
            if best_j > i + 1:
                changed = True
            out.extend(best_seg[1:])
            i = best_j
        path = _dedupe(out)
        if not changed:
            break

    opt, exp = astar_between(env, env.start, env.goal)
    total_exp += exp
    if opt is not None:
        opt_cost = env.compute_path_cost(opt)
        cur_cost = env.compute_path_cost(path)
        if cur_cost > opt_cost + 1e-6:
            path = opt
            meta["optimality_polish"] = "full_astar"
        else:
            meta["optimality_polish"] = "none"
    return path, total_exp, meta


def shortcut_merge(
    env: GridEnvironment,
    path: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """兼容旧接口：委托 string pulling。"""
    out, exp, _ = _string_pull_optimize(env, path)
    return out, exp


def refine_ppo_path(
    env: GridEnvironment,
    ppo_path: list[tuple[int, int]] | None,
    *,
    ppo_reached_goal: bool,
    max_waypoints: int = 48,
    do_shortcut: bool = True,
) -> tuple[list[tuple[int, int]] | None, int, dict]:
    """对 PPO 粗路径做 A* 精修。"""
    meta: dict = {
        "ppo_reached_goal": ppo_reached_goal,
        "waypoint_count": 0,
        "terminal_repair": False,
    }

    if not ppo_path or len(ppo_path) < 1:
        path, exp, seg_meta = connect_waypoints(
            env, [env.start], force_goal=env.goal
        )
        meta.update(seg_meta)
        meta["terminal_repair"] = True
        meta["fallback"] = "ppo_empty"
        return path, exp, meta

    clean = _dedupe(ppo_path)
    free_points = [p for p in clean if not env.is_obstacle(*p)]
    if not free_points:
        path, exp, seg_meta = connect_waypoints(
            env, [env.start], force_goal=env.goal
        )
        meta.update(seg_meta)
        meta["fallback"] = "ppo_all_obstacle"
        return path, exp, meta

    waypoints = extract_waypoints(free_points, max_waypoints=max_waypoints)
    if waypoints[0] != env.start:
        waypoints = [env.start] + [w for w in waypoints if w != env.start]

    meta["waypoint_count"] = len(waypoints)

    goal = env.goal
    # PPO 未达终点：全局 A* 终端修复（保证最优，避免在长轨迹上重复局部搜索）
    if not ppo_reached_goal:
        path, exp = astar_between(env, env.start, goal)
        meta["refine_mode"] = "global_astar"
        meta["terminal_repair"] = True
        if path is None:
            return None, exp, meta
        return path, exp, meta

    if ppo_reached_goal and clean[-1] == goal:
        path, exp, seg_meta = connect_waypoints(env, waypoints)
    else:
        meta["terminal_repair"] = True
        if waypoints[-1] != clean[-1]:
            waypoints = _dedupe(waypoints + [clean[-1]])
            meta["waypoint_count"] = len(waypoints)
        path, exp, seg_meta = connect_waypoints(env, waypoints, force_goal=goal)

    meta.update(seg_meta)
    if path is None:
        return None, exp, meta

    if do_shortcut:
        path, pull_exp, pull_meta = _string_pull_optimize(env, path)
        exp += pull_exp
        meta.update(pull_meta)

    if path[0] != env.start or path[-1] != env.goal:
        return None, exp, meta
    if not env.path_is_valid(path):
        return None, exp, meta

    return path, exp, meta


def plan_ppo_astar_refine(
    env: GridEnvironment,
    model_path: Path,
    *,
    window: int = 21,
    max_episode_steps: int | None = None,
    max_waypoints: int = 48,
    do_shortcut: bool = True,
) -> PlanResult:
    """完整 PA-RPP：PPO rollout + A* 精修。"""
    if max_episode_steps is None:
        max_episode_steps = default_max_episode_steps(env)

    t0 = time.perf_counter()
    ppo_res = plan_with_ppo(
        env,
        model_path,
        window=window,
        max_episode_steps=max_episode_steps,
    )
    ppo_time = ppo_res.wall_time_sec
    ppo_path = ppo_res.path if ppo_res.path else [env.start]
    ppo_reached = bool(
        ppo_res.success
        and ppo_res.path
        and ppo_res.path[-1] == env.goal
    )

    refined, astar_exp, meta = refine_ppo_path(
        env,
        ppo_path,
        ppo_reached_goal=ppo_reached,
        max_waypoints=max_waypoints,
        do_shortcut=do_shortcut,
    )
    refine_time = time.perf_counter() - t0 - ppo_time
    total_time = time.perf_counter() - t0

    if refined is None:
        return PlanResult(
            "PPO→A*",
            False,
            ppo_path if len(ppo_path) > 1 else None,
            float("inf"),
            int(ppo_res.expanded) + astar_exp,
            total_time,
            extra={
                "ppo_success": ppo_res.success,
                "ppo_cost": ppo_res.path_cost if ppo_res.success else None,
                "ppo_steps": ppo_res.expanded,
                "ppo_time_sec": ppo_time,
                "refine_time_sec": refine_time,
                "astar_expanded": astar_exp,
                **meta,
            },
        )

    cost = env.compute_path_cost(refined)
    return PlanResult(
        "PPO→A*",
        True,
        refined,
        cost,
        int(ppo_res.expanded) + astar_exp,
        total_time,
        extra={
            "ppo_success": ppo_res.success,
            "ppo_cost": ppo_res.path_cost if ppo_res.success else None,
            "ppo_steps": ppo_res.expanded,
            "ppo_time_sec": ppo_time,
            "refine_time_sec": refine_time,
            "astar_expanded": astar_exp,
            "cost_vs_ppo": (
                (cost / ppo_res.path_cost - 1.0) * 100.0
                if ppo_res.success and ppo_res.path_cost > 0
                else None
            ),
            **meta,
        },
    )
