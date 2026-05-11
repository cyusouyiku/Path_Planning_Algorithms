"""
离散栅格上的 RRT*（随机采样 + 渐进朝向样本 + 半径内选父 / 尝试重连）。
终点通过显式「若与当前点八邻域相连则直接接 goal」加入树中。
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import GridEnvironment

from common import PlanResult


class _Node:
    __slots__ = ("p", "parent", "cost")

    def __init__(
        self,
        p: tuple[int, int],
        parent: _Node | None,
        cost: float,
    ):
        self.p = p
        self.parent = parent
        self.cost = cost


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extend_toward(
    env: "GridEnvironment",
    origin: tuple[int, int],
    target: tuple[int, int],
    step_max: int,
) -> tuple[list[tuple[int, int]], float] | None:
    path_pts: list[tuple[int, int]] = []
    cur = origin
    total = 0.0
    for _ in range(step_max):
        fr, fc = cur
        best_nb = None
        best_d = _dist(cur, target)
        for vr, vc, w in env.get_neighbors(fr, fc):
            d = _dist((vr, vc), target)
            if d < best_d - 1e-9:
                best_d = d
                best_nb = (vr, vc, w)
        if best_nb is None:
            break
        vr, vc, w = best_nb
        cur = (vr, vc)
        total += w
        path_pts.append(cur)
        if cur == target:
            break
    if not path_pts:
        return None
    return path_pts, total


def plan(
    env: "GridEnvironment",
    max_iter: int = 40_000,
    goal_bias: float = 0.1,
    seed: int = 0,
    step_max: int = 24,
    rewire_radius: float = 20.0,
) -> PlanResult:
    rng = random.Random(seed)
    start = env.start
    goal = env.goal

    nodes: list[_Node] = [_Node(start, None, 0.0)]

    def nearest(q: tuple[int, int]) -> _Node:
        best = nodes[0]
        bd = _dist(best.p, q)
        for n in nodes[1:]:
            d = _dist(n.p, q)
            if d < bd:
                bd, best = d, n
        return best

    samples = 0
    last_it = 0
    for it in range(max_iter):
        last_it = it
        if rng.random() < goal_bias:
            qrand = goal
        else:
            qrand = None
            for _ in range(100):
                r = rng.randint(1, env.rows - 2)
                c = rng.randint(1, env.cols - 2)
                if not env.is_obstacle(r, c):
                    qrand = (r, c)
                    break
            if qrand is None:
                continue
        samples += 1

        near = nearest(qrand)
        ext = _extend_toward(env, near.p, qrand, step_max)
        if ext is None:
            continue
        seg_pts, _seg_cost = ext

        running = near
        run_cost = near.cost
        new_nodes_segment: list[_Node] = []
        for p in seg_pts:
            ec = env.edge_cost(running.p[0], running.p[1], p[0], p[1])
            if ec is None:
                break
            run_cost += ec
            node = _Node(p, running, run_cost)
            new_nodes_segment.append(node)
            running = node
            nodes.append(node)

        if not new_nodes_segment:
            continue

        x = new_nodes_segment[-1]
        if _dist(x.p, goal) <= rewire_radius:
            for n in nodes:
                if n is x:
                    continue
                if _dist(n.p, x.p) > rewire_radius:
                    continue
                ec = env.edge_cost(n.p[0], n.p[1], x.p[0], x.p[1])
                if ec is None:
                    continue
                alt = n.cost + ec
                if alt + 1e-6 < x.cost:
                    x.parent = n
                    x.cost = alt

        g_ec = env.edge_cost(x.p[0], x.p[1], goal[0], goal[1])
        if g_ec is not None:
            goal_cost = x.cost + g_ec
            goal_node = _Node(goal, x, goal_cost)
            nodes.append(goal_node)
            break
    else:
        goal_nodes = [n for n in nodes if n.p == goal]
        if not goal_nodes:
            return PlanResult(
                "RRT*",
                False,
                None,
                float("inf"),
                expanded=samples,
                wall_time_sec=0.0,
                extra={"iterations": last_it + 1},
            )
        gn = min(goal_nodes, key=lambda n: n.cost)
        path = []
        cur: _Node | None = gn
        while cur is not None:
            path.append(cur.p)
            cur = cur.parent
        path.reverse()
        pc = env.compute_path_cost(path)
        return PlanResult(
            "RRT*",
            True,
            path,
            pc,
            expanded=samples,
            wall_time_sec=0.0,
            extra={"iterations": last_it + 1},
        )

    gn = nodes[-1]
    path: list[tuple[int, int]] = []
    cur: _Node | None = gn
    while cur is not None:
        path.append(cur.p)
        cur = cur.parent
    path.reverse()
    pc = env.compute_path_cost(path)
    return PlanResult(
        "RRT*",
        True,
        path,
        pc,
        expanded=samples,
        wall_time_sec=0.0,
        extra={"iterations": last_it + 1},
    )


def plan_timed(env: "GridEnvironment", **kwargs) -> PlanResult:
    import time

    t0 = time.perf_counter()
    res = plan(env, **kwargs)
    res.wall_time_sec = time.perf_counter() - t0
    return res
