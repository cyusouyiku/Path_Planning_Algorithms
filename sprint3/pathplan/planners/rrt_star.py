"""离散栅格 RRT*。"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathplan.env.grid import GridEnvironment

from pathplan.common import PlanResult


class _Node:
    __slots__ = ("p", "parent", "cost")

    def __init__(
        self,
        p: tuple[int, int],
        parent: "_Node | None",
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
    for _ in range(step_max):
        fr, fc = cur
        best_nb, best_d = None, _dist(cur, target)
        for vr, vc, w in env.get_neighbors(fr, fc):
            d = _dist((vr, vc), target)
            if d < best_d - 1e-9:
                best_d, best_nb = d, (vr, vc, w)
        if best_nb is None:
            break
        vr, vc, w = best_nb
        cur = (vr, vc)
        path_pts.append(cur)
        if cur == target:
            break
    return (path_pts, 0.0) if path_pts else None


def _scale_params(env: "GridEnvironment") -> dict:
    side = max(env.rows, env.cols)
    if side >= 900:
        max_iter = 12_000
        step_max = max(20, side // 50)
    elif side >= 450:
        max_iter = 18_000
        step_max = max(16, side // 40)
    else:
        max_iter = min(80_000, max(20_000, side * 60))
        step_max = max(12, side // 35)
    return {
        "max_iter": max_iter,
        "step_max": step_max,
        "rewire_radius": max(12.0, side / 25.0),
    }


def plan(
    env: "GridEnvironment",
    max_iter: int | None = None,
    goal_bias: float = 0.12,
    seed: int = 0,
    step_max: int | None = None,
    rewire_radius: float | None = None,
) -> PlanResult:
    scaled = _scale_params(env)
    max_iter = max_iter if max_iter is not None else scaled["max_iter"]
    step_max = step_max if step_max is not None else scaled["step_max"]
    rewire_radius = (
        rewire_radius if rewire_radius is not None else scaled["rewire_radius"]
    )

    rng = random.Random(seed)
    start, goal = env.start, env.goal
    nodes: list[_Node] = [_Node(start, None, 0.0)]

    def nearest(q: tuple[int, int]) -> _Node:
        return min(nodes, key=lambda n: _dist(n.p, q))

    samples = 0
    last_it = 0
    for it in range(max_iter):
        last_it = it
        if rng.random() < goal_bias:
            qrand = goal
        else:
            qrand = None
            for _ in range(120):
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
        seg_pts, _ = ext
        running, run_cost = near, near.cost
        new_nodes: list[_Node] = []
        for p in seg_pts:
            ec = env.edge_cost(running.p[0], running.p[1], p[0], p[1])
            if ec is None:
                break
            run_cost += ec
            node = _Node(p, running, run_cost)
            new_nodes.append(node)
            running = node
            nodes.append(node)
        if not new_nodes:
            continue
        x = new_nodes[-1]
        for n in nodes:
            if n is x or _dist(n.p, x.p) > rewire_radius:
                continue
            ec = env.edge_cost(n.p[0], n.p[1], x.p[0], x.p[1])
            if ec is not None and n.cost + ec + 1e-6 < x.cost:
                x.parent, x.cost = n, n.cost + ec
        g_ec = env.edge_cost(x.p[0], x.p[1], goal[0], goal[1])
        if g_ec is not None:
            nodes.append(_Node(goal, x, x.cost + g_ec))
            break
    else:
        goal_nodes = [n for n in nodes if n.p == goal]
        if not goal_nodes:
            return PlanResult(
                "RRT*",
                False,
                None,
                float("inf"),
                samples,
                0.0,
                extra={"iterations": last_it + 1},
            )
        gn = min(goal_nodes, key=lambda n: n.cost)
        path = _trace(gn)
        return PlanResult(
            "RRT*",
            True,
            path,
            env.compute_path_cost(path),
            samples,
            0.0,
            extra={"iterations": last_it + 1},
        )

    path = _trace(nodes[-1])
    return PlanResult(
        "RRT*",
        True,
        path,
        env.compute_path_cost(path),
        samples,
        0.0,
        extra={"iterations": last_it + 1},
    )


def _trace(gn: _Node) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: _Node | None = gn
    while cur is not None:
        path.append(cur.p)
        cur = cur.parent
    path.reverse()
    return path


def plan_timed(env: "GridEnvironment", **kwargs) -> PlanResult:
    import time

    t0 = time.perf_counter()
    res = plan(env, **kwargs)
    res.wall_time_sec = time.perf_counter() - t0
    return res
