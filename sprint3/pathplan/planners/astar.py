"""A*（八连通 + 八距离启发）。"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathplan.env.grid import GridEnvironment

from pathplan.common import PlanResult, octile_heuristic


def plan(env: "GridEnvironment") -> PlanResult:
    sr, sc = env.start
    gr, gc = env.goal
    start = (sr, sc)
    goal = (gr, gc)

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    open_heap: list[tuple[float, float, tuple[int, int]]] = [
        (octile_heuristic(sr, sc, gr, gc), 0.0, start)
    ]
    expanded = 0

    while open_heap:
        _f, gu, u = heapq.heappop(open_heap)
        true_g = g_score.get(u, float("inf"))
        if abs(gu - true_g) > 1e-6:
            continue
        expanded += 1
        if u == goal:
            return PlanResult(
                "A*",
                True,
                _reconstruct(came, goal),
                g_score[goal],
                expanded,
                0.0,
            )
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

    return PlanResult("A*", False, None, float("inf"), expanded, 0.0)


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


def plan_timed(env: "GridEnvironment") -> PlanResult:
    import time

    t0 = time.perf_counter()
    res = plan(env)
    res.wall_time_sec = time.perf_counter() - t0
    return res
