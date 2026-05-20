"""Dijkstra 八网格最短路径。"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathplan.env.grid import GridEnvironment

from pathplan.common import PlanResult


def plan(env: "GridEnvironment") -> PlanResult:
    start = env.start
    goal = env.goal
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    expanded = 0

    while open_heap:
        gu, u = heapq.heappop(open_heap)
        if abs(gu - g_score.get(u, -1.0)) > 1e-9:
            continue
        expanded += 1
        if u == goal:
            return PlanResult(
                "Dijkstra",
                True,
                _reconstruct(came, goal),
                g_score[goal],
                expanded,
                0.0,
            )
        ur, uc = u
        for vr, vc, w in env.get_neighbors(ur, uc):
            v = (vr, vc)
            ng = gu + w
            old = g_score.get(v)
            if old is None or ng < old - 1e-9:
                g_score[v] = ng
                came[v] = u
                heapq.heappush(open_heap, (ng, v))

    return PlanResult("Dijkstra", False, None, float("inf"), expanded, 0.0)


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
