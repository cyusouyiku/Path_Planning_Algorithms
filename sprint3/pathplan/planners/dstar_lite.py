"""D* Lite 静态栅格首次规划。"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathplan.env.grid import GridEnvironment

from pathplan.common import PlanResult, octile_heuristic


def plan(env: "GridEnvironment") -> PlanResult:
    s_start = env.start
    s_goal = env.goal
    sr0, sc0 = s_start

    free: set[tuple[int, int]] = set()
    for r in range(env.rows):
        for c in range(env.cols):
            if not env.is_obstacle(r, c):
                free.add((r, c))

    g: dict[tuple[int, int], float] = {u: math.inf for u in free}
    rhs: dict[tuple[int, int], float] = {u: math.inf for u in free}
    open_heap: list[tuple[float, float, tuple[int, int]]] = []

    def h(u: tuple[int, int]) -> float:
        return octile_heuristic(u[0], u[1], sr0, sc0)

    def calculate_key(u: tuple[int, int]) -> tuple[float, float]:
        gu, rh = g[u], rhs[u]
        return (min(gu, rh) + h(u), min(gu, rh))

    def edges_from(u: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        out: list[tuple[tuple[int, int], float]] = []
        ur, uc = u
        for vr, vc, w in env.get_neighbors(ur, uc):
            v = (vr, vc)
            if v in free:
                out.append((v, w))
        return out

    def update_vertex(u: tuple[int, int]) -> None:
        if u != s_goal:
            best = math.inf
            for v, w in edges_from(u):
                cand = w + g[v]
                if cand < best:
                    best = cand
            rhs[u] = best
        if abs(g[u] - rhs[u]) > 1e-9:
            k1, k2 = calculate_key(u)
            heapq.heappush(open_heap, (k1, k2, u))

    def peek_valid_top_key() -> tuple[float, float] | None:
        while open_heap:
            k1, k2, u = open_heap[0]
            ck1, ck2 = calculate_key(u)
            if abs(k1 - ck1) > 1e-9 or abs(k2 - ck2) > 1e-9:
                heapq.heappop(open_heap)
                continue
            if abs(g[u] - rhs[u]) < 1e-9:
                heapq.heappop(open_heap)
                continue
            return (k1, k2)
        return None

    def pop() -> tuple[int, int]:
        while open_heap:
            k1, k2, u = heapq.heappop(open_heap)
            ck1, ck2 = calculate_key(u)
            if abs(k1 - ck1) > 1e-9 or abs(k2 - ck2) > 1e-9:
                continue
            if abs(g[u] - rhs[u]) < 1e-9:
                continue
            return u
        raise RuntimeError("empty open")

    rhs[s_goal] = 0.0
    update_vertex(s_goal)
    expanded = 0

    def compute_shortest_path() -> bool:
        nonlocal expanded
        while True:
            inconsistent = abs(rhs[s_start] - g[s_start]) > 1e-9
            if not open_heap:
                return not inconsistent
            ks = calculate_key(s_start)
            tk = peek_valid_top_key()
            if tk is None:
                continue
            if not (tk < ks or inconsistent):
                return True
            expanded += 1
            u = pop()
            if g[u] > rhs[u]:
                g[u] = rhs[u]
                for pv, _ in edges_from(u):
                    update_vertex(pv)
            else:
                g[u] = math.inf
                for pv, _ in edges_from(u):
                    update_vertex(pv)
                update_vertex(u)

    ok = compute_shortest_path()
    if not ok or math.isinf(g[s_start]):
        return PlanResult("D* Lite", False, None, float("inf"), expanded, 0.0)

    path: list[tuple[int, int]] = [s_start]
    cur = s_start
    guard = 0
    while cur != s_goal:
        guard += 1
        if guard > len(free) * 4:
            return PlanResult("D* Lite", False, None, float("inf"), expanded, 0.0)
        best_n, best_c = None, math.inf
        for v, w in edges_from(cur):
            cand = w + g[v]
            if cand < best_c - 1e-9:
                best_c, best_n = cand, v
        if best_n is None:
            return PlanResult("D* Lite", False, None, float("inf"), expanded, 0.0)
        path.append(best_n)
        cur = best_n

    return PlanResult(
        "D* Lite",
        True,
        path,
        env.compute_path_cost(path),
        expanded,
        0.0,
    )


def plan_timed(env: "GridEnvironment") -> PlanResult:
    import time

    t0 = time.perf_counter()
    res = plan(env)
    res.wall_time_sec = time.perf_counter() - t0
    return res
