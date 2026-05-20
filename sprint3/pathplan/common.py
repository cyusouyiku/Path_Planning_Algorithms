"""共享类型、启发函数与计时工具。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def octile_heuristic(
    r: int,
    c: int,
    gr: int,
    gc: int,
    d_diag: float = 1.4142135623730951,
) -> float:
    """八连通下从 (r,c) 到 (gr,gc) 的可采纳下界。"""
    dr, dc = abs(r - gr), abs(c - gc)
    return d_diag * min(dr, dc) + abs(dr - dc)


@dataclass
class PlanResult:
    algorithm: str
    success: bool
    path: list[tuple[int, int]] | None
    path_cost: float
    expanded: int
    wall_time_sec: float
    extra: dict = field(default_factory=dict)


def timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
