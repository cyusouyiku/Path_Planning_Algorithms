"""可配置规模栅格环境：船舶管道布局 / 旧版撒点（八连通）。"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pathplan.common import octile_heuristic
from pathplan.env.ship_pipe import ShipPipeParams, build_with_reachability

LayoutKind = Literal["ship_pipe", "scatter"]


@dataclass
class EnvBuildConfig:
    rows: int = 500
    cols: int = 500
    seed: int | None = 0
    layout: LayoutKind = "ship_pipe"
    # scatter 专用
    obstacle_target_ratio: tuple[float, float] = (0.12, 0.16)
    pool_weights_edge: tuple[float, float] = (0.28, 0.40)
    pool_weights_center: tuple[float, float] = (0.22, 0.34)
    center_band_frac: float = 0.50
    # ship_pipe 专用
    ship_pipe: ShipPipeParams = field(default_factory=ShipPipeParams)
    safe_radius: int = 3
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    force_no_obstacles: bool = False
    max_build_trials: int = 500

    def margin_cells(self) -> int:
        return max(5, min(self.rows, self.cols) // 20)

    def default_start_goal(self) -> tuple[tuple[int, int], tuple[int, int]]:
        m = self.margin_cells()
        return (m, m), (self.rows - 1 - m, self.cols - 1 - m)


class GridEnvironment:
    """
    默认 **船舶管道** 布局：大矩形舱室块 + 外板/贴墙加厚，主走廊保持空旷；
    亦可切换为旧版撒点法（layout=scatter）。
    """

    FREE = 0
    OBSTACLE = 1

    def __init__(self, config: EnvBuildConfig | None = None, **kwargs):
        cfg = config or EnvBuildConfig(**kwargs)
        self.config = cfg
        self._rng = np.random.default_rng(cfg.seed)
        self.rows = int(cfg.rows)
        self.cols = int(cfg.cols)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        ds, dg = cfg.default_start_goal()
        self.start = cfg.start if cfg.start is not None else ds
        self.goal = cfg.goal if cfg.goal is not None else dg

        self.layout = cfg.layout
        if cfg.force_no_obstacles:
            self.grid.fill(self.FREE)
        elif cfg.layout == "ship_pipe":
            self._build_ship_pipe()
        else:
            self._build_obstacles_scatter()

    def _forbidden_mask(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        r_safe = self.config.safe_radius
        for cr, cc in (self.start, self.goal):
            for dr in range(-r_safe, r_safe + 1):
                for dc in range(-r_safe, r_safe + 1):
                    r, c = cr + dr, cc + dc
                    if self.in_bounds(r, c):
                        out.add((r, c))
        return out

    def _goal_reachable(self) -> bool:
        return self._shortest_path_cost() != float("inf")

    def _shortest_path_cost(self) -> float:
        sr, sc = self.start
        gr, gc = self.goal
        if self.is_obstacle(sr, sc) or self.is_obstacle(gr, gc):
            return float("inf")
        start = (sr, sc)
        goal = (gr, gc)
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        while open_heap:
            gu, u = heapq.heappop(open_heap)
            if abs(gu - g_score.get(u, -1.0)) > 1e-9:
                continue
            if u == goal:
                return g_score[u]
            ur, uc = u
            for nr, nc, w in self.get_neighbors(ur, uc):
                v = (nr, nc)
                ng = gu + w
                old = g_score.get(v)
                if old is None or ng < old - 1e-9:
                    g_score[v] = ng
                    heapq.heappush(open_heap, (ng, v))
        return float("inf")

    def _center_bounds(self) -> tuple[int, int, int, int]:
        f = self.config.center_band_frac
        r0 = int(self.rows * (0.5 - f / 2))
        r1 = int(self.rows * (0.5 + f / 2))
        c0 = int(self.cols * (0.5 - f / 2))
        c1 = int(self.cols * (0.5 + f / 2))
        return r0, r1, c0, c1

    def _build_ship_pipe(self) -> None:
        fb = self._forbidden_mask()
        params = self.config.ship_pipe
        sr, sc = self.start
        gr, gc = self.goal
        ideal_cost = octile_heuristic(sr, sc, gr, gc)
        stretch = float(params.min_path_stretch)

        def acceptable(grid: np.ndarray) -> bool:
            self.grid = grid
            if not self._goal_reachable():
                return False
            if stretch <= 1.0:
                return True
            cost = self._shortest_path_cost()
            return cost >= ideal_cost * stretch

        self.grid = build_with_reachability(
            self.rows,
            self.cols,
            self.start,
            self.goal,
            fb,
            self._rng,
            params,
            acceptable,
        )

    def _build_obstacles_scatter(self) -> None:
        fb = self._forbidden_mask()
        lo, hi = self.config.obstacle_target_ratio
        r0, r1, c0, c1 = self._center_bounds()

        placeable = self.rows * self.cols - len(fb)
        for _ in range(self.config.max_build_trials):
            self.grid.fill(self.FREE)
            ratio = float(self._rng.uniform(lo, hi))
            target = int(placeable * ratio)

            edge: list[tuple[int, int]] = []
            center: list[tuple[int, int]] = []
            other: list[tuple[int, int]] = []

            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in fb:
                        continue
                    on_border = (
                        r == 0
                        or r == self.rows - 1
                        or c == 0
                        or c == self.cols - 1
                    )
                    if on_border:
                        edge.append((r, c))
                    elif r0 <= r < r1 and c0 <= c < c1:
                        center.append((r, c))
                    else:
                        other.append((r, c))

            w_e = float(
                self._rng.uniform(
                    self.config.pool_weights_edge[0],
                    self.config.pool_weights_edge[1],
                )
            )
            w_c = float(
                self._rng.uniform(
                    self.config.pool_weights_center[0],
                    self.config.pool_weights_center[1],
                )
            )
            n_e = min(len(edge), max(0, int(target * w_e)))
            n_c = min(len(center), max(0, int(target * w_c)))
            n_o = min(len(other), max(0, target - n_e - n_c))

            def take(cells: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
                if n <= 0 or not cells:
                    return []
                k = min(n, len(cells))
                idx = self._rng.choice(len(cells), size=k, replace=False)
                return [cells[int(i)] for i in idx]

            for r, c in take(edge, n_e) + take(center, n_c) + take(other, n_o):
                self.grid[r, c] = self.OBSTACLE

            if self._goal_reachable():
                return

        # 退化：稀疏随机
        self.grid.fill(self.FREE)
        inner = [
            (r, c)
            for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
            if (r, c) not in fb
        ]
        if inner:
            self._rng.shuffle(inner)
            fallback = max(1, int(placeable * lo * 0.5))
            for r, c in inner[:fallback]:
                self.grid[r, c] = self.OBSTACLE
        if not self._goal_reachable():
            self.grid.fill(self.FREE)

    def obstacle_count(self) -> int:
        return int(self.grid.sum())

    def obstacle_ratio(self) -> float:
        return self.obstacle_count() / (self.rows * self.cols)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return bool(self.grid[row, col] == self.OBSTACLE)

    def get_neighbors(self, row: int, col: int, allow_diagonal: bool = True):
        neighbors = []
        cardinals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in cardinals:
            nr, nc = row + dr, col + dc
            if not self.is_obstacle(nr, nc):
                neighbors.append((nr, nc, 1.0))

        if allow_diagonal:
            for dr, dc in diagonals:
                nr, nc = row + dr, col + dc
                if (
                    not self.is_obstacle(nr, nc)
                    and not self.is_obstacle(row + dr, col)
                    and not self.is_obstacle(row, col + dc)
                ):
                    neighbors.append((nr, nc, 1.4142135623730951))

        return neighbors

    def edge_cost(self, row: int, col: int, nrow: int, ncol: int) -> float | None:
        for nr, nc, w in self.get_neighbors(row, col):
            if (nr, nc) == (nrow, ncol):
                return w
        return None

    def path_is_valid(self, path: list[tuple[int, int]] | None) -> bool:
        if not path or len(path) < 2:
            return False
        if path[0] != self.start or path[-1] != self.goal:
            return False
        for i, (r, c) in enumerate(path):
            if self.is_obstacle(r, c):
                return False
            if i + 1 < len(path):
                nr, nc = path[i + 1]
                if self.edge_cost(r, c, nr, nc) is None:
                    return False
        return True

    def compute_path_cost(self, path: list[tuple[int, int]] | None) -> float:
        if not path or len(path) < 2:
            return float("inf")
        if path[0] != self.start or path[-1] != self.goal:
            return float("inf")
        total = 0.0
        for i in range(len(path) - 1):
            r, c = path[i]
            nr, nc = path[i + 1]
            w = self.edge_cost(r, c, nr, nc)
            if w is None:
                return float("inf")
            total += w
        return total

    def occupancy_map(self) -> np.ndarray:
        return self.grid.copy()

    def render(
        self,
        path: list[tuple[int, int]] | None = None,
        title: str | None = None,
        *,
        save_path: str | None = None,
        show: bool = False,
        dpi: int = 150,
    ):
        """绘制栅格地图；大图自动稀疏刻度，默认仅保存不弹窗。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.setdefault(
            "font.sans-serif",
            [
                "PingFang SC",
                "Hiragino Sans GB",
                "Heiti TC",
                "STHeiti",
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
        )
        plt.rcParams["axes.unicode_minus"] = False

        side = max(self.rows, self.cols)
        tick_step = max(10, side // 10)
        if side >= 800:
            tick_step = 100
        elif side >= 400:
            tick_step = 50

        if title is None:
            kind = "船舶管道" if getattr(self, "layout", "ship_pipe") == "ship_pipe" else "栅格"
            title = f"{self.rows}×{self.cols} {kind}环境"

        img = np.ones((self.rows, self.cols, 3), dtype=float)
        img[self.grid == self.OBSTACLE] = [0.25, 0.25, 0.25]

        figsize = min(12.0, 6.0 + side / 120.0)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        ax.imshow(img, origin="upper", interpolation="nearest")

        if path and len(path) > 1:
            rs = [p[0] for p in path]
            cs = [p[1] for p in path]
            lw = max(0.6, min(2.0, 400.0 / side))
            ax.plot(
                cs,
                rs,
                color="royalblue",
                linewidth=lw,
                label=f"路径 ({len(path)} 步)",
                zorder=4,
            )

        sr, sc = self.start
        gr, gc = self.goal
        ms = max(4.0, min(11.0, 500.0 / side))
        ax.plot(
            sc,
            sr,
            "o",
            color="limegreen",
            markersize=ms,
            markeredgecolor="k",
            label=f"起点 {self.start}",
            zorder=5,
        )
        ax.plot(
            gc,
            gr,
            "*",
            color="red",
            markersize=ms + 2,
            markeredgecolor="k",
            label=f"终点 {self.goal}",
            zorder=5,
        )

        ratio = self.obstacle_ratio() * 100
        ax.set_title(
            f"{title}\n障碍 {self.obstacle_count()} 格 ({ratio:.1f}%)",
            fontsize=12,
        )
        ax.set_xlabel("列 (col)")
        ax.set_ylabel("行 (row)")
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        ax.legend(loc="upper right", fontsize=8)

        ax.set_xticks(np.arange(-0.5, self.cols, tick_step))
        ax.set_yticks(np.arange(-0.5, self.rows, tick_step))
        ax.set_xticklabels(np.arange(0, self.cols + 1, tick_step))
        ax.set_yticklabels(np.arange(0, self.rows + 1, tick_step))
        ax.grid(color="lightgray", linewidth=0.3, zorder=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def __repr__(self) -> str:
        n_obs = self.obstacle_count()
        ratio = n_obs / (self.rows * self.cols) * 100
        return (
            f"GridEnvironment({self.rows}×{self.cols}, "
            f"障碍={n_obs} ({ratio:.1f}%), start={self.start}, goal={self.goal})"
        )
