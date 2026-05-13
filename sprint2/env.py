from __future__ import annotations

import numpy as np


class GridEnvironment:
    """
    100×100 路径规划网格环境。
    障碍物采用 **撒点法（第一版）**：在「地图边界格」「中部中心带 [25,75)×[25,75)」
    与「其余内部格」三类候选池中随机抽取整格置障；起终点 **5×5** 邻域不置障；重复采样直至八连通存在通路。

    网格值: 0 = 可通行, 1 = 障碍物
    坐标约定: (row, col)，左上角为 (0, 0)
    """

    FREE     = 0
    OBSTACLE = 1

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.rows  = 100
        self.cols  = 100
        self.grid  = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # 人工定义的起点和终点
        self.start = (5, 5)
        self.goal  = (94, 94)

        self._build_obstacles()

    def _forbidden_mask(self) -> set[tuple[int, int]]:
        """起、终点及周围若干格不置障，避免孤立。"""
        out: set[tuple[int, int]] = set()
        for cr, cc in (self.start, self.goal):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = cr + dr, cc + dc
                    if self.in_bounds(r, c):
                        out.add((r, c))
        return out

    def _goal_reachable(self) -> bool:
        """八邻域下起点能否到达终点（障碍格不可走）。"""
        from collections import deque

        sr, sc = self.start
        gr, gc = self.goal
        if self.is_obstacle(sr, sc) or self.is_obstacle(gr, gc):
            return False
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        vis: set[tuple[int, int]] = {(sr, sc)}
        while q:
            r, c = q.popleft()
            if (r, c) == (gr, gc):
                return True
            for nr, nc, _ in self.get_neighbors(r, c):
                if (nr, nc) not in vis:
                    vis.add((nr, nc))
                    q.append((nr, nc))
        return False

    def _build_obstacles(self):
        """
        第一版撒点：边界池 / 中心带池 / 其它内部池，各自按随机权重取整格障碍；
        总障碍格约 700–980；不通则重试。
        """
        fb = self._forbidden_mask()
        max_trials = 500

        for _ in range(max_trials):
            self.grid.fill(0)
            target = int(self._rng.integers(700, 981))

            edge: list[tuple[int, int]] = []
            center: list[tuple[int, int]] = []
            other: list[tuple[int, int]] = []

            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in fb:
                        continue
                    on_border = r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1
                    if on_border:
                        edge.append((r, c))
                    elif 25 <= r < 75 and 25 <= c < 75:
                        center.append((r, c))
                    else:
                        other.append((r, c))

            w_e = float(self._rng.uniform(0.30, 0.48))
            w_c = float(self._rng.uniform(0.26, 0.42))
            n_e = min(len(edge), max(0, int(target * w_e)))
            n_c = min(len(center), max(0, int(target * w_c)))
            n_o = min(len(other), max(0, target - n_e - n_c))

            def take(cells: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
                if n <= 0 or not cells:
                    return []
                idx = self._rng.choice(len(cells), size=min(n, len(cells)), replace=False)
                return [cells[int(i)] for i in idx]

            placed: list[tuple[int, int]] = []
            placed.extend(take(edge, n_e))
            placed.extend(take(center, n_c))
            placed.extend(take(other, n_o))

            for r, c in placed:
                self.grid[r, c] = self.OBSTACLE

            if self._goal_reachable():
                return

        self.grid.fill(0)
        inner = [
            (r, c)
            for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
            if (r, c) not in fb
        ]
        if inner:
            self._rng.shuffle(inner)
            for r, c in inner[:400]:
                self.grid[r, c] = self.OBSTACLE
        if not self._goal_reachable():
            self.grid.fill(0)

    def in_bounds(self, row: int, col: int) -> bool:
        """坐标是否在网格范围内。"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, row: int, col: int) -> bool:
        """该格是否为障碍物（越界视为障碍）。"""
        if not self.in_bounds(row, col):
            return True
        return bool(self.grid[row, col] == self.OBSTACLE)

    def get_neighbors(self, row: int, col: int, allow_diagonal: bool = True):
        """
        返回可通行邻居列表，每项为 (nr, nc, cost)。
        对角移动代价为 √2 ≈ 1.414，且不穿越对角障碍角。
        """
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
                if (not self.is_obstacle(nr, nc)
                        and not self.is_obstacle(row + dr, col)
                        and not self.is_obstacle(row, col + dc)):
                    neighbors.append((nr, nc, 1.414))

        return neighbors

    def edge_cost(self, row: int, col: int, nrow: int, ncol: int) -> float | None:
        """若从 (row,col) 可一步到达 (nrow,ncol)，返回边代价，否则 None。"""
        for nr, nc, w in self.get_neighbors(row, col):
            if (nr, nc) == (nrow, ncol):
                return w
        return None

    def path_is_valid(self, path: list[tuple[int, int]] | None) -> bool:
        """路径是否连续、可通行且从 start 到 goal。"""
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
        """路径总代价（与 get_neighbors 中代价一致）；非法路径返回 inf。"""
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
        """返回障碍物栅格副本（uint8），供 RL/可视化复用。"""
        return self.grid.copy()

    # ------------------------------------------------------------------
    # 起终点设置
    # ------------------------------------------------------------------

    def set_start(self, row: int, col: int):
        if self.is_obstacle(row, col):
            raise ValueError(f"起点 ({row}, {col}) 位于障碍物上！")
        self.start = (row, col)

    def set_goal(self, row: int, col: int):
        if self.is_obstacle(row, col):
            raise ValueError(f"终点 ({row}, {col}) 位于障碍物上！")
        self.goal = (row, col)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def render(
        self,
        path=None,
        title="100×100 路径规划环境",
        *,
        save_path: str | None = None,
        show: bool = True,
        dpi: int = 150,
    ):
        """
        绘制网格地图。
        path: [(row, col), ...] 规划出的路径，可选。
        save_path: 若给定则保存为图片（如 .png），不依赖交互窗口。
        show: 是否在保存后仍弹出窗口（默认 True；批处理时可设 False）。
        dpi: 保存分辨率。
        """
        import matplotlib.pyplot as plt

        # 尽量使用系统常见中文字体，避免图例/轴标签在 PNG 中缺字
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

        # 构建 RGB 图像
        img = np.ones((self.rows, self.cols, 3), dtype=float)   # 白色背景
        img[self.grid == self.OBSTACLE] = [0.25, 0.25, 0.25]    # 障碍物深灰

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(img, origin="upper", interpolation="nearest")

        # 路径
        if path and len(path) > 1:
            rs = [p[0] for p in path]
            cs = [p[1] for p in path]
            ax.plot(cs, rs, color="royalblue", linewidth=1.8,
                    label=f"路径 ({len(path)} 步)", zorder=4)

        # 起点 / 终点
        sr, sc = self.start
        gr, gc = self.goal
        ax.plot(sc, sr, "o", color="limegreen", markersize=11,
                markeredgecolor="k", label=f"起点 {self.start}", zorder=5)
        ax.plot(gc, gr, "*", color="red", markersize=14,
                markeredgecolor="k", label=f"终点 {self.goal}", zorder=5)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("列 (col)")
        ax.set_ylabel("行 (row)")
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(self.rows - 0.5, -0.5)
        ax.legend(loc="upper right", fontsize=9)

        # 轻量网格线（每 10 格）
        ax.set_xticks(np.arange(-0.5, self.cols, 10), minor=False)
        ax.set_yticks(np.arange(-0.5, self.rows, 10), minor=False)
        ax.set_xticklabels(np.arange(0, self.cols + 1, 10))
        ax.set_yticklabels(np.arange(0, self.rows + 1, 10))
        ax.grid(color="lightgray", linewidth=0.4, zorder=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def __repr__(self):
        n_obs = int(self.grid.sum())
        n_free = self.rows * self.cols - n_obs
        ratio = n_obs / (self.rows * self.cols) * 100
        return (f"GridEnvironment(size={self.rows}×{self.cols}, "
                f"障碍={n_obs} ({ratio:.1f}%), 可通行={n_free}, "
                f"start={self.start}, goal={self.goal})")


# ----------------------------------------------------------------------
# 快速预览
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="预览或导出 GridEnvironment 地图")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="保存 PNG 路径；不设则仅弹窗预览",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="与 --out 同时使用时仍弹出预览窗口",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机地图种子（默认 0，与实验一致）",
    )
    args = p.parse_args()

    env = GridEnvironment(seed=args.seed)
    print(env)
    save = args.out.strip() or None
    if save:
        env.render(save_path=save, show=args.show, dpi=200)
        print(f"已保存: {os.path.abspath(save)}")
    else:
        env.render()
