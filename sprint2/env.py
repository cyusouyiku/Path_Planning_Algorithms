from __future__ import annotations

import numpy as np


class GridEnvironment:
    """
    100x100 路径规划网格环境。
    障碍物采用撒点法，放置在网格边缘（围墙）及内部结构化墙段中心区域。
    
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

    def _build_obstacles(self):
        """依次添加边界围墙、内部横纵墙段、中心方块。"""
        self._add_border_walls()
        self._add_horizontal_barriers()
        self._add_vertical_barriers()
        self._add_center_blocks()

    def _add_border_walls(self):
        """外围一圈围墙（边上撒点）。"""
        self.grid[0,  :]  = self.OBSTACLE   # 上边
        self.grid[-1, :]  = self.OBSTACLE   # 下边
        self.grid[:,  0]  = self.OBSTACLE   # 左边
        self.grid[:, -1]  = self.OBSTACLE   # 右边

    def _add_horizontal_barriers(self):
        """
        水平隔离墙，每段预留一个缺口保证可通行性。
        格式: (行, 列起, 列止, 缺口列起, 缺口列止)
        """
        h_barriers = [
            (20,  5,  60, 28, 38),   # 上区横墙，缺口偏中
            (38, 42,  95, 58, 68),   # 中区右横墙，缺口偏右
            (60,  5,  65, 22, 32),   # 中区左横墙，缺口偏左
            (75, 28,  95, 48, 58),   # 下区横墙，缺口居中
        ]
        for row, c0, c1, gs, ge in h_barriers:
            for col in range(c0, c1 + 1):
                if not (gs <= col <= ge):
                    self.grid[row, col] = self.OBSTACLE

    def _add_vertical_barriers(self):
        """
        垂直隔离墙，每段预留一个缺口保证可通行性。
        格式: (列, 行起, 行止, 缺口行起, 缺口行止)
        """
        v_barriers = [
            (25,  5,  55, 18, 28),   # 左区竖墙
            (50,  5,  35, 14, 24),   # 上中竖墙
            (70, 42,  90, 58, 68),   # 右下竖墙
            (80,  5,  38, 18, 28),   # 右上竖墙
        ]
        for col, r0, r1, gs, ge in v_barriers:
            for row in range(r0, r1 + 1):
                if not (gs <= row <= ge):
                    self.grid[row, col] = self.OBSTACLE

    def _add_center_blocks(self):
        """
        在网格各区域中心撒置 5×5 方块障碍（中心撒点）。
        各中心坐标均经过验证不覆盖起终点与墙段缺口。
        """
        centers = [
            (12, 70), (12, 85),
            (30, 10), (30, 82),
            (50, 58), (50, 82),
            (65, 10), (65, 45),
            (85, 15), (85, 55), (85, 82),
        ]
        for (r, c) in centers:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 1 <= nr < self.rows - 1 and 1 <= nc < self.cols - 1:
                        self.grid[nr, nc] = self.OBSTACLE

    # ------------------------------------------------------------------
    # 查询接口（供算法调用）
    # ------------------------------------------------------------------

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
                # 同时检查两侧格子，防止穿越墙角
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
        n_obs  = int(self.grid.sum())
        n_free = self.rows * self.cols - n_obs
        ratio  = n_obs / (self.rows * self.cols) * 100
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
    args = p.parse_args()

    env = GridEnvironment()
    print(env)
    save = args.out.strip() or None
    if save:
        env.render(save_path=save, show=args.show, dpi=200)
        print(f"已保存: {os.path.abspath(save)}")
    else:
        env.render()
