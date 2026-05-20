"""船舶管道模拟布局：大矩形块 + 沿墙加密；可选主走廊保护。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShipPipeParams:
    """布局参数（随地图尺度在 preset 中配置）。"""

    num_blocks: tuple[int, int] = (6, 12)
    block_size_frac: tuple[float, float] = (0.06, 0.18)
    # 0 = 不保护起终点连线走廊（仅 safe 区禁障）；>0 为走廊半宽（切比雪夫距）
    corridor_half_width: int = 0
    hull_depth: int = 2
    hull_fill_prob: float = 0.82
    perimeter_shell: int = 1
    wall_thickness_prob: float = 0.58
    max_build_trials: int = 120
    # 最短路径代价 / 无障碍八连通下界，须 >1 才接受（用于拒绝“直线穿障”的简单图）
    min_path_stretch: float = 1.0


def corridor_mask(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    half_width: int,
    safe: set[tuple[int, int]],
) -> np.ndarray:
    """
    管路主走廊：起终点连线带状区域 + 安全区，这些格永不随机加厚。
    使用切比雪夫距离，形成沿对角/主轴的宽带通道。
    """
    mask = np.zeros((rows, cols), dtype=bool)
    sr, sc = start
    gr, gc = goal
    if half_width <= 0:
        return mask

    hw = half_width

    for r in range(rows):
        for c in range(cols):
            if (r, c) in safe:
                mask[r, c] = True
                continue
            # 点到线段 (start->goal) 的近似：参数 t 投影
            dr, dc = gr - sr, gc - sc
            if dr == 0 and dc == 0:
                dist = max(abs(r - sr), abs(c - sc))
            else:
                t = ((r - sr) * dr + (c - sc) * dc) / (dr * dr + dc * dc + 1e-9)
                t = max(0.0, min(1.0, t))
                pr = sr + t * dr
                pc = sc + t * dc
                dist = max(abs(r - pr), abs(c - pc))
            if dist <= hw:
                mask[r, c] = True

    return mask


def build_ship_pipe_grid(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    safe: set[tuple[int, int]],
    rng: np.random.Generator,
    params: ShipPipeParams,
) -> np.ndarray:
    """
    生成 0/1 障碍栅格。
    1) 外板/舱壁条带；2) 随机大矩形舱室块；3) 仅沿障碍周边与边界加厚，不进入走廊。
    """
    free, obs = 0, 1
    pipe_corridor = corridor_mask(
        rows, cols, start, goal, params.corridor_half_width, safe
    )

    def can_place_obstacle(r: int, c: int) -> bool:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if (r, c) in safe:
            return False
        if pipe_corridor[r, c]:
            return False
        return True

    for _trial in range(params.max_build_trials):
        grid = np.zeros((rows, cols), dtype=np.uint8)

        # --- 外轮廓舱壁（靠边，走廊内不铺）---
        depth = params.hull_depth
        for r in range(rows):
            for c in range(cols):
                on_hull = (
                    r < depth
                    or r >= rows - depth
                    or c < depth
                    or c >= cols - depth
                )
                if not on_hull:
                    continue
                if not can_place_obstacle(r, c):
                    continue
                if rng.random() < params.hull_fill_prob:
                    grid[r, c] = obs

        # --- 大矩形舱室 / 设备块 ---
        n_lo, n_hi = params.num_blocks
        n_blocks = int(rng.integers(n_lo, n_hi + 1))
        side = max(rows, cols)
        placed_rects: list[tuple[int, int, int, int]] = []

        for _ in range(n_blocks * 4):
            if len(placed_rects) >= n_blocks:
                break
            fh = float(rng.uniform(*params.block_size_frac))
            fw = float(rng.uniform(*params.block_size_frac))
            rh = max(4, int(rows * fh))
            rw = max(4, int(cols * fw))
            r0 = int(rng.integers(depth + 2, max(depth + 3, rows - rh - depth - 1)))
            c0 = int(rng.integers(depth + 2, max(depth + 3, cols - rw - depth - 1)))
            r1, c1 = r0 + rh, c0 + rw

            overlap_corridor = False
            for r in range(r0, min(r1, rows)):
                for c in range(c0, min(c1, cols)):
                    if pipe_corridor[r, c]:
                        overlap_corridor = True
                        break
                if overlap_corridor:
                    break
            if overlap_corridor:
                continue

            for r in range(r0, min(r1, rows)):
                for c in range(c0, min(c1, cols)):
                    if can_place_obstacle(r, c):
                        grid[r, c] = obs
            placed_rects.append((r0, c0, r1, c1))

        # --- 沿障碍周边 / 贴墙加厚（仅 wall-adjacent 候选）---
        shell = params.perimeter_shell
        wall_candidates: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == obs or not can_place_obstacle(r, c):
                    continue
                near_obs = False
                for dr in range(-shell, shell + 1):
                    for dc in range(-shell, shell + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == obs:
                            near_obs = True
                            break
                    if near_obs:
                        break
                if not near_obs:
                    continue
                on_outer = (
                    r < depth + 1
                    or r >= rows - depth - 1
                    or c < depth + 1
                    or c >= cols - depth - 1
                )
                # 贴墙或贴矩形边才加厚；开阔腹地不加
                if on_outer or near_obs:
                    wall_candidates.append((r, c))

        rng.shuffle(wall_candidates)
        for r, c in wall_candidates:
            if grid[r, c] == obs:
                continue
            if rng.random() < params.wall_thickness_prob:
                grid[r, c] = obs

        yield grid


def build_with_reachability(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    safe: set[tuple[int, int]],
    rng: np.random.Generator,
    params: ShipPipeParams,
    is_reachable,
) -> np.ndarray:
    """反复生成直到八连通可达或用尽试验次数。"""
    last = np.zeros((rows, cols), dtype=np.uint8)
    for grid in build_ship_pipe_grid(
        rows, cols, start, goal, safe, rng, params
    ):
        last = grid
        if is_reachable(grid):
            return grid
    if is_reachable(last):
        return last
    return np.zeros((rows, cols), dtype=np.uint8)
