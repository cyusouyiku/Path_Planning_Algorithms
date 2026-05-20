"""从 YAML 加载并构建标准对比环境。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pathplan.env.grid import EnvBuildConfig, GridEnvironment
from pathplan.env.ship_pipe import ShipPipeParams

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "env_presets.yaml"


def load_presets() -> dict[str, dict[str, Any]]:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dict(data["presets"])


PRESET_NAMES: tuple[str, ...] = tuple(load_presets().keys())


def _tuple2(seq: list | tuple, cast=float) -> tuple:
    return (cast(seq[0]), cast(seq[1]))


def _ship_params(raw: dict[str, Any] | None) -> ShipPipeParams:
    defaults = ShipPipeParams()
    if not raw:
        return defaults
    bs = raw.get("block_size_frac")
    nb = raw.get("num_blocks")
    return ShipPipeParams(
        num_blocks=(
            (int(nb[0]), int(nb[1]))
            if nb
            else defaults.num_blocks
        ),
        block_size_frac=_tuple2(bs) if bs else defaults.block_size_frac,
        corridor_half_width=int(raw.get("corridor_half_width", defaults.corridor_half_width)),
        hull_depth=int(raw.get("hull_depth", defaults.hull_depth)),
        hull_fill_prob=float(raw.get("hull_fill_prob", defaults.hull_fill_prob)),
        perimeter_shell=int(raw.get("perimeter_shell", defaults.perimeter_shell)),
        wall_thickness_prob=float(
            raw.get("wall_thickness_prob", defaults.wall_thickness_prob)
        ),
        max_build_trials=int(raw.get("max_build_trials", defaults.max_build_trials)),
        min_path_stretch=float(raw.get("min_path_stretch", defaults.min_path_stretch)),
    )


def build_preset(name: str) -> GridEnvironment:
    presets = load_presets()
    if name not in presets:
        raise KeyError(f"未知 preset: {name}，可选 {list(presets)}")
    p = presets[name]
    layout = p.get("layout", "ship_pipe")

    if layout == "scatter":
        pw = p["pool_weights"]
        cfg = EnvBuildConfig(
            rows=int(p["rows"]),
            cols=int(p["cols"]),
            seed=int(p.get("seed", 0)),
            layout="scatter",
            obstacle_target_ratio=_tuple2(p["obstacle_target_ratio"]),
            pool_weights_edge=_tuple2(pw["edge"]),
            pool_weights_center=_tuple2(pw["center"]),
            center_band_frac=float(p.get("center_band_frac", 0.5)),
            safe_radius=int(p.get("safe_radius", 3)),
        )
    else:
        cfg = EnvBuildConfig(
            rows=int(p["rows"]),
            cols=int(p["cols"]),
            seed=int(p.get("seed", 0)),
            layout="ship_pipe",
            safe_radius=int(p.get("safe_radius", 3)),
            ship_pipe=_ship_params(p.get("ship_pipe")),
        )

    env = GridEnvironment(cfg)
    env.preset_name = name  # type: ignore[attr-defined]
    env.preset_description = p.get("description", "")  # type: ignore[attr-defined]
    return env
