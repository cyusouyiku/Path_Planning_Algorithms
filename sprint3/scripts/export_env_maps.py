#!/usr/bin/env python3
"""导出 README 中三种 preset 环境示意图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathplan.env.presets import PRESET_NAMES, build_preset, load_presets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "maps",
    )
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--with-astar", action="store_true", help="叠加 A* 参考路径")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    presets = load_presets()

    for name in PRESET_NAMES:
        env = build_preset(name)
        desc = presets[name].get("description", name)
        path = None
        out = args.out_dir / f"{name}.png"
        if args.with_astar:
            from pathplan.planners.astar import plan

            res = plan(env)
            if res.success:
                path = res.path
                out = args.out_dir / f"{name}_with_astar.png"

        env.render(
            path=path,
            title=f"{name}\n{desc}",
            save_path=str(out),
            show=False,
            dpi=args.dpi,
        )
        print(f"已保存: {out.resolve()}")


if __name__ == "__main__":
    main()
