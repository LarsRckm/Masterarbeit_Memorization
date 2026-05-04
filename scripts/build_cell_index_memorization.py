"""Build a CT cell_index.json using the memorization debug config.

This script is self-contained and does not import from the main `model/` package.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import config as debug_config
from .build_cell_index_local import build_index


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build CT cell index (memorization debug config)")
    p.add_argument(
        "--base-path",
        default=None,
        help="Dataset BASE_PATH. If omitted, uses model_debug_memorization.config.BASE_PATH",
    )
    p.add_argument("--min-rel-depth", type=float, default=0.1)
    p.add_argument("--max-rel-depth", type=float, default=0.9)
    p.add_argument(
        "--out",
        default=os.path.join("model_debug_memorization", "data", "cell_index.json"),
        help="Output JSON path",
    )
    args = p.parse_args(argv)

    base_path = args.base_path if args.base_path is not None else str(getattr(debug_config, "BASE_PATH", ""))
    base_path = str(base_path)
    if not base_path:
        raise SystemExit(
            "No BASE_PATH provided. Set model_debug_memorization/config.py::BASE_PATH "
            "or pass --base-path."
        )

    index = build_index(base_path, min_rel_depth=args.min_rel_depth, max_rel_depth=args.max_rel_depth)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    n_cells = len(index.get("cells", {}))
    n_images = sum(len(v.get("images", [])) for v in index.get("cells", {}).values())
    print(f"Wrote index: {args.out}  (cells={n_cells}, images={n_images})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
