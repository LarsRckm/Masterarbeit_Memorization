"""Precompute per-cell geometry for memorization debug runs.

This is a thin wrapper around `model.CT_scan_model.scripts.precompute_geometry`.
It exists so the entire memorization pipeline can be run using
`model_debug_memorization.*` entrypoints.

The dataset path is taken from the provided index JSON's `base_path`.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from model.CT_scan_model.scripts import precompute_geometry as _pre


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Precompute geometry (memorization debug wrapper)")
    p.add_argument(
        "--index",
        default=os.path.join("model_debug_memorization", "data", "cell_index.json"),
        help="Input cell_index.json path",
    )
    p.add_argument(
        "--out",
        default=os.path.join("model_debug_memorization", "data", "cell_geometry.json"),
        help="Output cell_geometry.json path",
    )
    p.add_argument("--target-rel-depth", type=float, default=0.5)
    args = p.parse_args(argv)

    return int(
        _pre.main(
            [
                "--index",
                str(args.index),
                "--out",
                str(args.out),
                "--target-rel-depth",
                str(float(args.target_rel_depth)),
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
