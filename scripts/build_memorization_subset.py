"""Build a tiny memorization/overfit subset (5 cells, fixed conditions).

This script creates a small subset of:
  - cell_index.json
  - cell_geometry.json
  - splits.json

so that training can be run deterministically on exactly N cells with constant
conditioning (cell_format + manufacturer).

It also restricts each cell to exactly **one** slice (mid-depth) to avoid
introducing slice variation during a memorization test.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple


def _choose_mid_slice(images: List[dict], target_rel_depth: float = 0.5) -> dict:
    return min(images, key=lambda d: abs(float(d.get("rel_depth", 0.0)) - float(target_rel_depth)))


def _filter_cell_ids(
    geometry_cells: Dict[str, dict],
    cell_format: str,
    manufacturer: str,
) -> List[str]:
    out: List[str] = []
    for cid, g in geometry_cells.items():
        if str(g.get("cell_format", "")) != str(cell_format):
            continue
        if str(g.get("manufacturer", "")) != str(manufacturer):
            continue
        out.append(cid)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build a memorization subset (N cells, fixed conditions)")
    p.add_argument("--index", required=True, help="Path to CT cell_index.json")
    p.add_argument("--geometry", required=True, help="Path to CT cell_geometry.json")
    p.add_argument("--outdir", required=True, help="Output directory for subset JSON files")
    p.add_argument("--cell-format", default="18650")
    p.add_argument("--manufacturer", default="EVE")
    p.add_argument("--n-cells", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target-rel-depth",
        type=float,
        default=0.5,
        help="Target rel_depth for selecting the single slice per cell.",
    )
    args = p.parse_args(argv)

    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)
    with open(args.geometry, "r", encoding="utf-8") as f:
        geometry = json.load(f)

    geometry_cells: Dict[str, dict] = geometry.get("cells", {})
    index_cells: Dict[str, dict] = index.get("cells", {})

    candidates = _filter_cell_ids(
        geometry_cells=geometry_cells,
        cell_format=str(args.cell_format),
        manufacturer=str(args.manufacturer),
    )

    # Ensure they exist in the index and have images.
    candidates = [cid for cid in candidates if cid in index_cells and index_cells[cid].get("images")]

    if len(candidates) < int(args.n_cells):
        raise SystemExit(
            f"Not enough candidate cells for (cell_format={args.cell_format}, manufacturer={args.manufacturer}). "
            f"Need {int(args.n_cells)}, found {len(candidates)}."
        )

    rng = random.Random(int(args.seed))
    candidates_sorted = sorted(candidates)
    rng.shuffle(candidates_sorted)
    chosen = sorted(candidates_sorted[: int(args.n_cells)])

    # Build subset index: keep only one mid-depth slice per cell.
    subset_cells: Dict[str, dict] = {}
    selected_samples: List[dict] = []
    for cid in chosen:
        cinfo = dict(index_cells[cid])
        imgs = list(cinfo.get("images", []))
        if not imgs:
            continue
        mid = _choose_mid_slice(imgs, target_rel_depth=float(args.target_rel_depth))
        cinfo["images"] = [mid]
        subset_cells[cid] = cinfo
        selected_samples.append({"cell_id": cid, "relpath": mid["relpath"], "rel_depth": float(mid.get("rel_depth", 0.0))})

    if len(subset_cells) != len(chosen):
        missing = [cid for cid in chosen if cid not in subset_cells]
        raise SystemExit(f"Failed to build subset for some cells (missing images?): {missing}")

    subset_index = {
        "base_path": index.get("base_path"),
        "min_rel_depth": index.get("min_rel_depth"),
        "max_rel_depth": index.get("max_rel_depth"),
        "cells": subset_cells,
    }

    subset_geometry = {
        "base_path": geometry.get("base_path", index.get("base_path")),
        "cells": {cid: geometry_cells[cid] for cid in chosen},
    }

    splits = {
        "seed": int(args.seed),
        "note": "memorization debug: train/val/test are identical",
        "cell_format": str(args.cell_format),
        "manufacturer": str(args.manufacturer),
        "n_cells": int(args.n_cells),
        "target_rel_depth": float(args.target_rel_depth),
        "train_cells": chosen,
        "val_cells": list(chosen),
        "test_cells": list(chosen),
        "selected_samples": selected_samples,
    }

    os.makedirs(args.outdir, exist_ok=True)
    out_index = os.path.join(args.outdir, f"cell_index_mem{int(args.n_cells)}_{args.cell_format}_{args.manufacturer}.json")
    out_geom = os.path.join(args.outdir, f"cell_geometry_mem{int(args.n_cells)}_{args.cell_format}_{args.manufacturer}.json")
    out_splits = os.path.join(args.outdir, f"splits_mem{int(args.n_cells)}_{args.cell_format}_{args.manufacturer}.json")

    with open(out_index, "w", encoding="utf-8") as f:
        json.dump(subset_index, f, indent=2)
    with open(out_geom, "w", encoding="utf-8") as f:
        json.dump(subset_geometry, f, indent=2)
    with open(out_splits, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("Wrote memorization subset:")
    print("-", out_index)
    print("-", out_geom)
    print("-", out_splits)
    print(f"Selected cells ({len(chosen)}): {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
