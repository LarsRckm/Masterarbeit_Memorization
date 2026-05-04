"""Minimal metadata helpers for the memorization debug pipeline.

Copied from the main project to avoid importing from `model/`.
"""

from __future__ import annotations

import os


def extract_cell_format(format_dir_name: str) -> str:
    """Extract cell format (e.g. 4680, 2170, 18650) from directory token."""
    if "_" in format_dir_name:
        return format_dir_name.split("_")[0]
    return format_dir_name


def get_max_height_from_format(cell_format: str):
    """Return the max height derived from the format string.

    Examples: 18650 -> 65, 2170 -> 70, 4680 -> 80.
    """
    if len(cell_format) >= 4:
        return float(cell_format[2:4])
    return None


def extract_absolute_depth(filename: str) -> float:
    """Extract absolute slice depth from the filename (last '_' token).

    Files end in .png; depth is assumed to be the last part before extension.
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")
    try:
        return float(parts[-1])
    except ValueError:
        return 0.0


def determine_manufacturer(width: int, height: int, relative_path: str) -> str:
    """Infer manufacturer from image size and optional path hints."""
    if width == 1340 and height == 1340:
        return "Samsung"
    if width == 1370 and height == 1370:
        return "Vapcell"
    if width == 1342 and height == 1342:
        return "BYD"
    if width == 1320 and height == 1320:
        if "sodium" in relative_path.lower():
            return "HAKADI"
        return "EVE"
    return "Unknown"


def determine_chemistry(manufacturer: str) -> str:
    """Infer chemistry from manufacturer."""
    if manufacturer in ["HAKADI", "Vapcell"]:
        return "Sodium-ion"
    return "Lithium-ion"


def determine_voxel_size(cell_format: str):
    """Voxel size in µm derived from the cell format."""
    if "18650" in cell_format:
        return 14.4
    if "2170" in cell_format:
        return 16.4
    if "4680" in cell_format:
        return 35.0
    return None
