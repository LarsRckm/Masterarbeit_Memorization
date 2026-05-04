"""Configuration for memorization debug runs.

This config is intentionally separate from `model/config.py` so that debug runs
do not accidentally pick up a BASE_PATH from the main project.
"""

# Dataset root containing the CT folder structure.
# Example (Windows):
#   BASE_PATH = "D:/Data/Battery_CT_Scans/cylindrical"
BASE_PATH = "/data/Data/projects/GLIMPSE/cylindrical"
