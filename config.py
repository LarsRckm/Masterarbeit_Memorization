"""Configuration for memorization debug runs.

This config is intentionally separate so that debug runs do not import anything
from the main `model/` package.

Only lightweight constants live here (no torch imports).
"""

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

# Dataset root containing the CT folder structure.
# Example (Windows):
#   BASE_PATH = "D:/Data/Battery_CT_Scans/cylindrical"
BASE_PATH = ""

# -----------------------------------------------------------------------------
# Polar representation (model input)
# -----------------------------------------------------------------------------

POLAR_THETA_BINS = 1024
POLAR_R_MODEL = 704
POLAR_R_VALID_REF_MAX = 662

# -----------------------------------------------------------------------------
# UNet / DDPM model parameters (architecture only)
# -----------------------------------------------------------------------------

UNET_NUM_DOWNS = 5
UNET_IN_CHANNELS = 2
UNET_OUT_CHANNELS = 1
UNET_BASE_CHANNELS = 16

TIME_EMB_DIM = 512

ATTN_ENABLED = True
ATTN_HEADS = 4
ATTN_MAX_TOKENS = 4096

# Padding behavior for polar tensors.
# Theta is periodic -> circular padding; radius is non-periodic -> reflect padding.
PAD_R_MODE = "reflect"
PAD_THETA_MODE = "circular"

# -----------------------------------------------------------------------------
# Conditioning (categorical + continuous)
# -----------------------------------------------------------------------------

CELL_FORMAT_VOCAB = ["18650", "2170", "4680", "Unknown"]
MANUFACTURER_VOCAB = ["Samsung", "Vapcell", "BYD", "HAKADI", "EVE", "Unknown"]
CHEMISTRY_VOCAB = ["Lithium-ion", "Sodium-ion", "Unknown"]

CELL_FORMAT_VOCAB_SIZE = len(CELL_FORMAT_VOCAB)
MANUFACTURER_VOCAB_SIZE = len(MANUFACTURER_VOCAB)
CHEMISTRY_VOCAB_SIZE = len(CHEMISTRY_VOCAB)

COND_CONT_DIM = 3
COND_CAT_EMB_DIM = 64
COND_EMB_OUT_DIM = TIME_EMB_DIM
