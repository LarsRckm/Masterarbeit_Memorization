# model_debug_memorization

Small, controlled **memorization (overfit)** debug test for the polar CT DDPM.

Goal: train on **5 distinct cells** but with **constant conditioning**:

- `cell_format = 18650`
- `manufacturer = EVE`

Validation and test use the **same cells** (`val_cells = test_cells = train_cells`).

This does **not** change the model, loss, or sampling code in `model/CT_scan_model/`.
It only creates a tiny subset of the training data and provides a separate
training script that works with the one-format subset.

## Prerequisites (generate index + geometry)

This debug setup keeps its own dataset root.

1) Set the dataset path in `model_debug_memorization/config.py`:

```py
BASE_PATH = "D:/.../cylindrical"
```

2) Build index + geometry (written into `model_debug_memorization/data/`):

```powershell
python -m model_debug_memorization.scripts.build_cell_index_memorization \
  --out model_debug_memorization/data/cell_index.json

python -m model.CT_scan_model.scripts.precompute_geometry \
  --index model_debug_memorization/data/cell_index.json \
  --out   model_debug_memorization/data/cell_geometry.json
```

## 1) Build memorization subset

```powershell
python -m model_debug_memorization.scripts.build_memorization_subset \
  --index    model_debug_memorization/data/cell_index.json \
  --geometry model_debug_memorization/data/cell_geometry.json \
  --outdir   model_debug_memorization/data \
  --cell-format 18650 \
  --manufacturer EVE \
  --n-cells 5 \
  --seed 42
```

Outputs (in `model_debug_memorization/data/`):

- `cell_index_mem5_18650_EVE.json` (only the 5 selected cells; **1 slice per cell**)
- `cell_geometry_mem5_18650_EVE.json`
- `splits_mem5_18650_EVE.json` (train/val/test all identical)

## 2) Train memorization run

Recommended settings:

- `--batch-size 5` (exactly one batch per epoch; avoids padding duplicates)
- `--p-uncond 0.0` (do not drop conditioning)
- `--epochs` high (e.g. 1000+) until loss ~0 and samples match inputs

```powershell
python -m model_debug_memorization.scripts.train_ct_ddpm_memorization \
  --index    model_debug_memorization/data/cell_index_mem5_18650_EVE.json \
  --geometry model_debug_memorization/data/cell_geometry_mem5_18650_EVE.json \
  --splits   model_debug_memorization/data/splits_mem5_18650_EVE.json \
  --batch-size 5 \
  --p-uncond 0.0 \
  --epochs 1000 \
  --sample-every 50 \
  --sample-n 2
```

Run outputs go to `runs/ct_scan_model_memorization/<timestamp>/` by default.
