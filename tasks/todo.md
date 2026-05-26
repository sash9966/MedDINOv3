# MedDINOv3 3D Training Fix — Task List

**Goal**: Fix 3D Ashwin/centering plateau at 0.15 Dice (2D hits 0.88).  
**Dataset**: Dataset030_imageCHD_HU

## Root Causes (verified)
- SGD used instead of AdamW (critical — ViTs don't converge with SGD)
- Backbone LR default 0.05 is too conservative
- Ashwin inflation missing /d_patch normalisation (16× activation scale bug — kept as baseline)
- D_PATCH already fixed to 16 in existing scripts (prior session)

---

## Tasks

- [x] A: Replace SGD→AdamW + 10-ep warmup in `meddinov3_3d_primus_multiscale_Trainer.configure_optimizers`
      File: `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py`
      Gate: `python3 -m py_compile` passes

- [x] B: Bump backbone LR default 0.05→0.3; add `MEDDINOV3_NUM_EPOCHS` env var to `__init__`
      File: `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py`
      Gate: syntax check passes

- [x] C: Write `CHD_Dataset030_MedDINO_Centering_d8.sh` (D_PATCH=8, centering, 500 epochs)
      Gate: shellcheck passes (or no syntax errors visible)

- [x] D: Write `verify_inflation.py` at repo root
      Gate: `python3 verify_inflation.py --help` runs without import errors

- [x] Dashboard: Add "MedDINOv3 3D centering d=8 AdamW" experiment row + Ashwin /d_patch bug entry
      File: `dashboard.html`

- [ ] Smoke test on server (fold 0, 5–10 epochs):
      - Optimizer line prints "AdamW"
      - Loss negative by epoch 5
      - Pseudo-Dice > 0.3 by epoch 20 (if not: stop, re-plan)
