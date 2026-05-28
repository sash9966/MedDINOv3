# MedDINOv3 3D Training Fix — Task List

**Goal**: Fix 3D Ashwin/centering plateau at 0.15 Dice (2D hits 0.88).  
**Dataset**: Dataset030_imageCHD_HU

## Root Causes (verified)
- SGD used instead of AdamW (critical — ViTs don't converge with SGD)
- Backbone LR default 0.05 is too conservative
- Ashwin inflation missing /d_patch normalisation (16× activation scale bug — kept as baseline)
- D_PATCH already fixed to 16 in existing scripts (prior session)

---

## Round 1 Tasks

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

---

## Round 2 — Opus 4.7 review of stalling 3D run (2026-05-27)

Additional root causes identified by reviewing `dinov3Trainer.py` / inflation scripts:
- `clip_grad_norm_(..., 12)` (parent dinov3Trainer) is the SGD-era default; effectively
  no clipping for AdamW on a ViT.
- `patch_embed_3d` had `weight_decay=5e-2`, decaying the inflated pretrained kernel
  (and for centering inflation, decaying non-centre slices away from 0).
- `MEDDINOV3_BACKBONE_LR_SCALE=0.3` → 9e-5 effective; high end of ViT FT range and
  destabilised the freshly-initialised decoder during warmup.
- Token-count OOD at d=8 (1200 tok, 6.1×) too aggressive for first 3D run on D030.

Fixes applied:

- [x] E: Add `self.grad_clip_norm` attribute (parent default 12, 3D trainer override 1.0)
      File: `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py`
      (parent __init__, train_step lines 1006/1011, 3D __init__)
      Gate: `python3 -m py_compile` passes

- [x] F: `patch_embed_3d` weight_decay → 0 in `meddinov3_3d_primus_multiscale_Trainer.configure_optimizers`
      File: `nnUNet/nnunetv2/training/nnUNetTrainer/dinov3Trainer.py`

- [x] G: Default `MEDDINOV3_BACKBONE_LR_SCALE` 0.3 → 0.1; update existing
      `CHD_Dataset030_MedDINO_Centering_d8.sh` explicit export to 0.1 as well

- [x] H: New safe-baseline script `CHD_Dataset030_MedDINO_Centering_d16.sh`
      (centering, d=16, 600 tok, 500 ep, AdamW + new clip + new wd + new lr_scale)
      Gate: `bash -n` passes

- [x] I: Dashboard updated — interventions table, new d=16 experiment row, new bug entry

Validation steps (server):
- [ ] Run `verify_inflation.py` against centering d=16 checkpoint:
      expect mean cos-sim ≥ 0.95, act_scale ≈ 1.0
- [ ] Submit `CHD_Dataset030_MedDINO_Centering_d16.sh` fold 0
      Gate: pseudo-Dice > 0.3 by epoch 20; if not → stop & re-plan
      (Do NOT submit d=8 until d=16 has converged.)
