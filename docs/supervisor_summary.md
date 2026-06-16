# CHD MedDINO — Supervisor Summary

**Overview.** We are testing whether a 3D-inflated MedDINO ViT (2D CT-pretrained
weights inflated to 3D, with a 1×1×1 contrast adapter) can match or beat nnU-Net
for congenital-heart-disease whole-heart CT segmentation, and whether
diagnosis-aware conditioning and topology-aware losses can recover the hardest
structures (pulmonary/great arteries). Current MedDINO whole-heart Dice ≈ 0.89 on
~100 cases — slightly below nnU-Net — with small vessels as the main weakness. We
have built a config-driven ablation harness so every comparison is on the same
split and is re-runnable on the coming ~10× expert cohort without code changes.

## Current results (same split; fill blanks as runs complete)
| Model | WH Dice | Mean class Dice | Ao Dice | PA Dice | HD95 | clDice (Ao/PA) | Notes |
|------|--------|-----------------|---------|---------|------|----------------|-------|
| nnU-Net baseline | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _n/a_ | reference |
| MedDINO d16 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | 600 tok |
| MedDINO d8 | ~0.89* | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | current main |
| MedDINO d4 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | finest depth |
| MedDINO d4 (Ashwin) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | inflation variant |
| MedDINO + FiLM (oracle dx) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | upper bound |
| + clDice (planned) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | vessel continuity |
| + post-processing (planned) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | plausibility |
| Best / ensemble | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | — |

*whole-heart ~0.89 is the current approximate figure; per-class and same-split
nnU-Net numbers are pending `collect_metrics.py`.

## Ablation priorities
1. Same-split nnU-Net + MedDINO d16/d8/d4 confirmation.
2. Depth d16 vs d8 vs d4.
3. Inflated vs random-3D init.
4. Adapter on vs off.
5. Encoder freeze (frozen / partial / full).
6. Vessel-aware sampling.
7. Class-balanced / Tversky / focal.
8. clDice / topology loss on Ao+PA.
9. Post-processing.
10. Diagnosis multi-task → FiLM variants → combined best.

## Phasing (relative order, not time commitments)
| Phase | Content | Status |
|------|---------|--------|
| P0 | Harness + dashboard + same-split baselines | done (harness) / baselines pending metrics |
| P1 | Depth + inflation + adapter ablations | queued |
| P2 | Small-vessel loss & sampling (clDice, vessel sampling) | planned (hooks pending) |
| P3 | Diagnosis conditioning (multi-task, FiLM variants) | partly runnable (FiLM); multi-task planned |
| P4 | Post-processing + combined best | planned |
| P5 | Re-run winners on ~10× expert cohort | planned |

## How to reproduce
`python experiments/run_ablation.py --list` · generate SLURM with
`experiments/generate_slurm.py --priority high` · after runs,
`collect_metrics.py` then `update_dashboard_data.py` refreshes the dashboard.
