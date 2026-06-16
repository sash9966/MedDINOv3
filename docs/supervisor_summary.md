# CHD MedDINO — Supervisor Summary

**Overview.** We are testing whether a 3D-inflated MedDINO ViT (2D CT-pretrained
weights inflated to 3D, with a 1×1×1 contrast adapter) can match or beat nnU-Net
for congenital-heart-disease whole-heart CT segmentation, and whether
diagnosis-aware conditioning and topology-aware losses can recover the hardest
structures (pulmonary/great arteries). Current MedDINO whole-heart Dice ≈ 0.89 on
~100 cases — slightly below nnU-Net — with small vessels as the main weakness. We
have built a config-driven ablation harness so every comparison is on the same
split and is re-runnable on the coming ~10× expert cohort without code changes.

## Current results — held-out test set (32 cases, same split)
WH = union-mask whole-heart Dice; **Mean cls** = mean of the 7 structure Dice;
Δcls = mean-class gap vs nnU-Net. HD95 / clDice pending `tools/evaluate_topology.py`.

| Model | WH | Mean cls | Δcls | LV-BP | RV-BP | LA | RA | Myo | Ao | PA |
|------|----|----------|------|-------|-------|----|----|-----|----|----|
| **nnU-Net (DA5 baseline)** | **0.909** | **0.832** | ref | 0.876 | 0.852 | 0.874 | 0.892 | 0.718 | 0.837 | **0.774** |
| MedDINO **d4** | 0.889 | 0.827 | −0.005 | 0.871 | 0.861 | 0.868 | 0.884 | 0.714 | 0.827 | 0.765 |
| MedDINO **Ashwin d4** | 0.888 | 0.830 | −0.002 | 0.864 | 0.863 | 0.867 | 0.883 | **0.747** | 0.823 | 0.763 |
| MedDINO **d8** | 0.886 | 0.821 | −0.011 | 0.865 | 0.846 | 0.857 | 0.869 | 0.733 | 0.821 | 0.758 |
| MedDINO **FiLM d8** (oracle dx) | 0.882 | 0.813 | −0.019 | 0.875 | 0.830 | 0.853 | 0.855 | 0.721 | 0.809 | 0.747 |
| MedDINO **d16** | 0.880 | 0.803 | −0.029 | 0.852 | 0.828 | 0.840 | 0.860 | 0.704 | 0.814 | 0.724 |
| + clDice (planned) | — | — | — | — | — | — | — | — | — | — |
| + post-processing (planned) | — | — | — | — | — | — | — | — | — | — |

### Key findings so far
- **nnU-Net still leads** (0.832 mean-class), but MedDINO d4 / Ashwin-d4 are within
  **0.002–0.005** — effectively on par given n=32.
- **Depth helps:** d4 > d8 > d16 (0.827 > 0.821 > 0.803) — finer through-plane
  resolution clearly beats coarser, consistent with the small-structure hypothesis.
- **Ashwin d4 gives the best myocardium** (0.747, > nnU-Net's 0.718) — the
  channel-averaged, depth-symmetric inflation helps that class specifically.
- **FiLM diagnosis conditioning did not help** (0.813 < 0.821 plain d8) — even with
  oracle diagnosis. Consistent with the FiLM caveat; not pursuing further as-is.
- **PA is the universal weak class** (0.72–0.77 across all models) — the clear
  target for the planned clDice / vessel-sampling phase.

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
