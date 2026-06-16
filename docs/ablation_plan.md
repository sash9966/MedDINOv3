# CHD MedDINO 3D Inflation — Ablation Plan

*Source of truth for experiments: [`experiments/ablation_registry.yaml`](../experiments/ablation_registry.yaml).
Live status: the "CHD MedDINO 3D Inflation Ablation Plan" section of `dashboard.html`.*

## 1. Research question
Can a 3D-inflated MedDINO model use pretrained representations, 3D anatomy, and
disease-topology information to **match or surpass nnU-Net** for CHD CT
segmentation — especially as the dataset scales from ~100 cases to a larger
expert-annotated cohort?

## 2. Current state
- ~100 contrast-enhanced CT cases, each with a segmentation mask and a diagnosis label.
- MedDINO whole-heart Dice ≈ **0.89**; nnU-Net is slightly higher (exact TODO, same split).
- 8 labels: LV-BP, RV-BP, LA, RA, Myo, Ao, PA.
- Hardest structures: **pulmonary arteries / great arteries / small vessels** —
  small, elongated, topology- and continuity-sensitive.
- Working depth variants: **d16, d8, d4** (centering inflation), plus an Ashwin
  channel-averaging d4 variant and an identity-init FiLM diagnosis-conditioned d8.

## 3. Why nnU-Net is hard to beat
Strong medical-image inductive bias; automated preprocessing/configuration;
reliable augmentation and patch sampling; extremely competitive in small-data 3D.

## 4. Why MedDINO may still be worth it
Pretrained dense features; potential to **scale better** with more/diverse data;
ability to exploit diagnosis/topology metadata; potentially better global context
than a purely convolutional baseline.

## 5. Ablation groups
1. **Pretraining / inflation** — inflated vs random-3D init (d16/d8/d4); centering vs Ashwin.
2. **Input / contrast adapter** — 1×1×1 adapter on/off, init modes.
3. **Depth / context** — d16 vs d8 vs d4 (whole-heart vs small-vessel trade-off).
4. **Fine-tuning** — frozen / last-N-blocks / full / LoRA.
5. **Small-vessel loss & sampling** — class-balanced, focal/Tversky, clDice, vessel-aware sampling.
6. **Diagnosis conditioning** — multi-task head, FiLM (bridge/decoder/multiscale), oracle vs predicted.
7. **FiLM / text supervision** — injection location & strength; text-embedding placeholder.
8. **Post-processing** — class-specific connected components, vessel continuity.
9. **Combined best** — best depth + adapter + (vessel sampling | clDice | diagnosis), and the future-cohort candidate.

## 6. Decision criteria
- **Primary:** mean class Dice and whole-heart Dice.
- **Small-vessel:** Ao/PA Dice, HD95, clDice / centerline connectivity.
- **Clinical plausibility:** fewer disconnected vessel fragments / missing structures (component counts, Ao/PA switch).
- **Statistics:** paired per-case differences with bootstrap 95% CI + Wilcoxon (same split only).
- **Cost:** GPU memory, training time.

## 7. What result would be convincing
- Inflated > random-3D → pretraining transfer is useful in low data.
- 1×1×1 adapter > no adapter → contrast adaptation matters.
- d4 > d16 on small vessels → less z-context preserves fine structures.
- Diagnosis multi-task improves abnormal-anatomy classes → disease labels carry topology.
- FiLM helps with oracle but not predicted diagnosis → scientifically useful, less deployable.
- clDice improves PA connectivity while preserving Dice → topology-aware learning is worth keeping.
- Improvements only on the larger cohort → the 100-case data is underpowered.

## 8. Checklist
- [ ] Confirm fixed split file for current 100-case dataset.
- [ ] Confirm nnU-Net baseline metrics on the same split.
- [ ] Confirm current MedDINO d16/d8/d4 metrics on the same split.
- [ ] Run adapter ablation: no adapter vs 1×1×1 adapter.
- [ ] Run inflation ablation: inflated vs random 3D initialization.
- [ ] Run depth ablation: d16 vs d8 vs d4.
- [ ] Run freeze ablation: frozen vs partial vs full fine-tune.
- [ ] Run vessel-aware sampling ablation.
- [ ] Run class-balanced / focal / Tversky loss ablation.
- [ ] Run clDice / soft-skeleton topology loss for pulmonary/great arteries.
- [ ] Run post-processing ablation for small components and vessel continuity.
- [ ] Run diagnosis multi-task head ablation.
- [ ] Run oracle diagnosis-FiLM ablation.
- [ ] Run predicted-diagnosis conditioning ablation if clinically needed.
- [ ] Compare all completed models to nnU-Net with paired statistics.
- [ ] Generate supervisor summary table and plots.
- [ ] Re-run best ablations on larger expert-annotated cohort when available.

## 9. Implementation status (this phase)
Built now: the registry + runner + SLURM generator + metrics collector + dashboard
data + docs + tests, wired to the **existing** trainers (nnU-Net, centering
d16/d8/d4, Ashwin d4, FiLM d8 incl. bridge/decoder/multiscale + oracle). The
remaining hooks (random-3D init, encoder-freeze, focal/Tversky/clDice **loss**,
vessel-aware sampling, diagnosis multi-task head, post-processing) are tracked as
`implemented: false` registry entries and become their own later phases — no
training-loop changes were made in this phase.
