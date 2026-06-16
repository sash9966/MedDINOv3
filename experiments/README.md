# CHD MedDINO Ablation Harness

Config-driven infrastructure to **define, inspect, generate, run, and track** the
ablation study for the 3D-inflated MedDINO CHD segmentation project. The single
source of truth is [`ablation_registry.yaml`](ablation_registry.yaml); every other
tool derives from it. Switching to the future ~10× cohort is a one-line change
(`dataset_version` / `dataset_id`) — no code edits.

## Files
| File | Purpose |
|------|---------|
| `ablation_registry.yaml` | All experiments (full schema). `implemented:true` = runnable now via an existing trainer; `implemented:false` = planned, needs a training hook (`requires:[...]`). |
| `_common.py` | Shared loader + command/env composer + manifest writer. |
| `validate_registry.py` | Schema / unique-id / required-field / d16-d8-d4 checks. |
| `run_ablation.py` | `--list`, `--id ID --dry-run` / `--print-command` / `--run --yes`; filters; writes `results/<id>/manifest.json`. |
| `generate_slurm.py` | Emit `slurm_generated/<id>.sh` (mirrors cluster header + inflation). **Never submits.** |
| `collect_metrics.py` | Gather nnUNet `summary.json` + Phase-A `topology_eval/` → `results/summary.{csv,json}`, `dashboard_data/current_results.json`, paired `comparisons.{csv,json}`. (Use when results live in `$nnUNet_results`.) |
| `ingest_dice_csvs.py` | Pull external per-case Dice CSVs (the SegmentationDetailStandard `dice_*.csv` files) → `dashboard_data/current_results.json`. (Use when results come from that analysis, not local nnUNet dirs.) |
| `update_dashboard_data.py` | Build `dashboard_data/*.json` and embed into `dashboard.html`; derives `status: completed` + result summary from whichever metrics are present (preserves checklist ticks unless `--reset-checklist`). |

### Loading real results
Two sources feed `dashboard_data/current_results.json`; pick the one matching where your numbers live, then run `update_dashboard_data.py`:
```
# A) external per-case Dice CSVs (current setup):
python experiments/ingest_dice_csvs.py --dice_dir /path/to/Dataset030/dice_results
python experiments/update_dashboard_data.py

# B) local nnUNet result folders:
python experiments/collect_metrics.py
python experiments/update_dashboard_data.py
```
Note: `collect_metrics.py` overwrites `current_results.json`, so run `ingest_dice_csvs.py` *after* it (or use only one source). The dashboard derives completed-status from the metrics present, and HD95/clDice fill in once `tools/evaluate_topology.py` has run on the prediction folders.

## Typical flow
```
python experiments/validate_registry.py
python experiments/run_ablation.py --list --enabled-only
python experiments/run_ablation.py --id meddinov3_current_d4 --print-command
python experiments/generate_slurm.py --priority high      # then: sbatch experiments/slurm_generated/<id>.sh
# after runs finish:
python experiments/collect_metrics.py
python experiments/update_dashboard_data.py               # refresh dashboard.html
```

## Adding real metric results
`collect_metrics.py` reads each experiment's
`$nnUNet_results/<dataset>/<trainer>__nnUNetPlans__<config>/fold_<f>/validation/`:
- `summary.json` (nnUNet, always written by training) → per-class + whole-heart Dice;
- `topology_eval/aggregate_metrics.json` (run `tools/evaluate_topology.py` first) → HD95 / clDice / components.
Set the experiment's `status: completed` in the registry, rerun `collect_metrics.py`
then `update_dashboard_data.py`. Missing metrics become `null` with a warning — it
never crashes.

## Launching one high-priority ablation
```
python experiments/generate_slurm.py --id meddinov3_current_d4
sbatch experiments/slurm_generated/meddinov3_current_d4.sh
```

## Planned (not yet runnable) experiments
Anything with `implemented: false` lists the missing capability in `requires`
(e.g. `cldice_loss`, `vessel_sampling`, `encoder_freeze`, `random_3d_init`,
`diagnosis_multitask_head`, `postprocessing`). They appear in `--list`, the
dashboard matrix, and docs, but `--run` refuses them until the hook is built in a
later phase. This keeps the existing training loop untouched.
