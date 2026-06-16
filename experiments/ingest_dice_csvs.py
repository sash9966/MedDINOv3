"""
ingest_dice_csvs.py — pull real per-case Dice CSVs (from the external
SegmentationDetailStandard analysis) into the ablation dashboard.

Each CSV is `Patient_ID,WH,LV-BP,RV-BP,LA,RA,Myo,Ao,PA` (one held-out test case
per row). This maps the MedDINO-family files (+ the nnU-Net DA5 baseline) to the
matching registry experiments, computes mean/std per class, the mean subclass
Dice, and Δ vs the nnU-Net baseline, and writes dashboard_data/current_results.json.

Then run `python experiments/update_dashboard_data.py` to embed it into
dashboard.html (it derives status=completed from the presence of metrics).

    python experiments/ingest_dice_csvs.py \
        --dice_dir /Users/.../SegmentationDetailStandard/Dataset030/dice_results

Only the methods that are "ours" are mapped (MedDINO family + baseline). Other
methods in the folder (AuxDiag, CrossAttn, DA5 variants, FiLM_V3, TopoSched, ...)
are intentionally ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DASHBOARD_DATA_DIR, list_experiments, load_registry  # noqa: E402

DEFAULT_DICE_DIR = ("/Users/saschastocker/Documents/Stanford/AlisonMarsden/"
                    "SegmentationDetailStandard/Dataset030/dice_results")

CLASSES = ["LV-BP", "RV-BP", "LA", "RA", "Myo", "Ao", "PA"]
BASELINE_ID = "nnunet_baseline_same_split"

# CSV filename -> primary registry experiment id
PRIMARY = {
    "dice_DA5_Baseline.csv": "nnunet_baseline_same_split",
    "dice_MedDINO.csv": "meddinov3_current_d16",          # MedDINO_3d_center_d16_ensemble
    "dice_MedDINO_d8.csv": "meddinov3_current_d8",
    "dice_MedDINO_d4.csv": "meddinov3_current_d4",
    "dice_MedDINO_Ashwin_d4.csv": "ashwin_d4_inflation",
    "dice_MedDINO_CHD_FiLM_d8.csv": "diagnosis_film_bottleneck",
}
# experiments that share the SAME run/results as a primary id (aliased views)
ALIASES = {
    "meddinov3_current_d16": ["depth_d16", "d16_conv1x1x1_adapter"],
    "meddinov3_current_d8": ["depth_d8", "d8_conv1x1x1_adapter", "encoder_full_finetune"],
    "meddinov3_current_d4": ["depth_d4", "d4_conv1x1x1_adapter"],
    "diagnosis_film_bottleneck": ["oracle_diagnosis_conditioning"],
}


def _stats_for_csv(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    per_class = {}
    for c in CLASSES:
        vals = [float(r[c]) for r in rows]
        per_class[c] = round(st.mean(vals), 4)
    wh = [float(r["WH"]) for r in rows]
    per_case_meancls = [st.mean(float(r[c]) for c in CLASSES) for r in rows]
    return {
        "n_cases": n,
        "whole_heart_dice": round(st.mean(wh), 4),
        "whole_heart_std": round(st.pstdev(wh), 4) if n > 1 else 0.0,
        "mean_class_dice": round(st.mean(per_case_meancls), 4),
        "mean_class_std": round(st.pstdev(per_case_meancls), 4) if n > 1 else 0.0,
        "per_class": per_class,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dice_dir", default=DEFAULT_DICE_DIR)
    args = ap.parse_args()

    dice_dir = Path(args.dice_dir)
    if not dice_dir.is_dir():
        print(f"[error] dice_dir not found: {dice_dir}", file=sys.stderr)
        return 1

    registry = load_registry()
    exps = list_experiments(registry)
    by_id = {e["id"]: e for e in exps}

    # compute stats and fan out to primary + alias experiment ids
    stats_by_exp: dict[str, dict] = {}
    warnings = []
    for fname, pid in PRIMARY.items():
        p = dice_dir / fname
        if not p.is_file():
            warnings.append(f"missing CSV: {fname}")
            continue
        s = _stats_for_csv(p)
        s["source_csv"] = fname
        for eid in [pid] + ALIASES.get(pid, []):
            stats_by_exp[eid] = s

    base = stats_by_exp.get(BASELINE_ID)
    base_mc = base["mean_class_dice"] if base else None

    results = []
    for e in exps:
        eid = e["id"]
        s = stats_by_exp.get(eid)
        row = {
            "id": eid, "group": e.get("group"), "name": e["name"],
            "status": "completed" if s else e.get("status"),
            "implemented": e.get("implemented"), "depth": e.get("meddinov3_depth"),
            "dataset_version": e.get("dataset_version"), "trainer": e.get("trainer"),
            "whole_heart_dice": s["whole_heart_dice"] if s else None,
            "mean_class_dice": s["mean_class_dice"] if s else None,
            "whole_heart_std": s["whole_heart_std"] if s else None,
            "mean_class_std": s["mean_class_std"] if s else None,
            "n_cases": s["n_cases"] if s else None,
            "per_class": s["per_class"] if s else {c: None for c in CLASSES},
            "hd95": None, "cldice_Ao": None, "cldice_PA": None,  # not in these CSVs
            "delta_meancls_vs_nnunet": (round(s["mean_class_dice"] - base_mc, 4)
                                        if s and base_mc is not None and eid != BASELINE_ID else None),
            "source_csv": s.get("source_csv") if s else None,
        }
        results.append(row)

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "results": results,
        "warnings": warnings,
        "source": str(dice_dir),
        "baseline_id": BASELINE_ID,
        "note": "Held-out test-set Dice (32 cases). WH = union-mask whole-heart Dice; "
                "mean_class = mean of 7 structure Dice. Per-case CSVs from the external "
                "SegmentationDetailStandard analysis (dice_analysis.ipynb).",
    }
    (DASHBOARD_DATA_DIR / "current_results.json").write_text(json.dumps(out, indent=2))

    n_with = sum(1 for r in results if r["whole_heart_dice"] is not None)
    print(f"ingested {len(PRIMARY)} CSVs → {n_with} experiment rows with metrics "
          f"({len(stats_by_exp)} incl. aliases). {len(warnings)} warning(s).")
    for fname, pid in PRIMARY.items():
        s = stats_by_exp.get(pid)
        if s:
            d = ("" if pid == BASELINE_ID
                 else f"  Δcls={s['mean_class_dice'] - base_mc:+.3f}" if base_mc else "")
            print(f"  {pid:32s} WH={s['whole_heart_dice']:.3f}  cls={s['mean_class_dice']:.3f}{d}")
    print("\nNext: python experiments/update_dashboard_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
