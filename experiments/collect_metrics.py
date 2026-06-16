"""
collect_metrics.py — gather metrics from nnUNet/MedDINO result folders into
experiments/results/summary.{csv,json} + dashboard_data/current_results.json, and
paired comparisons into experiments/results/comparisons.{csv,json}.

Reuses what the pipeline already writes:
  - nnUNet  fold_<f>/validation/summary.json   (per-label Dice, foreground_mean)
  - Phase-A fold_<f>/validation/topology_eval/aggregate_metrics.json (+ per_case)

Never fails on missing data: absent metrics become null with a warning. Paired
comparisons only run between experiments whose split-file hash matches (fairness).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DASHBOARD_DATA_DIR, RESULTS_DIR, dataset_name, list_experiments, load_registry,
    results_dir_for, split_file_hash,
)

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

import numpy as np

WARN: list[str] = []


def warn(msg):
    WARN.append(msg)
    print(f"[warn] {msg}", file=sys.stderr)


def _label_names(exp, registry):
    return registry["meta"].get("label_map", {})


def _read_summary(val_dir: Path):
    p = val_dir / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        warn(f"could not parse {p}: {e}")
        return None


def _read_topology(val_dir: Path):
    p = val_dir / "topology_eval" / "aggregate_metrics.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("mean", {})
    except Exception as e:
        warn(f"could not parse {p}: {e}")
        return None


def _per_class_dice(summary, label_map):
    out = {}
    means = (summary or {}).get("mean", {})
    for lid, name in label_map.items():
        entry = means.get(str(lid)) or means.get(lid)
        out[name] = (entry or {}).get("Dice") if entry else None
    return out


def _per_case_fg_dice(summary):
    """case_id -> mean foreground Dice (averaged over present labels)."""
    out = {}
    for rec in (summary or {}).get("metric_per_case", []):
        ref = rec.get("reference_file") or rec.get("prediction_file") or ""
        cid = os.path.basename(ref).split(".")[0]
        vals = []
        for _lid, m in (rec.get("metrics") or {}).items():
            d = m.get("Dice")
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                vals.append(d)
        if cid and vals:
            out[cid] = float(np.mean(vals))
    return out


def collect_one(exp, registry):
    fold = exp.get("folds", [0])[0]
    label_map = _label_names(exp, registry)
    row = {
        "id": exp["id"], "group": exp.get("group"), "name": exp["name"],
        "status": exp.get("status"), "implemented": exp.get("implemented"),
        "dataset_version": exp.get("dataset_version"), "depth": exp.get("meddinov3_depth"),
        "fold": fold, "trainer": exp.get("trainer"),
        "whole_heart_dice": None, "mean_class_dice": None,
        "hd95": None, "cldice_Ao": None, "cldice_PA": None,
        "split_sha1": split_file_hash(exp, registry),
        "results_path": None, "per_class": {n: None for n in label_map.values()},
        "_per_case_fg": {},
    }
    val_dir = results_dir_for(exp, registry, fold)
    if val_dir is None or not val_dir.is_dir():
        if exp.get("status") == "completed":
            warn(f"{exp['id']}: marked completed but no results dir at {val_dir}")
        return row
    row["results_path"] = str(val_dir)

    summary = _read_summary(val_dir)
    if summary:
        row["whole_heart_dice"] = (summary.get("foreground_mean") or {}).get("Dice")
        pc = _per_class_dice(summary, label_map)
        row["per_class"] = pc
        present = [v for v in pc.values() if v is not None]
        row["mean_class_dice"] = float(np.mean(present)) if present else None
        row["_per_case_fg"] = _per_case_fg_dice(summary)
    else:
        warn(f"{exp['id']}: no summary.json in {val_dir}")

    topo = _read_topology(val_dir)
    if topo:
        row["hd95"] = topo.get("hd95_Ao") or topo.get("hd95_PA")
        row["cldice_Ao"] = topo.get("cldice_Ao")
        row["cldice_PA"] = topo.get("cldice_PA")
    return row


def _bootstrap_ci(diffs, n=2000, seed=0):
    if len(diffs) < 2:
        return [None, None]
    rng = np.random.RandomState(seed)
    arr = np.asarray(diffs, dtype=float)
    boot = [np.mean(arr[rng.randint(0, len(arr), len(arr))]) for _ in range(n)]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def compare(exp_row, base_row):
    a, b = exp_row["_per_case_fg"], base_row["_per_case_fg"]
    if exp_row["split_sha1"] and base_row["split_sha1"] and exp_row["split_sha1"] != base_row["split_sha1"]:
        warn(f"skip comparison {exp_row['id']} vs {base_row['id']}: different split hashes")
        return None
    common = sorted(set(a) & set(b))
    if not common:
        return None
    diffs = [a[c] - b[c] for c in common]
    p = None
    if _HAVE_SCIPY and len(diffs) >= 1 and any(d != 0 for d in diffs):
        try:
            p = float(wilcoxon(diffs).pvalue)
        except Exception:
            p = None
    elif not _HAVE_SCIPY:
        warn("scipy unavailable — skipping Wilcoxon p-values")
    return {
        "experiment": exp_row["id"], "baseline": base_row["id"], "n_cases": len(common),
        "mean_diff": float(np.mean(diffs)), "median_diff": float(np.median(diffs)),
        "ci95": _bootstrap_ci(diffs), "wilcoxon_p": p,
    }


def main() -> int:
    registry = load_registry()
    exps = list_experiments(registry)
    rows = [collect_one(e, registry) for e in exps]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # summary.json / current_results.json (drop the heavy per-case dict)
    public = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "_per_case_fg"}
        public.append(rr)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(public, indent=2))
    (DASHBOARD_DATA_DIR / "current_results.json").write_text(
        json.dumps({"results": public, "warnings": WARN}, indent=2))

    # summary.csv (flat)
    cols = ["id", "group", "name", "status", "dataset_version", "depth", "fold",
            "whole_heart_dice", "mean_class_dice", "hd95", "cldice_Ao", "cldice_PA", "results_path"]
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + list(registry["meta"]["label_map"].values()))
        w.writeheader()
        for r in public:
            flat = {c: r.get(c) for c in cols}
            flat.update(r.get("per_class", {}))
            w.writerow(flat)

    # comparisons vs key baselines
    by_id = {r["id"]: r for r in rows}
    baselines = [bid for bid in ("nnunet_baseline_same_split", "meddinov3_current_d8") if bid in by_id]
    comparisons = []
    for r in rows:
        if not r["_per_case_fg"]:
            continue
        for bid in baselines:
            if r["id"] == bid:
                continue
            c = compare(r, by_id[bid])
            if c:
                comparisons.append(c)
    (RESULTS_DIR / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    with open(RESULTS_DIR / "comparisons.csv", "w", newline="") as f:
        cc = ["experiment", "baseline", "n_cases", "mean_diff", "median_diff", "ci95", "wilcoxon_p"]
        w = csv.DictWriter(f, fieldnames=cc)
        w.writeheader()
        for c in comparisons:
            w.writerow(c)

    n_with = sum(1 for r in rows if r["whole_heart_dice"] is not None)
    print(f"collected {len(rows)} experiments · {n_with} with metrics · "
          f"{len(comparisons)} paired comparisons · {len(WARN)} warning(s)")
    print(f"wrote: {RESULTS_DIR}/summary.{{csv,json}}, comparisons.{{csv,json}}; "
          f"{DASHBOARD_DATA_DIR}/current_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
