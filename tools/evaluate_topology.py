"""
evaluate_topology.py — offline topology / surface / connectivity evaluation of an
existing 3D segmentation prediction folder against ground truth.

Runs on predictions nnUNet has already written (e.g. a fold's validation folder)
WITHOUT touching training or inference. Produces per-case, aggregate (incl.
hard-case), and optional diagnosis-stratified reports.

Example
-------
python tools/evaluate_topology.py \
--pred_dir /scratch/.../meddinov3_3d_centering_d8_..._3d_fullres/fold_0/validation \
--gt_dir /scratch/.../nnUNet_preprocessed/Dataset030_imageCHD_HU/gt_segmentations \
--dataset_json /scratch/.../nnUNet_raw/Dataset030_imageCHD_HU/dataset.json \
--xlsx /scratch/.../nnUNet_raw/Dataset030_imageCHD_HU/imageCHD_dataset_info.xlsx \
--out_dir /scratch/.../fold_0/validation/topology_eval

The --xlsx argument is optional; without it, diagnosis stratification is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

# Make the repo importable whether run from root or elsewhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (os.path.join(_REPO_ROOT, "nnUNet"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nnunetv2.evaluation.topology_metrics import evaluate_case  # noqa: E402

# Fallback label map (id -> name) if dataset.json is unavailable.
# Matches linear_probe_cardiac.py:79.
_FALLBACK_LABELS = {
    1: "LV-BP", 2: "RV-BP", 3: "LA", 4: "RA", 5: "Myo", 6: "Ao", 7: "PA",
}


def _read_sitk(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    spacing = img.GetSpacing()         # (sx, sy, sz)
    return arr, spacing


def _load_label_map(dataset_json_path):
    """Return {int_id: name} for foreground labels from a dataset.json.

    nnUNet dataset.json 'labels' maps name -> id (id may be int or list for regions).
    We invert it and drop background (id 0).
    """
    if not dataset_json_path or not os.path.isfile(dataset_json_path):
        print(f"[evaluate_topology] dataset.json not found; using fallback labels "
              f"{_FALLBACK_LABELS}")
        return dict(_FALLBACK_LABELS)
    with open(dataset_json_path) as f:
        dj = json.load(f)
    labels = dj.get("labels", {})
    out = {}
    for name, val in labels.items():
        if isinstance(val, (list, tuple)):
            continue  # region-style label, skip for per-class metrics
        try:
            lid = int(val)
        except (TypeError, ValueError):
            continue
        if lid == 0:
            continue  # background
        out[lid] = name
    if not out:
        print("[evaluate_topology] no scalar foreground labels parsed; using fallback")
        return dict(_FALLBACK_LABELS)
    return out


def _find_label_id(label_map, *candidates):
    """Find a label id whose name matches any candidate (case-insensitive substring)."""
    lname = {lid: name.lower() for lid, name in label_map.items()}
    for cand in candidates:
        c = cand.lower()
        for lid, name in lname.items():
            if name == c:
                return lid
    for cand in candidates:
        c = cand.lower()
        for lid, name in lname.items():
            if c in name:
                return lid
    return None


def _pair_files(pred_dir, gt_dir, suffix=".nii.gz"):
    preds = {f for f in os.listdir(pred_dir) if f.endswith(suffix)}
    gts = {f for f in os.listdir(gt_dir) if f.endswith(suffix)}
    common = sorted(preds & gts)
    only_pred = sorted(preds - gts)
    only_gt = sorted(gts - preds)
    if only_pred:
        print(f"[evaluate_topology] {len(only_pred)} prediction(s) without GT "
              f"(ignored): {only_pred[:5]}{'...' if len(only_pred) > 5 else ''}")
    if only_gt:
        print(f"[evaluate_topology] {len(only_gt)} GT(s) without prediction "
              f"(ignored): {only_gt[:5]}{'...' if len(only_gt) > 5 else ''}")
    return common


def _load_diagnosis_map(xlsx_path):
    """Return {numeric_case_id: [diag_names]} reusing the canonical parser."""
    try:
        from add_chd_diagnosis_to_properties import _load_xlsx, _extract_num
    except Exception as e:
        print(f"[evaluate_topology] could not import diagnosis parser ({e}); "
              f"skipping stratification")
        return None, None
    try:
        mapping = _load_xlsx(xlsx_path)
    except SystemExit as e:
        print(f"[evaluate_topology] diagnosis xlsx load failed ({e}); skipping")
        return None, None
    return mapping, _extract_num


def _nanmean(vals):
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Topology/surface/connectivity evaluation "
                                             "of an existing prediction folder.")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--dataset_json", default=None)
    ap.add_argument("--xlsx", default=None, help="imageCHD_dataset_info.xlsx for diagnosis stratification")
    ap.add_argument("--out_dir", default=None, help="defaults to <pred_dir>/topology_eval")
    ap.add_argument("--small_island_voxels", type=int, default=50)
    ap.add_argument("--nsd_tau_mm", type=float, default=1.0)
    ap.add_argument("--suffix", default=".nii.gz")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.pred_dir, "topology_eval")
    os.makedirs(out_dir, exist_ok=True)

    label_map = _load_label_map(args.dataset_json)
    tubular = list(label_map.keys())  # component/connectivity for all foreground; clDice within evaluate_case
    ao = _find_label_id(label_map, "Ao", "aorta")
    pa = _find_label_id(label_map, "PA", "pulmonary_artery", "pulmonary artery")
    print(f"[evaluate_topology] labels: {label_map}")
    print(f"[evaluate_topology] Ao label={ao}  PA label={pa}")

    cases = _pair_files(args.pred_dir, args.gt_dir, args.suffix)
    if not cases:
        sys.exit(f"No common cases between {args.pred_dir} and {args.gt_dir}")
    print(f"[evaluate_topology] {len(cases)} paired cases")

    diag_map, extract_num = (None, None)
    if args.xlsx:
        diag_map, extract_num = _load_diagnosis_map(args.xlsx)

    per_case_flat = []
    audits = {}
    for i, fname in enumerate(cases):
        pred, sp_p = _read_sitk(os.path.join(args.pred_dir, fname))
        gt, sp_g = _read_sitk(os.path.join(args.gt_dir, fname))
        if pred.shape != gt.shape:
            print(f"  [SKIP] shape mismatch {fname}: pred {pred.shape} vs gt {gt.shape}")
            continue
        res = evaluate_case(
            pred.astype(np.int32), gt.astype(np.int32), sp_g, label_map,
            tubular_labels=tubular, ao_label=ao, pa_label=pa,
            small_island_voxels=args.small_island_voxels, nsd_tau_mm=args.nsd_tau_mm,
        )
        row = {"case": fname}
        row.update(res["flat"])
        per_case_flat.append(row)
        audits[fname] = res["audit"]
        print(f"  [{i + 1}/{len(cases)}] {fname}")

    # ── per-case CSV + JSON ──
    all_keys = ["case"]
    for row in per_case_flat:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    csv_path = os.path.join(out_dir, "per_case_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for row in per_case_flat:
            w.writerow(row)
    with open(os.path.join(out_dir, "per_case_metrics.json"), "w") as f:
        json.dump(per_case_flat, f, indent=2)
    with open(os.path.join(out_dir, "per_case_audit.json"), "w") as f:
        json.dump(audits, f, indent=2)

    # ── aggregate + hard-case ──
    metric_keys = [k for k in all_keys if k != "case"]
    aggregate = {"num_cases": len(per_case_flat), "mean": {}}
    for k in metric_keys:
        aggregate["mean"][k] = _nanmean([r.get(k) for r in per_case_flat])

    def _worst(metric, mode="min"):
        vals = [(r["case"], r.get(metric)) for r in per_case_flat
                if r.get(metric) is not None and not (isinstance(r.get(metric), float) and np.isnan(r.get(metric)))]
        if not vals:
            return None
        return min(vals, key=lambda x: x[1]) if mode == "min" else max(vals, key=lambda x: x[1])

    hard = {}
    # worst Dice / clDice across foreground (use mean-over-classes per case)
    def _case_mean(row, prefix):
        vs = [v for kk, v in row.items() if kk.startswith(prefix) and v is not None
              and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vs)) if vs else float("nan")

    dice_means = [(r["case"], _case_mean(r, "dice_")) for r in per_case_flat]
    dice_means = [t for t in dice_means if not np.isnan(t[1])]
    if dice_means:
        hard["worst_mean_dice"] = min(dice_means, key=lambda x: x[1])
    ao_name = label_map.get(ao)
    pa_name = label_map.get(pa)
    if ao_name:
        hard["worst_cldice_Ao"] = _worst(f"cldice_{ao_name}", "min")
        hard["worst_hd95_Ao"] = _worst(f"hd95_{ao_name}", "max")
    if pa_name:
        hard["worst_cldice_PA"] = _worst(f"cldice_{pa_name}", "min")
        hard["worst_hd95_PA"] = _worst(f"hd95_{pa_name}", "max")

    # counts of fragmented structures (>1 predicted component) and switch flags
    def _count(pred_fn):
        return int(sum(1 for r in per_case_flat if pred_fn(r)))

    frag_counts = {}
    for name in label_map.values():
        frag_counts[name] = _count(lambda r, n=name: (r.get(f"ncomp_{n}") or 0) > 1)
    hard["cases_with_fragmentation_per_class"] = frag_counts
    hard["cases_with_ao_pa_switch"] = _count(lambda r: bool(r.get("ao_pa_switch_flag")))
    aggregate["hard_cases"] = hard

    with open(os.path.join(out_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    # ── diagnosis stratification ──
    strat_written = False
    if diag_map is not None and extract_num is not None:
        # group case -> diagnoses
        case_diags = {}
        for r in per_case_flat:
            stem = r["case"]
            for suf in (args.suffix,):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
            num = extract_num(stem)
            case_diags[r["case"]] = diag_map.get(num, [])
        # collect all diagnosis names present
        diag_names = sorted({d for ds in case_diags.values() for d in ds})
        strat = {}
        for d in diag_names:
            rows = [r for r in per_case_flat if d in case_diags[r["case"]]]
            if not rows:
                continue
            strat[d] = {"num_cases": len(rows), "mean": {}}
            for k in metric_keys:
                strat[d]["mean"][k] = _nanmean([r.get(k) for r in rows])
        with open(os.path.join(out_dir, "diagnosis_stratified.json"), "w") as f:
            json.dump(strat, f, indent=2)
        # CSV: one row per diagnosis, key metrics
        strat_csv = os.path.join(out_dir, "diagnosis_stratified.csv")
        with open(strat_csv, "w", newline="") as f:
            cols = ["diagnosis", "num_cases"] + metric_keys
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for d, info in strat.items():
                row = {"diagnosis": d, "num_cases": info["num_cases"]}
                row.update(info["mean"])
                w.writerow(row)
        strat_written = True
    else:
        print("[evaluate_topology] diagnosis stratification skipped (no --xlsx or parser)")

    # ── markdown summary ──
    md = [f"# Topology evaluation — {len(per_case_flat)} cases", ""]
    md.append(f"- pred_dir: `{args.pred_dir}`")
    md.append(f"- gt_dir: `{args.gt_dir}`")
    md.append("")
    md.append("## Mean per-class Dice / HD95 / clDice")
    md.append("")
    md.append("| class | Dice | HD95 (mm) | NSD | clDice | mean #comp | frag |")
    md.append("|---|---|---|---|---|---|---|")
    for name in label_map.values():
        d = aggregate["mean"].get(f"dice_{name}", float("nan"))
        h = aggregate["mean"].get(f"hd95_{name}", float("nan"))
        n = aggregate["mean"].get(f"nsd_{name}", float("nan"))
        c = aggregate["mean"].get(f"cldice_{name}", None)
        nc = aggregate["mean"].get(f"ncomp_{name}", float("nan"))
        fr = aggregate["mean"].get(f"frag_{name}", float("nan"))
        c_str = "—" if c is None or (isinstance(c, float) and np.isnan(c)) else f"{c:.3f}"
        md.append(f"| {name} | {d:.3f} | {h:.2f} | {n:.3f} | {c_str} | {nc:.2f} | {fr:.3f} |")
    md.append("")
    md.append("## Hard cases")
    if "worst_mean_dice" in hard:
        cse, val = hard["worst_mean_dice"]
        md.append(f"- worst mean Dice: **{val:.3f}** ({cse})")
    md.append(f"- cases with Ao/PA label switch: **{hard.get('cases_with_ao_pa_switch', 0)}**")
    md.append(f"- fragmentation (>1 component) per class: "
              f"{', '.join(f'{k}={v}' for k, v in frag_counts.items())}")
    if strat_written:
        md.append("")
        md.append("Diagnosis-stratified metrics written to `diagnosis_stratified.csv`.")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[evaluate_topology] wrote outputs to {out_dir}")
    print("  per_case_metrics.csv / .json, per_case_audit.json, aggregate_metrics.json, summary.md"
          + (", diagnosis_stratified.csv/.json" if strat_written else ""))


if __name__ == "__main__":
    main()
