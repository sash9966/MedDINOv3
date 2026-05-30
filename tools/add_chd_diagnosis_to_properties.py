"""
add_chd_diagnosis_to_properties.py — inject per-case multi-hot CHD diagnosis
vectors into nnUNet preprocessed case .pkl property files.

Reads imageCHD_dataset_info.xlsx (the spreadsheet at nnUNet_raw/Dataset030_*/),
which has one row per case (index=1001..1178) and 16 binary diagnosis columns.
Writes two keys into each case's .pkl:

    properties['diagnosis_vec']    list[float], len 18 (multi-hot)
    properties['diagnosis_names']  list[str] of diagnoses present

18-label canonical vector = 16 spreadsheet diagnoses + HLHS + TA
(the two NIH/CDC CCHD types absent from this cohort; they get all-zeros here
but the model slot is reserved so the vector is future-proof).

A case absent from the spreadsheet gets an all-zero vector (UNK = unconditioned).
"""

import argparse
import os
import re
import sys

from batchgenerators.utilities.file_and_folder_operations import (
    subfiles, load_pickle, write_pickle,
)

# ── Canonical 18-label vector (FROZEN — never reorder) ──────────────────────
# Columns 0-15 match the spreadsheet column order exactly.
# Columns 16-17 are the two NIH/CDC CCHDs not in this dataset (always 0 here).
DIAGNOSIS_ORDER = [
    "ASD",    # 0  Atrial Septal Defect
    "VSD",    # 1  Ventricular Septal Defect
    "AVSD",   # 2  Atrioventricular Septal Defect
    "ToF",    # 3  Tetralogy of Fallot
    "TGA",    # 4  Transposition of Great Arteries
    "DORV",   # 5  Double Outlet Right Ventricle
    "CAT",    # 6  Common Arterial Trunk (Truncus Arteriosus)
    "CA",     # 7  Coarctation of Aorta
    "AAH",    # 8  Aortic Arch Hypoplasia
    "DAA",    # 9  Double Aortic Arch
    "IAA",    # 10 Interrupted Aortic Arch
    "PA",     # 11 Pulmonary Atresia
    "APVC",   # 12 Anomalous Pulmonary Venous Connection (TAPVR)
    "DSVC",   # 13 Drainage of Superior Vena Cava variant
    "PDA",    # 14 Patent Ductus Arteriosus
    "PAS",    # 15 Pulmonary Artery Stenosis
    "HLHS",   # 16 Hypoplastic Left Heart Syndrome (NIH/CDC — absent in dataset)
    "TA",     # 17 Tricuspid Atresia (NIH/CDC — absent in dataset)
]
# CHD_NUM_DIAGNOSES = 18
# ────────────────────────────────────────────────────────────────────────────


def _load_xlsx(xlsx_path: str) -> dict:
    """Return {numeric_case_id_str: [diag, ...]} from the spreadsheet."""
    try:
        import openpyxl
    except ImportError:
        sys.exit("Install openpyxl: pip install openpyxl")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]  # ('index', 'ASD', 'VSD', ...)
    diag_cols = list(header[1:])  # the 16 diagnosis column names

    mapping = {}
    for row in rows[1:]:
        if row[0] is None:
            continue
        case_num = str(int(row[0]))
        names = [diag_cols[j] for j, v in enumerate(row[1:]) if v]
        mapping[case_num] = names
    return mapping


def _extract_num(identifier: str) -> str:
    """Extract trailing digit string from a case identifier.

    Examples:
        "1001"              -> "1001"
        "imageCHD_1001"     -> "1001"
        "ct_1001_0000"      -> "1001"  (first group of >=3 digits wins)
    """
    nums = re.findall(r'\d{3,}', identifier)
    return nums[0] if nums else identifier


def inject(preprocessed_folder: str, xlsx_path: str, dry_run: bool = False) -> None:
    mapping = _load_xlsx(xlsx_path)
    diag_idx = {name: i for i, name in enumerate(DIAGNOSIS_ORDER)}

    def to_multihot(names):
        vec = [0.0] * len(DIAGNOSIS_ORDER)
        for n in names:
            if n not in diag_idx:
                print(f"  WARNING: '{n}' not in DIAGNOSIS_ORDER — skipped")
                continue
            vec[diag_idx[n]] = 1.0
        return vec

    pkls = sorted(subfiles(preprocessed_folder, suffix=".pkl", join=True))
    if not pkls:
        sys.exit(f"No .pkl files found in {preprocessed_folder}")

    n_hit = n_unk = n_multi = 0
    for pkl in pkls:
        stem = os.path.basename(pkl)[:-4]          # e.g. "imageCHD_1001"
        num = _extract_num(stem)                    # e.g. "1001"
        names = mapping.get(num, [])

        if names:
            n_hit += 1
        else:
            n_unk += 1
        if len(names) > 1:
            n_multi += 1

        if not dry_run:
            props = load_pickle(pkl)
            props["diagnosis_names"] = names
            props["diagnosis_vec"] = to_multihot(names)
            write_pickle(props, pkl)
        else:
            print(f"  [DRY] {stem} ({num}) -> {names}")

    print()
    print(f"{'[DRY RUN] ' if dry_run else ''}Processed {len(pkls)} cases in:")
    print(f"  {preprocessed_folder}")
    print(f"  matched: {n_hit}  unk/zero: {n_unk}  multi-label: {n_multi}")
    print(f"  vector length: {len(DIAGNOSIS_ORDER)}  (CHD_NUM_DIAGNOSES=18)")
    if n_unk:
        print(f"  NOTE: {n_unk} cases have no diagnosis entry → all-zero vector (unconditioned)")


def main():
    ap = argparse.ArgumentParser(
        description="Inject 18-label multi-hot CHD diagnosis into nnUNet case .pkl files."
    )
    ap.add_argument(
        "--preprocessed_folder", required=True,
        help="Path to the nnUNetPlans_3d_fullres folder containing case .pkl files",
    )
    ap.add_argument(
        "--xlsx", required=True,
        help="Path to imageCHD_dataset_info.xlsx "
             "(at nnUNet_raw/Dataset030_imageCHD_HU/imageCHD_dataset_info.xlsx)",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Print mappings without writing anything",
    )
    args = ap.parse_args()
    inject(args.preprocessed_folder, args.xlsx, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
