"""
add_chd_diagnosis_to_properties.py — inject per-case multi-hot CHD diagnosis
vectors into nnUNet preprocessed case .pkl property files.

Reuses the EXISTING disease mapping in the nnunet_CHD folder so the conditioned
trainer compares fairly against prior work. Writes two keys into each case's
properties dict:
    properties['diagnosis_vec']    list[float], length = len(DIAGNOSIS_ORDER) (multi-hot)
    properties['diagnosis_names']  list[str] of the diagnoses present for the case

A case may have several diagnoses (multi-label) -> several 1s in the vector.
A case absent from the mapping gets an all-zero vector (== UNK / unconditioned).

-------------------------------------------------------------------------------
IMPORTANT — adapt load_case_to_diagnoses() to the real mapping file format.
The mapping lives on the server (/scratch/users/sastocke/nnunet_CHD), not in this
repo, so the parser below supports the two most common shapes and is selected by
file extension. Confirm the column/key names match your file, then freeze
DIAGNOSIS_ORDER (its order defines the vector layout and must never change).
-------------------------------------------------------------------------------
"""

import argparse
import csv
import json
import os

from batchgenerators.utilities.file_and_folder_operations import (
    subfiles, load_pickle, write_pickle, isfile,
)

# Canonical, FROZEN ordering of the multi-hot vector. Index = bit position.
# EDIT to match the label set in the nnunet_CHD mapping (and keep it fixed).
DIAGNOSIS_ORDER = [
    "VSD",
    "ASD",
    "Coarctation",
    "Pulmonary_Atresia",
]


def load_case_to_diagnoses(mapping_path: str) -> dict:
    """Return {case_id: [diagnosis_name, ...]} from the existing mapping file.

    Supported formats (selected by extension):
      .json : {"case_id": ["VSD", "ASD"], ...}  OR  {"case_id": "VSD", ...}
      .csv  : columns 'case_id' and 'diagnosis'; diagnosis may be a single label
              or a delimited list (';' or '|' or ',') for multi-label cases.
    Adapt the column/key names here if the real file differs.
    """
    ext = os.path.splitext(mapping_path)[1].lower()
    mapping = {}
    if ext == ".json":
        with open(mapping_path) as f:
            raw = json.load(f)
        for case_id, val in raw.items():
            mapping[case_id] = val if isinstance(val, list) else [val]
    elif ext == ".csv":
        with open(mapping_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row["case_id"]
                diag = row["diagnosis"].strip()
                for sep in (";", "|", ","):
                    if sep in diag:
                        names = [d.strip() for d in diag.split(sep) if d.strip()]
                        break
                else:
                    names = [diag] if diag else []
                mapping[case_id] = names
    else:
        raise ValueError(f"Unsupported mapping extension '{ext}' (use .json or .csv)")
    return mapping


def to_multihot(names) -> list:
    idx = {name: i for i, name in enumerate(DIAGNOSIS_ORDER)}
    vec = [0.0] * len(DIAGNOSIS_ORDER)
    for n in names:
        if n not in idx:
            raise KeyError(
                f"Diagnosis '{n}' not in DIAGNOSIS_ORDER {DIAGNOSIS_ORDER}. "
                "Add it (append, do not reorder) and keep the order frozen."
            )
        vec[idx[n]] = 1.0
    return vec


def inject(preprocessed_folder: str, mapping_path: str) -> None:
    case_to_diag = load_case_to_diagnoses(mapping_path)
    pkls = subfiles(preprocessed_folder, suffix=".pkl", join=True)
    n_hit = n_unk = 0
    for pkl in pkls:
        case_id = os.path.basename(pkl)[:-4]
        names = case_to_diag.get(case_id, [])
        if names:
            n_hit += 1
        else:
            n_unk += 1
        props = load_pickle(pkl)
        props["diagnosis_names"] = names
        props["diagnosis_vec"] = to_multihot(names)
        write_pickle(props, pkl)
    print(f"Injected diagnosis into {len(pkls)} cases in {preprocessed_folder}")
    print(f"  with diagnosis: {n_hit}   UNK/empty: {n_unk}   vec_len={len(DIAGNOSIS_ORDER)}")
    print(f"  CHD_NUM_DIAGNOSES={len(DIAGNOSIS_ORDER)}  (export this for training)")


def main():
    ap = argparse.ArgumentParser(description="Inject multi-hot CHD diagnosis into case .pkl files.")
    ap.add_argument("--preprocessed_folder", required=True,
                    help="e.g. .../nnUNet_preprocessed/Dataset030_imageCHD_HU/nnUNetPlans_3d_fullres")
    ap.add_argument("--mapping", required=True,
                    help="Path to the existing disease mapping (.json or .csv) in nnunet_CHD")
    args = ap.parse_args()
    inject(args.preprocessed_folder, args.mapping)


if __name__ == "__main__":
    main()
