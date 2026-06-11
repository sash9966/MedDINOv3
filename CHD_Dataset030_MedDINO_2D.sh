#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_2D.sh
#  Dataset030_imageCHD_HU — MedDINOv3 2D baseline (fold 0)
#
#  Trainer : meddinov3_base_primus_multiscale_Trainer
#  Config  : 2d
#  Epochs  : 100
#
#  RESUME SUPPORT
#    Each phase creates a .done marker. Resubmit to continue from last phase.
# =============================================================================
#SBATCH --job-name=D030-MedDINO-2D
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-2D_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-2D_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

export PYTHONPATH="/scratch/users/sastocke/MedDINOv3/nnUNet:${PYTHONPATH:-}"

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────
# 2.  Configuration
# ─────────────────────────────────────────────
DATASET_ID=30
DATASET_NAME="Dataset030_imageCHD_HU"
CONFIG_2D="2d"
TRAINER_2D="meddinov3_base_primus_multiscale_Trainer"

SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
RAW_CKPT="${SHARED_CKPT_DIR}/meddinov3_2d.pth"

SPLITS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
RAW_DATA_DIR="${nnUNet_raw}/${DATASET_NAME}"
PREPROCESSED_2D="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d"

export MEDDINOV3_2D_CHECKPOINT="${RAW_CKPT}"

IN_DIR="${RAW_DATA_DIR}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_MedDINO_2D"
START_TS=$(date +%s)

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}" "${SHARED_CKPT_DIR}"
mark_done()        { touch "${CKPT_DIR}/${1}.done"; }
is_done()          { [[ -f "${CKPT_DIR}/${1}.done" ]]; }
mark_shared_done() { touch "${SHARED_CKPT_DIR}/${1}.done"; }
is_shared_done()   { [[ -f "${SHARED_CKPT_DIR}/${1}.done" ]]; }

# ─────────────────────────────────────────────
# 4.  Banner
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_MedDINO_2D.sh  — START                         ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
printf "║  %-66s ║\n" "Config       : ${CONFIG_2D}"
printf "║  %-66s ║\n" "Trainer      : ${TRAINER_2D}"
printf "║  %-66s ║\n" "Epochs       : 100"
printf "║  %-66s ║\n" "Results      : ${nnUNet_results}/${DATASET_NAME}/${TRAINER_2D}__nnUNetPlans__2d/"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Completed steps (from previous runs, if any):"
ls "${CKPT_DIR}/"*.done 2>/dev/null \
    | xargs -I{} basename {} .done \
    | sort | sed 's/^/    [DONE] /' \
    || echo "    (none — fresh run)"
echo ""

# ─────────────────────────────────────────────
# Phase 1 — Download 2D checkpoint (shared)
# ─────────────────────────────────────────────
if is_shared_done "download_2d_ckpt"; then
    echo "[SKIP] Phase 1: 2D checkpoint already downloaded"
else
    echo "Phase 1: Downloading MedDINOv3 2D checkpoint from HuggingFace"
    cat > /tmp/d030_2d_download.py << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys
shared_dir, raw_ckpt = sys.argv[1], sys.argv[2]
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs(shared_dir, exist_ok=True)
shutil.copy(src, raw_ckpt)
print('Saved:', raw_ckpt)
PYEOF
    python3 /tmp/d030_2d_download.py "${SHARED_CKPT_DIR}" "${RAW_CKPT}"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Plan and preprocess (2D config)
# ─────────────────────────────────────────────
if is_done "p2_preprocess_2d"; then
    echo "[SKIP] Phase 2: 2D preprocess already done"
else
    echo "================================================================"
    echo "Phase 2: plan_and_preprocess — ${CONFIG_2D}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -c ${CONFIG_2D} \
        --verify_dataset_integrity
    mark_done "p2_preprocess_2d"
fi

# ─────────────────────────────────────────────
# Phase 3 — Regenerate splits_final.json from imagesTr only
# The existing splits_final.json may have been created when test cases
# were part of the dataset. This phase regenerates a clean split using
# only the current imagesTr cases (seed=12345, 5-fold CV, same as nnUNet).
# Backs up the old file before overwriting.
# ─────────────────────────────────────────────
if is_done "p3_clean_splits"; then
    echo "[SKIP] Phase 3: splits already regenerated from imagesTr"
else
    echo "================================================================"
    echo "Phase 3: Regenerating splits_final.json from imagesTr cases only"
    echo "================================================================"
    cat > /tmp/d030_2d_regen_splits.py << 'PYEOF'
import json, os, sys, re, shutil
import numpy as np
from sklearn.model_selection import KFold

splits_file  = sys.argv[1]
imagesTr_dir = sys.argv[2]
imagesTs_dir = sys.argv[3]

def strip_suffix(name):
    return re.sub(r'_\d{4}\.nii\.gz$', '', name)

tr_ids = sorted({strip_suffix(f) for f in os.listdir(imagesTr_dir) if f.endswith('.nii.gz')})
ts_ids = {strip_suffix(f) for f in os.listdir(imagesTs_dir) if f.endswith('.nii.gz')}

print(f"imagesTr cases : {len(tr_ids)}")
print(f"imagesTs cases : {len(ts_ids)}")

overlap = set(tr_ids) & ts_ids
if overlap:
    print(f"FATAL: {len(overlap)} cases exist in BOTH imagesTr and imagesTs — fix the dataset first.")
    for k in sorted(overlap): print(f"  {k}")
    sys.exit(1)

# Check if existing split is already clean
needs_regen = True
if os.path.isfile(splits_file):
    existing = json.load(open(splits_file))
    all_split_ids = set()
    for fold in existing:
        all_split_ids.update(fold.get('train', []))
        all_split_ids.update(fold.get('val', []))
    leaked = all_split_ids & ts_ids
    unknown = all_split_ids - set(tr_ids) - ts_ids
    if not leaked and not unknown:
        print("Existing splits_final.json is already clean — no regeneration needed.")
        print(f"Fold 0: {len(existing[0]['train'])} train / {len(existing[0]['val'])} val")
        needs_regen = False
    else:
        print(f"Existing splits_final.json has {len(leaked)} test-case leaks and {len(unknown)} unknown IDs — regenerating.")
        backup = splits_file + ".contaminated_backup"
        shutil.copy(splits_file, backup)
        print(f"Backed up to: {backup}")

if needs_regen:
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    splits = []
    arr = np.array(tr_ids)
    for train_idx, val_idx in kf.split(arr):
        splits.append({'train': list(arr[train_idx]), 'val': list(arr[val_idx])})
    json.dump(splits, open(splits_file, 'w'), indent=4)
    print(f"Wrote clean splits_final.json with {len(tr_ids)} imagesTr cases.")
    print(f"Fold 0: {len(splits[0]['train'])} train / {len(splits[0]['val'])} val")

# Final safety check: fold 0 must have zero test-case overlap
splits = json.load(open(splits_file))
f0_leak_tr  = [k for k in splits[0]['train'] if k in ts_ids]
f0_leak_val = [k for k in splits[0]['val']   if k in ts_ids]
if f0_leak_tr or f0_leak_val:
    print(f"\nFATAL: fold 0 still contaminated after regeneration — aborting.")
    sys.exit(2)
print("Fold 0 verified CLEAN — safe to train.")
PYEOF
    python3 /tmp/d030_2d_regen_splits.py \
        "${SPLITS_FILE}" \
        "${RAW_DATA_DIR}/imagesTr" \
        "${RAW_DATA_DIR}/imagesTs"
    mark_done "p3_clean_splits"
fi

# ─────────────────────────────────────────────
# Phase 4 — MedDINOv3 2D training (fold 0 only)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: MedDINOv3 2D training — fold 0  (100 epochs)"
echo "================================================================"
KEY="2d_D${DATASET_ID}_fold0"
if is_shared_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${CONFIG_2D} 0 \
        -tr ${TRAINER_2D} --npz
    mark_shared_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 5 — Inference on test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: Inference on imagesTs"
echo "================================================================"
mkdir -p "${PRED_BASE}"
PRED_2D="${PRED_BASE}/MedDINO_2d_fold0"
mkdir -p "${PRED_2D}"

if is_shared_done "p5_infer_2d_D${DATASET_ID}"; then
    echo "[SKIP] inference already done"
else
    echo "--- inference ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_2D}" \
        -d ${DATASET_ID} -c ${CONFIG_2D} \
        -f 0 \
        -tr ${TRAINER_2D}
    mark_shared_done "p5_infer_2d_D${DATASET_ID}"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
elapsed=$(( $(date +%s) - START_TS ))
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_MedDINO_2D.sh  — COMPLETE                      ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Elapsed : $(( elapsed/3600 ))h $(( (elapsed%3600)/60 ))m $(( elapsed%60 ))s"
printf "║  %-66s ║\n" "Results : ${nnUNet_results}/${DATASET_NAME}/${TRAINER_2D}__nnUNetPlans__2d/"
printf "║  %-66s ║\n" "Preds   : ${PRED_2D}"
echo "╚══════════════════════════════════════════════════════════════════╝"
