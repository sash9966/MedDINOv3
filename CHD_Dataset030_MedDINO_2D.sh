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
# Phase 3 — Verify splits file (no test-case leak)
# Checks that every case in splits_final.json exists
# in imagesTr and NOT in imagesTs.
# ─────────────────────────────────────────────
if is_done "p3_verify_splits"; then
    echo "[SKIP] Phase 3: splits verification already passed"
else
    echo "================================================================"
    echo "Phase 3: Verifying splits_final.json — no test-case contamination"
    echo "================================================================"
    cat > /tmp/d030_2d_verify_splits.py << 'PYEOF'
import json, os, sys, re

splits_file   = sys.argv[1]
imagesTr_dir  = sys.argv[2]
imagesTs_dir  = sys.argv[3]
preprocessed  = sys.argv[4]

# Build sets of known training and test case IDs
def strip_suffix(name):
    return re.sub(r'_\d{4}\.nii\.gz$', '', name)

tr_ids = {strip_suffix(f) for f in os.listdir(imagesTr_dir) if f.endswith('.nii.gz')}
ts_ids = {strip_suffix(f) for f in os.listdir(imagesTs_dir) if f.endswith('.nii.gz')}

print(f"imagesTr cases : {len(tr_ids)}")
print(f"imagesTs cases : {len(ts_ids)}")

# Check splits file exists
if not os.path.isfile(splits_file):
    print(f"ERROR: {splits_file} does not exist — run training first to generate it, or check preprocessing.")
    sys.exit(1)

splits = json.load(open(splits_file))
print(f"splits_final.json: {len(splits)} folds")

leaked_into_train = []
leaked_into_val   = []

for fold_idx, fold in enumerate(splits):
    for key in fold.get('train', []):
        if key in ts_ids:
            leaked_into_train.append((fold_idx, key))
    for key in fold.get('val', []):
        if key in ts_ids:
            leaked_into_val.append((fold_idx, key))
    # also check that all split cases are actually in imagesTr
    unknown = [k for k in fold.get('train', []) + fold.get('val', []) if k not in tr_ids]
    if unknown:
        print(f"  WARNING fold {fold_idx}: {len(unknown)} split cases not in imagesTr: {unknown[:5]}{'...' if len(unknown)>5 else ''}")

if leaked_into_train:
    print(f"\nDATA LEAK DETECTED: {len(leaked_into_train)} test cases in training splits:")
    for fi, k in leaked_into_train:
        print(f"  fold {fi} train: {k}")
    sys.exit(2)

if leaked_into_val:
    print(f"\nDATA LEAK DETECTED: {len(leaked_into_val)} test cases in validation splits:")
    for fi, k in leaked_into_val:
        print(f"  fold {fi} val: {k}")
    sys.exit(2)

# Check preprocessed folder only has training cases
preproc_ids = set()
if os.path.isdir(preprocessed):
    for f in os.listdir(preprocessed):
        if f.endswith('.npz') or f.endswith('.npy'):
            preproc_ids.add(re.sub(r'\.(npz|npy)$', '', f))
leaked_preproc = preproc_ids & ts_ids
if leaked_preproc:
    print(f"\nDATA LEAK DETECTED: {len(leaked_preproc)} test cases in preprocessed 2D folder:")
    for k in sorted(leaked_preproc):
        print(f"  {k}")
    sys.exit(2)

print("\nSplit verification PASSED — no test cases in training or validation splits.")
print(f"Fold 0: {len(splits[0]['train'])} train / {len(splits[0]['val'])} val cases")
PYEOF
    python3 /tmp/d030_2d_verify_splits.py \
        "${SPLITS_FILE}" \
        "${RAW_DATA_DIR}/imagesTr" \
        "${RAW_DATA_DIR}/imagesTs" \
        "${PREPROCESSED_2D}"
    mark_done "p3_verify_splits"
fi

# ─────────────────────────────────────────────
# Phase 4 — MedDINOv3 2D training (fold 0)
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
