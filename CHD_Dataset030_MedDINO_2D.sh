#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_2D.sh
#  Dataset030_imageCHD_HU — MedDINOv3 2D baseline (5-fold ensemble)
#
#  Trainer : meddinov3_base_primus_multiscale_Trainer
#  Config  : 2d
#  Epochs  : 100
#
#  RESUME SUPPORT
#    Each fold creates a .done marker. Resubmit to continue from last fold.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
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

export MEDDINOV3_2D_CHECKPOINT="${RAW_CKPT}"

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
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
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset030_MedDINO_2D.sh  — START                         ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Config       : ${CONFIG_2D}"
    printf "║  %-66s ║\n" "Trainer      : ${TRAINER_2D}"
    printf "║  %-66s ║\n" "Folds        : 0 only  (extend to 0-4 for full ensemble)"
    printf "║  %-66s ║\n" "Epochs       : 100"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "2D ckpt      : ${RAW_CKPT}"
    printf "║  %-66s ║\n" "Raw data     : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results      : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Marker dir   : ${CKPT_DIR}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Completed steps (from previous runs, if any):"
    ls "${CKPT_DIR}/"*.done 2>/dev/null \
        | xargs -I{} basename {} .done \
        | sort | sed 's/^/    [DONE] /' \
        || echo "    (none — fresh run)"
    echo ""
}

print_footer() {
    local elapsed=$(( $(date +%s) - START_TS ))
    local hh=$(( elapsed / 3600 ))
    local mm=$(( (elapsed % 3600) / 60 ))
    local ss=$(( elapsed % 60 ))
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset030_MedDINO_2D.sh  — COMPLETE                      ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed      : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "2D preds     : ${PRED_BASE}/MedDINO_2d_ensemble"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Next step:"
    printf "║  %-66s ║\n" "  nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIG_2D} -tr ${TRAINER_2D}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner

# ─────────────────────────────────────────────
# Phase 1 — Download 2D checkpoint (shared)
# ─────────────────────────────────────────────
if is_shared_done "download_2d_ckpt"; then
    echo "[SKIP] Phase 1: 2D checkpoint already downloaded"
else
    echo "================================================================"
    echo "Phase 1: Downloading MedDINOv3 2D checkpoint from HuggingFace"
    echo "================================================================"
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs('${SHARED_CKPT_DIR}', exist_ok=True)
shutil.copy(src, '${RAW_CKPT}')
print('Saved:', '${RAW_CKPT}')
"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Plan and preprocess
# ─────────────────────────────────────────────
if is_done "p2_preprocess"; then
    echo "[SKIP] Phase 2: preprocess already done"
else
    echo "================================================================"
    echo "Phase 2: plan_and_preprocess — ${CONFIG_2D}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -c ${CONFIG_2D} \
        --verify_dataset_integrity
    mark_done "p2_preprocess"
fi

# ─────────────────────────────────────────────
# Phase 3 — MedDINOv3 2D training (5 folds)
# Shared markers so parallel jobs skip already-done folds.
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 3: MedDINOv3 2D training — 5 folds"
echo "================================================================"
for FOLD in 0; do
    KEY="2d_D${DATASET_ID}_fold${FOLD}"
    if is_shared_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CONFIG_2D} ${FOLD} \
            -tr ${TRAINER_2D} --npz
        mark_shared_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4 — Inference on test set (5-fold ensemble)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Inference — 5-fold ensemble"
echo "================================================================"
mkdir -p "${PRED_BASE}"
PRED_2D="${PRED_BASE}/MedDINO_2d_ensemble"
mkdir -p "${PRED_2D}"

if is_shared_done "p6_infer_2d_D${DATASET_ID}"; then
    echo "[SKIP] inference (already done by another job)"
else
    echo "--- inference ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_2D}" \
        -d ${DATASET_ID} -c ${CONFIG_2D} \
        -f 0 \
        -tr ${TRAINER_2D}
    mark_shared_done "p6_infer_2d_D${DATASET_ID}"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
