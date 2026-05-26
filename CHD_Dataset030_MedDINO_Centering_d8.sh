#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_Centering_d8.sh
#  Dataset030_imageCHD_HU — MedDINOv3 3D centering inflation, d_patch=8, 500 ep
#
#  Experiments (fold 0):
#    1. MedDINOv3 2D  (meddinov3_base_primus_multiscale_Trainer, 2d)
#       Shared with CHD_Dataset030_MedDINO.sh — skipped if already done.
#    2. MedDINOv3 3D centering d=8  (meddinov3_3d_primus_multiscale_Trainer, 3d_fullres)
#       Inflation: centering (activation-preserving; 2D weights on centre slice).
#       d_patch=8  → (96/8)*(160/16)*(160/16) = 1200 tokens (6.1x pretrained).
#       Token budget will print *** HIGH — intentional experiment testing whether
#       finer depth resolution (8 vs 16) compensates for the higher OOD ratio
#       when combined with activation-preserving inflation + AdamW.
#
#  Optimizer: AdamW (fixed vs earlier SGD runs), warmup=10ep, 500 epochs.
#
#  RESUME SUPPORT
#    Resubmit the same script to continue from where it stopped.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-MedDINO-C8
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-C8_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-C8_%j.err

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
CONFIG_3D="3d_fullres"
TRAINER_2D="meddinov3_base_primus_multiscale_Trainer"
TRAINER_3D="meddinov3_3d_primus_multiscale_Trainer"

# d_patch=8: finer depth resolution vs. d_patch=16 (activation-preserving centering
# can tolerate the higher token count better than Ashwin).
# Token count: (96/8)*(160/16)*(160/16) = 1200 tokens (6.1x pretrained — HIGH).
D_PATCH=8

REPO="/scratch/users/sastocke/MedDINOv3"
SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
RAW_CKPT="${SHARED_CKPT_DIR}/meddinov3_2d.pth"
INFLATED_CKPT="${SHARED_CKPT_DIR}/meddinov3_inflated_center_d8.pth"
INFLATE_SCRIPT="${REPO}/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/inflate_weights_3d.py"

export MEDDINOV3_2D_CHECKPOINT="${RAW_CKPT}"
export MEDDINOV3_3D_CHECKPOINT="${INFLATED_CKPT}"
export MEDDINOV3_D_PATCH="${D_PATCH}"
export MEDDINOV3_NUM_EPOCHS=500
export MEDDINOV3_BACKBONE_LR_SCALE=0.3

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_MedDINO_Centering_d8"
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
# 4.  Banner helpers
# ─────────────────────────────────────────────
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset030_MedDINO_Centering_d8.sh  — START               ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Configs      : ${CONFIG_2D}  |  ${CONFIG_3D}"
    printf "║  %-66s ║\n" "Folds        : 0 only"
    printf "║  %-66s ║\n" "Epochs       : 2D=100  3D=${MEDDINOV3_NUM_EPOCHS}"
    printf "║  %-66s ║\n" "d_patch      : ${D_PATCH}  (1200 tokens, 6.1x pretrained — HIGH)"
    printf "║  %-66s ║\n" "Inflation    : centering (activation-preserving)"
    printf "║  %-66s ║\n" "Optimizer    : AdamW betas=(0.9,0.98), warmup=10ep"
    printf "║  %-66s ║\n" "Backbone LR  : ${MEDDINOV3_BACKBONE_LR_SCALE}"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1 — 2D   : ${TRAINER_2D}"
    printf "║  %-66s ║\n" "Exp 2 — 3D   : ${TRAINER_3D}"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "2D ckpt      : ${RAW_CKPT}"
    printf "║  %-66s ║\n" "3D ckpt      : ${INFLATED_CKPT}"
    printf "║  %-66s ║\n" "Raw data     : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results      : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Inference    : ${PRED_BASE}"
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
    echo "║  CHD_Dataset030_MedDINO_Centering_d8.sh  — COMPLETE            ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed      : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "2D preds     : ${PRED_BASE}/MedDINO_2d_ensemble"
    printf "║  %-66s ║\n" "3D preds     : ${PRED_BASE}/MedDINO_3d_center_d8_ensemble"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Next step:"
    printf "║  %-66s ║\n" "  nnUNetv2_find_best_configuration ${DATASET_ID} \\"
    printf "║  %-66s ║\n" "    -c ${CONFIG_2D} ${CONFIG_3D} \\"
    printf "║  %-66s ║\n" "    -tr ${TRAINER_2D} ${TRAINER_3D}"
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
    cat > /tmp/c8_download_ckpt.py << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys
shared_dir = sys.argv[1]
raw_ckpt   = sys.argv[2]
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs(shared_dir, exist_ok=True)
shutil.copy(src, raw_ckpt)
print('Saved:', raw_ckpt)
PYEOF
    python3 /tmp/c8_download_ckpt.py "${SHARED_CKPT_DIR}" "${RAW_CKPT}"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Inflate weights (centering, d=8, shared)
# ─────────────────────────────────────────────
if is_shared_done "inflate_center_d8"; then
    echo "[SKIP] Phase 2: centering-inflated d=8 checkpoint already exists"
else
    echo "================================================================"
    echo "Phase 2: Centering inflation  d_patch=8"
    echo "================================================================"
    python3 "${INFLATE_SCRIPT}" \
        --checkpoint "${RAW_CKPT}" \
        --d_patch "${D_PATCH}" \
        --inflation centering \
        --out "${INFLATED_CKPT}"
    mark_shared_done "inflate_center_d8"
fi

# ─────────────────────────────────────────────
# Phase 3 — Plan and preprocess
# ─────────────────────────────────────────────
if is_done "p3_preprocess"; then
    echo "[SKIP] Phase 3: preprocess already done"
else
    echo "================================================================"
    echo "Phase 3: plan_and_preprocess — ${CONFIG_2D} | ${CONFIG_3D}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -c ${CONFIG_2D} ${CONFIG_3D} \
        --verify_dataset_integrity
    mark_done "p3_preprocess"
fi

# ─────────────────────────────────────────────
# Phase 4 — MedDINOv3 2D training (fold 0)
# Shared marker — skip if CHD_Dataset030_MedDINO.sh already ran it.
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: MedDINOv3 2D training — fold 0"
echo "================================================================"
for FOLD in 0; do
    KEY="2d_D${DATASET_ID}_fold${FOLD}"
    if is_shared_done "${KEY}"; then
        echo "[SKIP] ${KEY} (already done by another job)"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CONFIG_2D} ${FOLD} \
            -tr ${TRAINER_2D} --npz
        mark_shared_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 5 — MedDINOv3 3D centering d=8 training (fold 0)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: MedDINOv3 3D centering d=8 training — fold 0  (${MEDDINOV3_NUM_EPOCHS} epochs)"
echo "================================================================"
for FOLD in 0; do
    KEY="p5_3d_center_d8_fold${FOLD}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CONFIG_3D} ${FOLD} \
            -tr ${TRAINER_3D} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 6 — Inference on test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 6: Inference"
echo "================================================================"
mkdir -p "${PRED_BASE}"

PRED_2D="${PRED_BASE}/MedDINO_2d_ensemble"
PRED_3D="${PRED_BASE}/MedDINO_3d_center_d8_ensemble"
mkdir -p "${PRED_2D}" "${PRED_3D}"

if is_shared_done "p6_infer_2d_D${DATASET_ID}"; then
    echo "[SKIP] p6_infer_2d (already done by another job)"
else
    echo "--- p6_infer_2d ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_2D}" \
        -d ${DATASET_ID} -c ${CONFIG_2D} \
        -f 0 \
        -tr ${TRAINER_2D}
    mark_shared_done "p6_infer_2d_D${DATASET_ID}"
fi

if is_done "p6_infer_3d_center_d8"; then
    echo "[SKIP] p6_infer_3d_center_d8"
else
    echo "--- p6_infer_3d_center_d8 ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_3D}" \
        -d ${DATASET_ID} -c ${CONFIG_3D} \
        -f 0 \
        -tr ${TRAINER_3D}
    mark_done "p6_infer_3d_center_d8"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
