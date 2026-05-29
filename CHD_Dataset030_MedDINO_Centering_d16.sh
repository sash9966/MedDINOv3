#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_Centering_d16.sh
#  Dataset030_imageCHD_HU — MedDINOv3 3D centering inflation, d_patch=16, 500 ep
#
#  PURPOSE — safe baseline for 3D training.
#  This is the lowest-risk 3D configuration: centering inflation (activation-
#  preserving) + d_patch=16 (token count closest to pretraining distribution).
#  Run this BEFORE d=8 / Ashwin variants to confirm the pipeline can converge
#  at all on Dataset030.
#
#  Token budget:
#    (96/16) * (160/16) * (160/16) = 6 * 10 * 10 = 600 tokens (3.1x pretrained)
#
#  Trainer fixes vs earlier runs:
#    - AdamW (not SGD) with betas=(0.9, 0.98), 10-ep warmup
#    - clip_grad_norm = 1.0   (ViT-appropriate; was 12 → effectively no clip)
#    - patch_embed_3d weight_decay = 0  (don't decay the inflated pretrained kernel)
#    - depth_pos_embed weight_decay = 0
#    - backbone_lr_scale = 0.1  (=> 3e-5 effective; was 0.3, too aggressive early)
#
#  Experiments (fold 0):
#    1. MedDINOv3 2D  (meddinov3_base_primus_multiscale_Trainer, 2d)
#       Shared with other CHD_Dataset030_MedDINO*.sh — skipped if already done.
#    2. MedDINOv3 3D centering d=16  (meddinov3_3d_primus_multiscale_Trainer)
#
#  RESUME SUPPORT
#    Resubmit the same script to continue from where it stopped.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-MedDINO-C16
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-C16_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-C16_%j.err

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

# d_patch=16: keeps token count closest to pretraining (600 tok, 3.1x pretrained 196).
# Safe baseline before pushing finer depth resolution (d=8 → 1200 tok, 6.1x).
D_PATCH=16

REPO="/scratch/users/sastocke/MedDINOv3"
SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
RAW_CKPT="${SHARED_CKPT_DIR}/meddinov3_2d.pth"
INFLATED_CKPT="${SHARED_CKPT_DIR}/meddinov3_inflated_center_d16.pth"
INFLATE_SCRIPT="${REPO}/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/inflate_weights_3d.py"

export MEDDINOV3_2D_CHECKPOINT="${RAW_CKPT}"
export MEDDINOV3_3D_CHECKPOINT="${INFLATED_CKPT}"
export MEDDINOV3_D_PATCH="${D_PATCH}"
export MEDDINOV3_NUM_EPOCHS=500
export MEDDINOV3_BACKBONE_LR_SCALE=0.1

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_MedDINO_Centering_d16"
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
    echo "║  CHD_Dataset030_MedDINO_Centering_d16.sh — START               ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Configs      : ${CONFIG_2D}  |  ${CONFIG_3D}"
    printf "║  %-66s ║\n" "Folds        : 0 only"
    printf "║  %-66s ║\n" "Epochs       : 2D=100  3D=${MEDDINOV3_NUM_EPOCHS}"
    printf "║  %-66s ║\n" "d_patch      : ${D_PATCH}  (600 tokens, 3.1x pretrained — SAFE)"
    printf "║  %-66s ║\n" "Inflation    : centering (activation-preserving)"
    printf "║  %-66s ║\n" "Optimizer    : AdamW betas=(0.9,0.98), warmup=10ep, clip=1.0"
    printf "║  %-66s ║\n" "Backbone LR  : ${MEDDINOV3_BACKBONE_LR_SCALE}  (patch_embed+depth_pe wd=0)"
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
    echo "║  CHD_Dataset030_MedDINO_Centering_d16.sh — COMPLETE            ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed      : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "2D preds     : ${PRED_BASE}/MedDINO_2d_ensemble"
    printf "║  %-66s ║\n" "3D preds     : ${PRED_BASE}/MedDINO_3d_center_d16_ensemble"
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
    cat > /tmp/c16_download_ckpt.py << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys
shared_dir = sys.argv[1]
raw_ckpt   = sys.argv[2]
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs(shared_dir, exist_ok=True)
shutil.copy(src, raw_ckpt)
print('Saved:', raw_ckpt)
PYEOF
    python3 /tmp/c16_download_ckpt.py "${SHARED_CKPT_DIR}" "${RAW_CKPT}"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Inflate weights (centering, d=16, shared)
# ─────────────────────────────────────────────
if is_shared_done "inflate_center_d16"; then
    echo "[SKIP] Phase 2: centering-inflated d=16 checkpoint already exists"
else
    echo "================================================================"
    echo "Phase 2: Centering inflation  d_patch=16"
    echo "================================================================"
    python3 "${INFLATE_SCRIPT}" \
        --checkpoint "${RAW_CKPT}" \
        --d_patch "${D_PATCH}" \
        --inflation centering \
        --out "${INFLATED_CKPT}"
    mark_shared_done "inflate_center_d16"
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
# Phase 3.5 — Pre-unpack dataset (shared, locked)
# WHY: nnUNetv2_train calls unpack_dataset() at on_train_start(). If two
# training jobs start at the same time on the same shared filesystem
# (nnUNet_preprocessed on Oak/Scratch), both find no .npy files and race
# to write them — corrupting writes and killing one job.
# Fix: unpack once here with flock so subsequent trainers skip unpacking.
# ─────────────────────────────────────────────
UNPACK_LOCK="${SHARED_CKPT_DIR}/unpack_D${DATASET_ID}.lock"
for _CFG in "${CONFIG_2D}" "${CONFIG_3D}"; do
    KEY="unpack_D${DATASET_ID}_${_CFG}"
    if is_shared_done "${KEY}"; then
        echo "[SKIP] Phase 3.5: ${_CFG} already unpacked"
    else
        echo "================================================================"
        echo "Phase 3.5: Pre-unpack ${_CFG} (flock-protected)"
        echo "================================================================"
        (
            flock -x 9
            if ! is_shared_done "${KEY}"; then
                cat > /tmp/c16_unpack.py << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get('PYTHONPATH', '').split(':')[0])
from nnunetv2.training.dataloading.utils import unpack_dataset
folder = sys.argv[1]
print(f"Unpacking: {folder}")
unpack_dataset(folder, unpack_segmentation=True, overwrite_existing=False, num_processes=4, verify_npy=True)
print("Done.")
PYEOF
                _PREP_FOLDER="${nnUNet_preprocessed}/${DATASET_NAME}"
                if [[ "${_CFG}" == "2d" ]]; then
                    _DATA_ID="nnUNetPlans_2d"
                else
                    _DATA_ID="nnUNetPlans_3d_fullres"
                fi
                python3 /tmp/c16_unpack.py "${_PREP_FOLDER}/${_DATA_ID}"
                mark_shared_done "${KEY}"
            fi
        ) 9>"${UNPACK_LOCK}_${_CFG}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4 — MedDINOv3 2D training (fold 0)
# Shared marker — skip if another CHD_Dataset030_MedDINO*.sh already ran it.
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
# Phase 5 — MedDINOv3 3D centering d=16 training (fold 0)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: MedDINOv3 3D centering d=16 training — fold 0  (${MEDDINOV3_NUM_EPOCHS} epochs)"
echo "================================================================"
for FOLD in 0; do
    KEY="p5_3d_center_d16_fold${FOLD}"
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
PRED_3D="${PRED_BASE}/MedDINO_3d_center_d16_ensemble"
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

if is_done "p6_infer_3d_center_d16"; then
    echo "[SKIP] p6_infer_3d_center_d16"
else
    echo "--- p6_infer_3d_center_d16 ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_3D}" \
        -d ${DATASET_ID} -c ${CONFIG_3D} \
        -f 0 \
        -tr ${TRAINER_3D}
    mark_done "p6_infer_3d_center_d16"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
