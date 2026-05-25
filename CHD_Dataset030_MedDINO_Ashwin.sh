#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_Ashwin.sh
#  Dataset030_imageCHD_HU — MedDINOv3 2D + 3D Ashwin_3d_inflation
#
#  Experiments (5-fold ensemble):
#    1. MedDINOv3 2D  (meddinov3_base_primus_multiscale_Trainer, 2d)
#    2. MedDINOv3 3D  (meddinov3_3d_ashwin_primus_multiscale_Trainer, 3d_fullres)
#       3D weights use Ashwin_3d_inflation: channel-sum -> redistribute -> depth-tile.
#
#  RESUME SUPPORT
#    Each training run creates a .done marker in CKPT_DIR.
#    Resubmit the same script to continue from where it stopped.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-MeDINO-A
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-Ashwin_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-MedDINO-Ashwin_%j.err

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
TRAINER_3D="meddinov3_3d_ashwin_primus_multiscale_Trainer"

# d_patch controls how many CT slices are merged into one depth token.
# Larger d_patch = fewer tokens, closer to the ViT's pretraining distribution (~196 tokens).
#   d_patch=2  → ~2400 tokens  (12x over pretrained — too many, ViT cannot learn)
# Dataset030 3d_fullres patch_size = [96, 160, 160].  Token counts:
#   d_patch=8  → (96/8)*(160/16)*(160/16) = 1200 tokens  (6.1x pretrained — too many)
#   d_patch=16 → (96/16)*(160/16)*(160/16) =  600 tokens  (3.1x pretrained — recommended)
D_PATCH=16

REPO="/scratch/users/sastocke/MedDINOv3"
SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
RAW_CKPT="${SHARED_CKPT_DIR}/meddinov3_2d.pth"
INFLATED_CKPT="${SHARED_CKPT_DIR}/meddinov3_inflated_ashwin_d${D_PATCH}.pth"
INFLATE_SCRIPT="${REPO}/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/inflate_weights_3d_ashwin.py"

export MEDDINOV3_2D_CHECKPOINT="${RAW_CKPT}"
export MEDDINOV3_3D_CHECKPOINT="${INFLATED_CKPT}"
export MEDDINOV3_D_PATCH="${D_PATCH}"

# Backbone LR scale: fraction of decoder LR applied to the pretrained ViT weights.
# Set to 0 to freeze the backbone entirely for the first run (decoder-only warm-up),
# then set back to 0.05 on subsequent submissions to fine-tune the backbone.
export MEDDINOV3_BACKBONE_LR_SCALE=0.05

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_MedDINO_Ashwin"
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
    echo "║  CHD_Dataset030_MedDINO_Ashwin.sh  — START                     ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Configs      : ${CONFIG_2D}  |  ${CONFIG_3D}"
    printf "║  %-66s ║\n" "Folds        : 0 only  (extend to 0-4 for full ensemble)"
    printf "║  %-66s ║\n" "Epochs       : 2D=100  3D=200"
    printf "║  %-66s ║\n" "d_patch      : ${D_PATCH}  (3D depth tokenisation)"
    printf "║  %-66s ║\n" "Inflation    : Ashwin_3d_inflation (channel-sum)"
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
    echo "║  CHD_Dataset030_MedDINO_Ashwin.sh  — COMPLETE                  ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed      : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "2D preds     : ${PRED_BASE}/MedDINO_2d_ensemble"
    printf "║  %-66s ║\n" "3D preds     : ${PRED_BASE}/MedDINO_3d_ashwin_ensemble"
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
# Phase 2 — Inflate weights (Ashwin_3d_inflation, shared)
# ─────────────────────────────────────────────
if is_shared_done "inflate_ashwin_d${D_PATCH}"; then
    echo "[SKIP] Phase 2: Ashwin-inflated checkpoint already exists"
else
    echo "================================================================"
    echo "Phase 2: Ashwin_3d_inflation  d_patch=${D_PATCH}"
    echo "================================================================"
    python3 "${INFLATE_SCRIPT}" \
--checkpoint "${RAW_CKPT}" \
--d_patch "${D_PATCH}" \
--out "${INFLATED_CKPT}"
    mark_shared_done "inflate_ashwin_d${D_PATCH}"
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
# Phase 4 — MedDINOv3 2D training (5 folds)
# Shared markers: skipped if CHD_Dataset030_MedDINO.sh
# already ran the 2D folds.
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: MedDINOv3 2D training — 5 folds"
echo "================================================================"
for FOLD in 0; do
    KEY="2d_D${DATASET_ID}_fold${FOLD}"
    if is_shared_done "${KEY}"; then
        echo "[SKIP] ${KEY} (already done by centering job)"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CONFIG_2D} ${FOLD} \
            -tr ${TRAINER_2D} --npz
        mark_shared_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 5 — MedDINOv3 3D Ashwin training (5 folds)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: MedDINOv3 3D Ashwin_3d_inflation training — 5 folds  (d_patch=${D_PATCH})"
echo "================================================================"
for FOLD in 0; do
    KEY="p5_3d_ashwin_MedDINO_fold${FOLD}"
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
# Phase 6 — Inference on test set (5-fold ensemble)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 6: Inference — 5-fold ensemble"
echo "================================================================"
mkdir -p "${PRED_BASE}"

PRED_2D="${PRED_BASE}/MedDINO_2d_ensemble"
PRED_3D="${PRED_BASE}/MedDINO_3d_ashwin_ensemble"
mkdir -p "${PRED_2D}" "${PRED_3D}"

if is_shared_done "p6_infer_2d_D${DATASET_ID}"; then
    echo "[SKIP] p6_infer_2d (already done by centering job)"
else
    echo "--- p6_infer_2d ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_2D}" \
        -d ${DATASET_ID} -c ${CONFIG_2D} \
        -f 0 \
        -tr ${TRAINER_2D}
    mark_shared_done "p6_infer_2d_D${DATASET_ID}"
fi

if is_done "p6_infer_3d_ashwin"; then
    echo "[SKIP] p6_infer_3d_ashwin"
else
    echo "--- p6_infer_3d_ashwin ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_3D}" \
        -d ${DATASET_ID} -c ${CONFIG_3D} \
        -f 0 \
        -tr ${TRAINER_3D}
    mark_done "p6_infer_3d_ashwin"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
