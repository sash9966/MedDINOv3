#!/bin/bash
# =============================================================================
#  CHD_Dataset050_MedDINO_Centering_d4.sh
#  Dataset050_imageCHD_DiseaseLandmarks — MedDINOv3 3D centering inflation,
#  d_patch=4, 500 ep
#
#  Direct counterpart to CHD_Dataset030_MedDINO_Centering_d4.sh, retargeted at
#  Dataset050_imageCHD_DiseaseLandmarks (adds a susceptible-defect label on top
#  of the Dataset030 label set — everything else is identical). Finest depth
#  resolution rung: d=4 = 2400 tokens (~12.2x pretrained) vs d=8 = 1200 (6.1x)
#  and d=16 = 600 (3.1x). Self-attention is O(N^2), so this is ~4x the d=8
#  attention memory — START AT batch_size=1. Watch the per-epoch [GPU] line in
#  the training log; if peak_reserved leaves clear headroom under total, bump
#  MEDDINOV3_BATCH_SIZE to 2 and resubmit.
#
#  Same centering inflation on the SAME base 2D weights, re-inflated d_patch=4.
#
#  Trainer:  meddinov3_3d_centering_d4_primus_multiscale_Trainer
#  Results:  nnUNet_results/.../meddinov3_3d_centering_d4_primus_multiscale_Trainer__nnUNetPlans__3d_fullres/
#
#  RESUME SUPPORT — resubmit the same script to continue.
# =============================================================================
#SBATCH --job-name=D050-MedDINO-C4
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D050-MedDINO-C4_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D050-MedDINO-C4_%j.err

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
DATASET_ID=50
DATASET_NAME="Dataset050_imageCHD_DiseaseLandmarks"
CONFIG_3D="3d_fullres"
TRAINER_3D="meddinov3_3d_centering_d4_primus_multiscale_Trainer"
D_PATCH=4

REPO="/scratch/users/sastocke/MedDINOv3"
SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
RAW_CKPT="${SHARED_CKPT_DIR}/meddinov3_2d.pth"
INFLATED_CKPT="${SHARED_CKPT_DIR}/meddinov3_inflated_center_d4.pth"
INFLATE_SCRIPT="${REPO}/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/inflate_weights_3d.py"

export MEDDINOV3_2D_CHECKPOINT="${RAW_CKPT}"
export MEDDINOV3_3D_CHECKPOINT="${INFLATED_CKPT}"
export MEDDINOV3_D_PATCH="${D_PATCH}"
export MEDDINOV3_NUM_EPOCHS=500
export MEDDINOV3_BACKBONE_LR_SCALE=0.1
# bs=2 (matches the Dataset030 d=4 runs). Watch the per-epoch [GPU] line in the
# training log; if it OOMs at the 2400-token budget, drop back to 1.
export MEDDINOV3_BATCH_SIZE=2

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_DIR="${nnUNet_results}/${DATASET_NAME}/predictions/MedDINO_3d_center_d4"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset050_MedDINO_Centering_d4"
START_TS=$(date +%s)

# ─────────────────────────────────────────────
# 3.  Helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}" "${SHARED_CKPT_DIR}"
mark_done()        { touch "${CKPT_DIR}/${1}.done"; }
is_done()          { [[ -f "${CKPT_DIR}/${1}.done" ]]; }
mark_shared_done() { touch "${SHARED_CKPT_DIR}/${1}.done"; }
is_shared_done()   { [[ -f "${SHARED_CKPT_DIR}/${1}.done" ]]; }

# ─────────────────────────────────────────────
# 4.  Banner + GPU readout
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset050_MedDINO_Centering_d4.sh — START                ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
printf "║  %-66s ║\n" "Trainer      : ${TRAINER_3D}"
printf "║  %-66s ║\n" "d_patch      : ${D_PATCH}  (2400 tokens, ~12.2x pretrained)"
printf "║  %-66s ║\n" "batch_size   : ${MEDDINOV3_BATCH_SIZE}  (drop to 1 if the [GPU] log OOMs)"
printf "║  %-66s ║\n" "Inflation    : centering (same approach as d=8/d=16)"
printf "║  %-66s ║\n" "Optimizer    : AdamW, warmup=10ep, clip=1.0, backbone_lr=0.1"
printf "║  %-66s ║\n" "Epochs       : ${MEDDINOV3_NUM_EPOCHS}"
printf "║  %-66s ║\n" "Results      : nnUNet_results/${DATASET_NAME}/${TRAINER_3D}__nnUNetPlans__3d_fullres/"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# nvidia-smi snapshot so the .out log records the exact GPU + total VRAM up front.
echo "──────── GPU at job start (nvidia-smi) ────────"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version \
    --format=csv 2>/dev/null || echo "  (nvidia-smi unavailable)"
echo "───────────────────────────────────────────────"
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
    echo "Phase 1: Downloading MedDINOv3 2D checkpoint"
    cat > /tmp/c4_download_ckpt.py << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys
shared_dir = sys.argv[1]
raw_ckpt   = sys.argv[2]
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs(shared_dir, exist_ok=True)
shutil.copy(src, raw_ckpt)
print('Saved:', raw_ckpt)
PYEOF
    python3 /tmp/c4_download_ckpt.py "${SHARED_CKPT_DIR}" "${RAW_CKPT}"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Inflate weights (centering, d=4, shared)
# ─────────────────────────────────────────────
if is_shared_done "inflate_center_d4"; then
    echo "[SKIP] Phase 2: centering-inflated d=4 checkpoint already exists"
else
    echo "Phase 2: Centering inflation d_patch=4"
    python3 "${INFLATE_SCRIPT}" \
        --checkpoint "${RAW_CKPT}" \
        --d_patch "${D_PATCH}" \
        --inflation centering \
        --out "${INFLATED_CKPT}"
    mark_shared_done "inflate_center_d4"
fi

# ─────────────────────────────────────────────
# Phase 3 — 3D centering d=4 training (fold 0)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 3: 3D centering d=4 training — fold 0  (${MEDDINOV3_NUM_EPOCHS} epochs)"
echo "================================================================"
KEY="p3_3d_center_d4_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${CONFIG_3D} 0 \
        -tr ${TRAINER_3D} --npz
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 4 — Inference on test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Inference"
echo "================================================================"
mkdir -p "${PRED_DIR}"

if is_done "p4_infer_3d_center_d4"; then
    echo "[SKIP] p4_infer_3d_center_d4"
else
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_DIR}" \
        -d ${DATASET_ID} -c ${CONFIG_3D} \
        -f 0 \
        -tr ${TRAINER_3D}
    mark_done "p4_infer_3d_center_d4"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
elapsed=$(( $(date +%s) - START_TS ))
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset050_MedDINO_Centering_d4.sh — COMPLETE             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Elapsed : $(( elapsed/3600 ))h $(( (elapsed%3600)/60 ))m $(( elapsed%60 ))s"
printf "║  %-66s ║\n" "Results : ${nnUNet_results}/${DATASET_NAME}/${TRAINER_3D}__nnUNetPlans__3d_fullres/"
printf "║  %-66s ║\n" "Preds   : ${PRED_DIR}"
echo "╚══════════════════════════════════════════════════════════════════╝"
