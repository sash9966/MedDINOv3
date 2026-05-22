#!/bin/bash
# =============================================================================
#  CHD_Dataset030_LinearProbe.sh
#  Dataset030_imageCHD_HU — MedDINOv3 linear probe segmentation baseline
#
#  Trains ONLY a single 1×1 conv head on top of frozen MedDINOv3 features.
#  The ViT backbone is never updated — pure test of pretrained representation.
#
#  Architecture:
#    MedDINOv3 (frozen) → blocks [2,5,8,11] → cat(3072-d) → Conv2d(3072,8,1)
#    → bilinear upsample → segmentation mask
#
#  Two passes:
#    Pass A (this script):  pure linear probe (backbone_lr=0)
#    Pass B (uncomment):    optional low-LR backbone fine-tuning (backbone_lr=3e-6)
#
#  RESUME SUPPORT
#    Done markers are written per-pass.  Resubmit to skip completed passes.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-LinProbe
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-LinProbe_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-LinProbe_%j.err

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
export PYTHONUNBUFFERED=1

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"

# ─────────────────────────────────────────────
# 2.  Configuration
# ─────────────────────────────────────────────
DATASET_ID=30
DATASET_NAME="Dataset030_imageCHD_HU"
FOLD=0

REPO="/scratch/users/sastocke/MedDINOv3"
SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
RAW_CKPT="${SHARED_CKPT_DIR}/meddinov3_2d.pth"

PREPROCESSED="${nnUNet_preprocessed}/${DATASET_NAME}"
OUT_BASE="${nnUNet_results}/${DATASET_NAME}/linear_probe"

# Pass A: pure linear probe (backbone_lr=0 → backbone is frozen)
OUT_A="${OUT_BASE}/pretrained_linear_fold${FOLD}"
EPOCHS_A=50
HEAD_LR_A=1e-3
BACKBONE_LR_A=0.0
BATCH_SIZE=16

# Pass B: optional low-LR backbone fine-tuning (uncomment Phase 5 below)
# OUT_B="${OUT_BASE}/pretrained_finetune_fold${FOLD}"
# EPOCHS_B=50
# HEAD_LR_B=1e-3
# BACKBONE_LR_B=3e-6

CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_LinearProbe"
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
    echo "║  CHD_Dataset030_LinearProbe.sh  — START                        ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Fold         : ${FOLD}"
    printf "║  %-66s ║\n" "Backbone     : FROZEN  (pure linear probe)"
    printf "║  %-66s ║\n" "Head arch    : Conv2d(3072, 8, kernel=1) + bilinear"
    printf "║  %-66s ║\n" "Features     : blocks [2,5,8,11]  embed=768×4=3072"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Checkpoint   : ${RAW_CKPT}"
    printf "║  %-66s ║\n" "Preprocessed : ${PREPROCESSED}"
    printf "║  %-66s ║\n" "Output A     : ${OUT_A}"
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
    echo "║  CHD_Dataset030_LinearProbe.sh  — COMPLETE                     ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed      : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Results A    : ${OUT_A}/results.json"
    printf "║  %-66s ║\n" "Log A        : ${OUT_A}/training_log.csv"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "To inspect results:"
    printf "║  %-66s ║\n" "  cat ${OUT_A}/results.json"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner

# ─────────────────────────────────────────────
# Phase 1 — Download 2D checkpoint (shared with other scripts)
# ─────────────────────────────────────────────
if is_shared_done "download_2d_ckpt"; then
    echo "[SKIP] Phase 1: 2D checkpoint already downloaded"
else
    echo "================================================================"
    echo "Phase 1: Downloading MedDINOv3 2D checkpoint from HuggingFace"
    echo "================================================================"
    cat > /tmp/lp_download_ckpt.py << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys
shared_dir = sys.argv[1]
raw_ckpt   = sys.argv[2]
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs(shared_dir, exist_ok=True)
shutil.copy(src, raw_ckpt)
print('Saved:', raw_ckpt)
PYEOF
    python3 /tmp/lp_download_ckpt.py "${SHARED_CKPT_DIR}" "${RAW_CKPT}"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Verify preprocessed data exists
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Checking preprocessed data"
echo "================================================================"
if [[ ! -f "${PREPROCESSED}/splits_final.json" ]]; then
    echo "ERROR: splits_final.json not found at ${PREPROCESSED}"
    echo "       Run nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -c 2d first"
    exit 1
fi
echo "  splits_final.json  : OK"
N_PLAN_DIRS=$(ls -d "${PREPROCESSED}/nnUNetPlans_"* 2>/dev/null | wc -l)
echo "  plan directories   : ${N_PLAN_DIRS} found"

# ─────────────────────────────────────────────
# Phase 3 — Preprocess if needed (2D config only, for slice data)
# ─────────────────────────────────────────────
if is_done "p3_preprocess_2d"; then
    echo "[SKIP] Phase 3: preprocess already done"
else
    echo "================================================================"
    echo "Phase 3: plan_and_preprocess — 2d (for linear probe slice data)"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -c 2d \
        --verify_dataset_integrity
    mark_done "p3_preprocess_2d"
fi

# ─────────────────────────────────────────────
# Phase 4 — Linear probe (Pass A: pure frozen backbone)
# ─────────────────────────────────────────────
if is_done "p4_linear_probe_fold${FOLD}"; then
    echo "[SKIP] Phase 4: linear probe (Pass A) already done"
else
    echo "================================================================"
    echo "Phase 4: Linear probe — backbone FROZEN, fold ${FOLD}, ${EPOCHS_A} epochs"
    echo "================================================================"
    mkdir -p "${OUT_A}"
    python3 "${REPO}/linear_probe_cardiac.py" \
--preprocessed "${PREPROCESSED}" \
--checkpoint "${RAW_CKPT}" \
--fold ${FOLD} \
--epochs ${EPOCHS_A} \
--batch_size ${BATCH_SIZE} \
--head_lr ${HEAD_LR_A} \
--backbone_lr ${BACKBONE_LR_A} \
--workers 8 \
--out "${OUT_A}"
    mark_done "p4_linear_probe_fold${FOLD}"
fi

echo ""
echo "================================================================"
echo "Pass A results:"
echo "================================================================"
if [[ -f "${OUT_A}/results.json" ]]; then
    cat "${OUT_A}/results.json"
else
    echo "  (results.json not found — training may have failed)"
fi

# ─────────────────────────────────────────────
# Phase 5 — Optional: low-LR backbone fine-tune (Pass B)
# Uncomment this block and set BACKBONE_LR_B to enable.
# ─────────────────────────────────────────────
# if is_done "p5_finetune_fold${FOLD}"; then
#     echo "[SKIP] Phase 5: fine-tune (Pass B) already done"
# else
#     echo "================================================================"
#     echo "Phase 5: Backbone fine-tune — backbone_lr=${BACKBONE_LR_B}, fold ${FOLD}"
#     echo "================================================================"
#     mkdir -p "${OUT_B}"
#     python3 "${REPO}/linear_probe_cardiac.py" \
# --preprocessed "${PREPROCESSED}" \
# --checkpoint "${RAW_CKPT}" \
# --fold ${FOLD} \
# --epochs ${EPOCHS_B} \
# --batch_size ${BATCH_SIZE} \
# --head_lr ${HEAD_LR_B} \
# --backbone_lr ${BACKBONE_LR_B} \
# --workers 8 \
# --out "${OUT_B}"
#     mark_done "p5_finetune_fold${FOLD}"
# fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
