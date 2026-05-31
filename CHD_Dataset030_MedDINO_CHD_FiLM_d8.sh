#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_CHD_FiLM_d8.sh
#  Dataset030_imageCHD_HU — MedDINOv3 3D CHD-conditioned FiLM, d_patch=8
#
#  Single experiment: 3D FiLM-conditioned training (rung 1 — bridge FiLM only).
#  Compare against CHD_Dataset030_MedDINO_Centering_d8.sh (unconditioned d=8).
#  No 2D training here — the 2D baseline is already done by other scripts.
#
#  Trainer: meddinov3_3d_chd_film_d8_Trainer
#  Results: nnUNet_results/.../meddinov3_3d_chd_film_d8_Trainer__nnUNetPlans__3d_fullres/
#
#  RESUME SUPPORT — resubmit the same script to continue.
# =============================================================================
#SBATCH --job-name=D030-CHD-FiLM-d8
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-CHD-FiLM-d8_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-CHD-FiLM-d8_%j.err

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
CONFIG_3D="3d_fullres"
TRAINER_3D="meddinov3_3d_chd_film_d8_Trainer"
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
export MEDDINOV3_BACKBONE_LR_SCALE=0.1

# ── CHD FiLM conditioning ──────────────────────────────────────────
export CHD_FILM_BRIDGE=1
export CHD_FILM_DECODER=0
export CHD_NUM_DIAGNOSES=18
CHD_XLSX="${nnUNet_raw}/${DATASET_NAME}/imageCHD_dataset_info.xlsx"

IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_DIR="${nnUNet_results}/${DATASET_NAME}/predictions/MedDINO_3d_chd_film_d8"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_MedDINO_CHD_FiLM_d8"
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
# 4.  Banner
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_MedDINO_CHD_FiLM_d8.sh — START                ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
printf "║  %-66s ║\n" "Trainer      : ${TRAINER_3D}"
printf "║  %-66s ║\n" "d_patch      : ${D_PATCH}  (1200 tokens, 6.1x pretrained)"
printf "║  %-66s ║\n" "FiLM         : bridge=${CHD_FILM_BRIDGE}  decoder=${CHD_FILM_DECODER}"
printf "║  %-66s ║\n" "Diagnoses    : ${CHD_NUM_DIAGNOSES}  (18-label NIH/CDC vector)"
printf "║  %-66s ║\n" "Epochs       : ${MEDDINOV3_NUM_EPOCHS}"
printf "║  %-66s ║\n" "Results      : nnUNet_results/${DATASET_NAME}/${TRAINER_3D}__nnUNetPlans__3d_fullres/"
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
    echo "Phase 1: Downloading MedDINOv3 2D checkpoint"
    cat > /tmp/chd_film_download.py << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys
shared_dir, raw_ckpt = sys.argv[1], sys.argv[2]
src = hf_hub_download(repo_id='ricklisz123/MedDINOv3-ViTB-16-CT-3M', filename='model.pth')
os.makedirs(shared_dir, exist_ok=True)
shutil.copy(src, raw_ckpt)
print('Saved:', raw_ckpt)
PYEOF
    python3 /tmp/chd_film_download.py "${SHARED_CKPT_DIR}" "${RAW_CKPT}"
    mark_shared_done "download_2d_ckpt"
fi

# ─────────────────────────────────────────────
# Phase 2 — Inflate weights (centering, d=8, shared)
# ─────────────────────────────────────────────
if is_shared_done "inflate_center_d8"; then
    echo "[SKIP] Phase 2: inflated checkpoint already exists"
else
    echo "Phase 2: Centering inflation d_patch=8"
    python3 "${INFLATE_SCRIPT}" \
        --checkpoint "${RAW_CKPT}" \
        --d_patch "${D_PATCH}" \
        --inflation centering \
        --out "${INFLATED_CKPT}"
    mark_shared_done "inflate_center_d8"
fi

# ─────────────────────────────────────────────
# Phase 3 — Inject CHD diagnosis (shared, locked)
# ─────────────────────────────────────────────
DIAG_KEY="inject_diag_D${DATASET_ID}_3d"
if is_shared_done "${DIAG_KEY}"; then
    echo "[SKIP] Phase 3: diagnosis already injected"
else
    echo "Phase 3: Inject CHD diagnosis vectors (flock-protected)"
    (
        flock -x 9
        if ! is_shared_done "${DIAG_KEY}"; then
            python3 "${REPO}/tools/add_chd_diagnosis_to_properties.py" \
                --preprocessed_folder "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres" \
                --xlsx "${CHD_XLSX}"
            mark_shared_done "${DIAG_KEY}"
        fi
    ) 9>"${SHARED_CKPT_DIR}/inject_diag_D${DATASET_ID}.lock"
fi

# ─────────────────────────────────────────────
# Phase 4 — 3D CHD FiLM training (fold 0)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: CHD FiLM 3D training — fold 0  (${MEDDINOV3_NUM_EPOCHS} epochs)"
echo "================================================================"
KEY="p4_3d_chd_film_d8_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${CONFIG_3D} 0 \
        -tr ${TRAINER_3D} --npz
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 5 — Inference on test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: Inference"
echo "================================================================"
mkdir -p "${PRED_DIR}"

if is_done "p5_infer_3d_chd_film_d8"; then
    echo "[SKIP] p5_infer_3d_chd_film_d8"
else
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_DIR}" \
        -d ${DATASET_ID} -c ${CONFIG_3D} \
        -f 0 \
        -tr ${TRAINER_3D}
    mark_done "p5_infer_3d_chd_film_d8"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
elapsed=$(( $(date +%s) - START_TS ))
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_MedDINO_CHD_FiLM_d8.sh — COMPLETE             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Elapsed : $(( elapsed/3600 ))h $(( (elapsed%3600)/60 ))m $(( elapsed%60 ))s"
printf "║  %-66s ║\n" "Results : ${nnUNet_results}/${DATASET_NAME}/${TRAINER_3D}__nnUNetPlans__3d_fullres/"
printf "║  %-66s ║\n" "Preds   : ${PRED_DIR}"
echo "╚══════════════════════════════════════════════════════════════════╝"
