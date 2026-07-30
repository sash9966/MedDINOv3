#!/bin/bash
# =============================================================================
#  CHD_Dataset030_MedDINO_Centering_d4_ClinicalInference.sh
#  Run the best trained model — Dataset030_imageCHD_HU, MedDINOv3 3D centering
#  inflation, d_patch=4, fold 0 (meddinov3_3d_centering_d4_primus_multiscale_Trainer)
#  — on new clinical CT cases dropped into a plain folder.
#
#  USAGE
#    1. Upload cases (any filenames, .nii.gz) into:
#         /scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/imagesTs
#    2. Run this script (sbatch or interactively on a GPU node).
#    3. Predictions land in:
#         /scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared/predictions
#
#  Files don't need to be pre-named with nnU-Net's "_0000" channel suffix —
#  this script stages a renamed copy if it's missing, so your uploads in
#  imagesTs are never modified.
#
#  Requires the trained checkpoint to already exist at:
#    nnUNet_results/Dataset030_imageCHD_HU/meddinov3_3d_centering_d4_primus_multiscale_Trainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
#  (produced by CHD_Dataset030_MedDINO_Centering_d4.sh)
# =============================================================================
#SBATCH --job-name=D030-Clinical-Infer
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-Clinical-Infer_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-Clinical-Infer_%j.err

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
TRAINER_3D="meddinov3_3d_centering_d4_primus_multiscale_Trainer"
FOLD=0
D_PATCH=4

SHARED_CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
# build_network_architecture reloads this inflated checkpoint to construct the
# model before nnU-Net overwrites its weights with the trained fold_0 checkpoint
# — it must be set even at inference time, and must match what training used.
export MEDDINOV3_3D_CHECKPOINT="${SHARED_CKPT_DIR}/meddinov3_inflated_center_d4.pth"
export MEDDINOV3_D_PATCH="${D_PATCH}"

BASE_DIR="/scratch/users/sastocke/nnunet_CHD/ClinicalImagesPHICleared"
IN_DIR="${BASE_DIR}/imagesTs"
STAGED_DIR="${BASE_DIR}/imagesTs_staged"
OUT_DIR="${BASE_DIR}/predictions"

CKPT_FINAL="${nnUNet_results}/${DATASET_NAME}/${TRAINER_3D}__nnUNetPlans__${CONFIG_3D}/fold_${FOLD}/checkpoint_final.pth"

# ─────────────────────────────────────────────
# 3.  Banner
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_MedDINO_Centering_d4_ClinicalInference.sh — START  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
printf "║  %-66s ║\n" "Model        : ${DATASET_NAME} / ${TRAINER_3D} / fold ${FOLD}"
printf "║  %-66s ║\n" "Input        : ${IN_DIR}"
printf "║  %-66s ║\n" "Output       : ${OUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [[ ! -f "${CKPT_FINAL}" ]]; then
    echo "ERROR: trained checkpoint not found:" >&2
    echo "  ${CKPT_FINAL}" >&2
    echo "Run CHD_Dataset030_MedDINO_Centering_d4.sh to completion first." >&2
    exit 1
fi

# ─────────────────────────────────────────────
# 4.  Stage input cases (ensure nnU-Net's "_0000" channel suffix; copies only,
#     never touches your uploads in imagesTs)
# ─────────────────────────────────────────────
mkdir -p "${IN_DIR}" "${STAGED_DIR}" "${OUT_DIR}"

shopt -s nullglob
cases=("${IN_DIR}"/*.nii.gz)
if [[ ${#cases[@]} -eq 0 ]]; then
    echo "ERROR: no .nii.gz files found in ${IN_DIR}" >&2
    echo "Upload cases there and re-run this script." >&2
    exit 1
fi

rm -f "${STAGED_DIR}"/*.nii.gz
for f in "${cases[@]}"; do
    base=$(basename "${f}" .nii.gz)
    if [[ "${base}" == *_0000 ]]; then
        cp "${f}" "${STAGED_DIR}/"
    else
        cp "${f}" "${STAGED_DIR}/${base}_0000.nii.gz"
    fi
done
echo "Staged ${#cases[@]} case(s) from ${IN_DIR} -> ${STAGED_DIR}"

# ─────────────────────────────────────────────
# 5.  Inference
# ─────────────────────────────────────────────
echo "================================================================"
echo "Running inference — fold ${FOLD}"
echo "================================================================"
nnUNetv2_predict \
    -i "${STAGED_DIR}" -o "${OUT_DIR}" \
    -d ${DATASET_ID} -c ${CONFIG_3D} \
    -f ${FOLD} \
    -tr ${TRAINER_3D}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_MedDINO_Centering_d4_ClinicalInference.sh — DONE   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Predictions : ${OUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════════╝"
