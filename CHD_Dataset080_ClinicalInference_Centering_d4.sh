#!/bin/bash
# =============================================================================
#  CHD_Dataset080_ClinicalInference_Centering_d4.sh
#  Run the best trained model — Dataset030_imageCHD_HU, MedDINOv3 3D centering
#  inflation, d_patch=4, fold 0 (meddinov3_3d_centering_d4_primus_multiscale_Trainer)
#  — on the unseen clinical cases in Dataset080_ClinicalCaseSanjibDetailed.
#
#  Dataset080 is already in nnU-Net format (imagesTr files already carry the
#  "_0000" channel suffix), so this just points nnUNetv2_predict straight at
#  imagesTr — no staging/renaming needed.
#
#  Input:
#    nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/imagesTr
#  Output:
#    nnUNet_raw/Dataset080_ClinicalCaseSanjibDetailed/predictions
#
#  Note: Dataset080 also has labelsTr (ground truth) — this script only runs
#  inference, it does not compute Dice. Ask if you want an eval pass added.
#
#  Requires the trained checkpoint to already exist at:
#    nnUNet_results/Dataset030_imageCHD_HU/meddinov3_3d_centering_d4_primus_multiscale_Trainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
#  (produced by CHD_Dataset030_MedDINO_Centering_d4.sh)
# =============================================================================
#SBATCH --job-name=D080-Clinical-Infer
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D080-Clinical-Infer_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D080-Clinical-Infer_%j.err

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
# Model being used for inference (trained on Dataset030).
MODEL_DATASET_ID=30
MODEL_DATASET_NAME="Dataset030_imageCHD_HU"
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

# Data being predicted on (unseen clinical dataset, already nnU-Net formatted).
DATA_DATASET_NAME="Dataset080_ClinicalCaseSanjibDetailed"
BASE_DIR="${nnUNet_raw}/${DATA_DATASET_NAME}"
IN_DIR="${BASE_DIR}/imagesTr"
OUT_DIR="${BASE_DIR}/predictions"

CKPT_FINAL="${nnUNet_results}/${MODEL_DATASET_NAME}/${TRAINER_3D}__nnUNetPlans__${CONFIG_3D}/fold_${FOLD}/checkpoint_final.pth"

# ─────────────────────────────────────────────
# 3.  Banner
# ─────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset080_ClinicalInference_Centering_d4.sh — START          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
printf "║  %-66s ║\n" "Model        : ${MODEL_DATASET_NAME} / ${TRAINER_3D} / fold ${FOLD}"
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

if [[ ! -d "${IN_DIR}" ]]; then
    echo "ERROR: input folder not found: ${IN_DIR}" >&2
    exit 1
fi

shopt -s nullglob
cases=("${IN_DIR}"/*.nii.gz)
if [[ ${#cases[@]} -eq 0 ]]; then
    echo "ERROR: no .nii.gz files found in ${IN_DIR}" >&2
    exit 1
fi

bad=0
for f in "${cases[@]}"; do
    base=$(basename "${f}" .nii.gz)
    if [[ "${base}" != *_0000 ]]; then
        echo "WARNING: ${f} is missing the nnU-Net channel suffix (_0000)" >&2
        bad=1
    fi
done
[[ ${bad} -eq 0 ]] && echo "Found ${#cases[@]} case(s) in ${IN_DIR}, all correctly named."

mkdir -p "${OUT_DIR}"

# ─────────────────────────────────────────────
# 4.  Inference
# ─────────────────────────────────────────────
echo "================================================================"
echo "Running inference — fold ${FOLD}"
echo "================================================================"
nnUNetv2_predict \
    -i "${IN_DIR}" -o "${OUT_DIR}" \
    -d ${MODEL_DATASET_ID} -c ${CONFIG_3D} \
    -f ${FOLD} \
    -tr ${TRAINER_3D}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset080_ClinicalInference_Centering_d4.sh — DONE            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Predictions : ${OUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════════╝"
