#!/bin/bash
#SBATCH --job-name=MedDINOv3_CHD
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu

set -euo pipefail

# -------------------------
# 1. Environment & Modules
# -------------------------
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

source /scratch/users/sastocke/conda_envs/nnunet310/etc/profile.d/conda.sh 2>/dev/null || true
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

# -------------------------
# 2. nnU-Net Paths
# -------------------------
export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"

# -------------------------
# 3. MedDINOv3 Configuration
# -------------------------
REPO="/scratch/users/sastocke/MedDINOv3"
CKPT_DIR="/scratch/users/sastocke/meddinov3_checkpoints"
D_PATCH=2

INFLATE_SCRIPT="$REPO/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/inflate_weights_3d.py"
INFLATED_CKPT="$CKPT_DIR/meddinov3_inflated_d${D_PATCH}.pth"
RAW_CKPT="$CKPT_DIR/meddinov3_2d.pth"

export MEDDINOV3_2D_CHECKPOINT="$RAW_CKPT"
export MEDDINOV3_3D_CHECKPOINT="$INFLATED_CKPT"
export MEDDINOV3_D_PATCH="$D_PATCH"

# -------------------------
# 4. Datasets
# -------------------------
DATASET_IDS=(1 13 20 30)
DATASET_NAMES=(
    "Dataset001_all_imageCHD"
    "Dataset013_Fanweidatacleaned"
    "Dataset020FanweiDataandImageCHD_HU"
    "Dataset030_imageCHD_HU"
)

# -------------------------
# 5. Install MedDINOv3 nnUNet
# -------------------------
echo "Installing MedDINOv3 nnUNet..."
pip install -e "$REPO/nnUNet" --quiet

# -------------------------
# 6. Download & Inflate Checkpoints (once)
# -------------------------
mkdir -p "$CKPT_DIR"

if [ ! -f "$RAW_CKPT" ]; then
    echo "Downloading MedDINOv3 2D checkpoint from HuggingFace..."
    python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import shutil
src = hf_hub_download(repo_id="ricklisz123/MedDINOv3-ViTB-16-CT-3M", filename="model.pth")
shutil.copy(src, "$RAW_CKPT")
print(f"Saved to $RAW_CKPT")
PYEOF
else
    echo "2D checkpoint already present: $RAW_CKPT"
fi

if [ ! -f "$INFLATED_CKPT" ]; then
    echo "Inflating weights for 3D (d_patch=$D_PATCH)..."
    python3 "$INFLATE_SCRIPT" --checkpoint "$RAW_CKPT" --d_patch "$D_PATCH" --out "$INFLATED_CKPT"
else
    echo "Inflated checkpoint already present: $INFLATED_CKPT"
fi

# -------------------------
# 7. Train all datasets
# -------------------------
for i in "${!DATASET_IDS[@]}"; do
    DATASET_ID="${DATASET_IDS[$i]}"
    DATASET_NAME="${DATASET_NAMES[$i]}"

    echo "================================================="
    echo "DATASET ${DATASET_ID} — ${DATASET_NAME}"
    echo "================================================="

    # --- Planning & Preprocessing ---
    echo "Planning and preprocessing ${DATASET_NAME}..."
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c 2d 3d_fullres --verify_dataset_integrity

    # --- 2D training (all folds) ---
    for FOLD in 0 1 2 3 4; do
        echo "----- 2D  fold ${FOLD}  dataset ${DATASET_ID} -----"
        nnUNetv2_train "${DATASET_ID}" 2d "${FOLD}" \
            -tr meddinov3_base_primus_multiscale_Trainer \
            --npz
    done

    # --- 3D training (all folds) ---
    for FOLD in 0 1 2 3 4; do
        echo "----- 3D  fold ${FOLD}  dataset ${DATASET_ID} -----"
        nnUNetv2_train "${DATASET_ID}" 3d_fullres "${FOLD}" \
            -tr meddinov3_3d_primus_multiscale_Trainer \
            --npz
    done

    echo "Dataset ${DATASET_ID} complete."
done

echo "All datasets and folds complete."
