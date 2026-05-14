#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MedDINOv3 segmentation — 2D and 3D training script
# =============================================================================
# Usage:
#   bash run_segmentation.sh [2d|3d|both|preprocess|inflate]
#
# Default (no argument): runs both 2D and 3D training.
#
# Prerequisites on the server:
#   1. pip install -e MedDINOv3/nnUNet
#   2. Dataset placed in $NNUNET_RAW following nnUNet folder conventions
#      (Dataset001_MyData/imagesTr, labelsTr, dataset.json)
#   3. For 3D: run this script with 'inflate' once before training
# =============================================================================

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------
DATASET_ID=1                          # nnUNet dataset number (e.g. 1 → Dataset001_...)
FOLD=0                                # Cross-validation fold (0-4, or 'all')
D_PATCH=2                             # Depth patch size for 3D inflation (power of 2)
NUM_GPUS=1                            # Set >1 to use torchrun multi-GPU

NNUNET_RAW="/data/nnunet/raw"
NNUNET_PREPROCESSED="/data/nnunet/preprocessed"
NNUNET_RESULTS="/data/nnunet/results"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFLATE_SCRIPT="$REPO_ROOT/nnUNet/nnunetv2/training/nnUNetTrainer/dinov3/inflate_weights_3d.py"
INFLATED_CKPT="$REPO_ROOT/meddinov3_inflated_d${D_PATCH}.pth"
# ---------------------------------------------------------------------------

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

MODE="${1:-both}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_train() {
    local config="$1" trainer="$2"
    log "Training  config=$config  trainer=$trainer  fold=$FOLD"
    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node="$NUM_GPUS" \
            "$(which nnUNetv2_train)" "$DATASET_ID" "$config" "$FOLD" -tr "$trainer"
    else
        nnUNetv2_train "$DATASET_ID" "$config" "$FOLD" -tr "$trainer"
    fi
}

preprocess() {
    log "Preprocessing dataset $DATASET_ID for 2d and 3d_fullres"
    nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -c 2d 3d_fullres --verify_dataset_integrity
}

inflate() {
    if [ -f "$INFLATED_CKPT" ]; then
        log "Inflated checkpoint already exists: $INFLATED_CKPT — skipping inflation."
        return
    fi
    log "Inflating MedDINOv3 weights with d_patch=$D_PATCH → $INFLATED_CKPT"
    python3 "$INFLATE_SCRIPT" --d_patch "$D_PATCH" --out "$INFLATED_CKPT"
    log "Inflation done."
}

train_2d() {
    log "=== 2D training (meddinov3_base_primus_multiscale_Trainer) ==="
    run_train "2d" "meddinov3_base_primus_multiscale_Trainer"
}

train_3d() {
    inflate
    export MEDDINOV3_3D_CHECKPOINT="$INFLATED_CKPT"
    export MEDDINOV3_D_PATCH="$D_PATCH"
    log "=== 3D training (meddinov3_3d_primus_multiscale_Trainer) ==="
    log "Checkpoint : $MEDDINOV3_3D_CHECKPOINT"
    log "d_patch    : $MEDDINOV3_D_PATCH"
    run_train "3d_fullres" "meddinov3_3d_primus_multiscale_Trainer"
}

case "$MODE" in
    preprocess)  preprocess ;;
    inflate)     inflate ;;
    2d)          preprocess; train_2d ;;
    3d)          preprocess; train_3d ;;
    both)        preprocess; train_2d; train_3d ;;
    *)
        echo "Usage: $0 [2d|3d|both|preprocess|inflate]"
        exit 1
        ;;
esac

log "Done."
