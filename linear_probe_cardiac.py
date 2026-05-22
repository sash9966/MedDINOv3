#!/usr/bin/env python3
"""
linear_probe_cardiac.py
=======================
Linear probe segmentation on top of frozen MedDINOv3 ViT-B/16 features.

The backbone is completely frozen by default.  Only a single 1×1 conv head
is trained — the minimal test of how much spatial-semantic information is
already in the pretrained features without any task-specific supervision.

Architecture
------------
  MedDINOv3 (frozen) ─► intermediate features from blocks [2,5,8,11]
                      ─► concatenate along channel: (B, 768×4, H/16, W/16)
                      ─► Conv2d(3072, n_classes, kernel_size=1)   ← only weights trained
                      ─► bilinear upsample to (H, W)
                      ─► segmentation mask

Why this matters
----------------
DINO's global self-attention encodes relative position and tissue identity
across the full image.  The viewer experiments confirmed that the heart
outline and chamber positions are already well-represented in the frozen
features.  A linear head can exploit this without any feature distortion.

Expected behaviour per class (Dataset030_imageCHD):
  LV-BP, RV-BP, LA, RA, Myo  → likely good: global context distinguishes chambers
  Ao, PA                      → harder: thin vessels need sub-patch resolution;
                                 ViT patch_size=16 ≈ 8–10 mm footprint on
                                 typical cardiac CT, often wider than the vessel lumen.

For Ao/PA improvement ideas:
  1. Use a ViT-B/8 backbone (finer patches — requires a re-pretrained model).
  2. Use the linear probe prediction as a coarse spatial prior, then refine
     with a lightweight convolutional module (thin-tube refinement).
  3. Apply test-time mask-and-re-run: mask out predicted heart region to focus
     attention on the residual vessels.

Usage
-----
# Pure linear probe (backbone frozen):
python linear_probe_cardiac.py \\
  --preprocessed /path/to/nnUNet_preprocessed/Dataset030_imageCHD_HU \\
  --checkpoint   /path/to/meddinov3_2d.pth \\
  --fold         0 \\
  --epochs       50 \\
  --out          linear_probe_fold0

# Low-LR backbone fine-tuning (optional, not pure linear probe):
python linear_probe_cardiac.py \\
  --preprocessed /path/to/nnUNet_preprocessed/Dataset030_imageCHD_HU \\
  --checkpoint   /path/to/meddinov3_2d.pth \\
  --fold         0 \\
  --backbone_lr  3e-6 \\
  --epochs       50 \\
  --out          linear_probe_finetune_fold0

After training, the script writes:
  <out>/checkpoint_best.pth   — best val-Dice head weights + backbone state
  <out>/results.json          — per-class and mean Dice on validation fold
  <out>/training_log.csv      — epoch-level loss and mean Dice
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ─── class names for Dataset030_imageCHD ──────────────────────────────────────
# index 0 = background, indices 1-7 = foreground structures
CHD_CLASSES = ["background", "LV-BP", "RV-BP", "LA", "RA", "Myo", "Ao", "PA"]
N_CLASSES    = len(CHD_CLASSES)   # 8

# ─── ViT interaction blocks (same as Primus_Multiscale) ───────────────────────
INTERACTION_INDICES = [2, 5, 8, 11]


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

class CardiacSliceDataset(Dataset):
    """
    Loads nnUNet preprocessed .npz cases and yields individual 2D slices.

    Each .npz contains:
      data : (C, D, H, W)  — normalised CT, C=1 for single-modality
      seg  : (1, D, H, W)  — integer labels, -1 = ignore

    Returns per slice:
      image : (1, H, W)  float32
      label : (H, W)     int64   (-1 entries masked in loss)
    """

    def __init__(self, preprocessed_dir: Path, case_ids: list):
        self.slices = []   # list of (npz_path, slice_idx)
        plan_dirs = sorted([d for d in preprocessed_dir.iterdir()
                            if d.is_dir() and 'nnUNetPlans' in d.name])
        if not plan_dirs:
            raise FileNotFoundError(
                f"No nnUNetPlans_* subdirectory found under {preprocessed_dir}. "
                "Run nnUNetv2_plan_and_preprocess first."
            )
        # Prefer 2d plan; fall back to first available.
        plan_dir = next((d for d in plan_dirs if '2d' in d.name), plan_dirs[0])
        print(f"[data] Loading from plan folder: {plan_dir.name}")

        missing = []
        for cid in case_ids:
            npz = plan_dir / f"{cid}.npz"
            if not npz.exists():
                missing.append(cid)
                continue
            d = np.load(npz, allow_pickle=False)
            n_slices = d['data'].shape[1] if d['data'].ndim == 4 else 1
            for s in range(n_slices):
                self.slices.append((npz, s))
        if missing:
            print(f"[data] WARNING: {len(missing)} case(s) not found in {plan_dir.name}")
        print(f"[data] {len(case_ids) - len(missing)} cases → {len(self.slices)} slices")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        npz_path, s = self.slices[idx]
        d = np.load(npz_path, allow_pickle=False)
        data = d['data']   # (C, D, H, W) or (C, H, W)
        seg  = d['seg']    # (1, D, H, W) or (1, H, W)

        if data.ndim == 4:
            img = data[:, s]     # (C, H, W)
            lbl = seg[0, s]      # (H, W)
        else:
            img = data           # (C, H, W)
            lbl = seg[0]         # (H, W)

        img = torch.from_numpy(img.astype(np.float32))    # (C, H, W)
        lbl = torch.from_numpy(lbl.astype(np.int64))      # (H, W)
        return img, lbl


def load_splits(preprocessed_dir: Path, fold: int):
    splits_file = preprocessed_dir / "splits_final.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"splits_final.json not found in {preprocessed_dir}")
    with open(splits_file) as f:
        splits = json.load(f)
    entry = splits[fold]
    return entry["train"], entry["val"]


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class LinearProbeSegmentor(nn.Module):
    """
    Frozen MedDINOv3 backbone + single 1×1 conv segmentation head.

    The 1×1 conv is the only trainable component in pure linear probe mode.
    With backbone_lr > 0, the backbone is also updated at that learning rate.
    """

    def __init__(self, backbone, n_classes: int,
                 embed_dim: int = 768,
                 interaction_indices: list = INTERACTION_INDICES):
        super().__init__()
        self.backbone            = backbone
        self.interaction_indices = interaction_indices
        # Single 1×1 conv = true linear probe.
        self.head = nn.Conv2d(embed_dim * len(interaction_indices), n_classes,
                              kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity='linear')
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W)  single-channel CT slice
        returns: (B, n_classes, H, W)
        """
        B, C, H, W = x.shape
        if C == 1:
            x = x.expand(-1, 3, -1, -1)    # repeat to 3 channels

        # Extract multi-scale patch features from frozen backbone.
        features = self.backbone.get_intermediate_layers(
            x, n=self.interaction_indices, reshape=True
        )   # list of (B, embed_dim, H/16, W/16)

        feat = torch.cat(features, dim=1)   # (B, embed_dim*4, H/16, W/16)
        logits = self.head(feat)            # (B, n_classes, H/16, W/16)

        # Upsample back to input resolution.
        logits = F.interpolate(logits, size=(H, W),
                               mode='bilinear', align_corners=False)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

class DiceCELoss(nn.Module):
    def __init__(self, n_classes: int, ignore_index: int = -1):
        super().__init__()
        self.n_classes    = n_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = self.ce(logits, targets)

        # Soft Dice over valid voxels.
        mask   = (targets != self.ignore_index)
        probs  = logits.softmax(dim=1)
        t_oh   = F.one_hot(targets.clamp(min=0), self.n_classes).permute(0, 3, 1, 2).float()
        dice_loss = 0.0
        eps = 1e-6
        for c in range(1, self.n_classes):   # skip background
            p = probs[:, c] * mask
            t = t_oh[:, c]  * mask
            dice_loss += 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        dice_loss /= (self.n_classes - 1)

        return 0.5 * ce + 0.5 * dice_loss


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, n_classes, device):
    """Returns per-class Dice dict and mean foreground Dice."""
    model.eval()
    intersection = torch.zeros(n_classes, device=device)
    denominator  = torch.zeros(n_classes, device=device)

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        preds  = logits.argmax(dim=1)   # (B, H, W)
        mask   = (labels >= 0)

        for c in range(n_classes):
            p = (preds == c) & mask
            t = (labels == c) & mask
            intersection[c] += (p & t).sum()
            denominator[c]  += p.sum() + t.sum()

    dice = {}
    for c, name in enumerate(CHD_CLASSES[:n_classes]):
        denom = denominator[c].item()
        dice[name] = (2 * intersection[c].item() / denom) if denom > 0 else float('nan')

    fg_dice = np.nanmean([v for k, v in dice.items() if k != 'background'])
    return dice, fg_dice


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] device = {device}")

    # ── backbone ──────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent / 'nnUNet'))
    from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.vision_transformer import vit_base

    backbone = vit_base(drop_path_rate=0., layerscale_init=1e-5,
                        n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'teacher' in ckpt:
        sd = {k.replace('backbone.', ''): v
              for k, v in ckpt['teacher'].items()
              if 'ibot' not in k and 'dino_head' not in k}
    elif 'network_weights' in ckpt:
        # nnUNet fine-tuned checkpoint
        sd = {k[len('dino_encoder.'):]: v
              for k, v in ckpt['network_weights'].items()
              if k.startswith('dino_encoder.')}
        print("[backbone] Loaded from nnUNet fine-tuned checkpoint")
    else:
        sd = ckpt
    missing, unexpected = backbone.load_state_dict(sd, strict=True)
    print(f"[backbone] Loaded  params={sum(p.numel() for p in backbone.parameters())/1e6:.1f}M")

    # Freeze backbone in pure linear probe mode.
    if args.backbone_lr == 0:
        backbone.requires_grad_(False)
        print("[backbone] FROZEN — pure linear probe")
    else:
        print(f"[backbone] Fine-tuning at lr={args.backbone_lr:.2e}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = LinearProbeSegmentor(backbone, n_classes=N_CLASSES).to(device)

    # ── data ──────────────────────────────────────────────────────────────────
    preprocessed = Path(args.preprocessed)
    train_ids, val_ids = load_splits(preprocessed, args.fold)
    print(f"[data] fold {args.fold}: {len(train_ids)} train cases, {len(val_ids)} val cases")

    train_ds = CardiacSliceDataset(preprocessed, train_ids)
    val_ds   = CardiacSliceDataset(preprocessed, val_ids)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # ── optimizer ─────────────────────────────────────────────────────────────
    param_groups = [{'params': model.head.parameters(), 'lr': args.head_lr}]
    if args.backbone_lr > 0:
        param_groups.append({'params': model.backbone.parameters(),
                             'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.head_lr * 0.01
    )
    criterion = DiceCELoss(N_CLASSES).to(device)

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_mean_dice'])

    best_dice   = -1.0
    best_epoch  = -1

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val mDice':>9}  {'Best':>6}")
    print('-' * 42)

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        if args.backbone_lr == 0:
            model.backbone.eval()   # keep BN/dropout frozen

        total_loss, n_batches = 0.0, 0
        for imgs, labels in train_dl:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ── validate ──────────────────────────────────────────────────────────
        dice_dict, mean_dice = evaluate(model, val_dl, N_CLASSES, device)

        flag = ''
        if mean_dice > best_dice:
            best_dice  = mean_dice
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'head': model.head.state_dict(),
                        'backbone': model.backbone.state_dict(),
                        'dice': dice_dict,
                        'mean_dice': mean_dice},
                       out_dir / 'checkpoint_best.pth')
            flag = ' ← best'

        print(f"{epoch:>6}  {avg_loss:>10.4f}  {mean_dice:>9.4f}{flag}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, mean_dice])

    # ── final evaluation ──────────────────────────────────────────────────────
    print(f"\nBest val mean Dice: {best_dice:.4f} at epoch {best_epoch}")
    print("\nPer-class Dice (best checkpoint):")
    best_ckpt = torch.load(out_dir / 'checkpoint_best.pth',
                           map_location=device, weights_only=False)
    model.head.load_state_dict(best_ckpt['head'])

    dice_dict, mean_dice = evaluate(model, val_dl, N_CLASSES, device)
    for name, d in dice_dict.items():
        if name == 'background':
            continue
        flag = '  ← low (vessels)' if name in ('Ao', 'PA') else ''
        print(f"  {name:>8}: {d:.4f}{flag}")
    print(f"  {'mean':>8}: {mean_dice:.4f}  (foreground classes)")

    results = {'fold': args.fold, 'best_epoch': best_epoch,
               'mean_dice': mean_dice, 'per_class': dice_dict}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Linear probe segmentation on MedDINOv3 features')
    p.add_argument('--preprocessed', required=True,
                   help='Path to nnUNet preprocessed dataset dir '
                        '(contains splits_final.json and nnUNetPlans_2d/)')
    p.add_argument('--checkpoint', required=True,
                   help='MedDINOv3 checkpoint (pretrained .pth or nnUNet fine-tuned .pth)')
    p.add_argument('--fold',        type=int,   default=0)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--head_lr',     type=float, default=1e-3,
                   help='Learning rate for the 1x1 conv head (default: 1e-3)')
    p.add_argument('--backbone_lr', type=float, default=0.0,
                   help='Learning rate for backbone (0 = fully frozen, pure linear probe)')
    p.add_argument('--workers',     type=int,   default=4)
    p.add_argument('--out',         type=str,   default='linear_probe_out',
                   help='Output directory for checkpoints and results')
    args = p.parse_args()

    print("=" * 60)
    print("MedDINOv3 Linear Probe — Cardiac Segmentation")
    print("=" * 60)
    print(f"  preprocessed : {args.preprocessed}")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  fold         : {args.fold}")
    print(f"  epochs       : {args.epochs}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  head_lr      : {args.head_lr}")
    print(f"  backbone_lr  : {args.backbone_lr}  "
          f"({'FROZEN' if args.backbone_lr == 0 else 'fine-tuning'})")
    print(f"  output       : {args.out}")
    print("=" * 60)

    train(args)


if __name__ == '__main__':
    main()
