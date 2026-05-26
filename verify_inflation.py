"""
verify_inflation.py — Check activation-preserving properties of 3D weight inflation.

Compares intermediate ViT features from the 2D MedDINOv3 model with the
central-slice features of the 3D inflated models, using a real preprocessed
case (or synthetic data if --preprocessed is not supplied).

Usage:
    python3 verify_inflation.py \
        --ckpt_2d    /path/to/meddinov3_2d.pth \
        --ckpt_center /path/to/meddinov3_inflated_center_d16.pth \
        --ckpt_ashwin /path/to/meddinov3_inflated_ashwin_d16.pth \
        --d_patch    16 \
        --preprocessed /path/to/nnUNetPlans_3d_fullres/

Expected results (before any fine-tuning):
    centering   : mean cos-sim > 0.95 on central slice, act_scale ≈ 1.0
    average     : mean cos-sim > 0.90 on central slice, act_scale ≈ 1.0
    ashwin      : cos-sim noticeably lower, act_scale ≈ d_patch (the bug)
"""

import argparse
import os
import sys
import glob

import torch
import torch.nn.functional as F
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DINO_ROOT = os.path.join(
    _HERE, "nnUNet", "nnunetv2", "training", "nnUNetTrainer", "dinov3"
)
for _p in [_DINO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dinov3.models.vision_transformer import vit_base, vit_base_3d


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_vol(preprocessed_dir):
    """Load first available case from a nnUNet preprocessed folder.
    Returns a float32 numpy array of shape (D, H, W)."""
    for ext in ("*.b2nd", "*.npz", "*.npy"):
        hits = [
            f for f in glob.glob(os.path.join(preprocessed_dir, ext))
            if "_seg" not in os.path.basename(f)
        ]
        if hits:
            path = sorted(hits)[0]
            print(f"  Loading case: {os.path.basename(path)}")
            if path.endswith(".b2nd"):
                import blosc2
                arr = blosc2.open(path)[:]
            elif path.endswith(".npz"):
                arr = np.load(path)["data"]
            else:
                arr = np.load(path)
            arr = np.squeeze(arr).astype(np.float32)
            while arr.ndim > 3:
                arr = arr[0]
            return arr
    return None


def _load_sd(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "teacher" in sd:
        sd = {k.replace("backbone.", ""): v for k, v in sd["teacher"].items()}
    return sd


def _build_2d_encoder(ckpt_path):
    enc = vit_base(patch_size=16)
    sd = _load_sd(ckpt_path)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    n_miss = len([k for k in missing if "patch_embed" not in k])
    print(f"  2D encoder loaded — {n_miss} missing non-patch-embed keys, "
          f"{len(unexpected)} unexpected")
    enc.eval()
    return enc


def _build_3d_encoder(ckpt_path, d_patch):
    enc = vit_base_3d(patch_size=16, d_patch=d_patch)
    sd = _load_sd(ckpt_path)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    n_miss = len([k for k in missing if "patch_embed" not in k])
    print(f"  3D encoder loaded — {n_miss} missing non-patch-embed keys, "
          f"{len(unexpected)} unexpected")
    enc.eval()
    return enc


@torch.no_grad()
def _extract_features_2d(enc, x_hw):
    """x_hw: (1, 1, H, W) tensor. Returns list of 4 tensors (1, H/16*W/16, 768)."""
    feats = enc.get_intermediate_layers(
        x_hw.expand(-1, 3, -1, -1),
        n=[2, 5, 8, 11], reshape=False
    )
    return feats


@torch.no_grad()
def _extract_features_3d_central(enc, x_dhw, d_patch):
    """x_dhw: (1, 1, D, H, W) tensor.
    Returns list of 4 tensors (1, H/16*W/16, 768) — the central depth slice."""
    D = x_dhw.shape[2]
    center_depth_token = (D // d_patch) // 2

    feats_3d = enc.get_intermediate_layers(
        x_dhw.expand(-1, 3, -1, -1, -1),
        n=[2, 5, 8, 11], reshape=True
    )
    central = []
    for f in feats_3d:
        # f: (1, 768, D', H', W')
        f_center = f[:, :, center_depth_token, :, :]  # (1, 768, H', W')
        B, C, Hp, Wp = f_center.shape
        central.append(f_center.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C))
    return central


def _cos_sim_row(feats_2d, feats_3d):
    """Returns per-block cosine similarity (averaged over spatial tokens)."""
    sims = []
    for f2, f3 in zip(feats_2d, feats_3d):
        f2_n = F.normalize(f2.float(), dim=-1)
        f3_n = F.normalize(f3.float(), dim=-1)
        sim = (f2_n * f3_n).sum(dim=-1).mean().item()
        sims.append(sim)
    return sims


def _act_scale(feats_3d, feats_2d):
    """Mean absolute activation ratio 3D/2D across all blocks."""
    scales = []
    for f2, f3 in zip(feats_2d, feats_3d):
        scales.append(f3.float().abs().mean().item() / (f2.float().abs().mean().item() + 1e-9))
    return float(np.mean(scales))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify 3D weight inflation quality")
    parser.add_argument("--ckpt_2d",     required=True,  help="Path to raw 2D MedDINOv3 checkpoint")
    parser.add_argument("--ckpt_center", default=None,   help="Path to centering-inflated 3D checkpoint")
    parser.add_argument("--ckpt_ashwin", default=None,   help="Path to Ashwin-inflated 3D checkpoint")
    parser.add_argument("--ckpt_avg",    default=None,   help="Path to average-inflated 3D checkpoint")
    parser.add_argument("--d_patch",     type=int, default=16)
    parser.add_argument("--preprocessed", default=None,
                        help="Path to nnUNetPlans_3d_fullres folder for a real case")
    parser.add_argument("--num_classes", type=int, default=8)
    args = parser.parse_args()

    # ── input data ──────────────────────────────────────────────────────────
    if args.preprocessed and os.path.isdir(args.preprocessed):
        print(f"\nLoading preprocessed case from: {args.preprocessed}")
        vol = _load_vol(args.preprocessed)
    else:
        vol = None

    if vol is None:
        print("\nNo preprocessed data found — using synthetic volume (D=96, H=160, W=160)")
        vol = np.random.randn(96, 160, 160).astype(np.float32)

    D, H, W = vol.shape
    print(f"  Volume shape: {D}×{H}×{W}")

    x_3d = torch.from_numpy(vol[None, None])                   # (1,1,D,H,W)
    x_2d = torch.from_numpy(vol[None, None, D // 2, :, :])    # (1,1,H,W)

    # ── 2D baseline ─────────────────────────────────────────────────────────
    print(f"\nBuilding 2D encoder from: {args.ckpt_2d}")
    enc_2d = _build_2d_encoder(args.ckpt_2d)
    feats_2d = _extract_features_2d(enc_2d, x_2d)

    # ── strategies ──────────────────────────────────────────────────────────
    strategies = []
    if args.ckpt_center:
        strategies.append(("centering", args.ckpt_center))
    if args.ckpt_avg:
        strategies.append(("average", args.ckpt_avg))
    if args.ckpt_ashwin:
        strategies.append(("ashwin", args.ckpt_ashwin))

    if not strategies:
        print("\nNo 3D checkpoints supplied — supply at least one of "
              "--ckpt_center / --ckpt_avg / --ckpt_ashwin.")
        return

    PASS_THRESHOLDS = {"centering": 0.95, "average": 0.90, "ashwin": 0.0}

    rows = []
    for label, ckpt in strategies:
        print(f"\nBuilding 3D encoder [{label}] from: {ckpt}")
        enc_3d = _build_3d_encoder(ckpt, args.d_patch)
        feats_3d = _extract_features_3d_central(enc_3d, x_3d, args.d_patch)
        sims = _cos_sim_row(feats_2d, feats_3d)
        scale = _act_scale(feats_3d, feats_2d)
        mean_sim = float(np.mean(sims))
        threshold = PASS_THRESHOLDS.get(label, 0.0)
        status = "PASS" if mean_sim >= threshold else "FAIL"
        rows.append((label, sims, mean_sim, scale, threshold, status))

    # ── table ────────────────────────────────────────────────────────────────
    header = (
        f"\n{'Strategy':<16} | {'Blk2':>6} {'Blk5':>6} {'Blk8':>6} {'Blk11':>6} "
        f"| {'Mean':>6} | {'Act.scale':>9} | {'Thresh':>6} | Result"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for label, sims, mean_sim, scale, thr, status in rows:
        b2, b5, b8, b11 = sims
        ashwin_note = "  ← /d_patch bug!" if label == "ashwin" and scale > 2.0 else ""
        print(
            f"{label:<16} | {b2:>6.3f} {b5:>6.3f} {b8:>6.3f} {b11:>6.3f} "
            f"| {mean_sim:>6.3f} | {scale:>9.2f}x | {thr:>6.2f} | {status}{ashwin_note}"
        )
    print(sep)

    if any(r[5] == "FAIL" for r in rows if r[4] > 0.0):
        print("\n[FAIL] One or more strategies below threshold — check inflation script.")
        sys.exit(1)
    else:
        print("\n[PASS] All checked strategies meet cosine similarity thresholds.")


if __name__ == "__main__":
    main()
