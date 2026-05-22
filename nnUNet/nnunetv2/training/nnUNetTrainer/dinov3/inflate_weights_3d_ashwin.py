"""
inflate_weights_3d_ashwin.py — Ashwin_3d_inflation strategy
Inflate MedDINOv3 2D patch embedding to 3D using Ashwin's channel-averaging approach.

Ashwin's strategy (faithfully reproduced, credit: Ashwin):
  1. Sum the 3 RGB input channels into 1:
       emb = emb.sum(1, keepdim=True)             # (C_out, 1, H, W)
  2. Redistribute equally back to n_chans:
       emb = emb.repeat(1, n_chans, 1, 1) / n_chans  # (C_out, n_chans, H, W)
  3. Inflate along depth by repeating d_patch times:
       emb = emb.unsqueeze(4).repeat(1, 1, 1, 1, d_patch)  # (C_out, n_chans, H, W, d_patch)

  A final .permute(0, 1, 4, 2, 3) converts from Ashwin's (H, W, D) axis ordering
  to PyTorch Conv3d's expected (D, H, W) ordering — this is a format-only adaptation,
  not a change to the inflation logic.

Key difference from average/centering inflation (inflate_weights_3d.py):
  - Channels are first averaged together before distributing to all input channels.
    This makes each spatial filter respond equally to all input channels (no per-channel
    specialisation is preserved). Average/centering inflation preserves channel identity.
  - Depth is NOT divided by d_patch — the channel averaging already renormalises the
    input-channel sum; depth copies preserve full channel-averaged response per slice.

Choosing d_patch
----------------
d_patch controls how many consecutive CT slices are merged into one depth token.
The pretrained ViT was trained on ~196 tokens (14x14 at 224x224). Larger d_patch
reduces token count and keeps the ViT closer to its pretraining distribution:

  d_patch | depth tokens (40-slice patch) | total tokens | vs pretrained
  --------+-------------------------------+--------------+---------------
       2  |              20               |     2400     |    12x over
       4  |              10               |     1200     |     6x over
       8  |               5               |      600     |     3x over  ← recommended
      16  |               2–3             |    240–360   |    ~2x over

  Recommendation: d_patch=8 balances depth resolution against token count.

Usage
-----
python inflate_weights_3d_ashwin.py --checkpoint meddinov3_2d.pth --d_patch 8 \
--out meddinov3_inflated_ashwin_d8.pth

After running:
  export MEDDINOV3_3D_CHECKPOINT=/path/to/meddinov3_inflated_ashwin_d8.pth
  export MEDDINOV3_D_PATCH=8
  nnUNetv2_train DATASET_ID 3d_fullres 0 -tr meddinov3_3d_ashwin_primus_multiscale_Trainer
"""

import argparse
import math
import os

import torch


HF_REPO = "ricklisz123/MedDINOv3-ViTB-16-CT-3M"
HF_FILENAME = "model.pth"


def download_hf_checkpoint() -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. "
            "Run: pip install huggingface_hub"
        )
    print(f"Downloading {HF_REPO}/{HF_FILENAME} from HuggingFace ...")
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    print(f"Downloaded to: {path}")
    return path


def inflate_patch_embed_ashwin(weight_2d: torch.Tensor, d_patch: int) -> torch.Tensor:
    """
    Ashwin_3d_inflation: channel-averaging + depth repetition.

    Steps (Ashwin's original logic):
      1. sum(axis=1, keepdim=True)     — collapse 3 input channels into 1
      2. repeat(1, n_chans, 1, 1) / n_chans  — redistribute equally to n_chans
      3. unsqueeze(4).repeat(..., d_patch)    — tile along depth axis (Ashwin's H,W,D ordering)
      4. permute(0,1,4,2,3)                  — reorder to PyTorch Conv3d (D,H,W) convention

    weight_2d: (C_out, C_in, P, P)
    returns:   (C_out, C_in, d_patch, P, P)
    """
    assert weight_2d.dim() == 4, f"Expected 4D weight, got {weight_2d.shape}"
    n_chans = weight_2d.shape[1]

    # Step 1: collapse all input channels into one (channel-average).
    emb = weight_2d.sum(1, keepdim=True)                   # (C_out, 1,      P, P)

    # Step 2: redistribute equally back to n_chans.
    emb = emb.repeat(1, n_chans, 1, 1) / n_chans          # (C_out, n_chans, P, P)

    # Step 3: inflate along depth — Ashwin appends depth as the last axis.
    emb = emb.unsqueeze(4).repeat(1, 1, 1, 1, d_patch)    # (C_out, n_chans, P, P, d_patch)

    # Step 4: reorder to PyTorch Conv3d convention (C_out, C_in, D, H, W).
    emb = emb.permute(0, 1, 4, 2, 3).contiguous()         # (C_out, n_chans, d_patch, P, P)

    return emb


def inflate_checkpoint_ashwin(ckpt_path: str, d_patch: int) -> dict:
    print(f"Loading checkpoint from: {ckpt_path}")
    chkpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Unwrap MedDINOv3 teacher format if present.
    if isinstance(chkpt, dict) and "teacher" in chkpt:
        state_dict = chkpt["teacher"]
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "ibot" not in k and "dino_head" not in k
        }
    else:
        state_dict = chkpt

    patch_embed_key = "patch_embed.proj.weight"
    if patch_embed_key not in state_dict:
        raise KeyError(
            f"'{patch_embed_key}' not found in checkpoint. "
            f"Available keys (first 20): {list(state_dict.keys())[:20]}"
        )

    w2d = state_dict[patch_embed_key]
    print(f"  patch_embed.proj.weight shape (2D): {tuple(w2d.shape)}")
    print(f"  Applying Ashwin_3d_inflation: channel-sum -> redistribute -> depth-tile")

    w3d = inflate_patch_embed_ashwin(w2d, d_patch)
    print(f"  patch_embed.proj.weight shape (3D): {tuple(w3d.shape)}")
    print(f"  Channel merge strategy  : sum {w2d.shape[1]} channels -> average across {w2d.shape[1]}")
    print(f"  Depth inflation         : repeat {d_patch}x (no depth normalisation)")

    inflated = dict(state_dict)
    inflated[patch_embed_key] = w3d
    return inflated


def main():
    parser = argparse.ArgumentParser(
        description="Ashwin_3d_inflation: inflate MedDINOv3 2D weights to 3D (channel-averaging strategy)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to local model.pth. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--d_patch", type=int, default=2,
        help="Depth patch size for 3D tokenisation (must be a power of 2, default: 2).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path. Defaults to meddinov3_inflated_ashwin_d{d_patch}.pth.",
    )
    args = parser.parse_args()

    n = math.log2(args.d_patch)
    assert n == int(n) and args.d_patch >= 1, f"d_patch must be a power of 2, got {args.d_patch}"

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = download_hf_checkpoint()

    inflated_state_dict = inflate_checkpoint_ashwin(ckpt_path, args.d_patch)

    out_path = args.out
    if out_path is None:
        out_path = f"meddinov3_inflated_ashwin_d{args.d_patch}.pth"
    out_path = os.path.abspath(out_path)

    torch.save(inflated_state_dict, out_path)
    print(f"\n[Ashwin_3d_inflation] Inflated checkpoint saved to: {out_path}")

    patch_size = 16
    print(f"\nToken count estimate for common 3d_fullres patch sizes (d_patch={args.d_patch}):")
    for depth in [32, 40, 48, 64]:
        for hw in [(192, 160), (224, 192)]:
            h, w = hw
            tokens = (depth // args.d_patch) * (h // patch_size) * (w // patch_size)
            print(f"  patch ({depth:3d},{h},{w}): {tokens:5d} tokens  "
                  f"({'OK' if tokens <= 400 else 'HIGH' if tokens <= 800 else 'VERY HIGH'})")
    print(f"  Pretrained distribution: ~196 tokens")

    print(f"\nTo use for training:")
    print(f"  export MEDDINOV3_3D_CHECKPOINT={out_path}")
    print(f"  export MEDDINOV3_D_PATCH={args.d_patch}")
    print(f"  nnUNetv2_train DATASET_ID 3d_fullres 0 -tr meddinov3_3d_ashwin_primus_multiscale_Trainer")


if __name__ == "__main__":
    main()
