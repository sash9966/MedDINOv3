"""
inflate_weights_3d.py — Inflate MedDINOv3 2D patch embedding to 3D.

Two inflation strategies (choose with --inflation):
  average   : w_3d = w_2d.unsqueeze(2).repeat(1,1,d_patch,1,1) / d_patch
              Preserves per-channel filter identity; all depth slices respond equally.
  centering : w_3d has the pretrained 2D weights at the center depth slice,
              zeros elsewhere. Model starts by focusing on the center slice and
              learns to use surrounding context through fine-tuning.

The 2D conv weight has shape (embed_dim, in_chans, P, P).
The inflated 3D conv weight has shape (embed_dim, in_chans, d_patch, P, P).
All other ViT weights are copied unchanged.

Choosing d_patch
----------------
d_patch controls how many consecutive CT slices are merged into one depth token.
The pretrained ViT was trained on ~196 tokens (14x14 at 224x224). Larger d_patch
reduces token count and keeps the ViT closer to its pretraining distribution:

  d_patch | depth tokens (40-slice patch) | total tokens | vs pretrained
  --------+-------------------------------+--------------+---------------
       2  |              20               |     2400     |    12x over
       4  |              10               |     1200     |     6x over
       8  |               5               |      600     |     3x over
      16  |               2–3             |    240–360   |    ~2x over  ← recommended

  Recommendation: d_patch=8 (good balance of depth resolution vs token count).
  Use d_patch=16 if training is still slow to converge.

Usage
-----
# Centering inflation, d_patch=8 (recommended):
python inflate_weights_3d.py --checkpoint meddinov3_2d.pth --d_patch 8 \
--inflation centering --out meddinov3_inflated_center_d8.pth

# Average inflation, d_patch=8:
python inflate_weights_3d.py --checkpoint meddinov3_2d.pth --d_patch 8 \
--out meddinov3_inflated_average_d8.pth

After running, export the output path before training:
  export MEDDINOV3_3D_CHECKPOINT=/path/to/meddinov3_inflated_center_d8.pth
  export MEDDINOV3_D_PATCH=8
  nnUNetv2_train DATASET_ID 3d_fullres 0 -tr meddinov3_3d_primus_multiscale_Trainer
"""

import argparse
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
    print(f"Downloading {HF_REPO}/{HF_FILENAME} from HuggingFace …")
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    print(f"Downloaded to: {path}")
    return path


def inflate_patch_embed_average(weight_2d: torch.Tensor, d_patch: int) -> torch.Tensor:
    """
    Average inflation: copy 2D kernel d_patch times along depth, divide by d_patch.

    weight_2d: (out_channels, in_channels, P, P)
    returns:   (out_channels, in_channels, d_patch, P, P)
    """
    assert weight_2d.dim() == 4, f"Expected 4D weight, got {weight_2d.shape}"
    w = weight_2d.unsqueeze(2)                     # (out, in, 1, P, P)
    w = w.repeat(1, 1, d_patch, 1, 1)             # (out, in, d_patch, P, P)
    w = w / d_patch                                # normalise
    return w


def inflate_patch_embed_centering(weight_2d: torch.Tensor, d_patch: int) -> torch.Tensor:
    """
    Centering inflation: pretrained weights go to the center slice; all other slices
    are zeroed.  Encourages the model to focus on the center slice initially.

    weight_2d: (out_channels, in_channels, P, P)
    returns:   (out_channels, in_channels, d_patch, P, P)
    """
    assert weight_2d.dim() == 4, f"Expected 4D weight, got {weight_2d.shape}"
    w = torch.zeros(
        weight_2d.shape[0], weight_2d.shape[1], d_patch,
        weight_2d.shape[2], weight_2d.shape[3],
        dtype=weight_2d.dtype,
    )
    center = d_patch // 2
    w[:, :, center, :, :] = weight_2d
    return w


def inflate_checkpoint(
    ckpt_path: str,
    d_patch: int,
    inflation: str = "average",
) -> dict:
    """
    Load checkpoint, inflate patch_embed.proj.weight, return inflated state dict.
    The returned dict is a flat state dict (not wrapped in 'teacher' etc.).
    """
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

    if inflation == "average":
        w3d = inflate_patch_embed_average(w2d, d_patch)
    elif inflation == "centering":
        w3d = inflate_patch_embed_centering(w2d, d_patch)
    else:
        raise ValueError(f"Unknown inflation method '{inflation}'. Choose 'average' or 'centering'.")

    print(f"  patch_embed.proj.weight shape (3D): {tuple(w3d.shape)}")

    inflated = dict(state_dict)
    inflated[patch_embed_key] = w3d
    return inflated


def main():
    parser = argparse.ArgumentParser(description="Inflate MedDINOv3 2D weights to 3D.")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to local model.pth. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--d_patch", type=int, default=2,
        help="Depth patch size for 3D tokenisation (must be a power of 2, default: 2).",
    )
    parser.add_argument(
        "--inflation", type=str, default="average", choices=["average", "centering"],
        help="Inflation strategy: 'average' (default) or 'centering'.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for inflated checkpoint. "
             "Defaults to meddinov3_inflated_{inflation}_d{d_patch}.pth in the current directory.",
    )
    args = parser.parse_args()

    # Validate d_patch is a power of 2.
    import math
    n = math.log2(args.d_patch)
    assert n == int(n) and args.d_patch >= 1, f"d_patch must be a power of 2, got {args.d_patch}"

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = download_hf_checkpoint()

    inflated_state_dict = inflate_checkpoint(ckpt_path, args.d_patch, args.inflation)

    out_path = args.out
    if out_path is None:
        out_path = f"meddinov3_inflated_{args.inflation}_d{args.d_patch}.pth"
    out_path = os.path.abspath(out_path)

    torch.save(inflated_state_dict, out_path)
    print(f"\nInflated checkpoint saved to: {out_path}")

    import math
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
    print(f"  nnUNetv2_train DATASET_ID 3d_fullres 0 -tr meddinov3_3d_primus_multiscale_Trainer")


if __name__ == "__main__":
    main()
