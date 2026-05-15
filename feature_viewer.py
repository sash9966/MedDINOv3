#!/usr/bin/env python3
"""
MedDINOv3 3D Feature Similarity Viewer

Slicer-style interactive viewer for exploring the 3D DINO feature space:
  - Top-left  : Axial    (Red,   Slicer convention)
  - Top-right : Sagittal (Yellow)
  - Bottom-left : Coronal  (Green)
  - Bottom-right: Feature map (Blue) — PCA coloring at startup,
                  cosine-similarity overlay after clicking a voxel.

Usage:
    python feature_viewer.py \\
        --nifti  path/to/image.nii.gz \\
        --checkpoint  /scratch/users/sastocke/meddinov3_checkpoints/meddinov3_inflated_d2.pth \\
        --d_patch 2

Controls:
    Left-click   : query features at voxel → similarity heatmap
    Right-click  : clear similarity, return to PCA view
    Scroll       : change slice in the panel under the cursor
    S            : save screenshot to viewer_screenshot.png
"""

import argparse
import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' or 'MacOSX' if TkAgg is missing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# ──────────────────────────────────────────────────────────────
# Path setup: make sure our nnUNet fork and dinov3 models are
# importable regardless of which Python env is active.
# ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_HERE, 'nnUNet'),
    os.path.join(_HERE, 'nnUNet', 'nnunetv2', 'training',
                 'nnUNetTrainer', 'dinov3'),
    os.path.join(_HERE, 'nnUNet', 'nnunetv2', 'training',
                 'nnUNetTrainer', 'dinov3', 'dinov3'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ──────────────────────────────────────────────────────────────
# NIfTI loading
# ──────────────────────────────────────────────────────────────

def load_nifti(path: str) -> np.ndarray:
    """Return volume as float32 (D, H, W) after reorienting to axial-first."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required:  pip install nibabel")

    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)

    # nibabel voxel order is (X, Y, Z).  Transpose to (Z, Y, X) so that
    # the first axis is the axial (inferior-superior) axis.
    data = data.transpose(2, 1, 0)
    return data


def preprocess_ct(
    volume: np.ndarray,
    target_shape: tuple = (64, 256, 256),
    clip_range: tuple = (-1000, 400),
) -> np.ndarray:
    """Clip HU, normalise to [0,1], resize with trilinear interpolation."""
    try:
        from scipy.ndimage import zoom as nd_zoom
    except ImportError:
        raise ImportError("scipy required:  pip install scipy")

    vol = np.clip(volume, clip_range[0], clip_range[1])
    lo, hi = clip_range
    vol = (vol - lo) / (hi - lo)

    zoom_factors = [t / s for t, s in zip(target_shape, vol.shape)]
    vol = nd_zoom(vol, zoom_factors, order=1)
    return vol.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    volume: np.ndarray,
    checkpoint_path: str,
    d_patch: int = 2,
    device: str = 'cpu',
    layer: int = 11,
) -> np.ndarray:
    """
    Run the inflated 3D MedDINOv3 backbone and return patch embeddings.

    Returns
    -------
    features : (Dp, Hp, Wp, embed_dim)  float32 numpy array
        Dp = D / d_patch,  Hp = H / 16,  Wp = W / 16
    """
    from models.vision_transformer import vit_base_3d

    model = vit_base_3d(
        patch_size=16,
        d_patch=d_patch,
        drop_path_rate=0.0,
        layerscale_init=1e-5,
        n_storage_tokens=4,
        qkv_bias=False,
        mask_k_bias=True,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle both formats: flat state dict (our inflated ckpt) or teacher-wrapped
    if isinstance(ckpt, dict) and 'teacher' in ckpt:
        sd = {
            k.replace('backbone.', ''): v
            for k, v in ckpt['teacher'].items()
            if k.startswith('backbone.')
        }
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    non_depth = [k for k in missing if 'depth_pos_embed' not in k]
    if non_depth:
        warnings.warn(f'Unexpected missing keys: {non_depth}')

    model = model.to(device).eval()

    D, H, W = volume.shape
    x = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)

    feats_list = model.get_intermediate_layers(x, n=[layer], reshape=True)
    feats = feats_list[0].squeeze(0)          # (embed_dim, Dp, Hp, Wp)
    feats = feats.permute(1, 2, 3, 0)         # (Dp, Hp, Wp, embed_dim)
    return feats.cpu().float().numpy()


# ──────────────────────────────────────────────────────────────
# PCA coloring (the "banana" visualisation)
# ──────────────────────────────────────────────────────────────

def pca_rgb(features: np.ndarray, fg_threshold: float = 0.5) -> np.ndarray:
    """
    Compute first 3 PCA components of patch features and map to RGB.

    1. PCA on all tokens → first component ≈ foreground mask
    2. Keep foreground tokens, re-run PCA
    3. Map 3 components to RGB

    Returns
    -------
    rgb : (Dp, Hp, Wp, 3)  float32 in [0, 1]
    """
    Dp, Hp, Wp, C = features.shape
    X = features.reshape(-1, C).astype(np.float32)      # (N, C)

    # Centre
    X -= X.mean(axis=0, keepdims=True)

    # First rough PCA via SVD (truncated to 3 components)
    try:
        from sklearn.decomposition import PCA as SKPCA
        pca1 = SKPCA(n_components=1)
        pc1 = pca1.fit_transform(X).flatten()   # (N,)
    except ImportError:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = X @ Vt[0]

    # Normalise pc1 to [0,1] and use as foreground mask
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    fg_mask = pc1 > fg_threshold

    if fg_mask.sum() < 10:
        fg_mask = np.ones(len(X), dtype=bool)

    X_fg = X[fg_mask]

    try:
        from sklearn.decomposition import PCA as SKPCA
        pca3 = SKPCA(n_components=3)
        comps = pca3.fit_transform(X_fg)
    except ImportError:
        _, _, Vt3 = np.linalg.svd(X_fg, full_matrices=False)
        comps = X_fg @ Vt3[:3].T

    rgb_full = np.zeros((len(X), 3), dtype=np.float32)
    rgb_full[fg_mask] = comps

    # Normalise each channel to [0, 1]
    for c in range(3):
        col = rgb_full[:, c]
        lo, hi = col.min(), col.max()
        rgb_full[:, c] = (col - lo) / (hi - lo + 1e-8)

    # Background → black
    rgb_full[~fg_mask] = 0.0

    return rgb_full.reshape(Dp, Hp, Wp, 3)


# ──────────────────────────────────────────────────────────────
# Upsample a (Dp, Hp, Wp) similarity/feature map to (D, H, W)
# ──────────────────────────────────────────────────────────────

def upsample_to_volume(arr: np.ndarray, target: tuple) -> np.ndarray:
    """arr: (Dp, Hp, Wp) or (Dp, Hp, Wp, C)  →  (D, H, W) or (D, H, W, C)"""
    if arr.ndim == 3:
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        out = F.interpolate(t, size=target, mode='trilinear', align_corners=False)
        return out.squeeze().numpy()
    else:  # (Dp, Hp, Wp, C)
        C = arr.shape[-1]
        t = torch.from_numpy(arr).float().permute(3, 0, 1, 2).unsqueeze(0)  # (1,C,D,H,W)
        out = F.interpolate(t, size=target, mode='trilinear', align_corners=False)
        return out.squeeze(0).permute(1, 2, 3, 0).numpy()  # (D, H, W, C)


# ──────────────────────────────────────────────────────────────
# Interactive viewer
# ──────────────────────────────────────────────────────────────

# Slicer-style crosshair colours per panel
_COL = dict(axial='#ff4444', sagittal='#ffcc00', coronal='#44ee44')
_TITLE_COL = dict(axial=_COL['axial'], sagittal=_COL['sagittal'],
                  coronal=_COL['coronal'], similarity='#44aaff')


class FeatureViewer:
    """
    2×2 Slicer-like viewer:
      [Axial]    [Sagittal]
      [Coronal]  [Similarity / PCA]
    """

    def __init__(
        self,
        volume: np.ndarray,         # (D, H, W) float32 [0,1]
        features: np.ndarray,       # (Dp, Hp, Wp, embed_dim) float32
        d_patch: int = 2,
    ):
        self.vol = volume
        self.feats = features
        self.d_patch = d_patch
        self.D, self.H, self.W = volume.shape
        self.Dp, self.Hp, self.Wp, self.C = features.shape

        self.pos = np.array([self.D // 2, self.H // 2, self.W // 2], dtype=int)
        self.sim_vol = None          # (D, H, W) cosine similarity, or None
        self._mode = 'pca'           # 'pca' | 'similarity'

        # Precompute PCA RGB and upsample once
        print('Computing PCA feature coloring…')
        pca_low = pca_rgb(self.feats)
        self.pca_full = upsample_to_volume(pca_low, (self.D, self.H, self.W))
        print('PCA done.')

        self._build_figure()
        self._redraw_all()
        self._connect_events()

    # ── Figure construction ──────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 10), facecolor='#111')
        self.fig.canvas.manager.set_window_title('MedDINOv3 Feature Viewer')

        gs = gridspec.GridSpec(
            2, 2, figure=self.fig,
            hspace=0.06, wspace=0.06,
            left=0.02, right=0.98, top=0.95, bottom=0.04,
        )
        self.ax = {
            'axial':      self.fig.add_subplot(gs[0, 0]),
            'sagittal':   self.fig.add_subplot(gs[0, 1]),
            'coronal':    self.fig.add_subplot(gs[1, 0]),
            'similarity': self.fig.add_subplot(gs[1, 1]),
        }

        for name, ax in self.ax.items():
            ax.set_facecolor('#111')
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor('#333')
            ax.set_title(name.capitalize(),
                         color=_TITLE_COL[name], fontsize=9, pad=3)

        d, h, w = self.pos

        # ── CT image handles ────────────────────────────────
        kw = dict(cmap='gray', vmin=0, vmax=1)
        self.im_ct = {
            'axial':    self.ax['axial'].imshow(
                            self.vol[d],           aspect='equal',  **kw),
            'sagittal': self.ax['sagittal'].imshow(
                            self.vol[:, h, :],     aspect='auto',   **kw),
            'coronal':  self.ax['coronal'].imshow(
                            self.vol[:, :, w],     aspect='auto',   **kw),
        }

        # ── Overlay handles ─────────────────────────────────
        # similarity / cosine overlay on CT panels
        kw_ov = dict(cmap='jet', vmin=0, vmax=1, alpha=0.0)
        sh = dict(
            axial    = (self.H, self.W),
            sagittal = (self.D, self.W),
            coronal  = (self.D, self.H),
        )
        asp = dict(axial='equal', sagittal='auto', coronal='auto')
        self.ov_sim = {
            k: self.ax[k].imshow(np.zeros(sh[k]), aspect=asp[k], **kw_ov)
            for k in ('axial', 'sagittal', 'coronal')
        }

        # PCA RGB overlay (shown by default)
        kw_pca = dict(alpha=0.0, aspect='equal', vmin=0, vmax=1)
        self.ov_pca = {
            'axial':    self.ax['axial'].imshow(
                            self.pca_full[d],        aspect='equal', alpha=0.0),
            'sagittal': self.ax['sagittal'].imshow(
                            self.pca_full[:, h, :],  aspect='auto',  alpha=0.0),
            'coronal':  self.ax['coronal'].imshow(
                            self.pca_full[:, :, w],  aspect='auto',  alpha=0.0),
        }

        # ── 4th panel (standalone feature view) ─────────────
        # Shows PCA-RGB at startup; switches to similarity heatmap after click
        self.im_4th = self.ax['similarity'].imshow(
            self.pca_full[d], aspect='equal')

        # ── Crosshairs ──────────────────────────────────────
        lkw = dict(linewidth=0.7, alpha=0.75)
        self.ch = {
            'axial':   [self.ax['axial'].axvline(w,   color=_COL['sagittal'], **lkw),
                        self.ax['axial'].axhline(h,   color=_COL['coronal'],  **lkw)],
            'sagittal':[self.ax['sagittal'].axvline(w, color=_COL['axial'],   **lkw),
                        self.ax['sagittal'].axhline(d, color=_COL['coronal'], **lkw)],
            'coronal': [self.ax['coronal'].axvline(h,  color=_COL['axial'],   **lkw),
                        self.ax['coronal'].axhline(d,  color=_COL['sagittal'],**lkw)],
        }

        # ── Status bar ──────────────────────────────────────
        self._status = self.fig.text(
            0.01, 0.005,
            'Left-click: query similarity  |  Scroll: change slice  |  '
            'Right-click: clear  |  S: screenshot',
            color='#555', fontsize=7, va='bottom',
        )
        self._title = self.fig.text(
            0.5, 0.97, 'MedDINOv3 Feature Viewer — PCA coloring',
            ha='center', va='top', color='#aaa', fontsize=10,
        )

        # Activate PCA overlays
        self._show_pca()

    # ── Drawing helpers ──────────────────────────────────────────

    def _show_pca(self):
        """Display PCA-RGB overlay on the three CT panels + 4th panel."""
        for k in ('axial', 'sagittal', 'coronal'):
            self.ov_sim[k].set_alpha(0.0)
            self.ov_pca[k].set_alpha(0.55)
        self._mode = 'pca'

    def _show_similarity(self):
        """Display cosine-similarity overlay on the three CT panels."""
        for k in ('axial', 'sagittal', 'coronal'):
            self.ov_pca[k].set_alpha(0.0)
            self.ov_sim[k].set_alpha(0.5)
        self._mode = 'similarity'

    def _update_slice_images(self):
        d, h, w = self.pos

        self.im_ct['axial'].set_data(self.vol[d])
        self.im_ct['sagittal'].set_data(self.vol[:, h, :])
        self.im_ct['coronal'].set_data(self.vol[:, :, w])

        self.ov_pca['axial'].set_data(self.pca_full[d])
        self.ov_pca['sagittal'].set_data(self.pca_full[:, h, :])
        self.ov_pca['coronal'].set_data(self.pca_full[:, :, w])

        if self.sim_vol is not None:
            sv = self.sim_vol
            self.ov_sim['axial'].set_data(sv[d])
            self.ov_sim['sagittal'].set_data(sv[:, h, :])
            self.ov_sim['coronal'].set_data(sv[:, :, w])
            if self._mode == 'similarity':
                self.im_4th.set_data(sv[d])
                self.im_4th.set_clim(0, 1)
                self.im_4th.set_cmap('inferno')
        else:
            if self._mode == 'pca':
                self.im_4th.set_data(self.pca_full[d])

    def _update_crosshairs(self):
        d, h, w = self.pos
        self.ch['axial'][0].set_xdata([w])
        self.ch['axial'][1].set_ydata([h])
        self.ch['sagittal'][0].set_xdata([w])
        self.ch['sagittal'][1].set_ydata([d])
        self.ch['coronal'][0].set_xdata([h])
        self.ch['coronal'][1].set_ydata([d])

    def _redraw_all(self):
        self._update_slice_images()
        self._update_crosshairs()
        self.fig.canvas.draw_idle()

    # ── Feature similarity computation ───────────────────────────

    def _compute_similarity(self, d: int, h: int, w: int) -> np.ndarray:
        """Return (D, H, W) cosine similarity map for the voxel at (d, h, w)."""
        # Map full-res voxel → patch-token index
        td = min(d // self.d_patch, self.Dp - 1)
        th = min(h // 16, self.Hp - 1)
        tw = min(w // 16, self.Wp - 1)

        query = self.feats[td, th, tw]                       # (C,)
        query = query / (np.linalg.norm(query) + 1e-8)

        all_f = self.feats.reshape(-1, self.C)               # (N, C)
        norms = np.linalg.norm(all_f, axis=1, keepdims=True) + 1e-8
        all_f = all_f / norms

        sims = (all_f @ query).reshape(self.Dp, self.Hp, self.Wp)

        # Normalise to [0, 1]
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

        # Upsample to full volume resolution
        return upsample_to_volume(sims, (self.D, self.H, self.W))

    # ── Event handlers ───────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None:
            return

        # Right-click → clear, return to PCA
        if event.button == 3:
            self.sim_vol = None
            self._show_pca()
            self.im_4th.set_data(self.pca_full[self.pos[0]])
            self.im_4th.set_clim(0, 1)
            self.im_4th.set_cmap('viridis')
            self._title.set_text('MedDINOv3 Feature Viewer — PCA coloring')
            self._redraw_all()
            return

        if event.button != 1:
            return

        ix = int(np.round(event.xdata))
        iy = int(np.round(event.ydata))
        d, h, w = self.pos

        name = self._axes_name(event.inaxes)
        if name == 'axial':
            w = int(np.clip(ix, 0, self.W - 1))
            h = int(np.clip(iy, 0, self.H - 1))
        elif name == 'sagittal':
            w = int(np.clip(ix, 0, self.W - 1))
            d = int(np.clip(iy, 0, self.D - 1))
        elif name == 'coronal':
            h = int(np.clip(ix, 0, self.H - 1))
            d = int(np.clip(iy, 0, self.D - 1))
        else:
            return

        self.pos = np.array([d, h, w])

        self._title.set_text(f'Computing similarity for voxel ({d}, {h}, {w})…')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        self.sim_vol = self._compute_similarity(d, h, w)
        self._show_similarity()

        td = min(d // self.d_patch, self.Dp - 1)
        th = min(h // 16, self.Hp - 1)
        tw = min(w // 16, self.Wp - 1)
        self._title.set_text(
            f'Similarity — query voxel ({d},{h},{w})  |  '
            f'patch token ({td},{th},{tw})'
        )
        self._redraw_all()

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        delta = int(event.step)
        d, h, w = self.pos

        name = self._axes_name(event.inaxes)
        if name == 'axial':
            d = int(np.clip(d + delta, 0, self.D - 1))
        elif name == 'sagittal':
            h = int(np.clip(h + delta, 0, self.H - 1))
        elif name == 'coronal':
            w = int(np.clip(w + delta, 0, self.W - 1))
        else:
            return

        self.pos = np.array([d, h, w])
        self._redraw_all()

    def _on_key(self, event):
        if event.key == 's':
            fname = 'viewer_screenshot.png'
            self.fig.savefig(fname, dpi=150, facecolor=self.fig.get_facecolor())
            print(f'Saved {fname}')

    def _axes_name(self, ax) -> str:
        for name, a in self.ax.items():
            if a is ax:
                return name
        return ''

    def _connect_events(self):
        c = self.fig.canvas
        c.mpl_connect('button_press_event', self._on_click)
        c.mpl_connect('scroll_event',       self._on_scroll)
        c.mpl_connect('key_press_event',    self._on_key)

    def show(self):
        plt.show()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MedDINOv3 3D Feature Similarity Viewer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nifti',      required=True, help='NIfTI file (.nii or .nii.gz)')
    parser.add_argument('--checkpoint', required=True, help='Inflated 3D checkpoint (.pth)')
    parser.add_argument('--d_patch',    type=int, default=2,
                        help='Depth patch size (must match the checkpoint)')
    parser.add_argument('--resize',     type=int, nargs=3, default=[64, 256, 256],
                        metavar=('D', 'H', 'W'),
                        help='Target shape for inference (must be divisible by d_patch×16)')
    parser.add_argument('--clip',       type=float, nargs=2, default=[-1000, 400],
                        metavar=('LO', 'HI'), help='HU clip range')
    parser.add_argument('--layer',      type=int, default=11,
                        help='ViT block index whose output to use as features (0–11)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Validate resize shape
    D, H, W = args.resize
    stride = args.d_patch * 16
    for dim, name in zip((D, H, W), ('D', 'H', 'W')):
        if dim % stride != 0:
            parser.error(
                f'--resize {name}={dim} must be divisible by d_patch×16={stride}'
            )

    print(f'[ 1/3 ] Loading {args.nifti}')
    raw = load_nifti(args.nifti)
    print(f'        Raw shape: {raw.shape}  dtype: {raw.dtype}  '
          f'range: [{raw.min():.0f}, {raw.max():.0f}]')

    print(f'[ 2/3 ] Preprocessing → {tuple(args.resize)}  clip={args.clip}')
    volume = preprocess_ct(raw, target_shape=args.resize, clip_range=args.clip)

    print(f'[ 3/3 ] Extracting features  device={args.device}  layer={args.layer}')
    features = extract_features(
        volume, args.checkpoint,
        d_patch=args.d_patch, device=args.device, layer=args.layer,
    )
    Dp, Hp, Wp, C = features.shape
    D2, H2, W2 = volume.shape
    print(f'        Volume: {D2}×{H2}×{W2}  →  '
          f'Patch grid: {Dp}×{Hp}×{Wp}  ({C}-d embeddings)')
    print(f'        Patch size per token: {args.d_patch}×16×16 voxels')
    print()
    print('Opening viewer…')
    print('  Left-click  : query cosine similarity')
    print('  Right-click : clear → return to PCA view')
    print('  Scroll      : change slice under cursor')
    print('  S           : save screenshot')

    viewer = FeatureViewer(volume, features, d_patch=args.d_patch)
    viewer.show()


if __name__ == '__main__':
    main()
