#!/usr/bin/env python3
"""
MedDINOv3 2D Feature Viewer — paper-style PCA + scrollable similarity

Reproduces the DINO-paper figure style for 3D CT:
  Left  : CT grayscale
  Right : Per-slice PCA feature coloring (vivid RGB, no CT bg)
          → switches to cosine-similarity heatmap after clicking

Controls:
  Scroll       : change axial slice
  Left-click   : query 3D similarity at cursor voxel
  Right-click/R: reset to PCA view
  G            : toggle global (cross-slice) vs local (per-slice) PCA
  S            : save screenshot

Usage:
  python feature_viewer_2d.py \\
      --nifti image.nii.gz \\
      --checkpoint meddinov3_inflated_d2.pth \\
      --d_patch 2
"""

import argparse, os, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_HERE, 'nnUNet'),
    os.path.join(_HERE, 'nnUNet', 'nnunetv2', 'training', 'nnUNetTrainer', 'dinov3'),
    os.path.join(_HERE, 'nnUNet', 'nnunetv2', 'training', 'nnUNetTrainer', 'dinov3', 'dinov3'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── shared helpers ────────────────────────────────────────────────────────────

def load_nifti(path):
    import nibabel as nib
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return data.transpose(2, 1, 0)

def preprocess_ct(vol, shape=(64, 256, 256), clip=(-1000, 400)):
    from scipy.ndimage import zoom
    v = np.clip(vol, clip[0], clip[1])
    v = (v - clip[0]) / (clip[1] - clip[0])
    return zoom(v, [t/s for t, s in zip(shape, v.shape)], order=1).astype(np.float32)

@torch.no_grad()
def extract_features(vol, ckpt_path, d_patch=2, device='cpu', layer=11):
    from models.vision_transformer import vit_base_3d
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ({k.replace('backbone.',''):v for k,v in ckpt['teacher'].items()
           if k.startswith('backbone.')} if 'teacher' in ckpt else ckpt)
    in_chans = sd['patch_embed.proj.weight'].shape[1]
    m = vit_base_3d(patch_size=16, d_patch=d_patch, in_chans=in_chans,
                    drop_path_rate=0., layerscale_init=1e-5,
                    n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    m.load_state_dict(sd, strict=False)
    m = m.to(device).eval()
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    if in_chans > 1:
        x = x.expand(-1, in_chans, -1, -1, -1).contiguous()
    f = m.get_intermediate_layers(x, n=[layer], reshape=True)[0]
    return f.squeeze(0).permute(1,2,3,0).cpu().float().numpy()  # (Dp,Hp,Wp,C)

def up2d(arr, H, W):
    """Upsample (Hp,Wp) or (Hp,Wp,3) → (H,W) or (H,W,3) via bilinear."""
    t = (torch.from_numpy(arr).float().permute(2,0,1).unsqueeze(0)
         if arr.ndim == 3
         else torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0))
    out = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    return (out.squeeze(0).permute(1,2,0).numpy() if arr.ndim == 3
            else out.squeeze().numpy())

def pca_rgb_2d(tokens_2d, fg_thr=0.5):
    """(Hp,Wp,C) → (Hp,Wp,3) vivid RGB via foreground-masked PCA."""
    Hp, Wp, C = tokens_2d.shape
    X = tokens_2d.reshape(-1, C).astype(np.float32)
    X -= X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = X @ Vt[0]
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    fg = pc1 > fg_thr
    if fg.sum() < 5: fg = np.ones(len(X), dtype=bool)
    _, _, Vt3 = np.linalg.svd(X[fg], full_matrices=False)
    comps = X[fg] @ Vt3[:3].T
    rgb = np.zeros((len(X), 3), np.float32)
    rgb[fg] = comps
    for c in range(3):
        lo, hi = rgb[fg, c].min(), rgb[fg, c].max()
        rgb[fg, c] = (rgb[fg, c] - lo) / (hi - lo + 1e-8)
    return rgb.reshape(Hp, Wp, 3)


# ── viewer ────────────────────────────────────────────────────────────────────

class Viewer2D:
    def __init__(self, vol, feats, d_patch=2):
        self.vol, self.feats, self.d_patch = vol, feats, d_patch
        self.D, self.H, self.W = vol.shape
        self.Dp, self.Hp, self.Wp, self.C = feats.shape
        self.d = self.D // 2
        self.query = None          # (d, h, w) or None
        self.sim_vol = None        # (D, H, W) similarity volume
        self.pca_mode = 'local'    # 'local' per-slice | 'global' 3-D PCA

        # Precompute global 3D PCA for all slices
        print('Pre-computing global PCA…')
        self._global_pca = self._compute_global_pca()
        print('Done.')

        self._build()
        self._redraw()

    # ── PCA helpers ─────────────────────────────────────────────────────────

    def _compute_global_pca(self):
        """PCA over ALL tokens → (Dp, Hp, Wp, 3), upsampled later per slice."""
        X = self.feats.reshape(-1, self.C).astype(np.float32)
        X -= X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = X @ Vt[0]
        pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
        fg = pc1 > 0.5
        if fg.sum() < 10: fg = np.ones(len(X), dtype=bool)
        _, _, Vt3 = np.linalg.svd(X[fg], full_matrices=False)
        comps = X[fg] @ Vt3[:3].T
        rgb = np.zeros((len(X), 3), np.float32)
        rgb[fg] = comps
        for c in range(3):
            lo, hi = rgb[fg,c].min(), rgb[fg,c].max()
            rgb[fg,c] = (rgb[fg,c]-lo)/(hi-lo+1e-8)
        return rgb.reshape(self.Dp, self.Hp, self.Wp, 3)

    def _feat_img(self):
        """Return (H, W, 3) feature image for current slice."""
        td = min(self.d // self.d_patch, self.Dp - 1)
        if self.pca_mode == 'local':
            low = pca_rgb_2d(self.feats[td])
        else:
            low = self._global_pca[td]
        return up2d(low, self.H, self.W)

    def _sim_img(self):
        """Return (H, W) similarity slice for current depth."""
        if self.sim_vol is None:
            return np.zeros((self.H, self.W), np.float32)
        td = min(self.d // self.d_patch, self.Dp - 1)
        # Compute similarity on current slice tokens against query
        qd, qh, qw = self.query
        td_q = min(qd // self.d_patch, self.Dp - 1)
        th_q = min(qh // 16, self.Hp - 1)
        tw_q = min(qw // 16, self.Wp - 1)
        q = self.feats[td_q, th_q, tw_q]
        q = q / (np.linalg.norm(q) + 1e-8)
        all_f = self.feats[td].reshape(-1, self.C)
        sims = (all_f / (np.linalg.norm(all_f, axis=1, keepdims=True) + 1e-8)) @ q
        sims = sims.reshape(self.Hp, self.Wp)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        return up2d(sims, self.H, self.W)

    def _compute_sim_vol(self, d, h, w):
        """Full 3D cosine similarity volume."""
        td = min(d // self.d_patch, self.Dp - 1)
        th = min(h // 16, self.Hp - 1)
        tw = min(w // 16, self.Wp - 1)
        q = self.feats[td, th, tw]
        q = q / (np.linalg.norm(q) + 1e-8)
        all_f = self.feats.reshape(-1, self.C)
        sims = (all_f / (np.linalg.norm(all_f, axis=1, keepdims=True) + 1e-8)) @ q
        sims = sims.reshape(self.Dp, self.Hp, self.Wp)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        t = torch.from_numpy(sims).float().unsqueeze(0).unsqueeze(0)
        return F.interpolate(t, (self.D, self.H, self.W),
                             mode='trilinear', align_corners=False).squeeze().numpy()

    # ── figure ───────────────────────────────────────────────────────────────

    def _build(self):
        self.fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor='#111')
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.91,
                                  bottom=0.05, wspace=0.03)
        self.fig.canvas.manager.set_window_title('MedDINOv3 2D Feature Viewer')
        self.ax_ct, self.ax_feat = axes

        for ax in axes:
            ax.set_facecolor('#111')
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor('#333')

        self.ax_ct.set_title('CT', color='#aaa', fontsize=12, pad=5)
        self.ax_feat.set_title('Features — PCA (local)', color='#6af', fontsize=12, pad=5)

        self.im_ct   = self.ax_ct.imshow(self.vol[self.d],
                                          cmap='gray', vmin=0, vmax=1)
        self.im_feat = self.ax_feat.imshow(self._feat_img())

        # crosshair on CT panel only
        lkw = dict(linewidth=0.6, alpha=0.6, color='#ff4')
        self._ch_v = self.ax_ct.axvline(self.W//2, **lkw)
        self._ch_h = self.ax_ct.axhline(self.H//2, **lkw)

        self._title = self.fig.text(0.5, 0.96,
            f'Axial  {self.d}/{self.D-1}  —  PCA coloring  '
            f'(G = global ↔ local,  click = similarity)',
            ha='center', va='top', color='#aaa', fontsize=9)
        self.fig.text(0.01, 0.01,
            'Scroll: slice  |  Click: similarity  |  Right-click/R: reset  |  '
            'G: toggle global/local PCA  |  S: screenshot',
            color='#555', fontsize=7)

        c = self.fig.canvas
        c.mpl_connect('scroll_event',       self._on_scroll)
        c.mpl_connect('button_press_event', self._on_click)
        c.mpl_connect('key_press_event',    self._on_key)

    def _redraw(self):
        self.im_ct.set_data(self.vol[self.d])

        if self.query is None:
            self.im_feat.set_data(self._feat_img())
            mode_str = f'PCA ({self.pca_mode})'
            self.ax_feat.set_title(f'Features — {mode_str}',
                                   color='#6af', fontsize=12, pad=5)
        else:
            sim = self._sim_img()
            self.im_feat.set_data(plt.cm.inferno(sim)[:, :, :3])
            qd, qh, qw = self.query
            self.ax_feat.set_title(
                f'Similarity  (query d={qd},h={qh},w={qw})',
                color='#fa6', fontsize=12, pad=5)

        self._title.set_text(f'Axial  {self.d}/{self.D-1}')
        self.fig.canvas.draw_idle()

    # ── events ───────────────────────────────────────────────────────────────

    def _on_scroll(self, ev):
        self.d = int(np.clip(self.d + int(ev.step), 0, self.D - 1))
        self._redraw()

    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None:
            return
        if ev.button == 3:
            self._reset(); return
        if ev.button != 1:
            return
        w = int(np.clip(ev.xdata,  0, self.W - 1))
        h = int(np.clip(ev.ydata,  0, self.H - 1))
        self._ch_v.set_xdata([w])
        self._ch_h.set_ydata([h])
        self._title.set_text('Computing similarity…')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)
        self.query   = (self.d, h, w)
        self.sim_vol = self._compute_sim_vol(self.d, h, w)
        self._redraw()

    def _reset(self):
        self.query   = None
        self.sim_vol = None
        self._redraw()

    def _on_key(self, ev):
        if ev.key == 'r':
            self._reset()
        elif ev.key == 'g':
            self.pca_mode = 'global' if self.pca_mode == 'local' else 'local'
            if self.query is None: self._redraw()
        elif ev.key == 's':
            fn = 'viewer2d_screenshot.png'
            self.fig.savefig(fn, dpi=150, facecolor='#111')
            print(f'Saved {fn}')

    def show(self): plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--nifti',      required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--d_patch',    type=int, default=2)
    ap.add_argument('--resize',     type=int, nargs=3, default=[64, 256, 256])
    ap.add_argument('--clip',       type=float, nargs=2, default=[-1000, 400])
    ap.add_argument('--layer',      type=int, default=11)
    ap.add_argument('--device',     default='mps' if torch.backends.mps.is_available()
                                    else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = ap.parse_args()

    print(f'Loading {args.nifti}')
    vol = preprocess_ct(load_nifti(args.nifti), args.resize, args.clip)
    print(f'Extracting features  device={args.device}  layer={args.layer}')
    feats = extract_features(vol, args.checkpoint, args.d_patch, args.device, args.layer)
    Dp,Hp,Wp,C = feats.shape
    print(f'Volume {vol.shape}  →  patch grid {Dp}×{Hp}×{Wp}  {C}-d')
    Viewer2D(vol, feats, args.d_patch).show()

if __name__ == '__main__':
    main()
