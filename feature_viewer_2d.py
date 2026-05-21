#!/usr/bin/env python3
"""
MedDINOv3 2D Feature Viewer — paper-style PCA + scrollable similarity

Uses the pretrained 2D vit_base on individual axial slices (not the inflated 3D
model). Each slice is resized to feat_size×feat_size, normalized with ImageNet
stats (matching pretraining), and forwarded through vit_base.

Controls:
  Scroll       : change axial slice
  Left-click   : query similarity at cursor voxel
  Right-click/R: reset to PCA view
  G            : toggle global (cross-slice) vs local (per-slice) PCA
  S            : save screenshot

Usage:
  python feature_viewer_2d.py \
  --nifti image.nii.gz \
  --checkpoint meddinov3_2d.pth
"""

import argparse, os, sys
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_nifti(path):
    import nibabel as nib
    data = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return data.transpose(2, 1, 0)  # (D, H, W) axial-first


def resize_vol(vol, target_dhw):
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_dhw, vol.shape)]
    return zoom(vol, factors, order=1).astype(np.float32)


def preprocess_slice_for_display(slc, clip=(-1000, 400)):
    v = np.clip(slc, clip[0], clip[1])
    return (v - clip[0]) / (clip[1] - clip[0])


# Windows for CECT cardiac: blood pool (200-600 HU) + myocardium visible in ch0,
# standard soft-tissue in ch1, wide/lung in ch2.
CECT_WINDOWS = [(-200, 600), (-160, 240), (-1000, 400)]


def slice_to_tensor_cect(slc_hw):
    """3-channel multi-window preprocessing for contrast-enhanced cardiac CT."""
    channels = []
    for lo, hi in CECT_WINDOWS:
        v = np.clip(slc_hw, lo, hi)
        v = (v - lo) / (hi - lo)
        channels.append(v.astype(np.float32))
    rgb = np.stack(channels, axis=0)
    rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(rgb).unsqueeze(0)


def slice_to_tensor(slc_hw, clip=(-1000, 400)):
    """CT slice → (1, 3, H, W) tensor with ImageNet normalization."""
    v = np.clip(slc_hw, clip[0], clip[1])
    v = (v - clip[0]) / (clip[1] - clip[0])
    rgb = np.stack([v, v, v], axis=0).astype(np.float32)
    rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(rgb).unsqueeze(0)


def up2d(arr, H, W):
    t = (torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0)
         if arr.ndim == 3
         else torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0))
    out = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    return (out.squeeze(0).permute(1, 2, 0).numpy() if arr.ndim == 3
            else out.squeeze().numpy())


def pca_rgb_2d(tokens_2d, fg_thr=0.5):
    """(Hp, Wp, C) → (Hp, Wp, 3) vivid RGB via foreground-masked PCA."""
    Hp, Wp, C = tokens_2d.shape
    X = tokens_2d.reshape(-1, C).astype(np.float32)
    X -= X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = X @ Vt[0]
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
    fg = pc1 > fg_thr
    if fg.sum() < 5:
        fg = np.ones(len(X), dtype=bool)
    _, _, Vt3 = np.linalg.svd(X[fg], full_matrices=False)
    comps = X[fg] @ Vt3[:3].T
    rgb = np.zeros((len(X), 3), np.float32)
    rgb[fg] = comps
    for c in range(3):
        lo, hi = rgb[fg, c].min(), rgb[fg, c].max()
        rgb[fg, c] = (rgb[fg, c] - lo) / (hi - lo + 1e-8)
    return rgb.reshape(Hp, Wp, 3)


@torch.no_grad()
def extract_features(vol, ckpt_path, device='cpu', layer=11,
                     feat_size=448, clip=(-1000, 400), batch_size=4, cect=False):
    """
    Extract per-slice features using the 2D pretrained vit_base.
    vol: (D, H, W) float32 raw HU values
    Returns: (D, Hp, Wp, C) where Hp = feat_size//16
    """
    from models.vision_transformer import vit_base

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'teacher' in ckpt:
        sd = {k.replace('backbone.', ''): v
              for k, v in ckpt['teacher'].items()
              if k.startswith('backbone.')}
    else:
        sd = ckpt

    pe_w = sd.get('patch_embed.proj.weight')
    if pe_w is not None and pe_w.dim() == 5:
        print(f'Detected 3D inflated checkpoint (patch_embed shape {list(pe_w.shape)}), '
              f'deflating to 2D by averaging over depth dim...')
        sd['patch_embed.proj.weight'] = pe_w.mean(dim=2)

    m = vit_base(drop_path_rate=0., layerscale_init=1e-5,
                 n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    missing, unexpected = m.load_state_dict(sd, strict=True)
    if missing:
        print(f'WARNING missing keys: {missing[:3]}')
    m = m.to(device).eval()
    print(f'Model loaded — {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')

    fs = feat_size - (feat_size % 16)
    D, H, W = vol.shape
    Hp, Wp = fs // 16, fs // 16

    feats_list = []
    if cect:
        print('CECT mode: using 3-channel cardiac windows (-200,600) | (-160,240) | (-1000,400)')
        slices = [slice_to_tensor_cect(vol[d]) for d in range(D)]
    else:
        slices = [slice_to_tensor(vol[d], clip) for d in range(D)]

    for start in range(0, D, batch_size):
        batch = torch.cat(slices[start:start + batch_size], dim=0).to(device)
        if batch.shape[-1] != fs or batch.shape[-2] != fs:
            batch = F.interpolate(batch, size=(fs, fs),
                                  mode='bilinear', align_corners=False)
        out = m.get_intermediate_layers(batch, n=[layer], reshape=True)[0]
        for i in range(out.shape[0]):
            feats_list.append(out[i].permute(1, 2, 0).cpu().float().numpy())
        print(f'  {min(start + batch_size, D)}/{D} slices')

    return np.stack(feats_list, axis=0)  # (D, Hp, Wp, C)


class Viewer2D:
    def __init__(self, vol_display, feats, clip=(-1000, 400)):
        self.vol = vol_display          # (D, H, W) display-ready [0,1]
        self.feats = feats              # (D, Hp, Wp, C)
        self.D, self.H, self.W = vol_display.shape
        self.Dp, self.Hp, self.Wp, self.C = feats.shape
        assert self.Dp == self.D, "feature depth must equal display depth"
        self.d = self.D // 2
        self.query = None
        self.sim_vol = None
        self.pca_mode = 'local'

        print('Pre-computing global PCA...')
        self._global_pca = self._compute_global_pca()
        print('Done.')

        self._build()
        self._redraw()

    def _compute_global_pca(self):
        X = self.feats.reshape(-1, self.C).astype(np.float32)
        X -= X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = X @ Vt[0]
        pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
        fg = pc1 > 0.5
        if fg.sum() < 10:
            fg = np.ones(len(X), dtype=bool)
        _, _, Vt3 = np.linalg.svd(X[fg], full_matrices=False)
        comps = X[fg] @ Vt3[:3].T
        rgb = np.zeros((len(X), 3), np.float32)
        rgb[fg] = comps
        for c in range(3):
            lo, hi = rgb[fg, c].min(), rgb[fg, c].max()
            rgb[fg, c] = (rgb[fg, c] - lo) / (hi - lo + 1e-8)
        return rgb.reshape(self.Dp, self.Hp, self.Wp, 3)

    def _feat_img(self):
        if self.pca_mode == 'local':
            low = pca_rgb_2d(self.feats[self.d])
        else:
            low = self._global_pca[self.d]
        return up2d(low, self.H, self.W)

    def _sim_img(self):
        if self.sim_vol is None:
            return np.zeros((self.H, self.W), np.float32)
        return self.sim_vol[self.d]

    def _compute_sim_vol(self, d, h, w):
        th = min(h * self.Hp // self.H, self.Hp - 1)
        tw = min(w * self.Wp // self.W, self.Wp - 1)
        q = self.feats[d, th, tw]
        q = q / (np.linalg.norm(q) + 1e-8)
        all_f = self.feats.reshape(-1, self.C)
        sims = (all_f / (np.linalg.norm(all_f, axis=1, keepdims=True) + 1e-8)) @ q
        sims = sims.reshape(self.Dp, self.Hp, self.Wp)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        t = torch.from_numpy(sims).float().unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, (self.D, self.H, self.W),
                            mode='trilinear', align_corners=False)
        return out.squeeze().numpy()

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

        self.im_ct   = self.ax_ct.imshow(self.vol[self.d], cmap='gray', vmin=0, vmax=1)
        self.im_feat = self.ax_feat.imshow(self._feat_img())

        lkw = dict(linewidth=0.6, alpha=0.6, color='#ff4')
        self._ch_v = self.ax_ct.axvline(self.W // 2, **lkw)
        self._ch_h = self.ax_ct.axhline(self.H // 2, **lkw)

        self._title = self.fig.text(
            0.5, 0.96,
            f'Axial  {self.d}/{self.D-1}  —  PCA coloring  '
            f'(G = global <-> local,  click = similarity)',
            ha='center', va='top', color='#aaa', fontsize=9)
        self.fig.text(
            0.01, 0.01,
            'Scroll: slice  |  Click: similarity  |  '
            'Right-click/R: reset  |  G: toggle global/local PCA  |  S: screenshot',
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
            self.ax_feat.set_title(f'Features — {mode_str}', color='#6af', fontsize=12, pad=5)
        else:
            sim = self._sim_img()
            self.im_feat.set_data(plt.cm.inferno(sim)[:, :, :3])
            qd, qh, qw = self.query
            self.ax_feat.set_title(
                f'Similarity  (query d={qd},h={qh},w={qw})',
                color='#fa6', fontsize=12, pad=5)
        self._title.set_text(f'Axial  {self.d}/{self.D-1}')
        self.fig.canvas.draw_idle()

    def _on_scroll(self, ev):
        self.d = int(np.clip(self.d + int(ev.step), 0, self.D - 1))
        self._redraw()

    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None:
            return
        if ev.button == 3:
            self._reset()
            return
        if ev.button != 1:
            return
        w = int(np.clip(ev.xdata, 0, self.W - 1))
        h = int(np.clip(ev.ydata, 0, self.H - 1))
        self._ch_v.set_xdata([w])
        self._ch_h.set_ydata([h])
        self._title.set_text('Computing similarity...')
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
            if self.query is None:
                self._redraw()
        elif ev.key == 's':
            fn = 'viewer2d_screenshot.png'
            self.fig.savefig(fn, dpi=150, facecolor='#111')
            print(f'Saved {fn}')

    def show(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--nifti',      required=True,  help='Input NIfTI volume')
    ap.add_argument('--checkpoint', required=True,  help='2D pretrained checkpoint (.pth)')
    ap.add_argument('--feat_size',  type=int, default=448,
                    help='Per-slice resolution fed to the model (multiple of 16)')
    ap.add_argument('--max_slices', type=int, default=0,
                    help='Subsample depth to this many slices (0 = all)')
    ap.add_argument('--clip',       type=float, nargs=2, default=[-1000, 400],
                    help='HU clip window for display and preprocessing')
    ap.add_argument('--cect',       action='store_true',
                    help='Use 3-channel cardiac CECT windows instead of single-clip grayscale')
    ap.add_argument('--layer',      type=int, default=11,
                    help='Transformer layer index for feature extraction')
    ap.add_argument('--batch_size', type=int, default=4,
                    help='Slices per forward pass')
    ap.add_argument('--device',     default='mps' if torch.backends.mps.is_available()
                                    else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = ap.parse_args()

    print(f'Loading {args.nifti}')
    raw = load_nifti(args.nifti)
    D, H, W = raw.shape
    print(f'Raw volume: {D}×{H}×{W}')

    if args.max_slices > 0 and D > args.max_slices:
        idx = np.linspace(0, D - 1, args.max_slices, dtype=int)
        raw = raw[idx]
        D = raw.shape[0]
        print(f'Subsampled to {D} slices')

    vol_display = np.stack([preprocess_slice_for_display(raw[d], args.clip)
                            for d in range(D)], axis=0)

    print(f'Extracting features  device={args.device}  layer={args.layer}  '
          f'feat_size={args.feat_size}')
    feats = extract_features(raw, args.checkpoint, args.device, args.layer,
                             args.feat_size, tuple(args.clip), args.batch_size,
                             cect=args.cect)
    Dp, Hp, Wp, C = feats.shape
    print(f'Volume {D}×{H}×{W}  →  feature grid {Dp}×{Hp}×{Wp}  {C}-d')

    Viewer2D(vol_display, feats, tuple(args.clip)).show()


if __name__ == '__main__':
    main()
