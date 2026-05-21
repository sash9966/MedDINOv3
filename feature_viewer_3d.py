#!/usr/bin/env python3
"""
MedDINOv3 3D Feature Viewer — Slicer-style with volumetric isosurface

Uses the pretrained 2D vit_base per axial slice (same as feature_viewer_2d.py).
Builds a full (D, Hp, Wp, C) feature volume, then shows it in a 4-panel layout.

Layout (matches 3D Slicer):
  Axial   [top-left]   |  3D isosurface + CT planes  [top-right]
  Coronal [bot-left]   |  Sagittal                   [bot-right]
                   [Threshold slider]

Controls:
  Scroll  (2D panels) : change slice
  Left-click (2D)     : query similarity
  Right-click / R     : clear
  Drag   (3D panel)   : rotate
  +  /  -             : raise / lower isosurface threshold
  S                   : screenshot

Requires: scikit-image for marching cubes (pip install scikit-image)

Usage:
  python feature_viewer_3d.py \
  --nifti image.nii.gz \
  --checkpoint meddinov3_2d.pth \
  --clip -1000 1000
"""

import argparse, os, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    warnings.warn('scikit-image not found; using voxel scatter instead. '
                  'pip install scikit-image for meshes.')

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
    return data.transpose(2, 1, 0)   # (D, H, W) axial-first


# Windows for CECT cardiac: blood pool (200-600 HU) + myocardium visible in ch0,
# standard soft-tissue in ch1, wide/lung in ch2.
CECT_WINDOWS = [(-200, 600), (-160, 240), (-1000, 400)]


def slice_to_tensor(slc_hw, clip):
    v = np.clip(slc_hw, clip[0], clip[1])
    v = (v - clip[0]) / (clip[1] - clip[0])
    rgb = np.stack([v, v, v], axis=0).astype(np.float32)
    rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(rgb).unsqueeze(0)


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


def preprocess_display(slc_hw, clip):
    v = np.clip(slc_hw, clip[0], clip[1])
    return (v - clip[0]) / (clip[1] - clip[0])


def up3d(arr, D, H, W):
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (D, H, W), mode='trilinear',
                         align_corners=False).squeeze().numpy()


def up3d_rgb(arr, D, H, W):
    """(Dp, Hp, Wp, 3) -> (D, H, W, 3)."""
    t = torch.from_numpy(arr).float().permute(3, 0, 1, 2).unsqueeze(0)
    out = F.interpolate(t, (D, H, W), mode='trilinear', align_corners=False)
    return out.squeeze(0).permute(1, 2, 3, 0).numpy()


def pca_rgb_global(feats):
    """(D, Hp, Wp, C) -> (D, Hp, Wp, 3) consistent-color global PCA."""
    D, Hp, Wp, C = feats.shape
    X = feats.reshape(-1, C).astype(np.float32)
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
    return rgb.reshape(D, Hp, Wp, 3)


@torch.no_grad()
def extract_features(raw_vol, ckpt_path, device='cpu', layer=11,
                     feat_size=448, clip=(-1000, 400), batch_size=4, cect=False):
    """
    Extract per-slice features with the 2D pretrained vit_base.
    raw_vol: (D, H, W) raw HU float32
    Returns: (D, Hp, Wp, C) where Hp = feat_size//16
    """
    from models.vision_transformer import vit_base

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'teacher' in ckpt:
        # MedDINOv3 pretrained format
        sd = {k.replace('backbone.', ''): v
              for k, v in ckpt['teacher'].items()
              if k.startswith('backbone.')}
    elif 'network_weights' in ckpt:
        # nnUNet fine-tuned checkpoint — extract dino_encoder submodule
        print('[Viewer] Detected nnUNet checkpoint — extracting dino_encoder weights')
        sd = {k[len('dino_encoder.'):]: v
              for k, v in ckpt['network_weights'].items()
              if k.startswith('dino_encoder.')}
    else:
        sd = ckpt

    pe_w = sd.get('patch_embed.proj.weight')
    if pe_w is not None and pe_w.dim() == 5:
        print(f'Inflated 3D checkpoint detected (shape {list(pe_w.shape)}), '
              f'deflating to 2D by averaging depth dim...')
        sd['patch_embed.proj.weight'] = pe_w.mean(dim=2)

    m = vit_base(drop_path_rate=0., layerscale_init=1e-5,
                 n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    missing, unexpected = m.load_state_dict(sd, strict=True)
    if missing:
        print(f'WARNING missing keys: {missing[:3]}')
    m = m.to(device).eval()
    print(f'Model loaded — {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')

    fs = feat_size - (feat_size % 16)
    D = raw_vol.shape[0]
    if cect:
        print('CECT mode: using 3-channel cardiac windows (-200,600) | (-160,240) | (-1000,400)')
        slices = [slice_to_tensor_cect(raw_vol[d]) for d in range(D)]
    else:
        slices = [slice_to_tensor(raw_vol[d], clip) for d in range(D)]
    feats_list = []

    for start in range(0, D, batch_size):
        batch = torch.cat(slices[start:start + batch_size], dim=0).to(device)
        if batch.shape[-1] != fs or batch.shape[-2] != fs:
            batch = F.interpolate(batch, size=(fs, fs),
                                  mode='bilinear', align_corners=False)
        out = m.get_intermediate_layers(batch, n=[layer], reshape=True)[0]
        for i in range(out.shape[0]):
            feats_list.append(out[i].permute(1, 2, 0).cpu().float().numpy())
        print(f'  {min(start + batch_size, D)}/{D} slices')

    return np.stack(feats_list, axis=0)   # (D, Hp, Wp, C)


_C = dict(axial='#ff4444', coronal='#44ee44', sagittal='#ffcc00', view3d='#8888ff')


class FeatureViewer3D:

    def __init__(self, vol, feats):
        self.vol   = vol     # (D, H, W) display [0,1]
        self.feats = feats   # (D, Hp, Wp, C)
        self.D, self.H, self.W = vol.shape
        self.Dp, self.Hp, self.Wp, self.C = feats.shape
        assert self.Dp == self.D
        self.pos = np.array([self.D // 2, self.H // 2, self.W // 2], int)
        self.sim_vol   = None
        self.threshold = 0.65
        self._scroll_accum = {'axial': 0.0, 'coronal': 0.0, 'sagittal': 0.0}

        print('Pre-computing global PCA...')
        pca_low = pca_rgb_global(feats)
        self.pca_full = up3d_rgb(pca_low, self.D, self.H, self.W)
        print('Done.')

        self._build()
        self._redraw_2d()
        self._redraw_3d()

    def _build(self):
        self.fig = plt.figure(figsize=(14, 11), facecolor='#111')
        self.fig.canvas.manager.set_window_title('MedDINOv3 3D Feature Viewer')

        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               left=0.02, right=0.98,
                               top=0.94, bottom=0.12,
                               hspace=0.06, wspace=0.06)

        self.ax2d = {
            'axial':    self.fig.add_subplot(gs[0, 0]),
            'coronal':  self.fig.add_subplot(gs[1, 0]),
            'sagittal': self.fig.add_subplot(gs[1, 1]),
        }
        self.ax3d = self.fig.add_subplot(gs[0, 1], projection='3d')

        for name, ax in self.ax2d.items():
            ax.set_facecolor('#111')
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor(_C[name])
            ax.set_title(name.capitalize(), color=_C[name], fontsize=9, pad=3)

        self.ax3d.set_facecolor('#111')
        self.ax3d.set_title('3D', color=_C['view3d'], fontsize=9, pad=3)
        for pane in (self.ax3d.xaxis.pane, self.ax3d.yaxis.pane,
                     self.ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#333')
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.label.set_color('#555')
            axis._axinfo['tick']['color'] = '#555'
        self.ax3d.view_init(elev=25, azim=-65)

        d, h, w = self.pos
        gkw = dict(cmap='gray', vmin=0, vmax=1)
        self.im_ct = {
            'axial':    self.ax2d['axial'].imshow(
                            self.vol[d],       aspect='equal', **gkw),
            'coronal':  self.ax2d['coronal'].imshow(
                            self.vol[:, :, w], aspect='auto',  **gkw),
            'sagittal': self.ax2d['sagittal'].imshow(
                            self.vol[:, h, :], aspect='auto',  **gkw),
        }
        self.ov_pca = {
            'axial':    self.ax2d['axial'].imshow(
                            self.pca_full[d],        aspect='equal', alpha=0.5),
            'coronal':  self.ax2d['coronal'].imshow(
                            self.pca_full[:, :, w],  aspect='auto',  alpha=0.5),
            'sagittal': self.ax2d['sagittal'].imshow(
                            self.pca_full[:, h, :],  aspect='auto',  alpha=0.5),
        }
        sh = dict(axial=(self.H, self.W),
                  coronal=(self.D, self.H),
                  sagittal=(self.D, self.W))
        self.ov_sim = {
            k: self.ax2d[k].imshow(
                np.zeros(sh[k]), cmap='inferno', vmin=0, vmax=1, alpha=0.0,
                aspect='equal' if k == 'axial' else 'auto')
            for k in ('axial', 'coronal', 'sagittal')
        }

        lkw = dict(linewidth=0.7, alpha=0.7)
        self.ch = {
            'axial':    [self.ax2d['axial'].axvline(w,    color=_C['sagittal'], **lkw),
                         self.ax2d['axial'].axhline(h,    color=_C['coronal'],  **lkw)],
            'coronal':  [self.ax2d['coronal'].axvline(h,  color=_C['axial'],    **lkw),
                         self.ax2d['coronal'].axhline(d,  color=_C['sagittal'], **lkw)],
            'sagittal': [self.ax2d['sagittal'].axvline(w, color=_C['axial'],    **lkw),
                         self.ax2d['sagittal'].axhline(d, color=_C['coronal'],  **lkw)],
        }

        ax_slice = self.fig.add_axes([0.15, 0.068, 0.70, 0.018], facecolor='#222')
        self._slice_slider = Slider(ax_slice, 'Axial slice', 0, self.D - 1,
                                    valinit=self.pos[0], valstep=1, color='#f44')
        self._slice_slider.label.set_color('#aaa')
        self._slice_slider.valtext.set_color('#f44')
        self._slice_slider.on_changed(self._on_slice_slider)

        ax_sl = self.fig.add_axes([0.15, 0.033, 0.70, 0.018], facecolor='#222')
        self._slider = Slider(ax_sl, 'Threshold', 0.40, 0.98,
                              valinit=self.threshold, color='#fa6')
        self._slider.label.set_color('#aaa')
        self._slider.valtext.set_color('#fa6')
        self._slider.on_changed(self._on_threshold)

        self._status = self.fig.text(
            0.5, 0.97,
            'MedDINOv3 3D Viewer — click a 2D panel to query similarity',
            ha='center', va='top', color='#888', fontsize=9)
        self.fig.text(
            0.01, 0.004,
            'Click 2D: query  |  Scroll / Up-Down arrows: axial slice  |  '
            'PgUp/PgDn: jump 10  |  Drag 3D: rotate  |  +/-: threshold  |  R: reset  |  S: screenshot',
            color='#444', fontsize=7)

        c = self.fig.canvas
        c.mpl_connect('button_press_event', self._on_click)
        c.mpl_connect('scroll_event',       self._on_scroll)
        c.mpl_connect('key_press_event',    self._on_key)

    def _redraw_2d(self):
        d, h, w = self.pos
        self.im_ct['axial'].set_data(self.vol[d])
        self.im_ct['coronal'].set_data(self.vol[:, :, w])
        self.im_ct['sagittal'].set_data(self.vol[:, h, :])

        self.ov_pca['axial'].set_data(self.pca_full[d])
        self.ov_pca['coronal'].set_data(self.pca_full[:, :, w])
        self.ov_pca['sagittal'].set_data(self.pca_full[:, h, :])

        if self.sim_vol is not None:
            sv = self.sim_vol
            self.ov_sim['axial'].set_data(sv[d])
            self.ov_sim['coronal'].set_data(sv[:, :, w])
            self.ov_sim['sagittal'].set_data(sv[:, h, :])
            for k in self.ov_sim:
                self.ov_pca[k].set_alpha(0.0)
                self.ov_sim[k].set_alpha(0.55)
        else:
            for k in self.ov_sim:
                self.ov_sim[k].set_alpha(0.0)
                self.ov_pca[k].set_alpha(0.5)

        self.ch['axial'][0].set_xdata([w]);    self.ch['axial'][1].set_ydata([h])
        self.ch['coronal'][0].set_xdata([h]);  self.ch['coronal'][1].set_ydata([d])
        self.ch['sagittal'][0].set_xdata([w]); self.ch['sagittal'][1].set_ydata([d])
        if self.sim_vol is None:
            self._status.set_text(
                f'Slice {d}/{self.D-1}  —  pos ({d}, {h}, {w})  —  click to query similarity')
        self.fig.canvas.draw_idle()

    def _redraw_3d(self):
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor('#111')
        ax.set_title('3D', color=_C['view3d'], fontsize=9, pad=3)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#222')

        d, h, w = self.pos
        step = max(self.H // 40, 1)
        h_idx = np.arange(0, self.H, step)
        w_idx = np.arange(0, self.W, step)
        d_idx = np.arange(0, self.D, max(self.D // 20, 1))

        def gray(arr): return plt.cm.gray(arr)

        HH, WW = np.meshgrid(h_idx, w_idx, indexing='ij')
        ax.plot_surface(np.full_like(HH, d), HH, WW,
                        facecolors=gray(self.vol[d, ::step, ::step]),
                        rstride=1, cstride=1, alpha=0.35, shade=False, linewidth=0)

        DD, HH2 = np.meshgrid(d_idx, h_idx, indexing='ij')
        ax.plot_surface(DD, HH2, np.full_like(DD, w),
                        facecolors=gray(self.vol[::max(self.D//20,1), ::step, w]),
                        rstride=1, cstride=1, alpha=0.30, shade=False, linewidth=0)

        DD2, WW2 = np.meshgrid(d_idx, w_idx, indexing='ij')
        ax.plot_surface(DD2, np.full_like(DD2, h), WW2,
                        facecolors=gray(self.vol[::max(self.D//20,1), h, ::step]),
                        rstride=1, cstride=1, alpha=0.30, shade=False, linewidth=0)

        if self.sim_vol is not None:
            self._draw_isosurface(ax)

        ax.set_xlim(0, self.D); ax.set_ylim(0, self.H); ax.set_zlim(0, self.W)
        ax.set_xlabel('D', color='#555', fontsize=7)
        ax.set_ylabel('H', color='#555', fontsize=7)
        ax.set_zlabel('W', color='#555', fontsize=7)
        ax.tick_params(colors='#333', labelsize=6)
        self.fig.canvas.draw_idle()

    def _draw_isosurface(self, ax):
        sv = self.sim_vol
        thr = self.threshold
        if sv.max() < thr:
            return
        if _HAS_SKIMAGE:
            verts, faces, _, _ = marching_cubes(sv, level=thr, step_size=3)
            if len(faces) == 0:
                return
            centroids = verts[faces].mean(axis=1).astype(int)
            c0 = np.clip(centroids[:, 0], 0, self.D - 1)
            c1 = np.clip(centroids[:, 1], 0, self.H - 1)
            c2 = np.clip(centroids[:, 2], 0, self.W - 1)
            face_colors = plt.cm.inferno(sv[c0, c1, c2])
            face_colors[:, 3] = 0.85
            mesh = Poly3DCollection(verts[faces], zsort='average')
            mesh.set_facecolor(face_colors)
            mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)
        else:
            step = max(self.D // 20, 1)
            mask = sv[::step, ::step, ::step] > thr
            di, hi, wi = np.where(mask)
            if len(di) == 0:
                return
            vals = sv[di * step, hi * step, wi * step]
            ax.scatter(di * step, hi * step, wi * step,
                       c=plt.cm.inferno(vals), s=8, alpha=0.6, depthshade=True)

    def _panel_name(self, ax):
        for n, a in self.ax2d.items():
            if a is ax: return n
        return None

    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None: return
        name = self._panel_name(ev.inaxes)
        if name is None: return
        if ev.button == 3:
            self._reset(); return
        if ev.button != 1: return

        ix = int(np.clip(ev.xdata, 0, self.W - 1))
        iy = int(np.clip(ev.ydata, 0, self.H - 1))
        d, h, w = self.pos

        if name == 'axial':
            w, h = ix, iy
        elif name == 'coronal':
            h, d = ix, iy
        elif name == 'sagittal':
            w, d = ix, iy

        self.pos = np.array([int(np.clip(d, 0, self.D - 1)),
                              int(np.clip(h, 0, self.H - 1)),
                              int(np.clip(w, 0, self.W - 1))], int)
        self._status.set_text(f'Computing similarity for ({d},{h},{w})...')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        self.sim_vol = self._compute_sim_vol(*self.pos)
        qd, qh, qw = self.pos
        self._status.set_text(
            f'Similarity — voxel ({qd},{qh},{qw})  threshold={self.threshold:.2f}')
        self._redraw_2d()
        self._redraw_3d()

    def _on_slice_slider(self, val):
        d = int(round(val))
        self.pos[0] = int(np.clip(d, 0, self.D - 1))
        self._redraw_2d()

    def _on_scroll(self, ev):
        if ev.inaxes is None: return
        name = self._panel_name(ev.inaxes)
        if name is None: return
        self._scroll_accum[name] += ev.step
        delta = int(self._scroll_accum[name])
        if delta == 0: return
        self._scroll_accum[name] -= delta
        d, h, w = self.pos
        if   name == 'axial':    d = int(np.clip(d + delta, 0, self.D - 1))
        elif name == 'coronal':  w = int(np.clip(w + delta, 0, self.W - 1))
        elif name == 'sagittal': h = int(np.clip(h + delta, 0, self.H - 1))
        self.pos = np.array([d, h, w], int)
        self._sync_slice_slider()
        self._redraw_2d()
        if self.sim_vol is not None:
            self._redraw_3d()

    def _sync_slice_slider(self):
        self._slice_slider.eventson = False
        self._slice_slider.set_val(self.pos[0])
        self._slice_slider.eventson = True

    def _on_threshold(self, val):
        self.threshold = float(val)
        if self.sim_vol is not None:
            self._redraw_3d()

    def _on_key(self, ev):
        if ev.key == 'r':
            self._reset()
        elif ev.key in ('up', 'down', 'pageup', 'pagedown'):
            step = {'up': 1, 'down': -1, 'pageup': 10, 'pagedown': -10}[ev.key]
            self.pos[0] = int(np.clip(self.pos[0] + step, 0, self.D - 1))
            self._sync_slice_slider()
            self._redraw_2d()
        elif ev.key in ('+', '='):
            self._slider.set_val(min(self.threshold + 0.02, 0.98))
        elif ev.key == '-':
            self._slider.set_val(max(self.threshold - 0.02, 0.40))
        elif ev.key == 's':
            fn = 'viewer3d_screenshot.png'
            self.fig.savefig(fn, dpi=150, facecolor='#111')
            print(f'Saved {fn}')

    def _reset(self):
        self.sim_vol = None
        self._status.set_text('MedDINOv3 3D Viewer — click a 2D panel to query')
        self._redraw_2d()
        self._redraw_3d()

    def _compute_sim_vol(self, d, h, w):
        th = min(h * self.Hp // self.H, self.Hp - 1)
        tw = min(w * self.Wp // self.W, self.Wp - 1)
        q  = self.feats[d, th, tw]
        q  = q / (np.linalg.norm(q) + 1e-8)
        af = self.feats.reshape(-1, self.C)
        sims = (af / (np.linalg.norm(af, axis=1, keepdims=True) + 1e-8)) @ q
        sims = sims.reshape(self.Dp, self.Hp, self.Wp)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        return up3d(sims, self.D, self.H, self.W)

    def show(self): plt.show()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--nifti',      required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--feat_size',  type=int, default=448,
                    help='Per-slice resolution fed to the model (multiple of 16)')
    ap.add_argument('--max_slices', type=int, default=0,
                    help='Subsample depth to at most N slices (0 = all)')
    ap.add_argument('--clip',       type=float, nargs=2, default=[-1000, 400])
    ap.add_argument('--cect',       action='store_true',
                    help='Use 3-channel cardiac CECT windows instead of single-clip grayscale')
    ap.add_argument('--layer',      type=int, default=11)
    ap.add_argument('--threshold',  type=float, default=0.65)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device',     default='mps' if torch.backends.mps.is_available()
                                    else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = ap.parse_args()

    print(f'Loading {args.nifti}')
    raw = load_nifti(args.nifti)
    D, H, W = raw.shape
    print(f'Raw volume: {D}x{H}x{W}')

    if args.max_slices > 0 and D > args.max_slices:
        idx = np.linspace(0, D - 1, args.max_slices, dtype=int)
        raw = raw[idx]
        D = raw.shape[0]
        print(f'Subsampled to {D} slices')

    clip = tuple(args.clip)
    vol_display = np.stack([preprocess_display(raw[d], clip) for d in range(D)])

    print(f'Extracting features  device={args.device}  layer={args.layer}  feat_size={args.feat_size}')
    feats = extract_features(raw, args.checkpoint, args.device, args.layer,
                             args.feat_size, clip, args.batch_size, cect=args.cect)
    print(f'Feature grid: {feats.shape[0]}x{feats.shape[1]}x{feats.shape[2]}  dim={feats.shape[3]}')

    if not _HAS_SKIMAGE:
        print('TIP: pip install scikit-image for marching-cubes meshes')

    v = FeatureViewer3D(vol_display, feats)
    v.threshold = args.threshold
    v.show()


if __name__ == '__main__':
    main()
