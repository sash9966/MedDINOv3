#!/usr/bin/env python3
"""
MedDINOv3 3D Feature Viewer — Slicer-style with volumetric isosurface

Layout (matches 3D Slicer):
  ┌──────────────────┬──────────────────┐
  │  Axial   [RED]   │  3D View         │  ← isosurface + CT planes
  ├──────────────────┼──────────────────┤
  │  Coronal [GREEN] │  Sagittal [YEL]  │
  └──────────────────┴──────────────────┘
                  [Threshold slider]

After clicking a voxel in any 2D panel:
  - Cosine-similarity isosurface appears in the 3D panel (marching cubes)
  - Color = inferno map (yellow = most similar, blue = least)
  - CT slice planes are rendered in 3D as context
  - All three 2D panels show the heatmap overlay

Controls:
  Scroll  (2D panels) : change slice
  Left-click (2D)     : query similarity
  Right-click / R     : clear
  Drag   (3D panel)   : rotate
  +  /  -             : raise / lower isosurface threshold
  S                   : screenshot

Requires: scikit-image for marching cubes (pip install scikit-image)
          Falls back to voxel scatter if not available.

Usage:
  python feature_viewer_3d.py \\
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
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    warnings.warn('scikit-image not found; using voxel scatter instead of '
                  'marching cubes.  pip install scikit-image for meshes.')

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
    return f.squeeze(0).permute(1,2,3,0).cpu().float().numpy()

def up3d(arr, D, H, W):
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, (D,H,W), mode='trilinear',
                         align_corners=False).squeeze().numpy()

def pca_rgb_global(feats):
    """(Dp,Hp,Wp,C) → (Dp,Hp,Wp,3) consistent-color 3D PCA."""
    Dp,Hp,Wp,C = feats.shape
    X = feats.reshape(-1, C).astype(np.float32)
    X -= X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = X @ Vt[0]
    pc1 = (pc1-pc1.min())/(pc1.max()-pc1.min()+1e-8)
    fg  = pc1 > 0.5
    if fg.sum() < 10: fg = np.ones(len(X), dtype=bool)
    _, _, Vt3 = np.linalg.svd(X[fg], full_matrices=False)
    comps = X[fg] @ Vt3[:3].T
    rgb = np.zeros((len(X), 3), np.float32)
    rgb[fg] = comps
    for c in range(3):
        lo, hi = rgb[fg,c].min(), rgb[fg,c].max()
        rgb[fg,c] = (rgb[fg,c]-lo)/(hi-lo+1e-8)
    return rgb.reshape(Dp, Hp, Wp, 3)


# ── 3-D viewer ────────────────────────────────────────────────────────────────

_C = dict(axial='#ff4444', coronal='#44ee44', sagittal='#ffcc00', view3d='#8888ff')


class FeatureViewer3D:

    def __init__(self, vol, feats, d_patch=2):
        self.vol, self.feats, self.d_patch = vol, feats, d_patch
        self.D, self.H, self.W = vol.shape
        self.Dp, self.Hp, self.Wp, self.C = feats.shape
        self.pos = np.array([self.D//2, self.H//2, self.W//2], int)
        self.sim_vol   = None
        self.threshold = 0.65

        print('Pre-computing PCA…')
        pca_low = pca_rgb_global(feats)   # (Dp, Hp, Wp, 3)
        self.pca_full = up3d_rgb(pca_low, self.D, self.H, self.W)
        print('Done.')

        self._build()
        self._redraw_2d()
        self._redraw_3d()

    # ── figure ───────────────────────────────────────────────────────────────

    def _build(self):
        self.fig = plt.figure(figsize=(14, 11), facecolor='#111')
        self.fig.canvas.manager.set_window_title('MedDINOv3 3D Feature Viewer')

        # Leave bottom 6% for the threshold slider
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               left=0.02, right=0.98,
                               top=0.94, bottom=0.08,
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
            ax.set_title(name.capitalize(),
                         color=_C[name], fontsize=9, pad=3)

        # 3D axis style
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

        # ── CT image handles ────────────────────────────────────────────────
        d, h, w = self.pos
        gkw = dict(cmap='gray', vmin=0, vmax=1)
        self.im_ct = {
            'axial':    self.ax2d['axial'].imshow(
                            self.vol[d],         aspect='equal', **gkw),
            'coronal':  self.ax2d['coronal'].imshow(
                            self.vol[:, :, w],   aspect='auto',  **gkw),
            'sagittal': self.ax2d['sagittal'].imshow(
                            self.vol[:, h, :],   aspect='auto',  **gkw),
        }
        # ── PCA overlay handles ─────────────────────────────────────────────
        self.ov_pca = {
            'axial':    self.ax2d['axial'].imshow(
                            self.pca_full[d],       aspect='equal', alpha=0.5),
            'coronal':  self.ax2d['coronal'].imshow(
                            self.pca_full[:, :, w], aspect='auto',  alpha=0.5),
            'sagittal': self.ax2d['sagittal'].imshow(
                            self.pca_full[:, h, :], aspect='auto',  alpha=0.5),
        }
        # ── Similarity overlay handles ──────────────────────────────────────
        sh = dict(axial=(self.H,self.W), coronal=(self.D,self.H), sagittal=(self.D,self.W))
        self.ov_sim = {
            k: self.ax2d[k].imshow(np.zeros(sh[k]), cmap='inferno',
                                    vmin=0, vmax=1, alpha=0.0,
                                    aspect='equal' if k=='axial' else 'auto')
            for k in ('axial','coronal','sagittal')
        }
        # ── Crosshairs ──────────────────────────────────────────────────────
        lkw = dict(linewidth=0.7, alpha=0.7)
        self.ch = {
            'axial':   [self.ax2d['axial'].axvline(w,   color=_C['sagittal'], **lkw),
                        self.ax2d['axial'].axhline(h,   color=_C['coronal'],  **lkw)],
            'coronal': [self.ax2d['coronal'].axvline(h, color=_C['axial'],    **lkw),
                        self.ax2d['coronal'].axhline(d, color=_C['sagittal'], **lkw)],
            'sagittal':[self.ax2d['sagittal'].axvline(w,color=_C['axial'],    **lkw),
                        self.ax2d['sagittal'].axhline(d,color=_C['coronal'],  **lkw)],
        }
        # ── Threshold slider ─────────────────────────────────────────────────
        ax_sl = self.fig.add_axes([0.15, 0.025, 0.70, 0.018],
                                   facecolor='#222')
        self._slider = Slider(ax_sl, 'Threshold', 0.40, 0.98,
                              valinit=self.threshold, color='#fa6')
        self._slider.label.set_color('#aaa')
        self._slider.valtext.set_color('#fa6')
        self._slider.on_changed(self._on_threshold)

        # ── Status ───────────────────────────────────────────────────────────
        self._status = self.fig.text(
            0.5, 0.97,
            'MedDINOv3 3D Viewer — click a 2D panel to query similarity',
            ha='center', va='top', color='#888', fontsize=9)
        self.fig.text(
            0.01, 0.002,
            'Click 2D: query  |  Scroll: slice  |  Drag 3D: rotate  |  '
            '+/-: threshold  |  R: reset  |  S: screenshot',
            color='#444', fontsize=7)

        c = self.fig.canvas
        c.mpl_connect('button_press_event', self._on_click)
        c.mpl_connect('scroll_event',       self._on_scroll)
        c.mpl_connect('key_press_event',    self._on_key)

    # ── drawing ───────────────────────────────────────────────────────────────

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

        # ── CT slice planes (downsampled) ────────────────────────────────────
        step = max(self.H // 40, 1)
        h_idx = np.arange(0, self.H, step)
        w_idx = np.arange(0, self.W, step)
        d_idx = np.arange(0, self.D, max(self.D // 20, 1))

        def gray(arr): return plt.cm.gray(arr)

        # Axial plane at d
        HH, WW = np.meshgrid(h_idx, w_idx, indexing='ij')
        ct_a = self.vol[d, ::step, ::step]
        ax.plot_surface(np.full_like(HH, d), HH, WW,
                        facecolors=gray(ct_a), rstride=1, cstride=1,
                        alpha=0.35, shade=False, linewidth=0)

        # Coronal plane at w
        DD, HH2 = np.meshgrid(d_idx, h_idx, indexing='ij')
        ct_c = self.vol[::max(self.D//20,1), ::step, w]
        ax.plot_surface(DD, HH2, np.full_like(DD, w),
                        facecolors=gray(ct_c), rstride=1, cstride=1,
                        alpha=0.30, shade=False, linewidth=0)

        # Sagittal plane at h
        DD2, WW2 = np.meshgrid(d_idx, w_idx, indexing='ij')
        ct_s = self.vol[::max(self.D//20,1), h, ::step]
        ax.plot_surface(DD2, np.full_like(DD2, h), WW2,
                        facecolors=gray(ct_s), rstride=1, cstride=1,
                        alpha=0.30, shade=False, linewidth=0)

        # ── Similarity isosurface ────────────────────────────────────────────
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
            return  # nothing above threshold

        if _HAS_SKIMAGE:
            verts, faces, _, _ = marching_cubes(sv, level=thr, step_size=3)
            if len(faces) == 0:
                return
            # Color each face by the mean similarity at its centroid
            centroids = verts[faces].mean(axis=1).astype(int)
            c0 = np.clip(centroids[:,0], 0, self.D-1)
            c1 = np.clip(centroids[:,1], 0, self.H-1)
            c2 = np.clip(centroids[:,2], 0, self.W-1)
            face_vals = sv[c0, c1, c2]
            face_colors = plt.cm.inferno(face_vals)
            face_colors[:, 3] = 0.85

            mesh = Poly3DCollection(verts[faces], zsort='average')
            mesh.set_facecolor(face_colors)
            mesh.set_edgecolor('none')
            ax.add_collection3d(mesh)
        else:
            # Fallback: scatter high-similarity voxels
            step = max(self.D // 20, 1)
            mask = sv[::step, ::step, ::step] > thr
            di, hi, wi = np.where(mask)
            vals = sv[di*step, hi*step, wi*step]
            if len(di) == 0: return
            cols = plt.cm.inferno(vals)
            ax.scatter(di*step, hi*step, wi*step,
                       c=cols, s=8, alpha=0.6, depthshade=True)

    # ── events ───────────────────────────────────────────────────────────────

    def _panel_name(self, ax):
        for n, a in self.ax2d.items():
            if a is ax: return n
        return None

    def _on_click(self, ev):
        if ev.inaxes is None or ev.xdata is None: return
        name = self._panel_name(ev.inaxes)
        if name is None: return          # 3D axis: handled by mpl internally

        if ev.button == 3:
            self._reset(); return
        if ev.button != 1: return

        ix = int(np.clip(ev.xdata, 0, self.W-1))
        iy = int(np.clip(ev.ydata, 0, self.H-1))
        d, h, w = self.pos

        if name == 'axial':
            w, h = ix, iy
        elif name == 'coronal':
            h, d = ix, iy
        elif name == 'sagittal':
            w, d = ix, iy

        self.pos = np.array([int(np.clip(d,0,self.D-1)),
                              int(np.clip(h,0,self.H-1)),
                              int(np.clip(w,0,self.W-1))], int)
        self._status.set_text(f'Computing similarity for ({d},{h},{w})…')
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        self.sim_vol = self._compute_sim_vol(*self.pos)
        qd,qh,qw = self.pos
        td = min(qd//self.d_patch, self.Dp-1)
        th = min(qh//16, self.Hp-1)
        tw = min(qw//16, self.Wp-1)
        self._status.set_text(
            f'Similarity — voxel ({qd},{qh},{qw})  token ({td},{th},{tw})  '
            f'threshold={self.threshold:.2f}')
        self._redraw_2d()
        self._redraw_3d()

    def _on_scroll(self, ev):
        if ev.inaxes is None: return
        name = self._panel_name(ev.inaxes)
        if name is None: return
        delta = int(ev.step)
        d, h, w = self.pos
        if name == 'axial':
            d = int(np.clip(d + delta, 0, self.D-1))
        elif name == 'coronal':
            w = int(np.clip(w + delta, 0, self.W-1))
        elif name == 'sagittal':
            h = int(np.clip(h + delta, 0, self.H-1))
        self.pos = np.array([d, h, w], int)
        self._redraw_2d()
        if self.sim_vol is not None:
            self._redraw_3d()

    def _on_threshold(self, val):
        self.threshold = float(val)
        if self.sim_vol is not None:
            self._redraw_3d()

    def _on_key(self, ev):
        if ev.key == 'r':
            self._reset()
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

    # ── similarity ────────────────────────────────────────────────────────────

    def _compute_sim_vol(self, d, h, w):
        td = min(d//self.d_patch, self.Dp-1)
        th = min(h//16, self.Hp-1)
        tw = min(w//16, self.Wp-1)
        q  = self.feats[td, th, tw]
        q  = q / (np.linalg.norm(q) + 1e-8)
        af = self.feats.reshape(-1, self.C)
        sims = (af / (np.linalg.norm(af, axis=1, keepdims=True) + 1e-8)) @ q
        sims = sims.reshape(self.Dp, self.Hp, self.Wp)
        sims = (sims-sims.min())/(sims.max()-sims.min()+1e-8)
        return up3d(sims, self.D, self.H, self.W)

    def show(self): plt.show()


# ── upsampling helper for RGB (not in up3d) ───────────────────────────────────

def up3d_rgb(arr, D, H, W):
    """(Dp,Hp,Wp,3) → (D,H,W,3)."""
    t = torch.from_numpy(arr).float().permute(3,0,1,2).unsqueeze(0)
    out = F.interpolate(t, (D,H,W), mode='trilinear', align_corners=False)
    return out.squeeze(0).permute(1,2,3,0).numpy()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--nifti',      required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--d_patch',    type=int, default=2)
    ap.add_argument('--resize',     type=int, nargs=3, default=[64, 256, 256])
    ap.add_argument('--clip',       type=float, nargs=2, default=[-1000, 400])
    ap.add_argument('--layer',      type=int, default=11)
    ap.add_argument('--threshold',  type=float, default=0.65)
    ap.add_argument('--device',     default='mps' if torch.backends.mps.is_available()
                                    else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = ap.parse_args()

    print(f'Loading {args.nifti}')
    vol = preprocess_ct(load_nifti(args.nifti), args.resize, args.clip)
    print(f'Extracting features  device={args.device}  layer={args.layer}')
    feats = extract_features(vol, args.checkpoint, args.d_patch, args.device, args.layer)
    Dp,Hp,Wp,C = feats.shape
    print(f'Volume {vol.shape}  →  patch grid {Dp}×{Hp}×{Wp}  {C}-d')
    if not _HAS_SKIMAGE:
        print('TIP: pip install scikit-image  for marching-cubes meshes')
    v = FeatureViewer3D(vol, feats, args.d_patch)
    v.threshold = args.threshold
    v.show()

if __name__ == '__main__':
    main()
