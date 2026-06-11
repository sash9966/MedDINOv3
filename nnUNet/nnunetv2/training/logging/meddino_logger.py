"""
meddino_logger.py — richer progress.png for the 3D MedDINOv3 trainers.

Extends nnUNetLogger with two families of debug signal, all plotted into a single
progress.png so there is nothing extra to open:

  Per-class pseudo Dice
    The base logger already records dice_per_class_or_region every epoch; this
    just unpacks it into one curve per cardiac structure so you can see *which*
    class is dragging the foreground mean (e.g. Ao/PA lagging the chambers).

  ViT architecture diagnostics (logged by the trainer each epoch)
    - per-param-group gradient L2 norm (decoder / patch_embed / depth_pe /
      backbone): the single best "is training healthy?" signal — vanishing or
      exploding groups show up immediately.
    - patch_embed depth-slice learning: mean |weight| of the centre depth slice
      vs. the off-centre slices. For *centering* inflation the off-centre slices
      start at 0 and growing them is exactly the model learning 3D depth context;
      for *Ashwin* the two start equal. This panel makes the inflation strategy's
      effect visible over training.
    - weight health: depth_pos_embed norm (is depth encoding drifting from its
      sinusoidal init?), backbone weight L2 (fine-tune drift), and input_adapter
      deviation from identity.

The trainer writes the ViT values via the extra keys below (see
dinov3Trainer.meddinov3_3d_primus_multiscale_Trainer._log_vit_diagnostics). If a
value is unavailable for an epoch it is logged as NaN and simply leaves a gap.
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.logging.nnunet_logger import nnUNetLogger


# Extra per-epoch scalar keys this logger expects the trainer to populate.
_VIT_KEYS = [
    'grad_decoder', 'grad_patch_embed', 'grad_depth_pe', 'grad_backbone',
    'patch_embed_center_absmean', 'patch_embed_offcenter_absmean',
    'depth_pe_weight_norm', 'backbone_weight_norm', 'adapter_identity_dev',
]


class MedDINOLogger(nnUNetLogger):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        for k in _VIT_KEYS:
            self.my_fantastic_logging[k] = []
        # Optional pretty names for the per-class panel; set by the trainer once
        # the label manager is available. Falls back to "class i".
        self.class_names = []

    # ── resume safety: old checkpoints lack the extra keys ──────────────────
    def load_checkpoint(self, checkpoint: dict):
        super().load_checkpoint(checkpoint)
        ref_len = len(self.my_fantastic_logging.get('lrs', []))
        for k in _VIT_KEYS:
            series = self.my_fantastic_logging.get(k, [])
            if len(series) < ref_len:
                series = list(series) + [float('nan')] * (ref_len - len(series))
            self.my_fantastic_logging[k] = series

    # ── helpers ─────────────────────────────────────────────────────────────
    def _series(self, key, upto):
        s = self.my_fantastic_logging.get(key, [])
        n = min(upto, len(s))
        return list(range(n)), s[:n]

    @staticmethod
    def _legend_if_any(ax, **kw):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(**kw)

    def plot_progress_png(self, output_folder):
        # Base the epoch count on the canonical curve, not min-over-all-keys, so a
        # transiently shorter extra series never corrupts the standard panels.
        core = self.my_fantastic_logging['mean_fg_dice']
        epoch = len(core) - 1
        if epoch < 0:
            return
        x = list(range(epoch + 1))

        sns.set(font_scale=2.5)
        n_panels = 7
        fig, ax_all = plt.subplots(n_panels, 1, figsize=(30, 16 * n_panels))

        # ── Panel 0: loss + pseudo Dice (identical to base) ────────────────
        ax = ax_all[0]; ax2 = ax.twinx()
        ax.plot(x, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label='loss_tr', linewidth=4)
        ax.plot(x, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label='loss_val', linewidth=4)
        ax2.plot(x, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label='pseudo dice', linewidth=3)
        ax2.plot(x, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label='pseudo dice (mov. avg.)', linewidth=4)
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax2.set_ylabel('pseudo dice')
        ax.legend(loc=(0, 1)); ax2.legend(loc=(0.2, 1))

        # ── Panel 1: epoch duration ─────────────────────────────────────────
        ax = ax_all[1]
        ax.plot(x, [e - s for e, s in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                          self.my_fantastic_logging['epoch_start_timestamps'][:epoch + 1])],
                color='b', ls='-', label='epoch duration', linewidth=4)
        ax.set(ylim=[0, ax.get_ylim()[1]]); ax.set_xlabel('epoch'); ax.set_ylabel('time [s]'); ax.legend(loc=(0, 1))

        # ── Panel 2: learning rate ──────────────────────────────────────────
        ax = ax_all[2]
        ax.plot(x, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label='learning rate', linewidth=4)
        ax.set_xlabel('epoch'); ax.set_ylabel('learning rate'); ax.legend(loc=(0, 1))

        # ── Panel 3: per-class pseudo Dice ──────────────────────────────────
        ax = ax_all[3]
        dpc = self.my_fantastic_logging['dice_per_class_or_region'][:epoch + 1]
        rows = [r for r in dpc if isinstance(r, (list, tuple, np.ndarray))]
        if rows:
            ncls = min(len(r) for r in rows)
            arr = np.array([list(r)[:ncls] for r in rows], dtype=float)  # (E, ncls)
            cmap = plt.get_cmap('tab10')
            for c in range(ncls):
                name = self.class_names[c] if c < len(self.class_names) else f'class {c + 1}'
                ax.plot(range(arr.shape[0]), arr[:, c], lw=2.5, color=cmap(c % 10), label=name)
            ax.set_ylim(0, 1)
        ax.set_xlabel('epoch'); ax.set_ylabel('per-class pseudo Dice')
        ax.set_title('Per-class pseudo Dice — which structure lags the mean?')
        self._legend_if_any(ax, loc=(0, 1), ncol=4)

        # ── Panel 4: per-group gradient norms (log y) ───────────────────────
        ax = ax_all[4]
        grad_spec = [('grad_decoder', '#1f77b4', 'decoder'),
                     ('grad_patch_embed', '#ff7f0e', 'patch_embed_3d'),
                     ('grad_depth_pe', '#2ca02c', 'depth_pos_embed'),
                     ('grad_backbone', '#d62728', 'backbone')]
        plotted = False
        for key, color, lbl in grad_spec:
            gx, gy = self._series(key, epoch + 1)
            if gy and np.isfinite(np.asarray(gy, dtype=float)).any():
                ax.plot(gx, gy, lw=3, color=color, label=lbl); plotted = True
        if plotted:
            ax.set_yscale('log')
        ax.set_xlabel('epoch'); ax.set_ylabel('grad L2 norm (per group)')
        ax.set_title('Gradient norm per param group (post-unscale, last train batch) — health check')
        self._legend_if_any(ax, loc=(0, 1))

        # ── Panel 5: patch_embed depth-slice learning ───────────────────────
        ax = ax_all[5]
        cx, cy = self._series('patch_embed_center_absmean', epoch + 1)
        ox, oy = self._series('patch_embed_offcenter_absmean', epoch + 1)
        if cy:
            ax.plot(cx, cy, lw=4, color='#9467bd', label='centre slice |w|')
        if oy:
            ax.plot(ox, oy, lw=4, color='#8c564b', ls='--', label='off-centre slices |w|')
        ax.set_xlabel('epoch'); ax.set_ylabel('mean |patch_embed weight|')
        ax.set_title('patch_embed depth learning — off-centre rising = model using 3D depth context')
        self._legend_if_any(ax, loc=(0, 1))
        ax3 = ax.twinx()
        px, py = self._series('depth_pe_weight_norm', epoch + 1)
        if py:
            ax3.plot(px, py, lw=3, color='#17becf', ls=':', label='depth_pe norm')
            ax3.set_ylabel('depth_pos_embed L2 norm'); ax3.legend(loc=(0.25, 1))

        # ── Panel 6: weight health (backbone drift + adapter deviation) ─────
        ax = ax_all[6]
        bx, by = self._series('backbone_weight_norm', epoch + 1)
        if by:
            ax.plot(bx, by, lw=4, color='#1f77b4', label='backbone weight L2')
        ax.set_xlabel('epoch'); ax.set_ylabel('backbone weight L2 norm')
        ax.set_title('Weight health — backbone fine-tune drift + input adapter deviation from identity')
        self._legend_if_any(ax, loc=(0, 1))
        ax4 = ax.twinx()
        ax_, ay = self._series('adapter_identity_dev', epoch + 1)
        if ay and np.isfinite(np.asarray(ay, dtype=float)).any():
            ax4.plot(ax_, ay, lw=3, color='#e377c2', ls='--', label='adapter dev. from identity')
            ax4.set_ylabel('||adapter − I|| + ||bias||'); ax4.legend(loc=(0.3, 1))

        plt.tight_layout()
        fig.savefig(join(output_folder, 'progress.png'))
        plt.close()
