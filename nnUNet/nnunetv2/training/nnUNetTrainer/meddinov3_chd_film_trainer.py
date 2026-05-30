"""
meddinov3_chd_film_trainer.py — Diagnosis-conditioned (FiLM) 3D MedDINOv3 trainer.

Kept in its own module (not appended to dinov3Trainer.py) because nnUNet's
recursive_find_python_class discovers trainer classes anywhere under the
nnUNetTrainer package. This keeps the large dinov3Trainer.py untouched, so the
currently-running d16/d8 centering jobs are unaffected.

meddinov3_3d_chd_film_d8_Trainer extends the d=8 centering trainer (the stronger
baseline). Conditioning is identity-initialised FiLM, so a fresh conditioned run
begins numerically identical to the unconditioned d=8 baseline; any divergence is
attributable to the diagnosis prior. FiLM scope is set by env toggles
(CHD_FILM_BRIDGE / CHD_FILM_DECODER) for one-change-at-a-time ablations.

Required env vars (in addition to the parent's):
  CHD_NUM_DIAGNOSES   length of the multi-hot diagnosis vector
  CHD_FILM_BRIDGE     "1" (default) -> FiLM on each ViT grid before concat
  CHD_FILM_DECODER    "1" -> FiLM after each decoder upsampling stage (default "0")

Each preprocessed case must carry properties['diagnosis_vec'] (multi-hot, length
CHD_NUM_DIAGNOSES); inject it with tools/add_chd_diagnosis_to_properties.py.
"""

import os

import torch
from torch import nn

# Import parent trainers + the symbols their get_dataloaders body relies on,
# straight from the dinov3Trainer module so they are guaranteed identical.
import nnunetv2.training.nnUNetTrainer.dinov3Trainer as _base
from nnunetv2.training.nnUNetTrainer.dinov3Trainer import (
    meddinov3_3d_primus_multiscale_Trainer,
    meddinov3_3d_centering_d8_primus_multiscale_Trainer,
)
from nnunetv2.training.dataloading.data_loader_3d_diagnosis import nnUNetDataLoader3D_Diagnosis
from nnunetv2.training.nnUNetTrainer.dinov3.dinov3.models.primus_chd import (
    Primus_Multiscale3D_CHD,
    film_flags_from_env,
)


class meddinov3_3d_chd_film_d8_Trainer(meddinov3_3d_centering_d8_primus_multiscale_Trainer):

    def _do_i_compile(self):
        # The diagnosis reaches forward() through a mutable attribute
        # (_pending_diagnosis), changed every step/case. Disable torch.compile so
        # dynamo cannot bake or guard it stale; correctness > a little speed here.
        return False

    # ------------------------------------------------------------------ network
    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        # Reuse the parent build entirely: it loads the inflated checkpoint, builds
        # vit_base_3d with the right d_patch/in_chans, verifies the checkpoint
        # d_patch, and sinusoidally initialises depth_pos_embed. We then steal the
        # fully-initialised encoder and rewrap it in the diagnosis-conditioned head.
        base = meddinov3_3d_primus_multiscale_Trainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

        d_patch = int(os.environ.get("MEDDINOV3_D_PATCH", "8"))
        if "CHD_NUM_DIAGNOSES" not in os.environ:
            raise RuntimeError(
                "CHD_NUM_DIAGNOSES is not set. Export it (= length of the multi-hot "
                "diagnosis vector written by add_chd_diagnosis_to_properties.py)."
            )
        num_diagnoses = int(os.environ["CHD_NUM_DIAGNOSES"])
        film_bridge, film_decoder = film_flags_from_env()

        net = Primus_Multiscale3D_CHD(
            embed_dim=768,
            patch_embed_size=16,
            d_patch=d_patch,
            num_classes=num_output_channels,
            num_diagnoses=num_diagnoses,
            film_bridge=film_bridge,
            film_decoder=film_decoder,
            dino_encoder=base.dino_encoder,
            interaction_indices=[2, 5, 8, 11],
            use_input_adapter=True,
        )
        print(
            f"[chd_film] Primus_Multiscale3D_CHD built — num_diagnoses={num_diagnoses} "
            f"film_bridge={film_bridge} film_decoder={film_decoder} (FiLM identity-init)"
        )
        return net

    # ------------------------------------------------------------- dataloaders
    def get_dataloaders(self):
        # Mirror of the parent get_dataloaders, but the 3D loader is the
        # diagnosis-carrying variant. 2D branch is unreachable here (3d_fullres)
        # but kept for parity.
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = _base.nnUNetDataLoader2D(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = _base.nnUNetDataLoader2D(
                dataset_val, self.batch_size, self.configuration_manager.patch_size,
                self.configuration_manager.patch_size, self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetDataLoader3D_Diagnosis(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader3D_Diagnosis(
                dataset_val, self.batch_size, self.configuration_manager.patch_size,
                self.configuration_manager.patch_size, self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = _base.get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = _base.SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = _base.SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = _base.NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = _base.NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # -------------------------------------------------------------- optimizer
    def configure_optimizers(self):
        # Parent builds the decoder/patch_embed/depth_pe/backbone groups + the
        # per-group poly-LR-with-warmup scheduler. Decoder-stage FiLM lives inside
        # net.up_projection, so it is ALREADY in the parent's 'decoder' group. Only
        # the conditioner + bridge FiLM sit outside the parent's explicit lists, so
        # add them as one extra group at the decoder LR (else they never train).
        optimizer, lr_scheduler = super().configure_optimizers()
        net = self.network
        extra = list(net.conditioner.parameters())
        if getattr(net, "bridge_film", None) is not None:
            for m in net.bridge_film:
                extra += list(m.parameters())
        lr = self.initial_lr
        optimizer.add_param_group({
            'params': extra, 'lr': lr, 'initial_lr': lr,
            'weight_decay': self.weight_decay, 'name': 'chd_conditioner',
        })
        print(f"[chd_film] added 'chd_conditioner' optim group "
              f"({sum(p.numel() for p in extra)} params) at lr={lr:.2e}")
        return optimizer, lr_scheduler

    # ----------------------------------------------------------- forward hooks
    def _apply_diagnosis(self, batch):
        diag = batch.get('diagnosis', None)
        if diag is not None:
            diag = diag.to(self.device, non_blocking=True)
        target = getattr(self.network, "_orig_mod", self.network)
        target.set_diagnosis(diag)

    def train_step(self, batch: dict) -> dict:
        self._apply_diagnosis(batch)
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        self._apply_diagnosis(batch)
        return super().validation_step(batch)

    def perform_actual_validation(self, save_probabilities: bool = False):
        # Conditioned end-of-training validation WITHOUT copying the long parent
        # method. The parent loops over val cases single-threaded, calling
        # dataset_val.load_case(k) immediately before
        # predictor.predict_sliding_window_return_logits(data) for that same case.
        # The dataset class is obtained via infer_dataset_class, which the parent
        # re-imports at call time. So we temporarily patch infer_dataset_class to
        # return a subclass whose load_case ALSO stashes that case's diagnosis on
        # the network (_pending_diagnosis). The base predictor then calls
        # network(x) with no diagnosis kwarg and the network uses the stashed
        # vector -> correct per-case conditioning. A [1, num_diag] vector
        # broadcasts over any tile-batch in the FiLM modulation.
        import torch as _torch
        import nnunetv2.training.dataloading.nnunet_dataset as _ds_mod

        net = getattr(self.network, "_orig_mod", self.network)
        orig_infer = _ds_mod.infer_dataset_class

        def _patched_infer(folder):
            base_cls = orig_infer(folder)

            class _DiagDataset(base_cls):
                def load_case(self_inner, identifier):
                    out = super().load_case(identifier)
                    props = out[-1]  # properties is the last element of the tuple
                    vec = props.get("diagnosis_vec") if isinstance(props, dict) else None
                    if vec is not None:
                        net.set_diagnosis(
                            _torch.as_tensor(vec, dtype=_torch.float32).unsqueeze(0)
                        )
                    else:
                        net.set_diagnosis(None)
                    return out

            return _DiagDataset

        _ds_mod.infer_dataset_class = _patched_infer
        try:
            print("[chd_film] perform_actual_validation: per-case diagnosis "
                  "conditioning active (via load_case hook)")
            return super().perform_actual_validation(save_probabilities)
        finally:
            _ds_mod.infer_dataset_class = orig_infer
            net.set_diagnosis(None)
