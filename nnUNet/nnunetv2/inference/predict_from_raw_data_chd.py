"""
predict_from_raw_data_chd.py — sliding-window predictor for the
diagnosis-conditioned 3D MedDINOv3 network.

The base nnUNetPredictor calls `self.network(x)` (no extra args) inside
`_internal_maybe_mirror_and_predict`. The CHD network instead expects a per-case
multi-hot diagnosis. Rather than reimplement the mirroring logic, this subclass
just stashes the current case's diagnosis on the network as `_pending_diagnosis`
(broadcast to the current batch) and delegates to the parent — the network's
forward falls back to `_pending_diagnosis` when called with no diagnosis kwarg.

Usage:
  predictor = nnUNetPredictorCHD(...)
  predictor.initialize_from_trained_model_folder(...)
  predictor.set_diagnosis_vec(case_multihot)   # 1-D [num_diagnoses], per case
  predictor.predict_single_npy_array(...) / predict_from_files(...)

If no diagnosis is set, prediction falls back to the zero/unconditioned vector.
"""

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class nnUNetPredictorCHD(nnUNetPredictor):
    current_diagnosis_vec = None  # 1-D multi-hot [num_diagnoses] for the active case

    def set_diagnosis_vec(self, vec):
        """Set the multi-hot diagnosis (1-D, length num_diagnoses) for the next case."""
        if vec is None:
            self.current_diagnosis_vec = None
        else:
            self.current_diagnosis_vec = torch.as_tensor(np.asarray(vec), dtype=torch.float32)

    def _set_pending_on_network(self, diagnosis):
        net = self.network
        # Unwrap a torch.compile wrapper if present so the attribute reaches the real module.
        target = getattr(net, "_orig_mod", net)
        if hasattr(target, "set_diagnosis"):
            target.set_diagnosis(diagnosis)
        else:
            setattr(target, "_pending_diagnosis", diagnosis)

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_diagnosis_vec is not None:
            diag = self.current_diagnosis_vec.to(x.device).unsqueeze(0).expand(x.shape[0], -1)
            self._set_pending_on_network(diag)
        else:
            self._set_pending_on_network(None)
        return super()._internal_maybe_mirror_and_predict(x)
