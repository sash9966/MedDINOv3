"""
data_loader_3d_diagnosis.py — 3D dataloader that also carries the per-case
multi-hot CHD diagnosis vector.

Identical to nnUNetDataLoader3D except each case's
`properties['diagnosis_vec']` (a multi-hot list/array of length num_diagnoses,
written into the case .pkl by tools/add_chd_diagnosis_to_properties.py) is
collected and returned under the batch key 'diagnosis' as a float tensor
[B, num_diagnoses]. The trainer moves it to device and hands it to the network.
"""

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D


class nnUNetDataLoader3D_Diagnosis(nnUNetDataLoader3D):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        diag_list = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            if 'diagnosis_vec' not in properties:
                raise KeyError(
                    f"Case '{i}' is missing properties['diagnosis_vec']. Run "
                    "tools/add_chd_diagnosis_to_properties.py on this preprocessed "
                    "folder before training the diagnosis-conditioned trainer."
                )
            diag_list.append(np.asarray(properties['diagnosis_vec'], dtype=np.float32))

            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        diag_all = torch.from_numpy(np.stack(diag_list, axis=0)).float()  # [B, num_diagnoses]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
                return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'diagnosis': diag_all}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys, 'diagnosis': diag_all}
