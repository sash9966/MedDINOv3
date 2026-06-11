"""
topology_metrics.py — topology / surface / connectivity metrics for 3D whole-heart
CHD segmentation. Pure functions: integer label maps in, dicts out.

This is Phase A of the structured-output evaluation roadmap. It exists to *quantify*
the failure modes that mean-Dice hides — fragmented myocardium, discontinuous /
switched Ao-PA, small-vessel dropout, anatomically implausible topology — so that
later training changes (losses, sampling, priors) are attributable. It touches no
training or inference code.

Conventions
-----------
- Volumes are 3D integer numpy arrays in (z, y, x) order, as produced by
  SimpleITK.GetArrayFromImage.
- `spacing_xyz` is (sx, sy, sz) in mm, exactly as SimpleITK.Image.GetSpacing()
  returns it. Voxel volume = sx*sy*sz.
- Connectivity for 3D components is 26-neighbour.
- Every function is safe on empty pred and/or empty target. Conventions:
  both empty -> perfect agreement (Dice/NSD/clDice = 1.0, distances = 0);
  exactly one empty -> total disagreement (overlap metrics = 0, HD95/ASSD = NaN).

Reuse
-----
Surface metrics (HD95 / ASSD / NSD) reuse the spacing-aware SimpleITK helpers in
`nnunetv2.compute_metrics`. That module imports MONAI at top level, which may be
absent in lightweight/local environments, so the import is guarded and falls back
to functionally identical inline implementations (the helpers themselves are pure
SimpleITK + numpy and do not use MONAI).
"""

from __future__ import annotations

import numpy as np

# ── Optional dependencies, resolved once at import ──────────────────────────
try:
    import cc3d  # connected-components-3d
    _HAVE_CC3D = True
except Exception:  # pragma: no cover - exercised only when cc3d missing
    _HAVE_CC3D = False

try:
    from scipy import ndimage as _ndi
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    from skimage.morphology import skeletonize as _skeletonize
    _HAVE_SKELETON = True
except Exception:  # pragma: no cover
    try:
        from skimage.morphology import skeletonize_3d as _skeletonize
        _HAVE_SKELETON = True
    except Exception:
        _skeletonize = None
        _HAVE_SKELETON = False

_SKELETON_WARNED = False


# ── Surface helpers: reuse compute_metrics.py, else inline equivalents ───────
try:
    from nnunetv2.compute_metrics import (
        compute_dice as _compute_dice,
        compute_hd95 as _compute_hd95,
        compute_average_surface_distance as _compute_assd,
        compute_nsd as _compute_nsd,
    )
except Exception:  # pragma: no cover - fallback when MONAI/compute_metrics absent
    try:
        import SimpleITK as sitk
        _HAVE_SITK = True
    except Exception:
        _HAVE_SITK = False

    def _compute_dice(pred, ref):
        inter = np.logical_and(pred, ref).sum()
        denom = pred.sum() + ref.sum()
        return 1.0 if denom == 0 else float(2.0 * inter / denom)

    def _surface_distance_arrays_mm(pred, ref, spacing_xyz):
        if pred.sum() == 0 and ref.sum() == 0:
            return np.array([0.0]), np.array([0.0])
        if pred.sum() == 0 or ref.sum() == 0:
            return np.array([np.inf]), np.array([np.inf])
        if not _HAVE_SITK:
            raise RuntimeError("SimpleITK required for surface metrics")
        ps = sitk.GetImageFromArray(pred.astype(np.uint8)); ps.SetSpacing(spacing_xyz)
        rs = sitk.GetImageFromArray(ref.astype(np.uint8));  rs.SetSpacing(spacing_xyz)
        contour = sitk.LabelContourImageFilter(); contour.SetFullyConnected(True)
        pred_surf = sitk.GetArrayFromImage(contour.Execute(ps)) > 0
        ref_surf = sitk.GetArrayFromImage(contour.Execute(rs)) > 0
        if not pred_surf.any() and pred.any():
            pred_surf = pred.astype(bool)
        if not ref_surf.any() and ref.any():
            ref_surf = ref.astype(bool)
        dt_ref = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
            rs, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
        dt_pred = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
            ps, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
        d_pr = np.abs(dt_ref[pred_surf]); d_rp = np.abs(dt_pred[ref_surf])
        if d_pr.size == 0 or d_rp.size == 0:
            return np.array([np.inf]), np.array([np.inf])
        return d_pr, d_rp

    def _compute_hd95(pred, ref, spacing_xyz):
        d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
        all_d = np.concatenate([d1, d2])
        return np.nan if np.isinf(all_d).all() else float(np.percentile(all_d, 95))

    def _compute_assd(pred, ref, spacing_xyz):
        d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
        if np.isinf(np.concatenate([d1, d2])).all():
            return np.nan
        return float(0.5 * (d1.mean() + d2.mean()))

    def _compute_nsd(pred, ref, spacing_xyz, tau_mm):
        if pred.sum() == 0 and ref.sum() == 0:
            return 1.0
        if pred.sum() == 0 or ref.sum() == 0:
            return 0.0
        d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
        if np.isinf(d1).all() and np.isinf(d2).all():
            return 0.0
        return float((np.sum(d1 <= tau_mm) + np.sum(d2 <= tau_mm)) / (d1.size + d2.size))


# ── Connected components ─────────────────────────────────────────────────────
def label_components(binary: np.ndarray, connectivity: int = 26):
    """Label connected components of a binary 3D volume (26-connectivity).

    Returns (labeled_array, num_components). Uses cc3d if available, otherwise
    scipy.ndimage with a full 3x3x3 structuring element.
    """
    b = np.ascontiguousarray(binary.astype(np.uint8))
    if b.sum() == 0:
        return np.zeros_like(b, dtype=np.int32), 0
    if _HAVE_CC3D:
        lab = cc3d.connected_components(b, connectivity=connectivity)
        return lab.astype(np.int32), int(lab.max())
    if _HAVE_SCIPY:
        structure = _ndi.generate_binary_structure(3, 3)  # 26-connectivity
        lab, n = _ndi.label(b, structure=structure)
        return lab.astype(np.int32), int(n)
    raise RuntimeError("Either cc3d or scipy is required for connected components")


def _component_sizes(labeled: np.ndarray, num: int) -> np.ndarray:
    """Voxel count per component label 1..num (excludes background 0)."""
    if num == 0:
        return np.array([], dtype=np.int64)
    counts = np.bincount(labeled.reshape(-1), minlength=num + 1)
    return counts[1:].astype(np.int64)


def component_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing_xyz,
    small_island_voxels: int = 50,
) -> dict:
    """Connected-component / fragmentation metrics for one binary class."""
    voxel_vol = float(np.prod(spacing_xyz))
    pred_lab, pred_n = label_components(pred_bin)
    gt_lab, gt_n = label_components(gt_bin)
    pred_sizes = _component_sizes(pred_lab, pred_n)
    pred_total = int(pred_bin.sum())

    if pred_n == 0:
        largest_vox = 0
        largest_ratio = 0.0
        n_small = 0
        frag = 0.0
    else:
        largest_vox = int(pred_sizes.max())
        largest_ratio = float(largest_vox / max(pred_total, 1))
        n_small = int(np.sum(pred_sizes < small_island_voxels))
        frag = float(1.0 - largest_ratio)

    # false disconnected: pred components with no voxel overlap to GT largest comp
    if gt_n == 0:
        false_disc = int(pred_n)  # any predicted component is spurious
    elif pred_n == 0:
        false_disc = 0
    else:
        gt_sizes = _component_sizes(gt_lab, gt_n)
        gt_largest_label = int(np.argmax(gt_sizes)) + 1
        gt_largest_mask = gt_lab == gt_largest_label
        false_disc = 0
        for c in range(1, pred_n + 1):
            comp = pred_lab == c
            if not np.logical_and(comp, gt_largest_mask).any():
                false_disc += 1

    return {
        "num_components": int(pred_n),
        "gt_num_components": int(gt_n),
        "largest_component_voxels": largest_vox,
        "largest_component_volume_mm3": float(largest_vox * voxel_vol),
        "largest_component_ratio": largest_ratio,
        "num_small_islands": n_small,
        "fragmentation_score": frag,
        "false_disconnected_components": int(false_disc),
    }


# ── Surface metrics (thin wrappers, empty-safe via reused helpers) ───────────
def surface_metrics(pred_bin, gt_bin, spacing_xyz, nsd_tau_mm: float = 1.0) -> dict:
    return {
        "hd95": float(_compute_hd95(pred_bin, gt_bin, spacing_xyz)),
        "assd": float(_compute_assd(pred_bin, gt_bin, spacing_xyz)),
        "nsd": float(_compute_nsd(pred_bin, gt_bin, spacing_xyz, nsd_tau_mm)),
    }


def dice(pred_bin, gt_bin) -> float:
    return float(_compute_dice(pred_bin, gt_bin))


# ── Connectivity / centerline metrics (tubular classes) ─────────────────────
def _skeleton(binary: np.ndarray):
    """3D skeleton (boolean) or None if skeletonization is unavailable.

    Fallback: skimage's 3D (Lee) thinning can collapse even-width prisms to an
    empty result. If a non-empty mask skeletonizes to nothing, return the mask
    itself as its own skeleton so connectivity metrics degrade gracefully instead
    of reporting a spurious zero.
    """
    global _SKELETON_WARNED
    if not _HAVE_SKELETON:
        if not _SKELETON_WARNED:
            print("[topology_metrics] skimage skeletonize unavailable — "
                  "connectivity metrics will be reported as None")
            _SKELETON_WARNED = True
        return None
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=bool)
    skel = np.asarray(_skeletonize(binary.astype(bool))).astype(bool)
    if not skel.any():
        return binary.astype(bool)
    return skel


def connectivity_metrics(pred_bin, gt_bin) -> dict:
    """clDice + centerline/skeleton/component recall for one tubular class.

    Returns None-valued keys if skeletonization is unavailable (graceful fallback).
    """
    none_result = {
        "cldice": None,
        "centerline_recall": None,
        "skeleton_recall": None,
        "vessel_component_recall": None,
    }
    pred_empty = pred_bin.sum() == 0
    gt_empty = gt_bin.sum() == 0

    skel_pred = _skeleton(pred_bin)
    skel_gt = _skeleton(gt_bin)
    if skel_pred is None or skel_gt is None:
        return dict(none_result)

    # clDice (Shit et al. 2021): Tprec on pred skeleton vs gt mask, Tsens on gt skel vs pred mask
    if pred_empty and gt_empty:
        cldice = 1.0
        cl_recall = 1.0
        skel_recall = 1.0
    elif pred_empty or gt_empty:
        cldice = 0.0
        cl_recall = 0.0
        skel_recall = 0.0
    else:
        n_sp = int(skel_pred.sum())
        n_sg = int(skel_gt.sum())
        tprec = float(np.logical_and(skel_pred, gt_bin).sum() / n_sp) if n_sp > 0 else 0.0
        tsens = float(np.logical_and(skel_gt, pred_bin).sum() / n_sg) if n_sg > 0 else 0.0
        cldice = 0.0 if (tprec + tsens) == 0 else float(2.0 * tprec * tsens / (tprec + tsens))
        cl_recall = tsens
        skel_recall = float(np.logical_and(skel_gt, skel_pred).sum() / n_sg) if n_sg > 0 else 0.0

    # vessel component recall: fraction of GT components that overlap any pred voxel
    gt_lab, gt_n = label_components(gt_bin)
    if gt_n == 0:
        vcr = 1.0 if pred_empty else 0.0
    elif pred_empty:
        vcr = 0.0
    else:
        hit = 0
        for c in range(1, gt_n + 1):
            if np.logical_and(gt_lab == c, pred_bin).any():
                hit += 1
        vcr = float(hit / gt_n)

    return {
        "cldice": cldice,
        "centerline_recall": cl_recall,
        "skeleton_recall": skel_recall,
        "vessel_component_recall": vcr,
    }


# ── Topology audit (multi-class, case level) ────────────────────────────────
def _dilate1(binary: np.ndarray) -> np.ndarray:
    """One-voxel 26-connected dilation (for adjacency tests)."""
    if _HAVE_SCIPY:
        structure = _ndi.generate_binary_structure(3, 3)
        return _ndi.binary_dilation(binary, structure=structure)
    # numpy fallback: 3x3x3 max via rolling (slower, rarely used)
    out = binary.copy()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= np.roll(np.roll(np.roll(binary, dz, 0), dy, 1), dx, 2)
    return out


def adjacency_matrix(pred: np.ndarray, label_ids) -> dict:
    """For each ordered pair (a, b) of foreground labels, whether dilated-a touches b.

    Returns {a_id: {b_id: bool}} for a != b. Symmetric in practice but computed
    directionally (dilate a, intersect b) for robustness.
    """
    masks = {lab: (pred == lab) for lab in label_ids}
    dil = {lab: _dilate1(masks[lab]) if masks[lab].any() else masks[lab] for lab in label_ids}
    out = {}
    for a in label_ids:
        row = {}
        for b in label_ids:
            if a == b:
                continue
            if not masks[a].any() or not masks[b].any():
                row[b] = False
            else:
                row[b] = bool(np.logical_and(dil[a], masks[b]).any())
        out[a] = row
    return out


def ao_pa_switch_indicator(
    pred: np.ndarray,
    gt: np.ndarray,
    ao_label: int,
    pa_label: int,
    swap_fraction_threshold: float = 0.05,
) -> dict:
    """Detect Ao/PA label switching along the great-vessel region.

    swap_voxels = |pred=Ao & gt=PA| + |pred=PA & gt=Ao|
    swap_fraction = swap_voxels / |gt in {Ao, PA}|
    switch_flag is True when the swap fraction exceeds threshold AND the predicted
    Ao and PA largest components are mutually adjacent (a continuous vessel whose
    labelling flips), which is the clinically meaningful failure mode.
    """
    pred_ao = pred == ao_label
    pred_pa = pred == pa_label
    gt_ao = gt == ao_label
    gt_pa = gt == pa_label

    swap = int(np.logical_and(pred_ao, gt_pa).sum() + np.logical_and(pred_pa, gt_ao).sum())
    gt_vol = int(np.logical_or(gt_ao, gt_pa).sum())
    swap_fraction = float(swap / gt_vol) if gt_vol > 0 else 0.0

    adjacent = False
    if pred_ao.any() and pred_pa.any():
        adjacent = bool(np.logical_and(_dilate1(pred_ao), pred_pa).any())

    return {
        "ao_pa_swap_voxels": swap,
        "ao_pa_swap_fraction": swap_fraction,
        "ao_pa_predicted_adjacent": adjacent,
        "ao_pa_switch_flag": bool(swap_fraction > swap_fraction_threshold and adjacent),
    }


# ── Per-case orchestration ──────────────────────────────────────────────────
def evaluate_case(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing_xyz,
    label_map: dict,
    tubular_labels=None,
    ao_label=None,
    pa_label=None,
    small_island_voxels: int = 50,
    nsd_tau_mm: float = 1.0,
) -> dict:
    """Compute all metrics for a single case.

    Parameters
    ----------
    pred, gt : 3D int label maps (z, y, x), same shape.
    spacing_xyz : (sx, sy, sz) mm.
    label_map : {label_id: name} for FOREGROUND classes (exclude background).
    tubular_labels : iterable of label ids to get connectivity/clDice metrics.
    ao_label, pa_label : ids for the Ao/PA switch indicator (or None to skip).

    Returns
    -------
    dict with:
      'flat'  : flat {metric_name: value} suitable for a CSV row
      'audit' : nested per-case topology audit (adjacency matrix etc.) for JSON
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
    if tubular_labels is None:
        tubular_labels = []
    tubular_labels = set(int(t) for t in tubular_labels)

    flat = {}
    per_class_comp_counts = {}

    for lab, name in label_map.items():
        pred_c = pred == lab
        gt_c = gt == lab

        flat[f"dice_{name}"] = dice(pred_c, gt_c)

        surf = surface_metrics(pred_c, gt_c, spacing_xyz, nsd_tau_mm)
        flat[f"hd95_{name}"] = surf["hd95"]
        flat[f"assd_{name}"] = surf["assd"]
        flat[f"nsd_{name}"] = surf["nsd"]

        comp = component_metrics(pred_c, gt_c, spacing_xyz, small_island_voxels)
        flat[f"ncomp_{name}"] = comp["num_components"]
        flat[f"ncomp_gt_{name}"] = comp["gt_num_components"]
        flat[f"largest_ratio_{name}"] = comp["largest_component_ratio"]
        flat[f"frag_{name}"] = comp["fragmentation_score"]
        flat[f"small_islands_{name}"] = comp["num_small_islands"]
        flat[f"false_disc_{name}"] = comp["false_disconnected_components"]
        per_class_comp_counts[name] = comp["num_components"]

        if lab in tubular_labels:
            conn = connectivity_metrics(pred_c, gt_c)
            flat[f"cldice_{name}"] = conn["cldice"]
            flat[f"clrecall_{name}"] = conn["centerline_recall"]
            flat[f"skelrecall_{name}"] = conn["skeleton_recall"]
            flat[f"vcomprecall_{name}"] = conn["vessel_component_recall"]

    # case-level topology
    total_islands = int(sum(flat.get(f"small_islands_{n}", 0) for n in label_map.values()))
    flat["total_isolated_islands"] = total_islands

    if ao_label is not None and pa_label is not None:
        sw = ao_pa_switch_indicator(pred, gt, int(ao_label), int(pa_label))
        flat.update(sw)

    audit = {
        "per_class_component_counts": per_class_comp_counts,
        "adjacency_matrix": {
            label_map[a]: {label_map[b]: v for b, v in row.items()}
            for a, row in adjacency_matrix(pred, list(label_map.keys())).items()
        },
    }

    return {"flat": flat, "audit": audit}
