"""
test_topology_metrics.py — standalone synthetic tests for topology_metrics.py.

No pytest dependency (matches tools/test_chd_conditioning.py). Run with:
    python tools/test_topology_metrics.py

Covers: connected-component counting, empty pred/target, perfect prediction,
fragmented vessel, single small island, Ao/PA label-swap detection, clDice
non-NaN + fallback, and anisotropic-spacing scaling of surface distances.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "nnUNet"))

from nnunetv2.evaluation import topology_metrics as tm  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def test_component_count():
    print("test_component_count")
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[2:5, 2:5, 2:5] = 1          # blob A
    vol[12:15, 12:15, 12:15] = 1    # blob B (disconnected)
    lab, n = tm.label_components(vol == 1)
    check(n == 2, f"two disconnected blobs -> 2 components (got {n})")

    m = tm.component_metrics(vol == 1, vol == 1, (1.0, 1.0, 1.0), small_island_voxels=5)
    check(m["num_components"] == 2, "component_metrics num_components == 2")
    check(m["largest_component_ratio"] == 0.5, f"largest ratio 0.5 (got {m['largest_component_ratio']})")


def test_empty_and_perfect():
    print("test_empty_and_perfect")
    gt = np.zeros((16, 16, 16), dtype=np.int32)
    gt[4:10, 4:10, 4:10] = 1
    empty = np.zeros_like(gt)

    # perfect
    check(abs(tm.dice(gt == 1, gt == 1) - 1.0) < 1e-9, "perfect Dice == 1.0")
    sm = tm.surface_metrics(gt == 1, gt == 1, (1.0, 1.0, 1.0))
    check(sm["hd95"] == 0.0, f"perfect HD95 == 0 (got {sm['hd95']})")
    check(abs(sm["nsd"] - 1.0) < 1e-9, "perfect NSD == 1.0")

    # both empty -> agreement conventions
    check(abs(tm.dice(empty == 1, empty == 1) - 1.0) < 1e-9, "both-empty Dice == 1.0")
    conn = tm.connectivity_metrics(empty == 1, empty == 1)
    check(conn["cldice"] is None or conn["cldice"] == 1.0, "both-empty clDice == 1.0 (or None fallback)")

    # one empty -> disagreement
    check(tm.dice(empty == 1, gt == 1) == 0.0, "pred-empty Dice == 0.0")
    sm2 = tm.surface_metrics(empty == 1, gt == 1, (1.0, 1.0, 1.0))
    check(np.isnan(sm2["hd95"]), "pred-empty HD95 is NaN")
    check(sm2["nsd"] == 0.0, "pred-empty NSD == 0.0")


def test_fragmented_vessel():
    print("test_fragmented_vessel")
    # GT: a single continuous tube along z
    gt = np.zeros((30, 12, 12), dtype=np.int32)
    gt[3:27, 5:7, 5:7] = 1
    # Pred: same tube but with a gap in the middle -> 2 components
    pred = gt.copy()
    pred[14:17, :, :] = 0
    m = tm.component_metrics(pred == 1, gt == 1, (1.0, 1.0, 1.0))
    check(m["num_components"] == 2, f"fragmented tube -> 2 pred components (got {m['num_components']})")
    check(m["gt_num_components"] == 1, "GT tube is 1 component")
    conn = tm.connectivity_metrics(pred == 1, gt == 1)
    if conn["cldice"] is not None:
        check(0.0 <= conn["cldice"] <= 1.0, f"clDice in [0,1] (got {conn['cldice']})")
        check(conn["centerline_recall"] < 1.0, "centerline recall < 1 for gapped tube")
    else:
        check(True, "clDice unavailable (skeleton fallback) — accepted")


def test_small_island():
    print("test_small_island")
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[2:12, 2:12, 2:12] = 1       # big blob
    vol[18, 18, 18] = 1             # 1-voxel island
    m = tm.component_metrics(vol == 1, vol == 1, (1.0, 1.0, 1.0), small_island_voxels=50)
    check(m["num_components"] == 2, "big blob + island -> 2 components")
    check(m["num_small_islands"] == 1, f"one small island (got {m['num_small_islands']})")


def test_ao_pa_switch():
    print("test_ao_pa_switch")
    AO, PA = 6, 7
    shape = (20, 20, 20)
    gt = np.zeros(shape, dtype=np.int32)
    pred = np.zeros(shape, dtype=np.int32)
    # GT: Ao on left half, PA on right half, adjacent
    gt[5:15, 5:15, 4:10] = AO
    gt[5:15, 5:15, 10:16] = PA
    # Pred: labels swapped along the same continuous region
    pred[5:15, 5:15, 4:10] = PA
    pred[5:15, 5:15, 10:16] = AO
    sw = tm.ao_pa_switch_indicator(pred, gt, AO, PA, swap_fraction_threshold=0.05)
    check(sw["ao_pa_swap_fraction"] > 0.5, f"large swap fraction (got {sw['ao_pa_swap_fraction']:.2f})")
    check(sw["ao_pa_predicted_adjacent"], "predicted Ao/PA adjacent")
    check(sw["ao_pa_switch_flag"], "switch flag fires on swapped labels")

    # no swap: identical pred == gt
    sw2 = tm.ao_pa_switch_indicator(gt, gt, AO, PA)
    check(sw2["ao_pa_swap_fraction"] == 0.0, "no swap when pred == gt")
    check(not sw2["ao_pa_switch_flag"], "switch flag off when pred == gt")


def test_clDice_no_nan():
    print("test_clDice_no_nan")
    gt = np.zeros((24, 10, 10), dtype=np.int32)
    gt[3:21, 4:6, 4:6] = 1
    pred = gt.copy()
    conn = tm.connectivity_metrics(pred == 1, gt == 1)
    if conn["cldice"] is None:
        check(True, "skeletonization unavailable — None fallback accepted")
    else:
        for k, v in conn.items():
            check(v is None or not (isinstance(v, float) and np.isnan(v)),
                  f"{k} is not NaN (got {v})")
        check(conn["cldice"] > 0.9, f"clDice high for near-perfect tube (got {conn['cldice']:.3f})")


def test_anisotropic_spacing():
    print("test_anisotropic_spacing")
    # A shell vs an eroded shell; HD95 should scale with z-spacing
    gt = np.zeros((20, 20, 20), dtype=np.int32)
    gt[5:15, 5:15, 5:15] = 1
    pred = np.zeros_like(gt)
    pred[6:14, 5:15, 5:15] = 1   # shifted by 1 voxel along z on both faces
    hd_iso = tm.surface_metrics(pred == 1, gt == 1, (1.0, 1.0, 1.0))["hd95"]
    hd_aniso = tm.surface_metrics(pred == 1, gt == 1, (1.0, 1.0, 3.0))["hd95"]
    check(not np.isnan(hd_iso) and not np.isnan(hd_aniso), "HD95 finite for both spacings")
    check(hd_aniso > hd_iso, f"HD95 grows with z-spacing ({hd_aniso:.2f} > {hd_iso:.2f})")


def test_evaluate_case_flat():
    print("test_evaluate_case_flat")
    label_map = {5: "Myo", 6: "Ao", 7: "PA"}
    shape = (20, 20, 20)
    gt = np.zeros(shape, dtype=np.int32)
    gt[5:15, 5:15, 4:9] = 6
    gt[5:15, 5:15, 9:15] = 7
    gt[2:4, 2:4, 2:4] = 5
    pred = gt.copy()
    res = tm.evaluate_case(pred, gt, (1.0, 1.0, 1.0), label_map,
                           tubular_labels=[6, 7], ao_label=6, pa_label=7)
    flat = res["flat"]
    check("dice_Ao" in flat and abs(flat["dice_Ao"] - 1.0) < 1e-9, "evaluate_case dice_Ao == 1.0")
    check("cldice_Ao" in flat, "evaluate_case includes cldice_Ao for tubular class")
    check("ao_pa_switch_flag" in flat and not flat["ao_pa_switch_flag"],
          "evaluate_case switch flag off for perfect pred")
    check("adjacency_matrix" in res["audit"], "audit includes adjacency_matrix")
    check(res["audit"]["adjacency_matrix"]["Ao"]["PA"] is True, "Ao adjacent to PA in audit")


def main():
    tests = [
        test_component_count,
        test_empty_and_perfect,
        test_fragmented_vessel,
        test_small_island,
        test_ao_pa_switch,
        test_clDice_no_nan,
        test_anisotropic_spacing,
        test_evaluate_case_flat,
    ]
    print(f"cc3d={tm._HAVE_CC3D}  scipy={tm._HAVE_SCIPY}  skeleton={tm._HAVE_SKELETON}\n")
    for t in tests:
        t()
        print()
    if _FAILURES:
        print(f"FAILED ({len(_FAILURES)}):")
        for m in _FAILURES:
            print(f"  - {m}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
