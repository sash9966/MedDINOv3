"""
test_ablation_registry.py — registry integrity. pytest-compatible AND runnable
standalone:  python tests/test_ablation_registry.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))

from _common import load_registry, list_experiments  # noqa: E402

REGISTRY = load_registry()
EXPS = list_experiments(REGISTRY)

# trainer classes that actually exist in the repo (kept in sync with dinov3Trainer.py)
EXISTING_TRAINERS = {
    "nnUNetTrainer",
    "meddinov3_3d_centering_d16_primus_multiscale_Trainer",
    "meddinov3_3d_centering_d8_primus_multiscale_Trainer",
    "meddinov3_3d_centering_d4_primus_multiscale_Trainer",
    "meddinov3_3d_ashwin_d4_primus_multiscale_Trainer",
    "meddinov3_3d_chd_film_d8_Trainer",
}


def test_loads_and_has_experiments():
    assert len(EXPS) >= 37


def test_unique_ids():
    ids = [e["id"] for e in EXPS]
    assert len(ids) == len(set(ids))


def test_required_fields():
    required = ["id", "name", "group", "enabled", "implemented", "priority",
                "dataset_id", "dataset_version", "folds", "meddinov3_depth",
                "status", "expected_rationale"]
    for e in EXPS:
        for f in required:
            assert f in e, f"{e.get('id')} missing {f}"


def test_depths_present():
    depths = {e.get("meddinov3_depth") for e in EXPS}
    assert {"d16", "d8", "d4"} <= depths


def test_implemented_map_to_existing_trainers():
    for e in EXPS:
        if e.get("implemented"):
            assert e.get("trainer") in EXISTING_TRAINERS, \
                f"{e['id']} implemented but trainer {e.get('trainer')!r} not found in repo"


def test_planned_have_requires():
    for e in EXPS:
        if not e.get("implemented"):
            assert e.get("requires"), f"{e['id']} planned but no 'requires'"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
