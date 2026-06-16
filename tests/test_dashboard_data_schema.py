"""
test_dashboard_data_schema.py — dashboard data generation works with no metrics
and emits the expected schema. pytest-compatible AND standalone:
    python tests/test_dashboard_data_schema.py
"""

import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "dashboard_data")


def _regen():
    # collect_metrics fills nulls when there are no result folders; update embeds JSON.
    subprocess.check_call([sys.executable, os.path.join(_ROOT, "experiments", "collect_metrics.py")],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, os.path.join(_ROOT, "experiments", "update_dashboard_data.py")],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_files_exist_and_parse():
    _regen()
    for fn in ("ablation_status.json", "current_results.json", "todo_items.json"):
        p = os.path.join(_DATA, fn)
        assert os.path.isfile(p), f"missing {fn}"
        json.loads(open(p).read())  # parses


def test_status_schema():
    _regen()
    data = json.loads(open(os.path.join(_DATA, "ablation_status.json")).read())
    assert "experiments" in data and len(data["experiments"]) >= 37
    e = data["experiments"][0]
    for k in ("id", "name", "group", "status", "implemented", "depth"):
        assert k in e


def test_results_nulls_when_no_metrics():
    _regen()
    data = json.loads(open(os.path.join(_DATA, "current_results.json")).read())
    assert "results" in data
    # with no result folders, whole_heart_dice should be null everywhere
    assert all(r.get("whole_heart_dice") is None for r in data["results"])


def test_checklist_seeded():
    _regen()
    data = json.loads(open(os.path.join(_DATA, "todo_items.json")).read())
    assert len(data["items"]) == 17
    assert all({"id", "text", "done"} <= set(it) for it in data["items"])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
