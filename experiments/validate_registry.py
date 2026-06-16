"""
validate_registry.py — schema / integrity checks for ablation_registry.yaml.

Exits nonzero on any failure. Run:  python experiments/validate_registry.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_registry, list_experiments  # noqa: E402

REQUIRED_FIELDS = [
    "id", "name", "group", "enabled", "implemented", "priority", "dataset_id",
    "dataset_version", "folds", "seed", "model_family", "config",
    "meddinov3_depth", "use_3d_inflation", "initialization", "input_adapter",
    "encoder_finetuning", "conditioning", "losses", "sampling", "augmentation",
    "postprocessing", "expected_rationale", "status", "notes",
]
VALID_STATUS = {"planned", "running", "completed", "failed"}
VALID_PRIORITY = {"high", "medium", "low"}


def main() -> int:
    reg = load_registry()
    exps = list_experiments(reg)
    errors: list[str] = []

    # meta
    if "meta" not in reg or "dataset_versions" not in reg["meta"]:
        errors.append("meta.dataset_versions missing")

    ids = [e.get("id") for e in exps]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        errors.append(f"duplicate experiment ids: {sorted(dupes)}")

    depths = set()
    for e in exps:
        eid = e.get("id", "<no-id>")
        for f in REQUIRED_FIELDS:
            if f not in e:
                errors.append(f"{eid}: missing field '{f}'")
        if e.get("status") not in VALID_STATUS:
            errors.append(f"{eid}: bad status {e.get('status')!r}")
        if e.get("priority") not in VALID_PRIORITY:
            errors.append(f"{eid}: bad priority {e.get('priority')!r}")
        if e.get("implemented") and not e.get("trainer"):
            errors.append(f"{eid}: implemented=true but no trainer mapped")
        if not e.get("implemented") and not e.get("requires"):
            errors.append(f"{eid}: implemented=false but 'requires' is empty")
        depths.add(e.get("meddinov3_depth"))

    for d in ("d16", "d8", "d4"):
        if d not in depths:
            errors.append(f"depth variant '{d}' not present in registry")

    if errors:
        print(f"REGISTRY INVALID — {len(errors)} problem(s):")
        for m in errors:
            print(f"  - {m}")
        return 1

    impl = sum(1 for e in exps if e.get("implemented"))
    print(f"REGISTRY OK — {len(exps)} experiments, {impl} implemented/runnable, "
          f"{len(exps) - impl} planned. d16/d8/d4 all present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
