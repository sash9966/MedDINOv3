"""
update_dashboard_data.py — refresh dashboard_data/*.json from the registry +
collected results, and embed them into dashboard.html (between markers) so the
board renders offline with no server.

    python experiments/update_dashboard_data.py
    python experiments/update_dashboard_data.py --reset-checklist

Preserves manual checklist tick state and manual `notes` unless --reset-checklist
is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DASHBOARD_DATA_DIR, REPO_ROOT, RESULTS_DIR, list_experiments, load_registry,
)

DASHBOARD_HTML = REPO_ROOT / "dashboard.html"
MARK_START = "<!--ABLATION_DATA_START-->"
MARK_END = "<!--ABLATION_DATA_END-->"

# Seed checklist (mirrors docs/ablation_plan.md and the dashboard).
CHECKLIST_SEED = [
    "Confirm fixed split file for current 100-case dataset",
    "Confirm nnU-Net baseline metrics on the same split",
    "Confirm current MedDINO d16/d8/d4 metrics on the same split",
    "Run adapter ablation: no adapter vs 1x1x1 adapter",
    "Run inflation ablation: inflated vs random 3D initialization",
    "Run depth ablation: d16 vs d8 vs d4",
    "Run freeze ablation: frozen vs partial vs full fine-tune",
    "Run vessel-aware sampling ablation",
    "Run class-balanced / focal / Tversky loss ablation",
    "Run clDice / soft-skeleton topology loss for pulmonary/great arteries",
    "Run post-processing ablation for small components and vessel continuity",
    "Run diagnosis multi-task head ablation",
    "Run oracle diagnosis-FiLM ablation",
    "Run predicted-diagnosis conditioning ablation if clinically needed",
    "Compare all completed models to nnU-Net with paired statistics",
    "Generate supervisor summary table and plots",
    "Re-run best ablations on larger expert-annotated cohort when available",
]


def load_results_index():
    """Map experiment id -> result row (if current_results.json has metrics)."""
    p = DASHBOARD_DATA_DIR / "current_results.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    idx = {}
    for r in data.get("results", []):
        if r.get("whole_heart_dice") is not None:
            idx[r["id"]] = r
    return idx


def build_status(registry):
    results_idx = load_results_index()
    out = []
    for e in list_experiments(registry):
        losses = e.get("losses", {})
        active_losses = [k for k in ("class_balanced", "focal", "tversky", "cldice", "boundary_loss")
                         if losses.get(k)]
        cond = e.get("conditioning", {})
        res = results_idx.get(e["id"])
        status = "completed" if res else e.get("status")
        if res:
            summary = f"WH {res['whole_heart_dice']:.3f} · cls {res['mean_class_dice']:.3f}"
            d = res.get("delta_meancls_vs_nnunet")
            if d is not None:
                summary += f" (Δcls {d:+.3f} vs nnU-Net)"
        else:
            summary = e.get("current_result")
        out.append({
            "id": e["id"], "name": e["name"], "group": e.get("group"),
            "priority": e.get("priority"), "status": status,
            "implemented": e.get("implemented"), "enabled": e.get("enabled"),
            "depth": e.get("meddinov3_depth"), "trainer": e.get("trainer"),
            "initialization": e.get("initialization"),
            "conditioning": cond.get("type") if cond.get("enabled") else "none",
            "losses": "+".join(["dice_ce"] + active_losses),
            "postprocessing": e.get("postprocessing", {}).get("mode", "none"),
            "requires": e.get("requires", []),
            "purpose": e.get("expected_rationale", ""),
            "result_summary": summary,
        })
    return out


def load_existing_checklist():
    p = DASHBOARD_DATA_DIR / "todo_items.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return {item["text"]: item.get("done", False) for item in data.get("items", [])}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-checklist", action="store_true")
    args = ap.parse_args()

    registry = load_registry()
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ablation_status.json
    status = build_status(registry)
    (DASHBOARD_DATA_DIR / "ablation_status.json").write_text(json.dumps({"experiments": status}, indent=2))

    # current_results.json — keep whatever collect_metrics wrote, else empty scaffold
    cur_path = DASHBOARD_DATA_DIR / "current_results.json"
    if cur_path.is_file():
        current_results = json.loads(cur_path.read_text())
    else:
        current_results = {"results": [], "warnings": ["run collect_metrics.py to populate"]}
        cur_path.write_text(json.dumps(current_results, indent=2))

    # todo_items.json — preserve ticks unless reset
    prev = {} if args.reset_checklist else load_existing_checklist()
    items = [{"id": f"chk{i:02d}", "text": t, "done": prev.get(t, False)}
             for i, t in enumerate(CHECKLIST_SEED)]
    (DASHBOARD_DATA_DIR / "todo_items.json").write_text(json.dumps({"items": items}, indent=2))

    # embed into dashboard.html between markers (if present)
    combined = {"ablation_status": {"experiments": status},
                "current_results": current_results,
                "todo_items": {"items": items}}
    blob = json.dumps(combined)
    if DASHBOARD_HTML.is_file():
        html = DASHBOARD_HTML.read_text()
        if MARK_START in html and MARK_END in html:
            pre = html.split(MARK_START)[0]
            post = html.split(MARK_END)[1]
            injected = (f'{pre}{MARK_START}\n'
                        f'<script id="ablation-data" type="application/json">{blob}</script>\n'
                        f'{MARK_END}{post}')
            DASHBOARD_HTML.write_text(injected)
            print("embedded ablation data into dashboard.html")
        else:
            print("[warn] dashboard.html has no ABLATION_DATA markers yet — "
                  "JSON written to dashboard_data/ only", file=sys.stderr)

    print(f"wrote dashboard_data/ablation_status.json ({len(status)} experiments), "
          f"todo_items.json ({len(items)} items), current_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
