"""
run_ablation.py — list / inspect / (guardedly) launch CHD MedDINO ablations.

Composes commands from experiments/ablation_registry.yaml — never hard-codes a
training command. Examples:

    python experiments/run_ablation.py --list
    python experiments/run_ablation.py --list --priority high --enabled-only
    python experiments/run_ablation.py --id meddinov3_current_d4 --dry-run
    python experiments/run_ablation.py --id meddinov3_current_d4 --print-command
    python experiments/run_ablation.py --id meddinov3_current_d4 --run --yes

--run executes the composed nnUNetv2_train locally and ONLY when --yes is given;
it refuses experiments whose hooks are not implemented, and never submits SLURM
(use generate_slurm.py + sbatch for the cluster).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    compose_train_command, env_export_lines, get_experiment, list_experiments,
    load_registry, resolve_env, write_manifest,
)

_STATUS_TAG = {"planned": "[plan]", "running": "[run ]", "completed": "[done]", "failed": "[fail]"}


def _filter(exps, args):
    out = []
    for e in exps:
        if args.priority and e.get("priority") != args.priority:
            continue
        if args.status and e.get("status") != args.status:
            continue
        if args.enabled_only and not e.get("enabled"):
            continue
        out.append(e)
    return out


def cmd_list(exps):
    by_group: dict[str, list[dict]] = {}
    for e in exps:
        by_group.setdefault(e.get("group", "?"), []).append(e)
    for group in sorted(by_group):
        print(f"\n=== {group} ===")
        for e in by_group[group]:
            tag = _STATUS_TAG.get(e.get("status"), "[?]")
            impl = "RUN " if e.get("implemented") else "plan"
            depth = e.get("meddinov3_depth") or "-"
            print(f"  {tag} {impl} {e['id']:<42} depth={depth:<4} prio={e.get('priority')}")
    impl = sum(1 for e in exps if e.get("implemented"))
    print(f"\n{len(exps)} shown · {impl} runnable now · {len(exps) - impl} planned")


def _show(exp, registry):
    fold = exp.get("folds", [0])[0]
    env = resolve_env(exp)
    if exp.get("implemented"):
        cmd = compose_train_command(exp, registry, fold)
    else:
        cmd = None
    print(f"id          : {exp['id']}")
    print(f"name        : {exp['name']}")
    print(f"group       : {exp.get('group')}   priority: {exp.get('priority')}   status: {exp.get('status')}")
    print(f"implemented : {exp.get('implemented')}   requires: {exp.get('requires') or '-'}")
    print(f"trainer     : {exp.get('trainer') or '(none — hook not built)'}")
    print(f"dataset     : id={exp.get('dataset_id')}  version={exp.get('dataset_version')}  fold={fold}")
    print(f"depth       : {exp.get('meddinov3_depth')}   init: {exp.get('initialization')}")
    print("env         :")
    for line in env_export_lines(env) or ["  (none)"]:
        print(f"  {line}")
    print(f"command     : {cmd or '(unavailable — implement the required hook first)'}")
    print(f"rationale   : {exp.get('expected_rationale')}")
    return env, cmd, fold


def cmd_print_command(exp, registry):
    fold = exp.get("folds", [0])[0]
    env = resolve_env(exp)
    if not exp.get("implemented"):
        print(f"# {exp['id']} is not implemented yet (requires: {exp.get('requires')})", file=sys.stderr)
        return 2
    cmd = compose_train_command(exp, registry, fold)
    for line in env_export_lines(env):
        print(line)
    print(cmd)
    return 0


def cmd_dry_run(exp, registry):
    print("=== DRY RUN (no training launched) ===")
    env, cmd, fold = _show(exp, registry)
    mpath = write_manifest(exp, registry, fold, cmd or "", env, extra_notes="dry-run")
    print(f"manifest    : {mpath}")
    return 0


def cmd_run(exp, registry, yes: bool):
    if not exp.get("implemented"):
        print(f"REFUSED: {exp['id']} needs an unbuilt hook (requires: {exp.get('requires')}). "
              f"Use --dry-run / --print-command, or implement the hook first.", file=sys.stderr)
        return 2
    fold = exp.get("folds", [0])[0]
    env = resolve_env(exp)
    cmd = compose_train_command(exp, registry, fold)
    mpath = write_manifest(exp, registry, fold, cmd, env, extra_notes="local --run")
    print(f"manifest    : {mpath}")
    for line in env_export_lines(env):
        print(line)
    print(cmd)
    if not yes:
        print("\nRefusing to launch without --yes (long job). Re-run with --run --yes to execute "
              "locally, or use generate_slurm.py + sbatch on the cluster.", file=sys.stderr)
        return 1
    run_env = dict(os.environ)
    run_env.update(env)
    print("\n[run_ablation] launching locally ...")
    return subprocess.call(cmd, shell=True, env=run_env)


def main() -> int:
    ap = argparse.ArgumentParser(description="CHD MedDINO ablation runner (dry-run first).")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-command", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm a local --run launch")
    ap.add_argument("--priority", choices=["high", "medium", "low"])
    ap.add_argument("--status", choices=["planned", "running", "completed", "failed"])
    ap.add_argument("--enabled-only", action="store_true")
    args = ap.parse_args()

    registry = load_registry()
    exps = _filter(list_experiments(registry), args)

    if args.list or not (args.id):
        cmd_list(exps)
        return 0

    exp = get_experiment(args.id, registry)
    if args.print_command:
        return cmd_print_command(exp, registry)
    if args.run:
        return cmd_run(exp, registry, args.yes)
    # default for --id is dry-run
    return cmd_dry_run(exp, registry)


if __name__ == "__main__":
    raise SystemExit(main())
