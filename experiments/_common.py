"""
_common.py — shared helpers for the CHD MedDINO ablation harness.

Loads experiments/ablation_registry.yaml and composes training commands /
env-var sets from the structured experiment definitions, mapping each to an
EXISTING nnUNet/MedDINO trainer. No training command is hard-coded; everything is
derived from the registry so the same definitions work for the current 100-case
dataset and a future larger cohort (switch `dataset_version` / `dataset_id`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "experiments" / "ablation_registry.yaml"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
SLURM_OUT_DIR = REPO_ROOT / "experiments" / "slurm_generated"
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard_data"

# Cluster defaults (overridable by environment). Documented, not hard-required:
# anything pointing off-cluster simply falls through to dry-run friendly behaviour.
DEFAULTS = {
    "nnUNet_raw": "/scratch/users/sastocke/nnunet_CHD/nnUNet_raw",
    "nnUNet_preprocessed": "/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed",
    "nnUNet_results": "/scratch/users/sastocke/nnunet_CHD/nnUNet_results",
    "SHARED_CKPT_DIR": "/scratch/users/sastocke/meddinov3_checkpoints",
    "REPO": "/scratch/users/sastocke/MedDINOv3",
}

_DEPTH_TO_DPATCH = {"d16": 16, "d8": 8, "d4": 4}


def env_or_default(key: str) -> str:
    return os.environ.get(key, DEFAULTS.get(key, ""))


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def list_experiments(registry: dict | None = None) -> list[dict]:
    reg = registry or load_registry()
    return reg.get("experiments", [])


def get_experiment(exp_id: str, registry: dict | None = None) -> dict:
    for e in list_experiments(registry):
        if e["id"] == exp_id:
            return e
    raise KeyError(f"experiment id not found: {exp_id}")


def dataset_name(exp: dict, registry: dict) -> str | None:
    dv = registry["meta"]["dataset_versions"].get(exp.get("dataset_version"), {})
    return dv.get("name")


def resolve_env(exp: dict) -> dict:
    """Compose the env-var dict for an experiment (MedDINO only; nnU-Net needs none)."""
    env: dict[str, str] = {}
    if exp.get("model_family") == "meddinov3":
        depth = exp.get("meddinov3_depth")
        if depth in _DEPTH_TO_DPATCH:
            env["MEDDINOV3_D_PATCH"] = str(_DEPTH_TO_DPATCH[depth])
        env.setdefault("MEDDINOV3_NUM_EPOCHS", "500")
        env.setdefault("MEDDINOV3_BACKBONE_LR_SCALE", "0.1")
    # explicit per-experiment env (e.g. MEDDINOV3_BATCH_SIZE, CHD_FILM_*) wins
    for k, v in (exp.get("env") or {}).items():
        env[k] = str(v)
    return env


def inflated_checkpoint_name(exp: dict) -> str | None:
    """Conventional shared checkpoint filename for a MedDINO experiment."""
    if exp.get("model_family") != "meddinov3":
        return None
    depth = exp.get("meddinov3_depth")
    n = _DEPTH_TO_DPATCH.get(depth)
    if n is None:
        return None
    if "ashwin" in (exp.get("trainer") or ""):
        return f"meddinov3_inflated_ashwin_d{n}.pth"
    return f"meddinov3_inflated_center_d{n}.pth"


def compose_train_command(exp: dict, registry: dict, fold: int) -> str:
    """The nnUNetv2_train invocation (env vars handled separately)."""
    ds_id = exp.get("dataset_id")
    config = exp.get("config", "3d_fullres")
    trainer = exp.get("trainer")
    if trainer is None:
        raise ValueError(f"{exp['id']}: no trainer mapped (implemented={exp.get('implemented')})")
    return f"nnUNetv2_train {ds_id} {config} {fold} -tr {trainer} --npz"


def env_export_lines(env: dict) -> list[str]:
    return [f"export {k}={v}" for k, v in env.items()]


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def split_file_hash(exp: dict, registry: dict) -> str | None:
    """sha1 of splits_final.json for the experiment's dataset (fairness guard)."""
    name = dataset_name(exp, registry)
    if not name:
        return None
    split_rel = registry["meta"].get("default_split_file", "splits_final.json")
    path = Path(env_or_default("nnUNet_preprocessed")) / name / split_rel
    if not path.is_file():
        return None
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except Exception:
        return None


def results_dir_for(exp: dict, registry: dict, fold: int) -> Path | None:
    """The nnUNet results validation folder for this experiment/fold, if resolvable."""
    name = dataset_name(exp, registry)
    trainer = exp.get("trainer")
    config = exp.get("config", "3d_fullres")
    if not (name and trainer):
        return None
    base = Path(env_or_default("nnUNet_results")) / name
    return base / f"{trainer}__nnUNetPlans__{config}" / f"fold_{fold}" / "validation"


def write_manifest(exp: dict, registry: dict, fold: int, command: str, env: dict,
                   extra_notes: str = "") -> Path:
    import platform
    import socket
    from datetime import datetime

    out = RESULTS_DIR / exp["id"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": exp["id"],
        "experiment": exp,
        "git_commit": git_commit_hash(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "dataset_id": exp.get("dataset_id"),
        "dataset_name": dataset_name(exp, registry),
        "fold": fold,
        "command": command,
        "composed_env": env,
        "nnUNet_raw": env_or_default("nnUNet_raw"),
        "nnUNet_preprocessed": env_or_default("nnUNet_preprocessed"),
        "nnUNet_results": env_or_default("nnUNet_results"),
        "split_file_sha1": split_file_hash(exp, registry),
        "notes": extra_notes,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
