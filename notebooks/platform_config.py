"""CMU 11-785: Shared TPU Platform Configuration.

Single source of truth for storage paths, dataset locations, Kueue queues,
and credential discovery across all notebooks, training scripts, and batch TPU jobs.
"""
from __future__ import annotations

import os
from pathlib import Path

# ==============================================================================
# 1. Central Storage Hierarchy (Tartan 3-Tier Storage)
# ==============================================================================
# Central Tartan course storage bucket (formerly / equivalent to: cmu-idl-ocean)
TARTAN_BUCKET = os.environ.get("TARTAN_BUCKET") or os.environ.get("OCEAN_BUCKET") or "cmu-idl-ocean"
TARTAN_URI = os.environ.get("TARTAN_URI", f"gs://{TARTAN_BUCKET}")
TARTAN_MOUNT = os.environ.get("TARTAN_MOUNT_PATH", "/tartan/11785")
# Legacy equivalent mount path for Bridges-2 alike backward compatibility
LEGACY_OCEAN_MOUNT = "/ocean"
SHARED_ENV_NAME = os.environ.get("SHARED_ENV_NAME", "idl-shared-env")
SHARED_ENV_PATH = os.environ.get("SHARED_ENV_DIR", f"{TARTAN_MOUNT}/envs/{SHARED_ENV_NAME}")

def setup_shared_env() -> None:
    """Dynamically discover and register shared environment packages into sys.path."""
    import sys
    lib_dir = Path(SHARED_ENV_PATH) / "lib"
    if lib_dir.exists():
        for p in lib_dir.glob("python*/site-packages"):
            if str(p) not in sys.path:
                sys.path.insert(1, str(p))

setup_shared_env()

# Aliases for backward compatibility
OCEAN_BUCKET = TARTAN_BUCKET
OCEAN_URI = TARTAN_URI

# ==============================================================================
# 2. Kubernetes & Kueue Substrate
# ==============================================================================
NAMESPACE = (
    os.environ.get("COURSE_NAMESPACE")
    or os.environ.get("NAMESPACE")
    or os.environ.get("NS")
    or "cmu-idl"
)
KUEUE_QUEUE = os.environ.get("KUEUE_LOCAL_QUEUE", "tpu")
PRIORITY_JOB = "student-tpu-job"
PRIORITY_NOTEBOOK = "student-notebook"
TPU_SERVICE_ACCOUNT = os.environ.get("TPU_SERVICE_ACCOUNT", "student")

# ==============================================================================
# 3. Dynamic Storage Path Resolvers (Zero-Hardcoding & No Repetition)
# ==============================================================================
def get_dataset_uri(assignment: str = "hw2p2") -> str:
    """Resolve high-speed local mount or GCS fallback for datasets.

    Prioritizes:
      1. /tartan/11785/datasets/<assignment> (Local NVMe cache via GCS FUSE)
      2. /ocean/datasets/<assignment> (Legacy equivalent dual-mount)
      3. gs://<TARTAN_BUCKET>/datasets/<assignment> (Direct GCS URI)
    """
    assignment_clean = assignment.lower().replace("-", "_")
    tartan_path = Path(TARTAN_MOUNT) / "datasets" / assignment_clean
    # Legacy mount check (equivalent to /tartan/11785)
    legacy_path = Path(LEGACY_OCEAN_MOUNT) / "datasets" / assignment_clean

    if tartan_path.exists():
        return str(tartan_path)
    if legacy_path.exists():
        return str(legacy_path)

    custom_uri = os.environ.get("DATA_URI")
    if custom_uri:
        return custom_uri

    return f"gs://{TARTAN_BUCKET}/datasets/{assignment_clean}"


def get_checkpoint_uri(assignment: str = "hw2p2", student: str = "anon") -> str:
    """Resolve Orbax checkpoint URI in durable Tartan store."""
    assignment_clean = assignment.lower().replace("-", "_")
    custom_root = os.environ.get("CKPT_ROOT")
    root = (custom_root or f"gs://{TARTAN_BUCKET}/checkpoints/{assignment_clean}").rstrip("/")
    return f"{root}/{student}"


def get_submission_uri(assignment: str = "hw2p2", student: str = "anon") -> str:
    """Resolve Kaggle / Gradescope submission CSV destination in Tartan store."""
    assignment_clean = assignment.lower().replace("-", "_")
    custom_root = os.environ.get("SUBMISSION_ROOT")
    root = (custom_root or f"gs://{TARTAN_BUCKET}/submissions/{assignment_clean}").rstrip("/")
    return f"{root}/{student}/submission.csv"
