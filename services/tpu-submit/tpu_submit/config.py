"""Service settings, read once from the environment at startup."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


def _parse_datasets(raw: str) -> dict[str, str]:
    """Parse `name=gs://bucket/path.tar,other=gs://...` into a dict."""
    out: dict[str, str] = {}
    for part in raw.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, uri = part.partition("=")
        if not sep or not name.strip() or not uri.strip():
            raise ValueError(f"DATASETS entry {part!r} is not name=uri")
        out[name.strip()] = uri.strip()
    return out


@dataclass(frozen=True)
class Settings:
    namespace: str
    public_url: str
    job_image: str
    student_bucket: str = ""
    # Filesystem store instead of GCS. For local development and tests only.
    store_dir: str = ""
    hub_api_url: str = "http://hub:8081/hub/api"
    service_name: str = "tpu-submit"
    require_access_scope: bool = True
    job_service_account: str = "tpu-job"
    queue_name: str = "tpu"
    priority_class: str = "student-tpu-job"
    tpu_accelerator: str = "tpu-v5-lite-podslice"
    tpu_topology: str = "1x1"
    datasets: Mapping[str, str] = field(default_factory=dict)
    default_dataset: str = ""
    max_job_hours: float = 8.0
    # 0 disables the budget.
    chip_hour_budget: float = 0.0
    max_active_jobs: int = 1
    max_snapshot_mb: int = 100
    max_file_mb: int = 4096
    job_disk: str = "100Gi"
    sync_interval: int = 300
    # Finished Jobs count against the namespace's Job quota. State and logs are already
    # copied to the bucket, so the Job object is only needed briefly.
    job_ttl_seconds: int = 3600
    reconcile_interval: int = 60
    spool_dir: str = "/tmp"
    port: int = 8080

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            return env.get(name, default).strip()

        namespace = get("NAMESPACE")
        if not namespace:
            raise ValueError("NAMESPACE is required")
        settings = cls(
            namespace=namespace,
            public_url=get("PUBLIC_URL", f"http://tpu-submit.{namespace}.svc.cluster.local:8080"),
            job_image=get("TPU_JOB_IMAGE"),
            student_bucket=get("STUDENT_BUCKET"),
            store_dir=get("STORE_DIR"),
            hub_api_url=get("HUB_API_URL", "http://hub:8081/hub/api").rstrip("/"),
            service_name=get("SERVICE_NAME", "tpu-submit"),
            require_access_scope=get("REQUIRE_ACCESS_SCOPE", "true").lower() in ("1", "true", "yes"),
            job_service_account=get("JOB_SERVICE_ACCOUNT", "tpu-job"),
            queue_name=get("QUEUE_NAME", "tpu"),
            priority_class=get("PRIORITY_CLASS", "student-tpu-job"),
            tpu_accelerator=get("TPU_ACCELERATOR", "tpu-v5-lite-podslice"),
            tpu_topology=get("TPU_TOPOLOGY", "1x1"),
            datasets=_parse_datasets(get("DATASETS")),
            default_dataset=get("DEFAULT_DATASET"),
            max_job_hours=float(get("MAX_JOB_HOURS", "8")),
            chip_hour_budget=float(get("CHIP_HOUR_BUDGET", "0")),
            max_active_jobs=int(get("MAX_ACTIVE_JOBS", "1")),
            max_snapshot_mb=int(get("MAX_SNAPSHOT_MB", "100")),
            max_file_mb=int(get("MAX_FILE_MB", "4096")),
            job_disk=get("JOB_DISK", "100Gi"),
            sync_interval=int(get("SYNC_INTERVAL", "300")),
            job_ttl_seconds=int(get("JOB_TTL_SECONDS", "3600")),
            reconcile_interval=int(get("RECONCILE_INTERVAL", "60")),
            spool_dir=get("SPOOL_DIR", "/tmp"),
            port=int(get("PORT", "8080")),
        )
        if not settings.job_image:
            raise ValueError("TPU_JOB_IMAGE is required")
        if not settings.student_bucket and not settings.store_dir:
            raise ValueError("STUDENT_BUCKET is required")
        if settings.default_dataset and settings.default_dataset not in settings.datasets:
            raise ValueError(
                f"DEFAULT_DATASET {settings.default_dataset!r} is not one of DATASETS "
                f"({', '.join(settings.datasets) or 'none configured'})"
            )
        return settings
