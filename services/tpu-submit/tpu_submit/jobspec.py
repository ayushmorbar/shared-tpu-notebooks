"""The one Job template every student run uses.

Nothing in here comes from the student except the script path, its arguments, the
dataset name (looked up in the host's list or Tartan store), and env values, which go
into a Secret. Queue label, chip count, image, priority, deadline, and service account
are fixed.

Shape notes forced by GKE, same as k8s/student-tpu-job.yaml:
  * One pod per TPU node, and the pod takes every chip on it. ct5lp-hightpu-1t has one.
  * The two nodeSelector labels are what make Autopilot pick ct5lp-hightpu-1t.
  * TPU nodes carry a NoSchedule taint, so the toleration is required.
  * TPU pods cannot use gVisor; host kernel access is unavoidable. The container runs
    as non-root with every capability dropped.
"""

from __future__ import annotations

import base64
from typing import Mapping, Sequence

from . import names
from .config import Settings

CONTAINER = "job"
WORK_DIR = "/work"
TARTAN_GCSFUSE_MOUNT_OPTIONS = (
    "implicit-dirs,file-cache:max-size-mb:0,"
    "metadata-cache:stat-cache-max-size-mb:2048,"
    "metadata-cache:ttl-secs:60,uid=1000,gid=1000,"
    "dir-mode=0777,file-mode=0777,ro"
)


def build_job(
    settings: Settings,
    *,
    user: str,
    job_id: str,
    script: str,
    args: Sequence[str],
    dataset_uri: str,
    token_sha: str,
) -> dict:
    name = names.job_name(user, job_id)
    labels = {
        names.LABEL_MANAGED_BY: names.MANAGED_BY,
        names.LABEL_USER_HASH: names.user_hash(user),
        names.LABEL_JOB_ID: job_id,
    }
    env = [
        {"name": "TPU_SUBMIT_URL", "value": settings.public_url},
        {"name": "TPU_JOB_NAME", "value": name},
        {"name": "TPU_JOB_ID", "value": job_id},
        {"name": "TPU_DATASET_URI", "value": dataset_uri},
        {"name": "TPU_SYNC_INTERVAL", "value": str(settings.sync_interval)},
        {"name": "TPU_WORK_DIR", "value": WORK_DIR},
        # Fail loudly instead of silently falling back to CPU. A CPU fallback produces
        # plausible numbers that are wrong.
        {"name": "JAX_PLATFORMS", "value": "tpu"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        # Hour-long progress bars otherwise write a line every 0.1 s.
        {"name": "TQDM_MININTERVAL", "value": "30"},
    ]
    volume_mounts = [{"name": "work", "mountPath": WORK_DIR}]
    volumes: list[dict] = [{"name": "work", "emptyDir": {"sizeLimit": settings.job_disk}}]
    template_metadata: dict = {"labels": labels}

    if settings.tartan_bucket:
        template_metadata["annotations"] = {"gke-gcsfuse/volumes": "true"}
        env.extend([
            {"name": "TARTAN_BUCKET", "value": settings.tartan_bucket},
            {"name": "TARTAN_MOUNT_PATH", "value": settings.tartan_mount_path},
            {"name": "SHARED_ENV_NAME", "value": settings.shared_env_name},
        ])
        volume_mounts.extend([
            {"name": "course-tartan", "mountPath": settings.tartan_mount_path, "readOnly": True},
            {"name": "course-tartan", "mountPath": "/ocean", "readOnly": True},
        ])
        volumes.append({
            "name": "course-tartan",
            "csi": {
                "driver": "gcsfuse.csi.storage.gke.io",
                "readOnly": True,
                "volumeAttributes": {
                    "bucketName": settings.tartan_bucket,
                    "mountOptions": TARTAN_GCSFUSE_MOUNT_OPTIONS,
                },
            },
        })

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": settings.namespace,
            "labels": {**labels, "kueue.x-k8s.io/queue-name": settings.queue_name},
            "annotations": {
                names.ANN_USER: user,
                names.ANN_JOB_ID: job_id,
                names.ANN_TOKEN_SHA: token_sha,
                names.ANN_SCRIPT: script,
            },
        },
        "spec": {
            # Kueue owns admission and unsuspends the Job when a chip is free.
            "suspend": True,
            "backoffLimit": 2,
            # The timer stops and resets while Kueue holds the Job suspended, so queue
            # time does not eat into the run.
            "activeDeadlineSeconds": int(settings.max_job_hours * 3600),
            "ttlSecondsAfterFinished": settings.job_ttl_seconds,
            # A pod pushed off its chip retries without counting against backoffLimit
            # and resumes from the last uploaded checkpoint. A non-zero exit from the
            # student's script ends the Job.
            "podFailurePolicy": {
                "rules": [
                    {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]},
                    {
                        "action": "FailJob",
                        "onExitCodes": {
                            "containerName": CONTAINER,
                            "operator": "NotIn",
                            "values": [0],
                        },
                    },
                ]
            },
            "template": {
                "metadata": template_metadata,
                "spec": {
                    "restartPolicy": "Never",
                    "priorityClassName": settings.priority_class,
                    # Workload Identity grants this account read access to the dataset
                    # bucket and nothing else. It has no Kubernetes RBAC.
                    "serviceAccountName": settings.job_service_account,
                    # Workload Identity goes through the GKE metadata server, not this
                    # token, so student code gets no Kubernetes credential at all.
                    "automountServiceAccountToken": False,
                    "enableServiceLinks": False,
                    "terminationGracePeriodSeconds": 120,
                    "nodeSelector": {
                        "cloud.google.com/gke-tpu-accelerator": settings.tpu_accelerator,
                        "cloud.google.com/gke-tpu-topology": settings.tpu_topology,
                    },
                    "tolerations": [
                        {"key": "google.com/tpu", "operator": "Exists", "effect": "NoSchedule"}
                    ],
                    "securityContext": {"fsGroup": 1000, "fsGroupChangePolicy": "OnRootMismatch"},
                    "containers": [
                        {
                            "name": CONTAINER,
                            "image": settings.job_image,
                            "command": ["python3", "/opt/tpu/tpu_entrypoint.py"],
                            "args": [script, *args],
                            "env": env,
                            "envFrom": [{"secretRef": {"name": name}}],
                            "resources": {
                                "requests": {
                                    "google.com/tpu": "1",
                                    "ephemeral-storage": settings.job_disk,
                                },
                                "limits": {
                                    "google.com/tpu": "1",
                                    "ephemeral-storage": settings.job_disk,
                                },
                            },
                            "securityContext": {
                                "runAsUser": 1000,
                                "runAsGroup": 1000,
                                "runAsNonRoot": True,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": volume_mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }


def build_secret(
    settings: Settings,
    *,
    name: str,
    token: str,
    env: Mapping[str, str],
    owner_uid: str,
) -> dict:
    """Per-job Secret: the job token plus the student's env values.

    Owned by the Job, so Kubernetes deletes it when the Job goes away.
    """
    data = {"TPU_JOB_TOKEN": token, **env}
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": settings.namespace,
            "labels": {names.LABEL_MANAGED_BY: names.MANAGED_BY},
            "ownerReferences": [
                {"apiVersion": "batch/v1", "kind": "Job", "name": name, "uid": owner_uid}
            ],
        },
        "type": "Opaque",
        "data": {k: base64.b64encode(v.encode()).decode() for k, v in data.items()},
    }
