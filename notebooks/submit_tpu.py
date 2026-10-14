"""Run code on a Cloud TPU chip from a CPU JupyterLab notebook.

This module provides the core bridge between student CPU JupyterLab pods and the
shared Cloud TPU v5e pool managed by Kueue on Google Kubernetes Engine (GKE Autopilot).

Two execution paths are provided:
1. `submit_tpu.run(code, ...)`:
   Submits code as an inline Python string for quick execution tests and micro-benchmarks.
   The Job is automatically deleted upon completion.

2. `submit_tpu.train(script, ...)`:
   Submits a full standalone training script (e.g. `hw2p2_train.py` or `train_hw2p2.py`)
   mounted into the TPU container via a Kubernetes ConfigMap. Provides:
   - High-throughput local NVMe scratch disk (`emptyDir` on `/tmp`)
   - In-RAM dataset loading (~24 GiB VmRSS on 47 GiB node)
   - Durable Orbax checkpointing to Google Cloud Storage
   - Real-time log streaming across multi-hour training epochs
   - Automated delivery of predictions (`submission.csv`) and checkpoints to the student's
     persistent JupyterHub home volume (`/home/jovyan/`)
   - Multi-tier private credential discovery (Kaggle, WandB, Hugging Face)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

# Cluster and Accelerator Defaults
TPU_IMAGE = os.environ.get(
    "TPU_IMAGE", "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest"
)
QUEUE = os.environ.get("KUEUE_LOCAL_QUEUE", "tpu")
TPU_ACCELERATOR = os.environ.get("TPU_ACCELERATOR", "tpu-v5-lite-podslice")
TPU_TOPOLOGY = os.environ.get("TPU_TOPOLOGY", "1x1")
TPU_SERVICE_ACCOUNT = os.environ.get("TPU_SERVICE_ACCOUNT", "student")


class JobInterrupted(RuntimeError):
    """The job went away before it could run.

    This exception is raised when the job is deleted from the cluster
    (e.g., via TTL or manual deletion) before it was admitted or finished.
    Nothing executed; re-running is safe.
    """


def _delete(batch: "client.BatchV1Api", name: str, ns: str) -> None:
    """Delete a Job, tolerating the case where it is already gone.

    Args:
        batch (client.BatchV1Api): The initialized Kubernetes Batch V1 API client.
        name (str): The name of the Job to delete.
        ns (str): The namespace where the Job resides.
    """
    try:
        batch.delete_namespaced_job(name, ns, propagation_policy="Background")
    except ApiException as e:
        if e.status != 404:
            raise


def _delete_config_map(core: "client.CoreV1Api", name: str, ns: str) -> None:
    """Delete a ConfigMap, tolerating 404 Not Found."""
    try:
        core.delete_namespaced_config_map(name, ns)
    except ApiException as e:
        if e.status != 404:
            raise


def _namespace() -> str:
    """Get the namespace this notebook pod runs in.

    Returns:
        str: The Kubernetes namespace string. Defaults to "cmu-idl" or "default".
    """
    path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return os.environ.get("POD_NAMESPACE", os.environ.get("NAMESPACE", "cmu-idl"))


def _load() -> None:
    """Load the Kubernetes configuration.

    Attempts to load the in-cluster config first, falling back to a local
    kubeconfig file if running outside a cluster.
    """
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def _discover_credentials(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Discover student credentials without leaking them into shared repositories.

    Hierarchy:
      1. Personal `/home/jovyan/.env` or `.env` in current directory.
      2. Kaggle configuration file `~/.kaggle/kaggle.json` or `/home/jovyan/.kaggle/kaggle.json`.
      3. Process environment variables (`os.environ`).
      4. Explicit `extra_env` dictionary argument (highest precedence).

    Supported keys:
      - Kaggle: `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`, `KAGGLE_KEY`
      - Weights & Biases: `WANDB_API_KEY`
      - Hugging Face: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`
      - Custom keys supplied by students or instructors for any homework assignment.
    """
    creds: dict[str, str] = {}

    # 1. Parse .env files (without external dependencies)
    env_paths = [
        os.path.expanduser("/home/jovyan/.env"),
        os.path.expanduser("~/.env"),
        os.path.abspath(".env"),
    ]
    for p in env_paths:
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k:
                            creds[k] = v
            except Exception:
                pass

    # 2. Parse kaggle.json
    kaggle_paths = [
        os.path.expanduser("/home/jovyan/.kaggle/kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
    ]
    for kp in kaggle_paths:
        if os.path.isfile(kp):
            try:
                with open(kp, encoding="utf-8") as fh:
                    data = json.load(fh)
                    if "username" in data and "KAGGLE_USERNAME" not in creds:
                        creds["KAGGLE_USERNAME"] = data["username"]
                    if "key" in data and "KAGGLE_API_TOKEN" not in creds:
                        creds["KAGGLE_API_TOKEN"] = data["key"]
            except Exception:
                pass

    # 3. Process environment variables
    standard_vars = [
        "WANDB_API_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ]
    for var in standard_vars:
        if var in os.environ and var not in creds:
            creds[var] = os.environ[var]

    # Ensure KAGGLE_API_TOKEN is mirrored if KAGGLE_KEY was set
    if "KAGGLE_KEY" in creds and "KAGGLE_API_TOKEN" not in creds:
        creds["KAGGLE_API_TOKEN"] = creds["KAGGLE_KEY"]
    if "KAGGLE_API_TOKEN" in creds and "KAGGLE_KEY" not in creds:
        creds["KAGGLE_KEY"] = creds["KAGGLE_API_TOKEN"]

    # 4. Merge extra_env
    if extra_env:
        creds.update(extra_env)

    return creds


def train(
    script: str,
    assignment: str | None = None,
    modules: list[str] | dict[str, str] | None = None,
    data_uri: str | None = None,
    ckpt_uri: str | None = None,
    submission_uri: str | None = None,
    epochs: int = 2,
    steps_per_epoch: int = 0,
    batch_size: int = 128,
    secret: str | None = "course-secrets",
    service_account_name: str | None = None,
    image: str | None = None,
    timeout: int = 28800,
    extra_env: dict[str, str] | None = None,
    grant_tier: str | None = None,
    queue: str | None = None,
    download_submission: bool = True,
    local_submission_path: str | None = None,
    sync_checkpoints: bool = True,
    local_ckpt_dir: str | None = None,
) -> str:
    """Run a full training script on one Cloud TPU v5e chip with live log streaming.

    Designed for long training runs across any course assignment (HW1, HW2, HW3, etc.).
    Unlike run(), train() mounts the script file from a ConfigMap, attaches high-speed
    NVMe scratch disk on /tmp, runs as an authenticated ServiceAccount with Workload Identity,
    streams container output live, and automatically delivers predictions and checkpoints
    back to the student's persistent JupyterHub home volume.

    Args:
        script (str): Path to the Python training script on the notebook disk.
        assignment (str, optional): Homework identifier (e.g. "hw2p2", "hw1p2", "hw3").
            If None, auto-inferred from script filename or os.environ["ASSIGNMENT"].
        data_uri (str, optional): Dataset URI. Auto-resolves /tartan/11785/datasets/{assignment},
            /ocean/datasets/{assignment}, or GCS bucket if not specified.
        ckpt_uri (str, optional): GCS base URI for Orbax model checkpoints.
            Defaults to f"{os.environ['CKPT_ROOT']}/{assignment}/{student}".
        submission_uri (str, optional): GCS URI where submission.csv will be saved.
            Defaults to f"{os.environ['SUBMISSION_ROOT']}/{assignment}/{student}/submission.csv".
        epochs (int, optional): Number of training epochs to execute. Defaults to 2.
        steps_per_epoch (int, optional): Steps per epoch (0 runs full epoch). Defaults to 0.
        batch_size (int, optional): Batch size per training step. Defaults to 128.
        secret (str, optional): Name of fallback Kubernetes Secret for course keys.
            Defaults to "course-secrets".
        service_account_name (str, optional): ServiceAccount for the TPU pod.
            Defaults to "student" (or os.environ["TPU_SERVICE_ACCOUNT"]).
        image (str, optional): Container image. Defaults to TPU_IMAGE.
        timeout (int, optional): Maximum timeout in seconds (default: 28800 = 8 hours).
        extra_env (dict[str, str], optional): Custom environment variables / secrets
            (e.g., HF_TOKEN, custom hyperparameters) forwarded directly to the pod.
        grant_tier (str, optional): Student project grant tier for Kueue budget tracking
            (e.g., "tier0-coursework", "tier1-capstone", "tier2-fellow").
        queue (str, optional): Specific Kueue LocalQueue to submit to. Defaults to "tpu".
        download_submission (bool, optional): Automatically download submission.csv
            from GCS to local_submission_path upon successful job completion. Defaults to True.
        local_submission_path (str, optional): Local destination for submission.csv.
            Defaults to f"/home/jovyan/{assignment}_submission.csv".
        sync_checkpoints (bool, optional): Automatically sync checkpoints from GCS
            to local_ckpt_dir upon job completion. Defaults to False.
        local_ckpt_dir (str, optional): Local destination directory for checkpoints.
            Defaults to f"/home/jovyan/model_checkpoints/{assignment}".

    Returns:
        str: The complete stdout/stderr logs from the training execution.

    Raises:
        TimeoutError: If the job does not start or finish within the timeout.
        RuntimeError: If the training job exits with failure.
    """
    _load()
    ns = _namespace()
    student = os.environ.get("JUPYTERHUB_USER", "anon").replace("_", "-").lower()

    # 1. Resolve assignment identifier generically
    if not assignment:
        assignment = os.environ.get("ASSIGNMENT")
        if not assignment:
            base_script = os.path.basename(script).lower()
            match = re.search(r"(hw\d+(?:p\d+)?|assignment\d+)", base_script)
            assignment = match.group(1) if match else "train"

    assignment_clean = assignment.replace("_", "-").lower()
    name = f"tpu-{assignment_clean}-{student}"[:63]

    # Resolve default storage URIs (Tartan-aware, Ocean-aware & assignment-aware)
    tartan_local_dir = f"/tartan/11785/datasets/{assignment_clean}"
    ocean_local_dir = f"/ocean/datasets/{assignment_clean}"
    if not data_uri:
        if os.path.exists(tartan_local_dir):
            data_uri = tartan_local_dir
        elif os.path.exists(ocean_local_dir):
            data_uri = ocean_local_dir
        elif assignment_clean == "hw2p2":
            data_uri = os.environ.get("DATA_URI", "gs://hw2p2-cached-data/img_cache")
        else:
            ocean_bucket = os.environ.get("OCEAN_BUCKET", "cmu-idl-ocean")
            data_uri = os.environ.get("DATA_URI", f"gs://{ocean_bucket}/datasets/{assignment_clean}")

    ckpt_root = os.environ.get(
        "CKPT_ROOT",
        "gs://hw2p2-cached-data/checkpoints" if assignment_clean == "hw2p2" else f"gs://cmu-idl-ocean/checkpoints/{assignment_clean}"
    ).rstrip("/")
    ckpt_uri = ckpt_uri or f"{ckpt_root}/{student}"

    sub_root = os.environ.get(
        "SUBMISSION_ROOT",
        "gs://hw2p2-cached-data/submissions" if assignment_clean == "hw2p2" else f"gs://cmu-idl-ocean/submissions/{assignment_clean}"
    ).rstrip("/")
    submission_uri = submission_uri or f"{sub_root}/{student}/submission.csv"

    # Resolve local download paths in student persistent home volume
    if local_submission_path is None:
        local_submission_path = f"/home/jovyan/{assignment_clean}_submission.csv"
    if local_ckpt_dir is None:
        local_ckpt_dir = f"/home/jovyan/checkpoints/{assignment_clean}"

    # Resolve ServiceAccount (defaults to student, configurable via argument or env)
    sa_name = service_account_name or os.environ.get(
        "TPU_SERVICE_ACCOUNT", TPU_SERVICE_ACCOUNT
    )

    batch = client.BatchV1Api()
    core = client.CoreV1Api()

    # Read the entrypoint script and package into a ConfigMap
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Training script not found: {script}")

    with open(script, encoding="utf-8") as fh:
        body = fh.read()

    cm_data = {"entrypoint.py": body}

    # Auto-discover or bundle modular helper Python files in script directory
    script_dir = os.path.dirname(os.path.abspath(script))
    if modules is None and os.path.isdir(script_dir):
        discovered = []
        for fname in os.listdir(script_dir):
            if fname.endswith(".py") and os.path.abspath(os.path.join(script_dir, fname)) != os.path.abspath(script):
                fpath = os.path.join(script_dir, fname)
                if os.path.getsize(fpath) < 150_000:
                    discovered.append(fpath)
        if discovered:
            print(f"[submit_tpu] Auto-bundling modular files: {[os.path.basename(f) for f in discovered]}", flush=True)
            modules = discovered

    if modules:
        if isinstance(modules, dict):
            for mod_name, mod_content in modules.items():
                cm_data[os.path.basename(mod_name)] = mod_content
        elif isinstance(modules, (list, tuple, set)):
            for mod_path in modules:
                if os.path.isfile(mod_path):
                    with open(mod_path, encoding="utf-8") as mfh:
                        cm_data[os.path.basename(mod_path)] = mfh.read()
                else:
                    raise FileNotFoundError(f"Module file not found: {mod_path}")
        elif isinstance(modules, str) and os.path.isfile(modules):
            with open(modules, encoding="utf-8") as mfh:
                cm_data[os.path.basename(modules)] = mfh.read()

    cm_name = f"{name}-src"
    cm = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=cm_name, namespace=ns),
        data=cm_data,
    )
    try:
        core.create_namespaced_config_map(ns, cm)
    except ApiException as e:
        if e.status != 409:
            raise
        core.replace_namespaced_config_map(cm_name, ns, cm)

    # Base environment variables
    env_vars = [
        client.V1EnvVar(name="JAX_PLATFORMS", value="tpu"),
        client.V1EnvVar(name="ASSIGNMENT", value=assignment_clean),
        client.V1EnvVar(name="DATA_URI", value=data_uri),
        client.V1EnvVar(name="CKPT_URI", value=ckpt_uri),
        client.V1EnvVar(name="SUBMISSION_URI", value=submission_uri),
        client.V1EnvVar(name="STUDENT", value=student),
        client.V1EnvVar(name="EPOCHS", value=str(epochs)),
        client.V1EnvVar(name="STEPS_PER_EPOCH", value=str(steps_per_epoch)),
        client.V1EnvVar(name="BATCH_SIZE", value=str(batch_size)),
        # Writable directory for gcloud, cache, and non-root execution
        client.V1EnvVar(name="HOME", value="/tmp"),
        # Ensure all mounted modules in /src are directly importable
        client.V1EnvVar(name="PYTHONPATH", value="/src"),
    ]

    # Discover and inject private credentials
    creds = _discover_credentials(extra_env)
    injected_keys = set()
    if creds:
        print("[submit_tpu] Discovered student credentials to forward:", flush=True)
    for k, v in creds.items():
        env_vars.append(client.V1EnvVar(name=k, value=str(v)))
        injected_keys.add(k)
        digest = hashlib.sha256(str(v).encode()).hexdigest()[:8]
        print(f"  • {k:20}: detected (len={len(str(v)):2}, sha256[:8]={digest})", flush=True)

    # Optional cluster Secret fallback (e.g. course-secrets)
    if secret:
        fallback_mappings = [
            ("WANDB_API_KEY", "wandb-api-key"),
            ("KAGGLE_USERNAME", "kaggle-username"),
            ("KAGGLE_API_TOKEN", "kaggle-api-token"),
            ("HF_TOKEN", "hf-token"),
        ]
        for env_key, secret_key in fallback_mappings:
            if env_key not in injected_keys:
                env_vars.append(
                    client.V1EnvVar(
                        name=env_key,
                        value_from=client.V1EnvVarSource(
                            secret_key_ref=client.V1SecretKeySelector(
                                name=secret, key=secret_key, optional=True
                            )
                        ),
                    )
                )

    # Clean up previous instance of this student's training job
    _delete(batch, name, ns)

    effective_queue = queue or os.environ.get("KUEUE_LOCAL_QUEUE", QUEUE)
    job_labels = {
        "kueue.x-k8s.io/queue-name": effective_queue,
        "cmu-course": "11-785",
        "cmu-student": student,
        "cmu-assignment": assignment_clean,
        "cmu-grant-tier": (grant_tier or os.environ.get("GRANT_TIER", "tier0-coursework")).lower(),
    }

    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=ns,
            labels=job_labels,
        ),
        spec=client.V1JobSpec(
            suspend=True,
            backoff_limit=0,
            active_deadline_seconds=timeout,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    service_account_name=sa_name,
                    priority_class_name="student-tpu-job",
                    node_selector={
                        "cloud.google.com/gke-tpu-accelerator": TPU_ACCELERATOR,
                        "cloud.google.com/gke-tpu-topology": TPU_TOPOLOGY,
                    },
                    tolerations=[
                        client.V1Toleration(
                            key="google.com/tpu",
                            operator="Exists",
                            effect="NoSchedule",
                        )
                    ],
                    volumes=[
                        client.V1Volume(
                            name="src",
                            config_map=client.V1ConfigMapVolumeSource(name=cm_name),
                        ),
                        # Local NVMe scratch disk for in-RAM cache loading
                        client.V1Volume(
                            name="scratch",
                            empty_dir=client.V1EmptyDirVolumeSource(),
                        ),
                    ],
                    containers=[
                        client.V1Container(
                            name="train",
                            image=image or TPU_IMAGE,
                            command=["/opt/venv/bin/python3", "/src/entrypoint.py"],
                            env=env_vars,
                            volume_mounts=[
                                client.V1VolumeMount(name="src", mount_path="/src"),
                                client.V1VolumeMount(
                                    name="scratch", mount_path="/tmp"
                                ),
                            ],
                            resources=client.V1ResourceRequirements(
                                limits={"google.com/tpu": "1"},
                                requests={"google.com/tpu": "1"},
                            ),
                            security_context=client.V1SecurityContext(
                                run_as_user=1000,
                                run_as_group=1000,
                                run_as_non_root=True,
                                allow_privilege_escalation=False,
                                capabilities=client.V1Capabilities(drop=["ALL"]),
                            ),
                        )
                    ],
                )
            ),
        ),
    )

    batch.create_namespaced_job(ns, job)
    print(f"submitted {name} to queue '{QUEUE}'", flush=True)
    print(f"  script:  {script}", flush=True)
    print(f"  data:    {data_uri}", flush=True)
    print(f"  ckpt:    {ckpt_uri}", flush=True)
    print(f"  sub:     {submission_uri}", flush=True)
    print(f"  sa:      {sa_name}", flush=True)
    if injected_keys:
        masked = [f"{k}(len={len(creds[k])})" for k in sorted(injected_keys)]
        print(f"  creds:   {', '.join(masked)}", flush=True)
    else:
        print("  creds:   none detected (Tip: copy student.env.template to /home/jovyan/.env)", flush=True)

    t0 = time.time()
    admitted_at = None
    last_heartbeat = t0
    pod_name = None
    collected: list[str] = []

    try:
        # 1. Wait for admission & node provisioning with 30s heartbeats
        while time.time() - t0 < timeout:
            try:
                j = batch.read_namespaced_job(name, ns)
            except ApiException as e:
                if e.status == 404:
                    raise JobInterrupted(
                        f"{name} was deleted while waiting for admission. Re-run to submit again."
                    ) from None
                raise

            if admitted_at is None and not j.spec.suspend:
                admitted_at = time.time()
                print(
                    f"  admitted after {admitted_at - t0:.0f}s in queue", flush=True
                )
            elif admitted_at is None and (time.time() - last_heartbeat) >= 30:
                elapsed = int(time.time() - t0)
                print(
                    f"  [Queue Update] Waiting in queue '{QUEUE}' ({elapsed}s elapsed)...",
                    flush=True,
                )
                last_heartbeat = time.time()

            pods = core.list_namespaced_pod(
                ns, label_selector=f"job-name={name}"
            ).items
            if pods:
                phase = pods[0].status.phase
                if phase in ("Running", "Succeeded", "Failed"):
                    pod_name = pods[0].metadata.name
                    break
                elif phase == "Pending" and (time.time() - last_heartbeat) >= 30:
                    elapsed = int(time.time() - t0)
                    print(
                        f"  [Node Build] TPU node provisioning in progress ({elapsed}s elapsed)...",
                        flush=True,
                    )
                    last_heartbeat = time.time()
            elif admitted_at is not None and (time.time() - last_heartbeat) >= 30:
                elapsed = int(time.time() - t0)
                print(
                    f"  [Starting] Cloud TPU pod initializing ({elapsed}s elapsed)...",
                    flush=True,
                )
                last_heartbeat = time.time()

            time.sleep(5)

        if not pod_name:
            raise TimeoutError(f"{name} never started within {timeout}s")

        print(
            f"  running in {pod_name} after {time.time() - t0:.0f}s\n", flush=True
        )

        # 2. Stream logs live
        stream = core.read_namespaced_pod_log(
            pod_name, ns, follow=True, _preload_content=False
        )
        for raw in stream:
            line = raw.decode(errors="replace")
            collected.append(line)
            print(line, end="", flush=True)

        # 3. Inspect final phase
        phase = core.read_namespaced_pod(pod_name, ns).status.phase
        print(f"\n  {phase} after {time.time() - t0:.0f}s total", flush=True)

        if phase == "Failed":
            raise RuntimeError(f"{name} failed; job retained for TA inspection")

        # 4. Post-run artifact delivery to student's persistent JupyterHub volume
        if phase == "Succeeded":
            if download_submission and submission_uri:
                os.makedirs(os.path.dirname(os.path.abspath(local_submission_path)), exist_ok=True)
                print(f"\n[Artifact Delivery] Syncing submission.csv to {local_submission_path}...", flush=True)
                try:
                    subprocess.run(
                        ["gcloud", "storage", "cp", submission_uri, local_submission_path],
                        check=False,
                        capture_output=True,
                    )
                    if os.path.isfile(local_submission_path):
                        sz = os.path.getsize(local_submission_path)
                        print(f"  ✓ {local_submission_path} ready ({sz:,} bytes)", flush=True)
                except Exception as ex:
                    print(f"  ⚠ Could not auto-download submission: {ex}", flush=True)

            if sync_checkpoints and ckpt_uri:
                os.makedirs(local_ckpt_dir, exist_ok=True)
                print(f"[Artifact Delivery] Syncing checkpoints to {local_ckpt_dir}...", flush=True)
                try:
                    subprocess.run(
                        ["gcloud", "storage", "rsync", "-r", ckpt_uri, local_ckpt_dir],
                        check=False,
                        capture_output=True,
                    )
                    print(f"  ✓ Checkpoints synced to {local_ckpt_dir}", flush=True)
                except Exception as ex:
                    print(f"  ⚠ Could not auto-sync checkpoints: {ex}", flush=True)

        return "".join(collected)

    except KeyboardInterrupt:
        print(f"\n  interrupted; deleting {name}", flush=True)
        _delete(batch, name, ns)
        raise


def run(code: str, timeout: int = 28800, keep: bool = False) -> str:
    """Run inline code on one v5e chip. Blocks until it finishes and returns stdout.

    The timeout covers the entire wait: queue time plus node provisioning
    plus the actual execution run. A cold chip can take several minutes to
    arrive, which is normal.

    The job name is deterministic (one per student). If a previous job with the
    same name is still running, the create call will fail with 409 Conflict,
    which prevents a student from hogging multiple chips at once.

    Args:
        code (str): The python code to execute on the TPU.
        timeout (int, optional): Maximum time in seconds to wait. Defaults to 28800 (8 hours).
        keep (bool, optional): Whether to keep the Job object around after execution
            instead of deleting it. Defaults to False.

    Returns:
        str: The standard output logs from the pod execution.

    Raises:
        JobInterrupted: If the job is deleted from the cluster while waiting.
        RuntimeError: If the job execution fails.
        TimeoutError: If the job does not complete within the timeout.
    """
    _load()
    ns = _namespace()
    name = f"tpu-{os.environ.get('JUPYTERHUB_USER', 'anon')}"
    name = name.replace("_", "-").lower()[:63]

    batch = client.BatchV1Api()
    core = client.CoreV1Api()

    job = client.V1Job(
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=ns,
            labels={"kueue.x-k8s.io/queue-name": QUEUE},
        ),
        spec=client.V1JobSpec(
            suspend=True,
            backoff_limit=0,
            ttl_seconds_after_finished=None if keep else 600,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    service_account_name=TPU_SERVICE_ACCOUNT,
                    priority_class_name="student-tpu-job",
                    node_selector={
                        "cloud.google.com/gke-tpu-accelerator": TPU_ACCELERATOR,
                        "cloud.google.com/gke-tpu-topology": TPU_TOPOLOGY,
                    },
                    tolerations=[
                        client.V1Toleration(
                            key="google.com/tpu",
                            operator="Exists",
                            effect="NoSchedule",
                        )
                    ],
                    containers=[
                        client.V1Container(
                            name="hw",
                            image=TPU_IMAGE,
                            command=["python3", "-c", code],
                            env=[client.V1EnvVar(name="JAX_PLATFORMS", value="tpu")],
                            resources=client.V1ResourceRequirements(
                                limits={"google.com/tpu": "1"},
                                requests={"google.com/tpu": "1"},
                            ),
                            security_context=client.V1SecurityContext(
                                run_as_user=1000,
                                run_as_group=1000,
                                run_as_non_root=True,
                                allow_privilege_escalation=False,
                                capabilities=client.V1Capabilities(drop=["ALL"]),
                            ),
                        )
                    ],
                )
            ),
        ),
    )

    batch.create_namespaced_job(ns, job)
    print(f"submitted {name} to queue '{QUEUE}'", flush=True)

    t0 = time.time()
    admitted_at = None
    last_heartbeat = t0

    try:
        while time.time() - t0 < timeout:
            try:
                j = batch.read_namespaced_job(name, ns)
            except ApiException as e:
                if e.status == 404:
                    raise JobInterrupted(
                        f"{name} was deleted while waiting for a chip. Nothing ran. "
                        f"Re-run the cell to submit again."
                    ) from None
                raise

            if admitted_at is None and not j.spec.suspend:
                admitted_at = time.time()
                print(
                    f"  admitted after {admitted_at - t0:.0f}s in queue", flush=True
                )
            elif admitted_at is None and (time.time() - last_heartbeat) >= 30:
                elapsed = int(time.time() - t0)
                print(
                    f"  [Queue Update] Still waiting for TPU allocation ({elapsed}s elapsed)...",
                    flush=True,
                )
                last_heartbeat = time.time()

            pods = core.list_namespaced_pod(
                ns, label_selector=f"job-name={name}"
            ).items
            if pods:
                phase = pods[0].status.phase
                if phase in ("Succeeded", "Failed"):
                    out = core.read_namespaced_pod_log(pods[0].metadata.name, ns)
                    print(f"  {phase} after {time.time() - t0:.0f}s total", flush=True)
                    if not keep:
                        _delete(batch, name, ns)
                    if phase == "Failed":
                        raise RuntimeError(f"job failed:\n{out}")
                    return out
                elif phase == "Pending" and (time.time() - last_heartbeat) >= 30:
                    elapsed = int(time.time() - t0)
                    print(
                        f"  [Starting] TPU container initializing ({elapsed}s elapsed)...",
                        flush=True,
                    )
                    last_heartbeat = time.time()
            elif admitted_at is not None and (time.time() - last_heartbeat) >= 30:
                elapsed = int(time.time() - t0)
                print(
                    f"  [Node Build] TPU node provisioning in progress ({elapsed}s elapsed)...",
                    flush=True,
                )
                last_heartbeat = time.time()

            time.sleep(5)

    except KeyboardInterrupt:
        print(f"  interrupted; deleting {name}", flush=True)
        _delete(batch, name, ns)
        raise

    _delete(batch, name, ns)
    raise TimeoutError(
        f"{name} did not finish within {timeout}s. The pool is busy or the zone is "
        f"out of v5e. Check: kubectl get workloads -n {ns}"
    )


def verify_credentials(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Inspect and verify student credentials without logging cleartext secrets.

    Returns the discovered credentials dictionary. Safe for student and TA debugging.
    """
    creds = _discover_credentials(extra_env)
    print("\n=== Student Credential Verification ===", flush=True)
    if not creds:
        print("  Notice: No personal credentials discovered in /home/jovyan/.env, ~/.kaggle, or environment.", flush=True)
        print("  Running with public/anonymous access.", flush=True)
    else:
        for k, v in sorted(creds.items()):
            digest = hashlib.sha256(str(v).encode()).hexdigest()[:8]
            print(f"  • {k:22}: present (length={len(str(v)):2}, sha256[:8]={digest})", flush=True)
    return creds


class TPUJob:
    """Object-oriented interface for managing Cloud TPU workloads.

    Provides a clean, modular abstraction so students and instructors can submit,
    track, and organize multi-file training runs cleanly.

    Usage:
        job = TPUJob(assignment="hw2p2")
        job.train("hw2p2_train.py", modules=["model.py", "dataset.py"], epochs=40)
    """

    def __init__(
        self,
        assignment: str | None = None,
        image: str | None = None,
        service_account_name: str | None = None,
        queue: str | None = None,
        grant_tier: str | None = None,
    ):
        self.assignment = assignment
        self.image = image
        self.service_account_name = service_account_name
        self.queue = queue or QUEUE
        self.grant_tier = grant_tier

    def train(
        self,
        script: str,
        modules: list[str] | dict[str, str] | None = None,
        epochs: int = 2,
        steps_per_epoch: int = 0,
        batch_size: int = 128,
        grant_tier: str | None = None,
        queue: str | None = None,
        **kwargs,
    ) -> str:
        """Submit a standalone training script and optional helper modules."""
        return train(
            script=script,
            assignment=self.assignment,
            modules=modules,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            image=self.image,
            service_account_name=self.service_account_name,
            grant_tier=grant_tier or self.grant_tier,
            queue=queue or self.queue,
            **kwargs,
        )

    def run(self, code: str, timeout: int = 600, keep: bool = False) -> str:
        """Run inline Python code on a Cloud TPU chip."""
        return run(code=code, image=self.image, timeout=timeout, keep=keep)

