"""Run code on a TPU chip from a CPU notebook.

The student's notebook has no accelerator. ``run()`` submits a Kubernetes Job
that Kueue queues against the shared v5e pool, waits for a chip, streams logs,
and returns. The chip is held only while the job runs.

    import submit_tpu
    submit_tpu.smoke()
    submit_tpu.run('''
        import jax
        print(jax.devices())
    ''')
    submit_tpu.submit()  # assemble gcp_barebones.ipynb (or SUBMIT_TPU_NOTEBOOK)

Homework-specific data (Kaggle competition ids, dataset paths, cell lists) is
not hardcoded. Override with arguments or environment variables.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

QUEUE = os.environ.get("KUEUE_LOCAL_QUEUE", "tpu")
TPU_ACCELERATOR = os.environ.get("TPU_ACCELERATOR", "tpu-v5-lite-podslice")
TPU_TOPOLOGY = os.environ.get("TPU_TOPOLOGY", "1x1")

# Default notebook Hub copies into the student home directory.
DEFAULT_NOTEBOOK = os.environ.get("SUBMIT_TPU_NOTEBOOK", "gcp_barebones.ipynb")

# Unique substrings of code cells in that notebook. Override with
# SUBMIT_TPU_MARKERS as a JSON list, or pass markers= to assemble()/submit().
DEFAULT_MARKERS = [
    "DEVICE = jax.devices()[0]",
    "def tpu_main():",
    "Skipping in-kernel TPU work",
]

_FORWARD_ENV = (
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "KAGGLE_API_TOKEN",
    "WANDB_API_KEY",
    "WANDB_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "DATA_ROOT",
    "HW_DATA_ROOT",
)

# Optional kaggle.json on the TPU pod when the notebook process already has creds.
# No competition id is baked in.
_BOOT = r"""
import os, json
from pathlib import Path
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/cache"
_user = os.environ.get("KAGGLE_USERNAME") or ""
_key = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY") or ""
if _user and _key:
    os.environ["KAGGLE_USERNAME"] = _user
    os.environ["KAGGLE_KEY"] = _key
    os.environ["KAGGLE_API_TOKEN"] = _key
    _cred = Path("/tmp/.kaggle")
    _cred.mkdir(parents=True, exist_ok=True)
    (_cred / "kaggle.json").write_text(json.dumps({"username": _user, "key": _key}))
    os.chmod(_cred / "kaggle.json", 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(_cred)
"""

_NOTEBOOK_BOOT = r"""
import os
from pathlib import Path

os.chdir("/tmp")
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/cache"

import jax
print("TPU devices:", jax.devices(), flush=True)
assert jax.devices()[0].platform == "tpu", jax.devices()

USE_TPU = True
TPU_JOB = True
"""


def _as_text(out) -> str:
    """Decode pod logs to a real str with real newlines."""
    if out is None:
        return ""
    if isinstance(out, memoryview):
        out = out.tobytes()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out).decode("utf-8", errors="replace")
    if isinstance(out, str):
        if len(out) >= 2 and out[0] == "b" and out[1] in ("'", '"'):
            try:
                recovered = ast.literal_eval(out)
            except (ValueError, SyntaxError):
                return out
            if isinstance(recovered, (bytes, bytearray)):
                return bytes(recovered).decode("utf-8", errors="replace")
        return out
    decode = getattr(out, "decode", None)
    if callable(decode):
        try:
            return decode("utf-8", errors="replace")
        except (TypeError, ValueError):
            pass
    data = getattr(out, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="replace")
    return str(out)


def _collapse_progress(text: str) -> str:
    """Keep only the last carriage-return rewrite on each newline-delimited line."""
    parts = text.split("\n")
    collapsed = [(p.split("\r")[-1] if "\r" in p else p) for p in parts]
    out = "\n".join(collapsed)
    if text.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    return out


def _write_visible(text: str) -> None:
    if not text:
        return
    if not text.endswith("\n"):
        text += "\n"
    sys.stdout.write(text)
    sys.stdout.flush()


def _pod_logs(core: "client.CoreV1Api", pod: str, ns: str) -> str:
    try:
        return _as_text(core.read_namespaced_pod_log(pod, ns))
    except ApiException:
        return ""


def _wait_reason(pod) -> str:
    """Why the pod is not printing yet (image pull, node, etc.)."""
    for cs in list(pod.status.container_statuses or []) + list(
        pod.status.init_container_statuses or []
    ):
        waiting = getattr(cs.state, "waiting", None)
        if waiting is not None and waiting.reason:
            return waiting.reason
    for cond in pod.status.conditions or []:
        if cond.status == "False" and cond.reason:
            return cond.reason
    return pod.status.phase or "waiting"


def _tpu_image() -> str:
    """Image for the v5e Job: TPU_IMAGE, else this notebook pod's image."""
    image = os.environ.get("TPU_IMAGE", "").strip()
    if image:
        return image
    inferred = _image_of_this_pod()
    if inferred:
        print(f"[submit_tpu] TPU_IMAGE unset; using notebook image {inferred}", flush=True)
        return inferred
    raise RuntimeError(
        "TPU_IMAGE is not set and this process is not a notebook pod. "
        "Export TPU_IMAGE to the image `make image` pushed."
    )


def _image_of_this_pod() -> str | None:
    """Read the current pod's container image (student Role allows get pods)."""
    name = os.environ.get("HOSTNAME") or os.environ.get("POD_NAME") or ""
    if not name:
        return None
    try:
        pod = client.CoreV1Api().read_namespaced_pod(name, _namespace())
    except Exception:
        return None
    for container in pod.spec.containers or []:
        if container.image:
            return container.image
    return None


class JobInterrupted(RuntimeError):
    """The job went away before it could run.

    Raised when the job is deleted from the cluster before it finished.
    """


def _delete(batch: "client.BatchV1Api", name: str, ns: str) -> None:
    """Delete a Job, tolerating the case where it is already gone."""
    try:
        batch.delete_namespaced_job(name, ns, propagation_policy="Background")
    except ApiException as e:
        if e.status != 404:
            raise


def _namespace() -> str:
    """Get the namespace this notebook pod runs in."""
    path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return os.environ.get("POD_NAMESPACE", "default")


def _load() -> None:
    """Load in-cluster config, else local kubeconfig."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def _job_env(extra_env: dict | None) -> list:
    env = [
        client.V1EnvVar(name="JAX_PLATFORMS", value="tpu"),
        client.V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
        client.V1EnvVar(name="HOME", value="/tmp"),
        client.V1EnvVar(name="XDG_CACHE_HOME", value="/tmp/cache"),
    ]
    merged: dict[str, str] = {}
    for name in _FORWARD_ENV:
        val = os.environ.get(name)
        if val:
            merged[name] = val
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    if token:
        merged.setdefault("KAGGLE_KEY", token)
        merged.setdefault("KAGGLE_API_TOKEN", token)
    if extra_env:
        merged.update(
            {
                k: str(v)
                for k, v in extra_env.items()
                if v is not None and k not in ("HOME", "XDG_CACHE_HOME")
            }
        )
    for name, val in merged.items():
        env.append(client.V1EnvVar(name=name, value=val))
    return env


def _markers(markers: list[str] | None) -> list[str]:
    if markers is not None:
        return list(markers)
    raw = os.environ.get("SUBMIT_TPU_MARKERS", "").strip()
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError("SUBMIT_TPU_MARKERS must be a JSON list of strings")
        return parsed
    return list(DEFAULT_MARKERS)


def assemble(
    notebook: Path | str | None = None,
    markers: list[str] | None = None,
    boot: str | None = None,
) -> str:
    """Concatenate boot + marked notebook code cells into one TPU program."""
    nb_path = Path(notebook or DEFAULT_NOTEBOOK)
    assert nb_path.exists(), (
        f"missing {nb_path.resolve()} — save the notebook in this directory"
    )
    cells = json.loads(nb_path.read_text())["cells"]
    selected = []
    for marker in _markers(markers):
        hits = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if marker in src:
                hits.append(src)
        if len(hits) != 1:
            raise RuntimeError(
                f"marker {marker!r} matched {len(hits)} cells; save {nb_path.name} and retry"
            )
        selected.append(hits[0])
    parts = [boot if boot is not None else _NOTEBOOK_BOOT]
    parts.extend(selected)
    return "\n\n".join(parts)


def run(
    code: str,
    timeout: int = 28800,
    keep: bool = False,
    env: dict | None = None,
) -> str:
    """Run code on one v5e chip. Blocks until it finishes and returns stdout."""
    _load()
    ns = _namespace()
    name = f"tpu-{os.environ.get('JUPYTERHUB_USER', 'anon')}"
    name = name.replace("_", "-").lower()[:63]
    image = _tpu_image()

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
                    priority_class_name="student-tpu-job",
                    node_selector={
                        "cloud.google.com/gke-tpu-accelerator": TPU_ACCELERATOR,
                        "cloud.google.com/gke-tpu-topology": TPU_TOPOLOGY,
                    },
                    tolerations=[
                        client.V1Toleration(
                            key="google.com/tpu", operator="Exists", effect="NoSchedule"
                        )
                    ],
                    containers=[
                        client.V1Container(
                            name="hw",
                            image=image,
                            command=["python3", "-c", _BOOT + "\n" + code],
                            env=_job_env(env),
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
    print(
        "Waiting for a v5e chip. A cold node often takes 1–3 minutes.",
        flush=True,
    )

    t0 = time.time()
    admitted_at = None
    last_phase = None
    last_reason = None
    last_beat = 0.0
    printed = 0
    heartbeat_s = 15

    try:
        while time.time() - t0 < timeout:
            try:
                j = batch.read_namespaced_job(name, ns)
            except ApiException as e:
                if e.status == 404:
                    raise JobInterrupted(
                        f"{name} was deleted while waiting. Re-run the cell."
                    ) from None
                raise

            if admitted_at is None and not j.spec.suspend:
                admitted_at = time.time()
                print(f"  admitted after {admitted_at - t0:.0f}s in queue", flush=True)

            elapsed = time.time() - t0
            pods = core.list_namespaced_pod(
                ns, label_selector=f"job-name={name}"
            ).items
            if pods:
                pod = pods[0]
                phase = pod.status.phase
                reason = _wait_reason(pod)
                if phase != last_phase or reason != last_reason:
                    print(
                        f"  pod {phase} ({reason}) after {elapsed:.0f}s",
                        flush=True,
                    )
                    last_phase = phase
                    last_reason = reason

                log = _as_text(_pod_logs(core, pod.metadata.name, ns))
                if len(log) > printed:
                    _write_visible(_collapse_progress(log[printed:]))
                    printed = len(log)

                if phase in ("Succeeded", "Failed"):
                    print(f"  {phase} after {elapsed:.0f}s total", flush=True)
                    if not keep:
                        _delete(batch, name, ns)
                    if phase == "Failed":
                        raise RuntimeError(f"job failed:\n{_as_text(log)}")
                    return log
            else:
                reason = "no pod yet (queued or unsuspending)"

            if printed == 0 and elapsed - last_beat >= heartbeat_s:
                print(
                    f"  still working ({reason}) {elapsed:.0f}s — "
                    "node provision / image pull; not stuck",
                    flush=True,
                )
                last_beat = elapsed
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


def smoke(timeout: int = 600) -> str:
    """Run jax.devices() on one v5e chip. Logs stream into this process."""
    print("Submitting TPU smoke test; stdout streams below.", flush=True)
    return run(
        """
import jax
devs = jax.devices()
print("devices:", devs, flush=True)
print("platform:", devs[0].platform, flush=True)
assert devs[0].platform == "tpu", f"not on TPU: {devs}"
""",
        timeout=timeout,
    )


def submit(
    notebook: Path | str | None = None,
    timeout: int = 28800,
    markers: list[str] | None = None,
    env: dict | None = None,
) -> str:
    """Assemble marked cells from the notebook and run them on one v5e chip."""
    code = assemble(notebook=notebook, markers=markers)
    print(f"Submitting {len(code):,} chars to v5e. Logs stream below.", flush=True)
    return run(code, timeout=timeout, env=env)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args and args[0] == "--smoke":
        smoke()
        return
    submit()


if __name__ == "__main__":
    main()
