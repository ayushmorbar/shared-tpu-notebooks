"""Shared fixtures. Nothing here talks to Kubernetes, GCS, or a real JupyterHub."""

from __future__ import annotations

import asyncio
import copy
import sys
import threading
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("services/tpu-submit", "notebooks", "docker/tpu-job"):
    sys.path.insert(0, str(ROOT / sub))

from tpu_submit import names  # noqa: E402
from tpu_submit.config import Settings  # noqa: E402
from tpu_submit.core import Core  # noqa: E402
from tpu_submit.store import LocalStore  # noqa: E402


def _now() -> str:
    from tpu_submit.state import now_iso

    return now_iso()


def _matches(labels: dict, selector: str) -> bool:
    for term in filter(None, selector.split(",")):
        key, _, value = term.partition("=")
        if labels.get(key) != value:
            return False
    return True


class FakeCluster:
    """Just enough of the Kubernetes API for the service, plus helpers to move Jobs along."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.secrets: dict[str, dict] = {}
        self.pods: dict[str, dict] = {}
        self.logs: dict[str, str] = {}
        self.fail_secret = False

    # --- the Cluster protocol
    def create_job(self, manifest: dict) -> dict:
        job = copy.deepcopy(manifest)
        name = job["metadata"]["name"]
        if name in self.jobs:
            raise RuntimeError("AlreadyExists")
        job["metadata"]["uid"] = str(uuid.uuid4())
        job["metadata"]["creationTimestamp"] = _now()
        job["status"] = {}
        self.jobs[name] = job
        return copy.deepcopy(job)

    def create_secret(self, manifest: dict) -> None:
        if self.fail_secret:
            raise RuntimeError("secret quota exceeded")
        self.secrets[manifest["metadata"]["name"]] = copy.deepcopy(manifest)

    def get_job(self, name: str) -> dict | None:
        job = self.jobs.get(name)
        return copy.deepcopy(job) if job else None

    def list_jobs(self, selector: str) -> list[dict]:
        return [copy.deepcopy(j) for j in self.jobs.values() if _matches(j["metadata"]["labels"], selector)]

    def delete_job(self, name: str) -> None:
        self.jobs.pop(name, None)
        self.secrets.pop(name, None)
        for pod in [p for p, v in self.pods.items() if v["metadata"]["labels"]["job-name"] == name]:
            del self.pods[pod]

    def list_pods(self, selector: str) -> list[dict]:
        return [copy.deepcopy(p) for p in self.pods.values() if _matches(p["metadata"]["labels"], selector)]

    def pod_log(self, pod: str) -> str:
        return self.logs.get(pod, "")

    # --- test helpers
    def token(self, name: str) -> str:
        import base64

        return base64.b64decode(self.secrets[name]["data"]["TPU_JOB_TOKEN"]).decode()

    def admit(self, name: str, pod_suffix: str = "a") -> str:
        job = self.jobs[name]
        job["spec"]["suspend"] = False
        job["status"]["startTime"] = job["status"].get("startTime") or _now()
        pod = f"{name}-{pod_suffix}"
        labels = dict(job["spec"]["template"]["metadata"]["labels"])
        labels.update({"job-name": name, "batch.kubernetes.io/job-name": name})
        self.pods[pod] = {
            "metadata": {"name": pod, "labels": labels, "creationTimestamp": _now()},
            "status": {"phase": "Running"},
        }
        return pod

    def finish(self, name: str, ok: bool = True, reason: str = "BackoffLimitExceeded") -> None:
        job = self.jobs[name]
        cond = {"type": "Complete" if ok else "Failed", "status": "True", "lastTransitionTime": _now()}
        if not ok:
            cond["reason"] = reason
            cond["message"] = f"Job failed: {reason}"
        else:
            job["status"]["completionTime"] = _now()
        job["status"]["conditions"] = [cond]
        for pod in self.pods.values():
            if pod["metadata"]["labels"]["job-name"] == name:
                pod["status"]["phase"] = "Succeeded" if ok else "Failed"


class FakeAuth:
    def __init__(self, tokens: dict[str, dict]):
        self.tokens = tokens

    def user_for_token(self, token: str):
        return self.tokens.get(token)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        namespace="course",
        public_url="http://tpu-submit.course.svc:8080",
        job_image="registry/tpu-job:test",
        store_dir=str(tmp_path / "bucket"),
        datasets={"hw3p2": "gs://data/hw3p2.tar"},
        default_dataset="hw3p2",
        spool_dir=str(tmp_path / "spool"),
        sync_interval=1,
    )


@pytest.fixture
def cluster() -> FakeCluster:
    return FakeCluster()


@pytest.fixture
def store(settings) -> LocalStore:
    return LocalStore(settings.store_dir)


@pytest.fixture
def core(settings, store, cluster) -> Core:
    return Core(settings, store, cluster)


@pytest.fixture
def make_snapshot(tmp_path):
    """Build a snapshot .tar.gz from {relpath: content} and return its path."""
    import tarfile
    import io

    counter = [0]

    def make(files: dict[str, str]) -> str:
        counter[0] += 1
        path = tmp_path / f"snap{counter[0]}.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for rel, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(rel)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return str(path)

    return make


@pytest.fixture
def server(settings, store, cluster):
    """The real tornado app on a random port, in a background thread."""
    from tornado.httpserver import HTTPServer
    from tornado.testing import bind_unused_port

    from tpu_submit.app import make_app

    auth = FakeAuth({
        "tok-alice": {"name": "alice", "admin": False},
        "tok-bob": {"name": "bob", "admin": False},
        "tok-admin": {"name": "instructor", "admin": True},
    })
    core = Core(settings, store, cluster)
    sock, port = bind_unused_port()
    ready = threading.Event()
    state = {}

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        srv = HTTPServer(make_app(core, auth, settings), max_buffer_size=64 * 1024 * 1024)
        srv.add_sockets([sock])
        state["loop"] = loop
        ready.set()
        loop.run_forever()
        srv.stop()
        loop.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    ready.wait(5)
    yield {"url": f"http://127.0.0.1:{port}", "core": core, "cluster": cluster}
    state["loop"].call_soon_threadsafe(state["loop"].stop)
    t.join(5)


__all__ = ["FakeCluster", "FakeAuth", "names"]
