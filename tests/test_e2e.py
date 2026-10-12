"""Notebook client -> service -> TPU wrapper -> service -> notebook, over real HTTP.

The cluster is fake (no Kueue, no chip): the test "admits" the Job and runs the real
tpu_entrypoint.py as a subprocess with the env the Job would give it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

import submit_tpu
from tpu_submit import names

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "tpu-job" / "tpu_entrypoint.py"

TRAIN = """
import os, sys, pathlib
from helper import greeting
assert "TPU_JOB_TOKEN" not in os.environ, "job token leaked to student code"
assert not os.environ.get("KUBERNETES_SERVICE_HOST")
run = pathlib.Path(os.environ["OUTPUT_DIR"]) / "run"
ckpt = run / "checkpoints" / "last.msgpack"
if ckpt.exists():
    print("resumed from", ckpt.read_text())
ckpt.parent.mkdir(parents=True, exist_ok=True)
ckpt.write_text("epoch-" + os.environ["EPOCH"])
(run / "results.csv").write_text("index,label\\n0,hi\\n")
print(greeting(), "args", sys.argv[1:], "wandb", os.environ.get("WANDB_API_KEY"))
"""

SLOW = """
import os, pathlib, signal, sys, time
run = pathlib.Path(os.environ["OUTPUT_DIR"]) / "run"
run.mkdir(parents=True, exist_ok=True)
def save(*_):
    (run / "last.msgpack").write_text("saved on SIGTERM")
    print("saved", flush=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, save)
print("training", flush=True)
time.sleep(60)
"""


@pytest.fixture
def client(server, monkeypatch):
    monkeypatch.setattr(submit_tpu, "SERVICE_URL", server["url"])
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "tok-alice")
    return server


@pytest.fixture
def homework(tmp_path):
    root = tmp_path / "hw"
    root.mkdir()
    (root / "train.py").write_text(TRAIN)
    (root / "slow.py").write_text(SLOW)
    (root / "helper.py").write_text("def greeting():\n    return 'hello from helper'\n")
    (root / "kaggle.json").write_text('{"key": "never-sent"}')
    (root / "notes.ipynb").write_text("{}")
    return root


def job_env(server, job, work: Path, extra=None) -> dict:
    cluster = server["cluster"]
    container = cluster.jobs[job["name"]]["spec"]["template"]["spec"]["containers"][0]
    env = {"PATH": os.environ["PATH"], "HOSTNAME": "test-node", "KUBERNETES_SERVICE_HOST": "10.0.0.1"}
    env.update({e["name"]: e["value"] for e in container["env"]})
    import base64
    env.update({k: base64.b64decode(v).decode() for k, v in cluster.secrets[job["name"]]["data"].items()})
    env.update({"TPU_SUBMIT_URL": server["url"], "TPU_WORK_DIR": str(work), "TPU_DATASET_URI": ""})
    env.pop("JAX_PLATFORMS", None)
    env.update(extra or {})
    return env


def run_wrapper(server, job, work: Path, extra=None, timeout=60) -> subprocess.CompletedProcess:
    args = server["cluster"].jobs[job["name"]]["spec"]["template"]["spec"]["containers"][0]["args"]
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *args], env=job_env(server, job, work, extra),
        capture_output=True, text=True, timeout=timeout,
    )


def test_submit_run_watch_pull_resume(client, homework, tmp_path, capsys, monkeypatch):
    cluster = client["cluster"]
    job = submit_tpu.submit(str(homework / "train.py"), args=["--config", "config.yaml"],
                            env={"WANDB_API_KEY": "wb-key", "EPOCH": "1"})
    assert job.info["state"] == "queued"

    snapshot = client["core"].store.get_bytes(names.job_prefix("alice", job.id) + "snapshot.tar.gz")
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(snapshot)) as tar:
        assert sorted(tar.getnames()) == ["helper.py", "slow.py", "train.py"]

    with pytest.raises(submit_tpu.SubmitError, match="already have job"):
        submit_tpu.submit(str(homework / "train.py"))

    pod = cluster.admit(job.info["name"])
    proc = run_wrapper(client, job.info, tmp_path / "work1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "hello from helper args ['--config', 'config.yaml'] wandb wb-key" in proc.stdout

    cluster.logs[pod] = proc.stdout
    cluster.finish(job.info["name"])
    capsys.readouterr()
    final = submit_tpu.watch(job, poll=0.01)
    assert final["state"] == "succeeded"
    assert "hello from helper" in capsys.readouterr().out

    dest = submit_tpu.pull(job, dest=tmp_path / "pulled")
    assert (dest / "expts/run/checkpoints/last.msgpack").read_text() == "epoch-1"
    assert (dest / "expts/run/results.csv").exists()
    log = (dest / "job.log").read_text()
    assert "running: python train.py --config config.yaml" in log and "exited with code 0" in log

    # resume_from: the new job starts from the old job's expts/.
    second = submit_tpu.submit(str(homework / "train.py"), env={"EPOCH": "2"}, resume_from=job)
    cluster.admit(second.info["name"])
    proc = run_wrapper(client, second.info, tmp_path / "work2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "resumed from epoch-1" in proc.stdout
    own = {f["path"] for f in submit_tpu.files(second)}
    assert "expts/run/checkpoints/last.msgpack" in own

    # A disruption: a new pod for the same Job resumes from that Job's own upload and
    # continues its log.
    proc = run_wrapper(client, second.info, tmp_path / "work3", extra={"EPOCH": "3"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "resumed from epoch-2" in proc.stdout
    log = submit_tpu.logs(second)
    assert log == ""  # the fake pod has no live log; stored log is used once the Job is gone
    cluster.finish(second.info["name"])
    del cluster.jobs[second.info["name"]]
    stored = submit_tpu.logs(second)
    assert "restarted on a new TPU node" in stored and stored.count("running: python train.py") == 2


def test_failing_script_still_uploads(client, homework, tmp_path):
    (homework / "train.py").write_text(
        "import os, pathlib\n"
        "p = pathlib.Path(os.environ['OUTPUT_DIR']) / 'run'; p.mkdir(parents=True)\n"
        "(p / 'partial.txt').write_text('x')\n"
        "raise SystemExit(3)\n"
    )
    job = submit_tpu.submit(str(homework / "train.py"))
    client["cluster"].admit(job.info["name"])
    proc = run_wrapper(client, job.info, tmp_path / "work")
    assert proc.returncode == 3
    paths = {f["path"] for f in submit_tpu.files(job)}
    assert {"job.log", "expts/run/partial.txt"} <= paths


def test_sigterm_lets_script_save_then_uploads(client, homework, tmp_path):
    job = submit_tpu.submit(str(homework / "slow.py"))
    client["cluster"].admit(job.info["name"])
    args = client["cluster"].jobs[job.info["name"]]["spec"]["template"]["spec"]["containers"][0]["args"]
    proc = subprocess.Popen(
        [sys.executable, str(ENTRYPOINT), *args], env=job_env(client, job.info, tmp_path / "work"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if (tmp_path / "work" / "job.log").exists() and "training" in (tmp_path / "work" / "job.log").read_text():
            break
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert "received SIGTERM" in out and "saved" in out
    files = {f["path"] for f in submit_tpu.files(job)}
    assert "expts/run/last.msgpack" in files


def test_http_access_rules(client, homework, tmp_path):
    url = client["url"]
    job = submit_tpu.submit(str(homework / "train.py"))

    assert requests.get(f"{url}/api/jobs").status_code == 401
    assert requests.get(f"{url}/api/jobs", headers={"Authorization": "token nope"}).status_code == 403

    bob = {"Authorization": "token tok-bob"}
    assert requests.get(f"{url}/api/jobs/{job.id}", headers=bob).status_code == 404
    assert requests.get(f"{url}/api/jobs/{job.id}/files/job.log", headers=bob).status_code == 404
    assert requests.delete(f"{url}/api/jobs/{job.id}", headers=bob).status_code == 404
    assert requests.get(f"{url}/api/jobs", headers=bob).json() == {"jobs": []}
    assert requests.get(f"{url}/api/jobs?user=alice", headers=bob).status_code == 403
    assert requests.get(f"{url}/api/jobs?all=1", headers=bob).status_code == 403

    admin = {"Authorization": "token tok-admin"}
    assert requests.get(f"{url}/api/jobs/{job.id}?user=alice", headers=admin).json()["id"] == job.id
    assert [j["id"] for j in requests.get(f"{url}/api/jobs?all=1", headers=admin).json()["jobs"]] == [job.id]

    # A job token only works for its own job.
    token = client["cluster"].token(job.info["name"])
    good = {"X-TPU-Job": job.info["name"], "Authorization": f"Bearer {token}"}
    assert requests.get(f"{url}/internal/snapshot", headers=good).status_code == 200
    assert requests.get(f"{url}/internal/snapshot", headers={**good, "Authorization": "Bearer x"}).status_code == 401
    assert requests.get(f"{url}/internal/snapshot", headers={"Authorization": "token tok-alice"}).status_code == 401
    r = requests.put(f"{url}/internal/files/..%2F..%2Fbob%2Fx", headers=good, data=b"x")
    assert r.status_code in (400, 404)
    # An upload cut off mid-body (pod killed during a sync) is not committed.
    import socket
    from urllib.parse import urlparse
    u = urlparse(url)
    with socket.create_connection((u.hostname, u.port)) as s:
        s.sendall(
            f"PUT /internal/files/expts/x.bin HTTP/1.1\r\nHost: x\r\nX-TPU-Job: {job.info['name']}\r\n"
            f"Authorization: Bearer {token}\r\nContent-Length: 1000\r\n\r\n".encode() + b"short"
        )
        time.sleep(0.3)
    time.sleep(0.3)
    assert "expts/x.bin" not in {f["path"] for f in submit_tpu.files(job)}
    r = requests.put(f"{url}/internal/files/expts/x.bin", headers=good, data=b"complete")
    assert r.status_code == 200
    assert {"path": "expts/x.bin", "size": 8} in submit_tpu.files(job)


def test_snapshot_size_cap_on_server(client, tmp_path, settings, monkeypatch):
    big = tmp_path / "big.tar.gz"
    big.write_bytes(os.urandom(settings.max_snapshot_mb * 1024 * 1024 + 10))
    with big.open("rb") as fh:
        r = requests.put(f"{client['url']}/api/uploads", data=fh,
                         headers={"Authorization": "token tok-alice"})
    assert r.status_code == 413


def test_cancel_from_client(client, homework):
    job = submit_tpu.submit(str(homework / "train.py"))
    assert submit_tpu.cancel()["state"] == "cancelled"
    assert job.info["name"] not in client["cluster"].jobs
    jobs = submit_tpu.status()
    assert jobs[0]["state"] == "cancelled"
