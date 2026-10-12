"""Per-student rules: validation, one active job, budget, ownership, job tokens."""

from __future__ import annotations

import json

import pytest

from tpu_submit import names, state
from tpu_submit.core import ApiError


def submit(core, make_snapshot, user="alice", **spec):
    spec.setdefault("script", "train.py")
    req = core.validate_request(user, spec)
    return core.submit(user, req, make_snapshot({"train.py": "print('hi')", "lib/x.py": ""}))


def test_submit_creates_job_secret_and_meta(core, cluster, store, make_snapshot):
    job = submit(core, make_snapshot, args=["--epochs", "1"], env={"WANDB_API_KEY": "wandb-secret-123"})
    assert job["state"] == state.QUEUED
    name = job["name"]
    assert name in cluster.jobs and name in cluster.secrets
    assert cluster.secrets[name]["metadata"]["ownerReferences"][0]["uid"] == cluster.jobs[name]["metadata"]["uid"]
    prefix = names.job_prefix("alice", job["id"])
    meta = json.loads(store.get_bytes(prefix + "meta.json"))
    assert meta["env_keys"] == ["WANDB_API_KEY"]
    assert "wandb-secret-123" not in json.dumps(meta)
    assert store.get_bytes(prefix + "snapshot.tar.gz")
    args = cluster.jobs[name]["spec"]["template"]["spec"]["containers"][0]["args"]
    assert args == ["train.py", "--epochs", "1"]


def test_default_dataset_is_resolved(core, cluster, make_snapshot):
    job = submit(core, make_snapshot)
    env = {e["name"]: e["value"] for e in cluster.jobs[job["name"]]["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["TPU_DATASET_URI"] == "gs://data/hw3p2.tar"
    assert job["dataset"] == "hw3p2"


@pytest.mark.parametrize("spec, message", [
    ({"script": "../train.py"}, "relative path"),
    ({"script": "/abs/train.py"}, "relative path"),
    ({"script": "train.sh"}, ".py"),
    ({"script": "train.py", "env": {"TPU_JOB_TOKEN": "x"}}, "cannot be overridden"),
    ({"script": "train.py", "env": {"LD_PRELOAD": "x"}}, "cannot be overridden"),
    ({"script": "train.py", "env": {"JAX_PLATFORMS": "cpu"}}, "cannot be overridden"),
    ({"script": "train.py", "env": {"lower": "x"}}, "must look like"),
    ({"script": "train.py", "args": "not a list"}, "args"),
    ({"script": "train.py", "dataset": "imagenet"}, "Unknown dataset"),
])
def test_validation(core, spec, message):
    with pytest.raises(ApiError) as err:
        core.validate_request("alice", spec)
    assert err.value.status == 400
    assert message in err.value.message


def test_one_active_job(core, cluster, make_snapshot):
    first = submit(core, make_snapshot)
    with pytest.raises(ApiError) as err:
        submit(core, make_snapshot)
    assert err.value.status == 409 and first["id"] in err.value.message
    # Another student is unaffected.
    submit(core, make_snapshot, user="bob")
    cluster.admit(first["name"])
    with pytest.raises(ApiError):
        submit(core, make_snapshot)
    cluster.finish(first["name"])
    submit(core, make_snapshot)


def test_budget(core, cluster, store, make_snapshot, settings):
    from dataclasses import replace

    core.settings = replace(settings, chip_hour_budget=1.0)
    job = submit(core, make_snapshot)
    cluster.admit(job["name"])
    cluster.finish(job["name"])
    core.status("alice", job["id"])
    key = names.job_prefix("alice", job["id"]) + "meta.json"
    meta = json.loads(store.get_bytes(key))
    meta["started_at"], meta["finished_at"] = "2026-10-01T00:00:00Z", "2026-10-01T01:30:00Z"
    store.put_bytes(key, json.dumps(meta).encode())
    assert core.usage_hours("alice") == pytest.approx(1.5)
    with pytest.raises(ApiError) as err:
        submit(core, make_snapshot)
    assert err.value.status == 403 and "1.50 of your 1" in err.value.message


def test_students_cannot_see_each_others_jobs(core, make_snapshot):
    job = submit(core, make_snapshot)
    for action in (core.status, core.files, core.cancel):
        with pytest.raises(ApiError) as err:
            action("bob", job["id"])
        assert err.value.status == 404
    with pytest.raises(ApiError):
        core.validate_request("bob", {"script": "train.py", "resume_from": job["id"]})
    assert core.list_jobs("bob") == []


def test_snapshot_must_contain_script_and_be_safe(core, make_snapshot):
    req = core.validate_request("alice", {"script": "train.py"})
    with pytest.raises(ApiError, match="not in the snapshot"):
        core.submit("alice", req, make_snapshot({"other.py": ""}))
    with pytest.raises(ApiError, match="unsafe path"):
        core.submit("alice", req, make_snapshot({"train.py": "", "../escape.py": ""}))


def test_failed_secret_removes_job(core, cluster, make_snapshot):
    cluster.fail_secret = True
    with pytest.raises(ApiError) as err:
        submit(core, make_snapshot)
    assert err.value.status == 500
    assert cluster.jobs == {}
    (meta,) = core.list_metas("alice")
    assert meta["state"] == state.FAILED
    submit_again = core.validate_request("alice", {"script": "train.py"})
    cluster.fail_secret = False
    core.check_can_submit("alice")
    assert submit_again


def test_status_follows_the_job(core, cluster, make_snapshot):
    job = submit(core, make_snapshot)
    assert core.status("alice", job["id"])["position"] == 1
    cluster.admit(job["name"])
    running = core.status("alice", job["id"])
    assert running["state"] == state.RUNNING and running["started_at"]
    cluster.finish(job["name"], ok=False, reason="DeadlineExceeded")
    assert core.status("alice", job["id"])["state"] == state.TIMED_OUT


def test_cancel(core, cluster, make_snapshot):
    job = submit(core, make_snapshot)
    out = core.cancel("alice", job["id"])
    assert out["state"] == state.CANCELLED
    assert job["name"] not in cluster.jobs
    assert core.cancel("alice", job["id"])["state"] == state.CANCELLED


def test_reconcile_records_state_and_lost_jobs(core, cluster, store, make_snapshot):
    job = submit(core, make_snapshot)
    cluster.admit(job["name"])
    core.reconcile()
    assert core.get_meta("alice", job["id"])["state"] == state.RUNNING
    del cluster.jobs[job["name"]]
    core.reconcile()
    meta = core.get_meta("alice", job["id"])
    assert meta["state"] == state.LOST and meta["finished_at"]


def test_logs_live_then_stored(core, cluster, store, make_snapshot):
    job = submit(core, make_snapshot)
    pod = cluster.admit(job["name"])
    cluster.logs[pod] = "epoch 1\n"
    first = core.logs("alice", job["id"])
    assert first["data"] == "epoch 1\n" and first["pod"] == pod
    cluster.logs[pod] += "epoch 2\n"
    second = core.logs("alice", job["id"], first["offset"], pod)
    assert second["data"] == "epoch 2\n"
    new_pod = cluster.admit(job["name"], pod_suffix="b")
    cluster.logs[new_pod] = "restarted\n"
    third = core.logs("alice", job["id"], second["offset"], pod)
    assert third["pod"] == new_pod and third["data"] == "restarted\n"
    cluster.finish(job["name"])
    store.put_bytes(names.job_prefix("alice", job["id"]) + "files/job.log", b"full log\n")
    del cluster.jobs[job["name"]]
    stored = core.logs("alice", job["id"], 0, None)
    assert stored["source"] == "stored" and stored["data"] == "full log\n" and stored["done"]


def test_job_token_auth(core, cluster, make_snapshot):
    job = submit(core, make_snapshot)
    token = cluster.token(job["name"])
    assert core.authenticate_job(job["name"], token) == ("alice", job["id"])
    for name, tok in ((job["name"], "wrong"), ("tpu-nobody-x", token), (job["name"], "")):
        with pytest.raises(ApiError) as err:
            core.authenticate_job(name, tok)
        assert err.value.status == 401


def test_inputs_prefer_own_checkpoints_then_resume_from(core, cluster, store, make_snapshot):
    first = submit(core, make_snapshot)
    p1 = names.job_prefix("alice", first["id"]) + "files/"
    store.put_bytes(p1 + "expts/run/checkpoints/last.msgpack", b"one")
    store.put_bytes(p1 + "job.log", b"log")
    cluster.finish(first["name"])

    second = submit(core, make_snapshot, resume_from=first["id"])
    files = core.inputs("alice", second["id"])["files"]
    assert files == [{"path": "expts/run/checkpoints/last.msgpack", "size": 3, "source": first["id"]}]
    fh, size = core.open_input("alice", second["id"], files[0]["path"], first["id"])
    assert fh.read() == b"one"
    fh.close()

    p2 = names.job_prefix("alice", second["id"]) + "files/"
    store.put_bytes(p2 + "expts/run/checkpoints/last.msgpack", b"two")
    files = core.inputs("alice", second["id"])["files"]
    assert {f["source"] for f in files} == {second["id"]}

    with pytest.raises(ApiError) as err:
        core.open_input("alice", first["id"], "expts/run/checkpoints/last.msgpack", second["id"])
    assert err.value.status == 403


def test_upload_writer_commits_only_complete_files(core, store, make_snapshot):
    job = submit(core, make_snapshot)
    w = core.open_upload("alice", job["id"], "expts/a.bin")
    w.write(b"partial")
    w.abort()
    assert core.files("alice", job["id"]) == []
    w = core.open_upload("alice", job["id"], "expts/a.bin")
    w.write(b"done")
    w.commit()
    assert core.files("alice", job["id"]) == [{"path": "expts/a.bin", "size": 4}]
    with pytest.raises(ApiError):
        core.open_upload("alice", job["id"], "../../bob/x")
