"""The fixed Job template: what students can and cannot influence."""

from __future__ import annotations

import base64
import re

from tpu_submit import names
from tpu_submit.jobspec import build_job, build_secret

DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def job_for(settings, user="alice", script="train.py", args=("--config", "config.yaml")):
    return build_job(settings, user=user, job_id="20261010-120000-abcd", script=script,
                     args=list(args), dataset_uri="gs://data/hw3p2.tar", token_sha="f" * 64)


def test_queue_label_and_suspended(settings):
    job = job_for(settings)
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "tpu"
    assert job["spec"]["suspend"] is True


def test_one_chip_on_v5e(settings):
    pod = job_for(settings)["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    assert container["resources"]["requests"]["google.com/tpu"] == "1"
    assert container["resources"]["limits"]["google.com/tpu"] == "1"
    assert pod["nodeSelector"] == {
        "cloud.google.com/gke-tpu-accelerator": "tpu-v5-lite-podslice",
        "cloud.google.com/gke-tpu-topology": "1x1",
    }
    assert pod["priorityClassName"] == "student-tpu-job"


def test_deadline_ttl_and_failure_policy(settings):
    spec = job_for(settings)["spec"]
    assert spec["activeDeadlineSeconds"] == 8 * 3600
    assert spec["ttlSecondsAfterFinished"] == settings.job_ttl_seconds
    rules = spec["podFailurePolicy"]["rules"]
    assert rules[0] == {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget"}]}
    assert rules[1]["action"] == "FailJob"


def test_identity_and_no_kubernetes_token(settings):
    pod = job_for(settings)["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "tpu-job"
    assert pod["automountServiceAccountToken"] is False
    sc = pod["containers"][0]["securityContext"]
    assert sc["runAsNonRoot"] and sc["capabilities"] == {"drop": ["ALL"]}
    assert not sc["allowPrivilegeEscalation"]


def test_script_and_args_are_plain_arguments(settings):
    container = job_for(settings, script="train.py", args=["--x", "$(HOME)"])["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == settings.job_image
    assert container["command"] == ["python3", "/opt/tpu/tpu_entrypoint.py"]
    assert container["args"] == ["train.py", "--x", "$(HOME)"]


def test_no_secrets_in_job_spec(settings):
    job = job_for(settings)
    container = job["spec"]["template"]["spec"]["containers"][0]
    env_names = {e["name"] for e in container["env"]}
    assert "TPU_JOB_TOKEN" not in env_names
    assert container["envFrom"] == [{"secretRef": {"name": job["metadata"]["name"]}}]
    assert job["metadata"]["annotations"][names.ANN_TOKEN_SHA] == "f" * 64


def test_secret_is_owned_by_job(settings):
    secret = build_secret(settings, name="tpu-alice-x", token="tok", env={"WANDB_API_KEY": "k"}, owner_uid="uid-1")
    (owner,) = secret["metadata"]["ownerReferences"]
    assert owner == {"apiVersion": "batch/v1", "kind": "Job", "name": "tpu-alice-x", "uid": "uid-1"}
    data = {k: base64.b64decode(v).decode() for k, v in secret["data"].items()}
    assert data == {"TPU_JOB_TOKEN": "tok", "WANDB_API_KEY": "k"}


def test_names_are_valid_for_awkward_usernames(settings):
    for user in ("alice", "a.b+c", "UPPER_case", "x" * 80, "..", "日本"):
        job = job_for(settings, user=user)
        name = job["metadata"]["name"]
        assert len(name) <= 63 and DNS_LABEL.match(name), name
        assert re.match(r"^[a-f0-9]{20}$", job["metadata"]["labels"][names.LABEL_USER_HASH])


def test_each_user_has_their_own_prefix():
    assert names.job_prefix("alice", "J") == "users/alice/jobs/J/"
    assert names.job_prefix("a.b+c", "J") == "users/a.b+c/jobs/J/"
    assert names.user_dir("..").startswith("h-")
    assert names.user_dir("../bob").startswith("h-")
    assert names.user_hash("alice") != names.user_hash("alice.")


def test_clean_relpath_rejects_escapes():
    for bad in ("", "/etc/passwd", "../x", "a/../../x", "a/..", "a\\b", "a\x00b"):
        assert names.clean_relpath(bad) is None, bad
    assert names.clean_relpath("a//b") == "a/b"
    assert names.clean_relpath("expts/run/checkpoints/last.msgpack") == "expts/run/checkpoints/last.msgpack"
