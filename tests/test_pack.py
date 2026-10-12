"""What submit_tpu.submit() sends, and what it refuses to send."""

from __future__ import annotations

import os
import tarfile

import pytest

import submit_tpu


def touch(root, rel, content="x"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def homework(tmp_path):
    root = tmp_path / "hw3"
    for rel in [
        "train.py", "config.yaml", "hw3lib/__init__.py", "hw3lib/data/asr_dataset.py",
        "hw3lib/data/tokenizer_jsons/tokenizer_char.json", "tests/test_x.py",
    ]:
        touch(root, rel)
    for rel in [
        ".git/config", "hw3lib/__pycache__/x.cpython-312.pyc", ".ipynb_checkpoints/a.ipynb",
        "expts/run/checkpoints/last.msgpack", "checkpoints/old.pkl", "wandb/run/file",
        "tpu_runs/2026/job.log", "hw3_data/hw3p2_data/train/x.npy",
        "hw3_data_subset/hw3p2_data/x.npy", "hw3p2-data/hw3p2_data/x.npy", "data/big.npy",
        "kaggle.json", ".env", ".netrc", "secret.key", "HW3P2.ipynb", "nested/.env.local",
        ".cache/pip/x",
    ]:
        touch(root, rel)
    return root


def test_sends_code_and_keeps_nested_data_package(homework):
    sent = {rel.as_posix() for rel, _ in submit_tpu.collect_files(homework)}
    assert sent == {
        "train.py", "config.yaml", "hw3lib/__init__.py", "hw3lib/data/asr_dataset.py",
        "hw3lib/data/tokenizer_jsons/tokenizer_char.json", "tests/test_x.py",
    }


def test_never_sends_credentials_or_notebooks(homework):
    sent = {rel.name for rel, _ in submit_tpu.collect_files(homework)}
    for name in ("kaggle.json", ".env", ".netrc", "secret.key", "HW3P2.ipynb", ".env.local"):
        assert name not in sent


def test_skips_symlinks(homework, tmp_path):
    outside = touch(tmp_path, "outside/secret.txt")
    os.symlink(outside, homework / "link.txt")
    os.symlink(outside.parent, homework / "linkdir")
    sent = {rel.as_posix() for rel, _ in submit_tpu.collect_files(homework)}
    assert "link.txt" not in sent and not any(s.startswith("linkdir") for s in sent)


def test_pack_writes_relative_names(homework, tmp_path):
    dest = tmp_path / "snap.tar.gz"
    submit_tpu.pack(homework, dest)
    with tarfile.open(dest) as tar:
        names = tar.getnames()
        assert "train.py" in names
        assert all(not n.startswith(("/", "..", "./")) for n in names)
        assert all(m.uid == 1000 and m.uname == "" for m in tar.getmembers())


def test_size_cap_names_the_big_files(homework, tmp_path):
    (homework / "features.npy").write_bytes(b"\0" * (2 * 1024 * 1024))
    with pytest.raises(submit_tpu.SubmitError) as err:
        submit_tpu.pack(homework, tmp_path / "snap.tar.gz", max_mb=1)
    assert "features.npy" in str(err.value)
    assert "limit is 1 MB" in str(err.value)


def test_submit_refuses_script_outside_root(homework, tmp_path, monkeypatch):
    monkeypatch.setattr(submit_tpu, "SERVICE_URL", "http://unused")
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "t")
    other = touch(tmp_path, "elsewhere/run.py")
    with pytest.raises(submit_tpu.SubmitError, match="not inside"):
        submit_tpu.submit(str(other), root=str(homework))


def test_submit_without_service_url_explains(homework, monkeypatch):
    monkeypatch.setattr(submit_tpu, "SERVICE_URL", "")
    with pytest.raises(submit_tpu.SubmitError, match="TPU_SUBMIT_URL"):
        submit_tpu.submit(str(homework / "train.py"))
