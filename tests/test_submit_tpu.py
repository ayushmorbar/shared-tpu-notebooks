"""Offline tests for notebooks/submit_tpu.py and gcp_barebones.ipynb.

kubernetes is stubbed so this runs without a cluster or the python client.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NB = NOTEBOOKS / "gcp_barebones.ipynb"
HELPER = NOTEBOOKS / "submit_tpu.py"


def _install_kubernetes_stub() -> None:
    if "kubernetes" in sys.modules:
        return
    kube = types.ModuleType("kubernetes")
    client = types.ModuleType("kubernetes.client")
    config = types.ModuleType("kubernetes.config")
    exceptions = types.ModuleType("kubernetes.client.exceptions")

    class ApiException(Exception):
        def __init__(self, status=0):
            super().__init__(status)
            self.status = status

    class _Dummy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    client.exceptions = exceptions
    exceptions.ApiException = ApiException
    client.ApiException = ApiException
    client.V1Job = _Dummy
    client.V1ObjectMeta = _Dummy
    client.V1JobSpec = _Dummy
    client.V1PodTemplateSpec = _Dummy
    client.V1PodSpec = _Dummy
    client.V1Toleration = _Dummy
    client.V1Container = _Dummy
    client.V1EnvVar = _Dummy
    client.V1ResourceRequirements = _Dummy
    client.V1SecurityContext = _Dummy
    client.V1Capabilities = _Dummy
    client.BatchV1Api = _Dummy
    client.CoreV1Api = _Dummy
    config.ConfigException = type("ConfigException", (Exception,), {})
    config.load_incluster_config = lambda: None
    config.load_kube_config = lambda: None
    kube.client = client
    kube.config = config
    sys.modules["kubernetes"] = kube
    sys.modules["kubernetes.client"] = client
    sys.modules["kubernetes.config"] = config
    sys.modules["kubernetes.client.exceptions"] = exceptions


def _load_helper():
    _install_kubernetes_stub()
    if str(NOTEBOOKS) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS))
    import submit_tpu

    return submit_tpu


def test_helper_and_notebook_parse():
    ast.parse(HELPER.read_text())
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    sources = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        sources.append(src)
        if cell["cell_type"] == "code":
            ast.parse(src)
    blob = "\n".join(sources)
    assert "!pip" not in blob
    assert "/content/" not in blob
    assert "/kaggle/" not in blob
    assert "/ocean/projects" not in blob
    assert "submit_tpu" in blob
    assert "KAGGLE_API_TOKEN" not in blob
    assert "WANDB_API_KEY" not in blob


def test_assemble_default_notebook():
    submit_tpu = _load_helper()
    code = submit_tpu.assemble(NB)
    ast.parse(code)
    assert "TPU_JOB = True" in code
    assert "def tpu_main():" in code
    assert "Skipping in-kernel TPU work" in code
    compile(code, "<assembled>", "exec")


def test_as_text_and_collapse():
    submit_tpu = _load_helper()
    assert submit_tpu._as_text(b"hello\n") == "hello\n"
    raw = "Train: 1%\rTrain: 2%\rTrain: 100%\ndone\n"
    out = submit_tpu._collapse_progress(raw)
    assert "Train: 1%" not in out
    assert "Train: 100%" in out
    assert "done" in out


def test_markers_env_override(monkeypatch=None):
    submit_tpu = _load_helper()
    markers = submit_tpu._markers(None)
    assert markers == submit_tpu.DEFAULT_MARKERS
    custom = ["only-this"]
    assert submit_tpu._markers(custom) == custom


if __name__ == "__main__":
    test_helper_and_notebook_parse()
    test_assemble_default_notebook()
    test_as_text_and_collapse()
    test_markers_env_override()
    print("ok")
