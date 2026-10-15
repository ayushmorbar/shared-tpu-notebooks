"""Wrapper that runs a student's script on a TPU node.

    python3 tpu_entrypoint.py <script> [args...]

1. Download the snapshot the student submitted and unpack it into /work/code.
2. Download inputs: this job's own expts/ and job.log if the pod is a restart after a
   disruption, otherwise the expts/ of the job named in resume_from.
3. Stream the dataset archive from the dataset bucket straight into /work/data. Training
   reads local disk, not a network mount.
4. Run `python <script> <args>` in /work/code with DATA_ROOT and OUTPUT_DIR=expts set,
   copying its output to the console (kubectl logs, live view) and to job.log.
5. Every TPU_SYNC_INTERVAL seconds, and once more at exit, upload job.log and any new or
   changed file under expts/ through the submission service.

The per-job token that authorizes uploads is removed from the environment before the
student's script starts. A SIGTERM (cancel, or the node being reclaimed) is passed to
the script, which gets most of the grace period to finish; the final upload follows.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import requests

URL = os.environ.get("TPU_SUBMIT_URL", "").rstrip("/")
JOB_NAME = os.environ.get("TPU_JOB_NAME", "")
JOB_ID = os.environ.get("TPU_JOB_ID", "")
DATASET_URI = os.environ.get("TPU_DATASET_URI", "")
SYNC_INTERVAL = int(os.environ.get("TPU_SYNC_INTERVAL", "300"))
WORK = Path(os.environ.get("TPU_WORK_DIR", "/work"))
CODE = WORK / "code"
DATA = WORK / "data"
HOME = WORK / "home"
LOG = WORK / "job.log"
OUTPUT_DIR = "expts"
# Seconds of the pod's termination grace period the script may use after SIGTERM.
CHILD_GRACE = 75
MAX_LOG_BYTES = 256 * 1024 * 1024
SKIP_SUFFIXES = (".tmp", ".part", ".partial", ".lock", ".swp")


class Console:
    """Everything goes to stdout (kubectl logs) and to job.log."""

    def __init__(self, path: Path):
        self._fh = path.open("ab")
        self._size = path.stat().st_size
        self._lock = threading.Lock()
        self._truncated = False

    def write(self, data: bytes) -> None:
        with self._lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            if self._size + len(data) > MAX_LOG_BYTES:
                if not self._truncated:
                    self._fh.write(b"\n[tpu] job.log is over 256 MB; further output is not saved.\n")
                    self._fh.flush()
                    self._truncated = True
                return
            self._fh.write(data)
            self._fh.flush()
            self._size += len(data)

    def say(self, msg: str) -> None:
        self.write(f"[tpu {time.strftime('%H:%M:%S')}] {msg}\n".encode())


class Service:
    def __init__(self, url: str, job_name: str, token: str):
        self.url = url
        self.s = requests.Session()
        self.s.headers.update({"X-TPU-Job": job_name, "Authorization": f"Bearer {token}"})

    def _request(self, method: str, path: str, *, attempts: int = 6, **kw) -> requests.Response:
        delay = 2.0
        for i in range(attempts):
            try:
                resp = self.s.request(method, self.url + path, timeout=(10, 600), **kw)
                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.HTTPError:
                raise
            except requests.RequestException as e:
                err = f"{e.__class__.__name__}: {e}"
            if i == attempts - 1:
                raise RuntimeError(f"{method} {path} failed: {err}")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            # A file body must be rewound before a retry.
            data = kw.get("data")
            if hasattr(data, "seek"):
                data.seek(0)
        raise AssertionError("unreachable")

    def download(self, path: str, dest: Path) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        resp = self._request("GET", path, stream=True)
        size = 0
        with tmp.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                out.write(chunk)
                size += len(chunk)
        tmp.replace(dest)
        return size

    def inputs(self) -> list[dict]:
        return self._request("GET", "/internal/inputs").json()["files"]

    def upload(self, rel: str, path: Path, size: int) -> None:
        with path.open("rb") as fh:
            self._request(
                "PUT", f"/internal/files/{rel}", data=_Limited(fh, size),
                headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"},
            )

    def upload_bytes(self, rel: str, data: bytes) -> None:
        self._request("PUT", f"/internal/files/{rel}", data=data,
                      headers={"Content-Type": "application/octet-stream"})

    def delete(self, rel: str) -> None:
        self._request("DELETE", f"/internal/files/{rel}")


class _Limited:
    """A file object that stops at `size` bytes, so a file still growing matches Content-Length."""

    def __init__(self, fh, size: int):
        self._fh = fh
        self._left = size
        self._size = size

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        n = self._left if n is None or n < 0 else min(n, self._left)
        data = self._fh.read(n)
        self._left -= len(data)
        return data

    def seek(self, pos: int, whence: int = 0) -> int:
        self._fh.seek(pos, whence)
        self._left = self._size - self._fh.tell()
        return self._fh.tell()

    def __len__(self) -> int:
        return self._size


class Syncer:
    """Mirrors expts/ and job.log to the service. Only new or changed files are sent."""

    def __init__(self, svc: Service, console: Console):
        self.svc = svc
        self.console = console
        self.sent: dict[str, tuple[int, int]] = {}
        self._log_sent = -1
        self._lock = threading.Lock()

    def mark_sent(self, rel: str, path: Path) -> None:
        st = path.stat()
        self.sent[rel] = (st.st_size, st.st_mtime_ns)

    def _local_files(self) -> dict[str, Path]:
        root = CODE / OUTPUT_DIR
        out = {}
        if not root.is_dir():
            return out
        for p in root.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            if p.name.startswith(".") or p.name.endswith(SKIP_SUFFIXES) or ".orbax-checkpoint-tmp" in str(p):
                continue
            out[p.relative_to(CODE).as_posix()] = p
        return out

    def sync(self, final: bool = False) -> None:
        with self._lock:
            uploaded, failed = 0, 0
            local = self._local_files()
            for rel, path in sorted(local.items()):
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue
                if self.sent.get(rel) == (st.st_size, st.st_mtime_ns):
                    continue
                # Skip a file that is still being written, unless this is the last chance.
                if not final and time.time() - st.st_mtime < 5:
                    continue
                try:
                    self.svc.upload(rel, path, st.st_size)
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns) == (st.st_size, st.st_mtime_ns):
                        self.sent[rel] = (st.st_size, st.st_mtime_ns)
                    uploaded += 1
                except (OSError, RuntimeError, requests.RequestException) as e:
                    failed += 1
                    self.console.say(f"upload of {rel} failed: {e}")
            for rel in sorted(set(self.sent) - set(local)):
                try:
                    self.svc.delete(rel)
                    del self.sent[rel]
                except (RuntimeError, requests.RequestException) as e:
                    self.console.say(f"delete of {rel} failed: {e}")
            if uploaded or failed:
                self.console.say(
                    f"synced {uploaded} file(s) from {OUTPUT_DIR}/" + (f", {failed} failed" if failed else "")
                )
            try:
                size = LOG.stat().st_size
                if size != self._log_sent:
                    self.svc.upload_bytes("job.log", LOG.read_bytes()[:size])
                    self._log_sent = size
            except (OSError, RuntimeError, requests.RequestException) as e:
                sys.stderr.write(f"[tpu] log upload failed: {e}\n")


def _extract(tar: tarfile.TarFile, dest: Path) -> None:
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
    else:
        tar.extractall(dest)


def fetch_snapshot(svc: Service, console: Console) -> None:
    archive = WORK / "snapshot.tar.gz"
    size = svc.download("/internal/snapshot", archive)
    CODE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        _extract(tar, CODE)
    archive.unlink()
    console.say(f"unpacked your files ({size / 2**20:.1f} MB)")


def fetch_inputs(svc: Service, syncer: Syncer, console: Console) -> bool:
    """Download checkpoints to resume from. Returns True if any expts/ files arrived."""
    files = svc.inputs()
    restored = 0
    for f in files:
        rel, source = f["path"], f["source"]
        if rel == "job.log":
            continue
        if not rel.startswith(OUTPUT_DIR + "/"):
            continue
        dest = CODE / rel
        svc.download(f"/internal/inputs/{source}/{rel}", dest)
        restored += 1
        # Files from this job's own folder are already uploaded. Files from resume_from
        # are copied into this job's folder on the first sync, so a later restart of this
        # job finds them here.
        if source == JOB_ID:
            syncer.mark_sent(rel, dest)
    if restored:
        sources = sorted({f["source"] for f in files if f["path"].startswith(OUTPUT_DIR + "/")})
        how = "this job's last upload" if sources == [JOB_ID] else f"job {', '.join(sources)}"
        console.say(f"restored {restored} file(s) into {OUTPUT_DIR}/ from {how}")
    return restored > 0


def restore_log(svc: Service) -> None:
    """After a restart, continue the earlier pod's job.log instead of replacing it."""
    for f in svc.inputs():
        if f["path"] == "job.log" and f["source"] == JOB_ID:
            svc.download(f"/internal/inputs/{JOB_ID}/job.log", LOG)
            with LOG.open("ab") as fh:
                fh.write(b"\n[tpu] ---- restarted on a new TPU node ----\n")
            return


def _unpack_local_tar(archive: Path, console: Console) -> Path:
    console.say(f"unpacking dataset {archive} ({archive.stat().st_size / 2**30:.2f} GB) to local disk")
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as tar:
        _extract(tar, DATA)
    entries = [p for p in DATA.iterdir() if not p.name.startswith(".")]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else DATA
    console.say(f"dataset ready at {root} in {time.time() - t0:.0f}s")
    return root


def fetch_dataset(console: Console) -> Path | None:
    if not DATASET_URI:
        return None
    if DATASET_URI.startswith("/"):
        p = Path(DATASET_URI)
        for ext in (".tar", ".tar.gz", ".tgz"):
            cand = Path(str(p) + ext) if not str(p).endswith(ext) else p
            if cand.is_file():
                return _unpack_local_tar(cand, console)
        if p.is_dir():
            console.say(f"dataset directory ready at {p}")
            return p
        raise RuntimeError(f"dataset path {DATASET_URI} does not exist")
    if not DATASET_URI.startswith("gs://"):
        raise RuntimeError(f"dataset URI must be gs://... or an absolute path, got {DATASET_URI!r}")
    from google.cloud import storage

    bucket_name, _, blob_name = DATASET_URI[len("gs://"):].partition("/")
    blob = storage.Client().bucket(bucket_name).get_blob(blob_name)
    if blob is None:
        raise RuntimeError(f"dataset archive {DATASET_URI} does not exist")
    console.say(f"downloading dataset {DATASET_URI} ({(blob.size or 0) / 2**30:.2f} GB) to local disk")
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    with blob.open("rb", chunk_size=64 * 1024 * 1024) as fh:
        with tarfile.open(fileobj=fh, mode="r|*") as tar:
            _extract(tar, DATA)
    entries = [p for p in DATA.iterdir() if not p.name.startswith(".")]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else DATA
    console.say(f"dataset ready at {root} in {time.time() - t0:.0f}s")
    return root


def run_script(argv: list[str], env: dict, console: Console) -> tuple[int, bool]:
    """Run the student's script. Returns (exit code, whether we were told to stop)."""
    stopping = threading.Event()
    proc = subprocess.Popen(
        [sys.executable, "-u", *argv], cwd=CODE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )

    def on_term(signum, _frame):
        if stopping.is_set():
            return
        stopping.set()
        console.say(f"received {signal.Signals(signum).name}; stopping your script "
                    f"(it has {CHILD_GRACE}s to exit)")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        def kill_later():
            time.sleep(CHILD_GRACE)
            if proc.poll() is None:
                console.say("script did not exit in time; killing it")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        threading.Thread(target=kill_later, daemon=True).start()

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        console.write(chunk)
    code = proc.wait()
    return (128 - code if code < 0 else code), stopping.is_set()


def child_env(data_root: Path | None, resumed: bool) -> dict:
    env = dict(os.environ)
    for key in list(env):
        if key == "TPU_JOB_TOKEN" or key.startswith("KUBERNETES_"):
            env.pop(key)
    HOME.mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(HOME),
        "USER": env.get("USER", "student"),
        "LOGNAME": env.get("LOGNAME", "student"),
        "OUTPUT_DIR": OUTPUT_DIR,
        "PYTHONPATH": str(CODE),
        "MPLBACKEND": "Agg",
        "TPU_RESUMED": "1" if resumed else "0",
        "XDG_CACHE_HOME": str(HOME / ".cache"),
    })
    if data_root is not None:
        env["DATA_ROOT"] = str(data_root)
        env.setdefault("DATA_URI", str(data_root))
    return env


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: tpu_entrypoint.py <script> [args...]", file=sys.stderr)
        return 2
    token = os.environ.pop("TPU_JOB_TOKEN", "")
    if not (URL and JOB_NAME and JOB_ID and token):
        print("[tpu] TPU_SUBMIT_URL, TPU_JOB_NAME, TPU_JOB_ID and TPU_JOB_TOKEN are required", file=sys.stderr)
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    svc = Service(URL, JOB_NAME, token)

    try:
        restore_log(svc)
    except Exception as e:  # a missing earlier log must not stop the run
        print(f"[tpu] could not restore earlier log: {e}", file=sys.stderr)
    console = Console(LOG)
    syncer = Syncer(svc, console)
    node = os.environ.get("HOSTNAME", "?")
    console.say(f"job {JOB_ID} starting on {node}")

    started = time.time()
    try:
        fetch_snapshot(svc, console)
        resumed = fetch_inputs(svc, syncer, console)
        data_root = fetch_dataset(console)
    except Exception as e:
        console.say(f"setup failed before your script ran: {e}")
        syncer.sync(final=True)
        return 1

    script, args = sys.argv[1], sys.argv[2:]
    console.say(f"running: python {script} {' '.join(args)}".rstrip())

    stop_sync = threading.Event()

    def sync_loop():
        while not stop_sync.wait(SYNC_INTERVAL):
            try:
                syncer.sync()
            except Exception as e:
                console.say(f"periodic sync failed: {e}")

    t = threading.Thread(target=sync_loop, daemon=True)
    t.start()
    code, stopped = run_script([script, *args], child_env(data_root, resumed), console)
    stop_sync.set()

    minutes = (time.time() - started) / 60
    if stopped:
        console.say(f"script stopped with exit code {code} after {minutes:.1f} min; uploading results")
    else:
        console.say(f"script exited with code {code} after {minutes:.1f} min; uploading results")
    syncer.sync(final=True)
    shutil.rmtree(DATA, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
