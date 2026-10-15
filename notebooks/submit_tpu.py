"""Run a training script on a TPU chip from a CPU notebook, like sbatch.

Your notebook has no accelerator. submit() packs the folder your script lives in, sends
it to the course's TPU submission service, and the service queues a job on the shared
v5e pool. When a chip is free the job unpacks your folder and the dataset on the chip's
machine (or mounts /tartan/11785 read-only), runs your script, and keeps uploading its
log and expts/ folder (checkpoints, results.csv) so you can watch it and download the
results here.

    import submit_tpu
    job = submit_tpu.submit("train.py", args=["--config", "config.yaml"])
    submit_tpu.watch(job)    # live log; stopping this cell does NOT stop the job
    submit_tpu.status()      # your jobs, and your place in line
    submit_tpu.pull(job)     # copy expts/ and job.log into ./tpu_runs/<job id>/
    submit_tpu.cancel(job)   # stop it and give the chip back

The chip is held only while your job runs. You can have one job queued or running at a
time. A job pushed off its chip by the cluster restarts on its own and resumes from the
last checkpoint it uploaded. To continue a finished or failed run, submit again with
resume_from=<old job>.

Credentials in /home/jovyan/.env (WANDB_API_KEY, KAGGLE_API_TOKEN, HF_TOKEN, etc.) or
~/.kaggle/kaggle.json are automatically forwarded to your TPU job as environment
variables via a per-job Kubernetes Secret—never bundled into the uploaded code archive.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import requests

SERVICE_URL = os.environ.get("TPU_SUBMIT_URL", "").rstrip("/")
MAX_SNAPSHOT_MB = int(os.environ.get("TPU_MAX_SNAPSHOT_MB", "100"))

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".venv", "venv", "node_modules", ".cache",
    ".local", ".config", ".jupyter", ".ipython", ".kaggle", ".ssh",
    "expts", "checkpoints", "wandb", "tpu_runs",
}
# Only at the top of the snapshot: code packages like hw3lib/data/ must still be sent.
EXCLUDE_TOP_DIRS = ("data", "datasets", "hw*_data", "hw*-data", "hw*_data_subset")
EXCLUDE_FILES = (
    "*.ipynb", "kaggle.json", ".env", ".env.*", "*.env", ".netrc", "_netrc",
    "*.key", "*.pem", "id_rsa*", "id_ed25519*", "credentials*.json",
    "*.pyc", ".DS_Store", "*.tar.gz.part",
)

_CREDENTIAL_KEYS = (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "KAGGLE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


class SubmitError(RuntimeError):
    """The service refused the request. The message says why."""


@dataclass
class Job:
    id: str
    info: dict

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"Job({self.id!r}, state={self.info.get('state')!r})"


# ----------------------------------------------------------------- credentials

def _load_dotenv_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE pairs from a .env file, skipping comments and empty values."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                out[key] = val
    except OSError:
        pass
    return out


def _discover_credentials(extra_env: dict | None = None) -> dict[str, str]:
    """Auto-discover student API keys from .env files, ~/.kaggle/kaggle.json, and os.environ.

    Precedence (highest wins):
      1. Explicit `extra_env` passed to submit(env=...)
      2. Current notebook process environment (os.environ)
      3. Local .env files (/home/jovyan/.env, ~/.env, ./.env)
      4. ~/.kaggle/kaggle.json
    """
    creds: dict[str, str] = {}

    # 4. ~/.kaggle/kaggle.json
    for kaggle_path in (Path("/home/jovyan/.kaggle/kaggle.json"), Path.home() / ".kaggle" / "kaggle.json"):
        if kaggle_path.is_file():
            try:
                kdata = json.loads(kaggle_path.read_text(encoding="utf-8"))
                if kdata.get("username"):
                    creds["KAGGLE_USERNAME"] = str(kdata["username"])
                if kdata.get("key"):
                    creds["KAGGLE_KEY"] = str(kdata["key"])
                    creds.setdefault("KAGGLE_API_TOKEN", str(kdata["key"]))
            except Exception:
                pass
            break

    # 3. .env files
    seen_paths: set[Path] = set()
    for env_path in (Path("/home/jovyan/.env"), Path.home() / ".env", Path(".env")):
        try:
            resolved = env_path.resolve()
        except OSError:
            continue
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        for k, v in _load_dotenv_file(resolved).items():
            if k in _CREDENTIAL_KEYS and v:
                creds[k] = v

    # 2. Process environment variables
    for k in _CREDENTIAL_KEYS:
        val = os.environ.get(k, "").strip()
        if val:
            creds[k] = val

    # Ensure KAGGLE_KEY and KAGGLE_API_TOKEN mirror each other if only one is set
    if "KAGGLE_API_TOKEN" in creds and "KAGGLE_KEY" not in creds:
        creds["KAGGLE_KEY"] = creds["KAGGLE_API_TOKEN"]
    elif "KAGGLE_KEY" in creds and "KAGGLE_API_TOKEN" not in creds:
        creds["KAGGLE_API_TOKEN"] = creds["KAGGLE_KEY"]

    # 1. Explicit env overrides
    if extra_env:
        for k, v in extra_env.items():
            creds[str(k)] = str(v)

    return creds


# --------------------------------------------------------------------------- HTTP

def _session() -> requests.Session:
    if not SERVICE_URL:
        raise SubmitError(
            "TPU_SUBMIT_URL is not set. submit_tpu only works from a notebook on the course hub."
        )
    token = os.environ.get("JUPYTERHUB_API_TOKEN")
    if not token:
        raise SubmitError("JUPYTERHUB_API_TOKEN is not set. Restart your server from the hub.")
    s = requests.Session()
    s.headers["Authorization"] = f"token {token}"
    return s


def _call(method: str, path: str, *, timeout=(10, 120), stream=False, **kw) -> requests.Response:
    try:
        resp = _session().request(method, SERVICE_URL + path, timeout=timeout, stream=stream, **kw)
    except requests.ConnectionError as e:
        raise SubmitError(
            f"Could not reach the TPU submission service at {SERVICE_URL}. "
            f"It may be restarting; try again in a minute. ({e.__class__.__name__})"
        ) from None
    if resp.status_code >= 400:
        try:
            message = resp.json().get("error") or resp.text
        except ValueError:
            message = resp.text or resp.reason
        raise SubmitError(f"{message} (HTTP {resp.status_code})")
    return resp


def _job_id(job) -> str:
    if isinstance(job, Job):
        return job.id
    if isinstance(job, dict):
        return job["id"]
    return str(job)


def _latest(active_only: bool = False) -> str:
    jobs_ = _call("GET", "/api/jobs").json()["jobs"]
    for j in jobs_:
        if not active_only or j["state"] in ("queued", "starting", "running"):
            return j["id"]
    raise SubmitError("You have no active jobs." if active_only else "You have no jobs yet.")


# ---------------------------------------------------------------------- packing

def _excluded(rel: Path, is_dir: bool) -> bool:
    name = rel.name
    if is_dir:
        if len(rel.parts) == 1 and any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_TOP_DIRS):
            return True
        return name in EXCLUDE_DIRS
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_FILES)


def collect_files(root: str | os.PathLike) -> list[tuple[Path, int]]:
    """Files submit() would send from root, as (relative path, size). Symlinks are skipped."""
    root = Path(root)
    out: list[tuple[Path, int]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if not (base / d).is_symlink() and not _excluded((base / d).relative_to(root), True)
        )
        for fname in sorted(filenames):
            path = base / fname
            rel = path.relative_to(root)
            if path.is_symlink() or not path.is_file() or _excluded(rel, False):
                continue
            out.append((rel, path.stat().st_size))
    return out


def pack(root: str | os.PathLike, dest: str | os.PathLike, max_mb: int = MAX_SNAPSHOT_MB) -> list[Path]:
    """Write root's sendable files to dest as .tar.gz. Returns the paths packed."""
    root = Path(root)
    files = collect_files(root)
    total = sum(size for _, size in files)
    if total > max_mb * 1024 * 1024:
        biggest = sorted(files, key=lambda f: f[1], reverse=True)[:5]
        listing = "\n".join(f"  {size / 2**20:8.1f} MB  {rel}" for rel, size in biggest)
        raise SubmitError(
            f"{root} has {total / 2**20:.0f} MB of files to send; the limit is {max_mb} MB. "
            f"Largest:\n{listing}\nMove data and old outputs out of this folder, or into a "
            f"folder named data/ or expts/, which are never sent."
        )

    def reset(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 1000
        info.uname = info.gname = ""
        return info

    with tarfile.open(dest, "w:gz", compresslevel=6) as tar:
        for rel, _ in files:
            tar.add(root / rel, arcname=rel.as_posix(), recursive=False, filter=reset)
    return [rel for rel, _ in files]


# ------------------------------------------------------------------- public API

def submit(
    script: str,
    args=(),
    env: dict | None = None,
    dataset: str | None = "default",
    resume_from=None,
    root: str | None = None,
    follow: bool = False,
) -> Job:
    """Queue `python <script> <args>` on one TPU chip. Returns the Job.

    script       path to the .py file, relative to the current folder or absolute
    args         command-line arguments for the script
    env          extra environment variables, e.g. {"WANDB_API_KEY": key}
                 (keys in /home/jovyan/.env or ~/.kaggle/kaggle.json are auto-discovered)
    dataset      which course dataset to unpack on the chip ("default", a name, or None)
    resume_from  an earlier job whose expts/ folder this job starts from
    root         folder to send; defaults to the script's folder
    follow       watch the log right away
    """
    script_path = Path(script).expanduser().resolve()
    if not script_path.is_file():
        raise SubmitError(f"{script} does not exist.")
    root_path = Path(root).expanduser().resolve() if root else script_path.parent
    try:
        rel_script = script_path.relative_to(root_path).as_posix()
    except ValueError:
        raise SubmitError(f"{script} is not inside {root_path}.") from None
    if _excluded(Path(rel_script), False):
        raise SubmitError(f"{rel_script} matches an excluded pattern and would not be sent.")

    merged_env = _discover_credentials(env)
    spec = {
        "script": rel_script,
        "args": [str(a) for a in args],
        "env": merged_env,
        "dataset": dataset,
        "resume_from": _job_id(resume_from) if resume_from else None,
    }
    if any(not v for v in spec["env"].values()):
        empty = [k for k, v in spec["env"].items() if not v]
        print(f"warning: env {', '.join(empty)} is empty", file=sys.stderr)

    _call("POST", "/api/preflight", json=spec)

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "snapshot.tar.gz"
        packed = pack(root_path, snapshot)
        size = snapshot.stat().st_size
        print(f"packed {len(packed)} files from {root_path} ({size / 2**20:.1f} MB)", flush=True)
        with snapshot.open("rb") as fh:
            upload = _call(
                "PUT", "/api/uploads", data=fh, timeout=(10, 600),
                headers={"Content-Type": "application/gzip"},
            ).json()

    info = _call("POST", "/api/jobs", json={**spec, "upload_id": upload["upload_id"]}).json()
    job = Job(info["id"], info)
    print(f"submitted job {job.id}: python {rel_script} {' '.join(spec['args'])}".rstrip(), flush=True)
    if follow:
        watch(job)
    else:
        print(f"watch it with submit_tpu.watch('{job.id}')", flush=True)
    return job


def status(job=None) -> dict | list[dict]:
    """With a job: its status. Without: print and return all your jobs."""
    if job is not None:
        return _call("GET", f"/api/jobs/{_job_id(job)}").json()
    jobs_ = _call("GET", "/api/jobs").json()["jobs"]
    me = usage()
    if not jobs_:
        print("no jobs yet")
    for j in jobs_:
        extra = f"  #{j['position']} in line" if j.get("position") else ""
        print(f"{j['id']}  {j['state']:<10} {j['chip_hours']:6.2f} chip-h  {j['script']}{extra}")
    budget = me.get("chip_hour_budget")
    used = me.get("chip_hours_used", 0)
    print(f"chip-hours used: {used:.2f}" + (f" of {budget:g}" if budget else ""))
    return jobs_


def jobs() -> list[dict]:
    """Your jobs, newest first, without printing."""
    return _call("GET", "/api/jobs").json()["jobs"]


def usage() -> dict:
    """Chip-hours used, your budget, and the datasets available."""
    return _call("GET", "/api/me").json()


def cancel(job=None) -> dict:
    """Stop a job (default: your active one). Its uploaded files stay downloadable."""
    job_id = _job_id(job) if job is not None else _latest(active_only=True)
    info = _call("DELETE", f"/api/jobs/{job_id}").json()
    print(f"job {job_id}: {info['state']}")
    return info


def logs(job=None) -> str:
    """The whole log so far, as a string."""
    job_id = _job_id(job) if job is not None else _latest()
    return _call("GET", f"/api/jobs/{job_id}/logs").json()["data"]


def watch(job=None, poll: float = 10.0) -> dict:
    """Print the job's log as it arrives until it finishes. Returns the final status.

    Stopping this cell only stops watching. The job keeps running; cancel() stops it.
    """
    job_id = _job_id(job) if job is not None else _latest()
    offset, pod, last_line = 0, None, None
    try:
        while True:
            r = _call("GET", f"/api/jobs/{job_id}/logs", params={"offset": offset, "pod": pod or ""}).json()
            info = r["job"]
            if r["pod"] != pod and r["source"] == "pod" and pod is not None:
                print(f"\n--- job restarted on a new chip ({r['pod']}); resuming from its last checkpoint ---\n", flush=True)
            if r["data"]:
                sys.stdout.write(r["data"])
                sys.stdout.flush()
            offset, pod = r["offset"], r["pod"]
            line = f"[{job_id}] {info['state']}"
            if info.get("position"):
                line += f", #{info['position']} in line"
            if line != last_line:
                print(line, flush=True)
                last_line = line
            if r["done"]:
                if info.get("detail"):
                    print(info["detail"], flush=True)
                print(f"finished: {info['state']} after {info['chip_hours']:.2f} chip-hours. "
                      f"submit_tpu.pull('{job_id}') downloads the results.", flush=True)
                return info
            time.sleep(poll)
    except KeyboardInterrupt:
        print(f"\nstopped watching. Job {job_id} is still running; "
              f"submit_tpu.cancel('{job_id}') stops it.", flush=True)
        return status(job_id)


def files(job=None) -> list[dict]:
    """Files the job has uploaded so far (job.log and expts/...)."""
    job_id = _job_id(job) if job is not None else _latest()
    return _call("GET", f"/api/jobs/{job_id}/files").json()["files"]


def pull(job=None, dest: str | os.PathLike | None = None, pattern: str = "*") -> Path:
    """Download the job's uploaded files into dest (default ./tpu_runs/<job id>/).

    Files already downloaded at the same size are skipped, so pulling again during a run
    fetches only what changed. `pattern` filters paths, e.g. "*results.csv".
    """
    job_id = _job_id(job) if job is not None else _latest()
    dest = Path(dest) if dest else Path("tpu_runs") / job_id
    listing = [f for f in files(job_id) if fnmatch.fnmatch(f["path"], pattern)]
    fetched = 0
    for f in listing:
        target = dest / f["path"]
        if target.is_file() and target.stat().st_size == f["size"] and f["path"] != "job.log":
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with _call("GET", f"/api/jobs/{job_id}/files/{f['path']}", stream=True, timeout=(10, 600)) as r:
            with tmp.open("wb") as out:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
        tmp.replace(target)
        fetched += 1
    print(f"{fetched} new or changed files of {len(listing)} in {dest}", flush=True)
    return dest


# ---------------------------------------------------- backward-compatible shims

def run(
    code: str,
    timeout: int = 28800,
    keep: bool = False,
    env: dict | None = None,
    dataset: str | None = None,
) -> str:
    """Run a self-contained Python snippet on a TPU chip and return its console output.

    Provided for quick experiments and backward compatibility with earlier notebooks.
    For multi-file projects and checkpointing, prefer `submit_tpu.submit("train.py")`.
    """
    del timeout, keep  # Deadline and TTL are enforced server-side by tpu-submit
    with tempfile.TemporaryDirectory() as tmp:
        script_file = Path(tmp) / "snippet.py"
        script_file.write_text(code, encoding="utf-8")
        job = submit(str(script_file), env=env, dataset=dataset, root=tmp, follow=True)
    info = status(job)
    out = logs(job)
    if info.get("state") != "succeeded":
        raise RuntimeError(f"TPU job {job.id} ended in state {info.get('state')!r}:\n{out}")
    return out


def train(
    script: str,
    assignment: str | None = None,
    dataset: str | None = None,
    extra_env: dict[str, str] | None = None,
    args=(),
    resume_from=None,
    timeout: int = 28800,
    keep: bool = False,
) -> str:
    """Submit a training script file (or inline code string), stream logs, and pull results.

    Compatible with Tartan HW2/HW0 notebooks (`submit_tpu.train("train.py", assignment="hw2p2")`).
    """
    del timeout, keep
    chosen_dataset = dataset if dataset is not None else (assignment or "default")
    cand = Path(script).expanduser()
    if "\n" not in script and cand.is_file():
        job = submit(
            str(cand),
            args=args,
            env=extra_env,
            dataset=chosen_dataset,
            resume_from=resume_from,
            follow=True,
        )
    else:
        return run(script, env=extra_env, dataset=chosen_dataset)

    info = status(job)
    out = logs(job)
    try:
        pull(job)
    except Exception:
        pass
    if info.get("state") != "succeeded":
        raise RuntimeError(f"TPU job {job.id} ended in state {info.get('state')!r}:\n{out}")
    return out


__all__ = [
    "Job", "SubmitError", "submit", "watch", "status", "jobs", "usage", "cancel",
    "logs", "files", "pull", "collect_files", "pack", "run", "train",
]
