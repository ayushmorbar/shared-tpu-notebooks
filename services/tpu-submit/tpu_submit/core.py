"""Everything the service does, without HTTP.

Blocking calls (Kubernetes, object store) live here; app.py runs them in a thread pool.
Bucket layout, all under users/<user>/jobs/<job-id>/:

    meta.json          state and timestamps, written by this service only
    snapshot.tar.gz    the homework folder as submitted
    files/job.log      the job's console output, uploaded by the TPU wrapper
    files/expts/...    whatever train.py wrote to expts/, uploaded by the TPU wrapper
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
import tarfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import BinaryIO

from . import names, state
from .cluster import Cluster
from .config import Settings
from .jobspec import build_job, build_secret
from .store import Store, Writer

_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_RESERVED_ENV = frozenset({
    "JAX_PLATFORMS", "DATA_ROOT", "OUTPUT_DIR", "HOME", "PATH", "PYTHONPATH",
    "PYTHONUNBUFFERED", "TQDM_MININTERVAL",
})
_RESERVED_ENV_PREFIXES = ("TPU_", "LD_", "KUBERNETES_", "GOOGLE_", "GCE_", "CLOUDSDK_")
_ACTIVE_LOST_GRACE = 120


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class JobRequest:
    script: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    dataset: str = ""
    dataset_uri: str = ""
    resume_from: str = ""


class Core:
    def __init__(self, settings: Settings, store: Store, cluster: Cluster, clock=time.time):
        self.settings = settings
        self.store = store
        self.cluster = cluster
        self.clock = clock
        self._user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        self._job_auth_cache: dict[str, tuple[float, str, str, str]] = {}
        self._reconcile_sigs: dict[str, tuple] = {}
        self._reconcile_live: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ metadata

    def _meta_key(self, user: str, job_id: str) -> str:
        return names.job_prefix(user, job_id) + "meta.json"

    def get_meta(self, user: str, job_id: str) -> dict | None:
        raw = self.store.get_bytes(self._meta_key(user, job_id))
        return json.loads(raw) if raw else None

    def _save_meta(self, meta: dict) -> None:
        meta["updated_at"] = state.now_iso()
        self.store.put_bytes(
            self._meta_key(meta["user"], meta["id"]), json.dumps(meta, indent=2).encode()
        )

    def _require_meta(self, user: str, job_id: str) -> dict:
        if not names.JOB_ID_RE.match(job_id or ""):
            raise ApiError(404, f"No job {job_id!r}.")
        meta = self.get_meta(user, job_id)
        if meta is None:
            raise ApiError(404, f"No job {job_id!r} for {user}.")
        return meta

    def list_metas(self, user: str) -> list[dict]:
        prefix = f"users/{names.user_dir(user)}/jobs/"
        metas = []
        for key, _ in self.store.list(prefix):
            if key.endswith("/meta.json"):
                raw = self.store.get_bytes(key)
                if raw:
                    metas.append(json.loads(raw))
        return sorted(metas, key=lambda m: m["id"], reverse=True)

    def _apply(self, meta: dict, derived: dict) -> bool:
        """Fold a derived Kubernetes state into meta. Returns True if meta changed."""
        if meta.get("state") in state.TERMINAL:
            return False
        changed = False
        if derived.get("started_at") and not meta.get("started_at"):
            meta["started_at"] = derived["started_at"]
            changed = True
        if meta.get("state") != derived["state"]:
            meta["state"] = derived["state"]
            changed = True
        if derived.get("finished_at") and meta.get("finished_at") != derived["finished_at"]:
            meta["finished_at"] = derived["finished_at"]
            changed = True
        if derived.get("detail") and meta.get("detail") != derived["detail"]:
            meta["detail"] = derived["detail"]
            changed = True
        return changed

    def _mark_lost(self, meta: dict) -> None:
        meta["state"] = state.LOST
        if meta.get("started_at") and not meta.get("finished_at"):
            meta["finished_at"] = meta.get("updated_at") or state.now_iso()
        meta["detail"] = "The Kubernetes Job was deleted outside the submission service."

    def chip_hours(self, meta: dict) -> float:
        start = state.parse_iso(meta.get("started_at"))
        if start is None:
            return 0.0
        end = state.parse_iso(meta.get("finished_at"))
        if end is None:
            if meta.get("state") in state.TERMINAL:
                end = state.parse_iso(meta.get("updated_at")) or start
            else:
                end = self.clock()
        return max(0.0, end - start) / 3600.0

    def usage_hours(self, user: str) -> float:
        return sum(self.chip_hours(m) for m in self.list_metas(user))

    def view(self, meta: dict) -> dict:
        keys = ("id", "user", "name", "state", "script", "args", "dataset", "resume_from",
                "created_at", "started_at", "finished_at", "detail")
        out = {k: meta.get(k) for k in keys}
        out["chip_hours"] = round(self.chip_hours(meta), 3)
        return out

    # ---------------------------------------------------------------- validation

    def validate_request(self, user: str, spec: dict) -> JobRequest:
        if not isinstance(spec, dict):
            raise ApiError(400, "The job spec must be a JSON object.")
        script = names.clean_relpath(str(spec.get("script") or ""))
        if not script or not script.endswith(".py"):
            raise ApiError(400, "script must be a relative path to a .py file, e.g. 'train.py'.")

        args = spec.get("args") or []
        if not isinstance(args, list) or len(args) > 64:
            raise ApiError(400, "args must be a list of at most 64 strings.")
        if not all(isinstance(a, str) and len(a) <= 4096 for a in args):
            raise ApiError(400, "Every arg must be a string of at most 4096 characters.")

        env = spec.get("env") or {}
        if not isinstance(env, dict) or len(env) > 32:
            raise ApiError(400, "env must be a mapping of at most 32 names to strings.")
        for key, value in env.items():
            if not isinstance(key, str) or not _ENV_KEY_RE.match(key):
                raise ApiError(400, f"env name {key!r} must look like WANDB_API_KEY.")
            if key in _RESERVED_ENV or key.startswith(_RESERVED_ENV_PREFIXES):
                raise ApiError(400, f"env name {key!r} is set by the TPU job and cannot be overridden.")
            if not isinstance(value, str) or len(value) > 4096:
                raise ApiError(400, f"env value for {key} must be a string of at most 4096 characters.")

        dataset = spec.get("dataset", "default")
        if dataset in (None, "", "none"):
            dataset = ""
        elif dataset == "default":
            dataset = self.settings.default_dataset
        elif dataset not in self.settings.datasets:
            known = ", ".join(sorted(self.settings.datasets)) or "none"
            raise ApiError(400, f"Unknown dataset {dataset!r}. Available: {known}.")
        dataset_uri = self.settings.datasets.get(dataset, "") if dataset else ""

        resume_from = spec.get("resume_from") or ""
        if resume_from:
            if not isinstance(resume_from, str):
                raise ApiError(400, "resume_from must be a job id.")
            self._require_meta(user, resume_from)

        return JobRequest(script=script, args=list(args), env=dict(env), dataset=dataset,
                          dataset_uri=dataset_uri, resume_from=resume_from)

    def _job_selector(self, user: str | None = None) -> str:
        sel = f"{names.LABEL_MANAGED_BY}={names.MANAGED_BY}"
        if user is not None:
            sel += f",{names.LABEL_USER_HASH}={names.user_hash(user)}"
        return sel

    def active_jobs(self, user: str) -> list[dict]:
        active = []
        for job in self.cluster.list_jobs(self._job_selector(user)):
            if (job.get("metadata") or {}).get("deletionTimestamp"):
                continue
            if state.derive(job, [])["state"] not in state.TERMINAL:
                active.append(job)
        return active

    def check_can_submit(self, user: str) -> None:
        active = self.active_jobs(user)
        if len(active) >= self.settings.max_active_jobs:
            ann = (active[0].get("metadata") or {}).get("annotations") or {}
            job_id = ann.get(names.ANN_JOB_ID, "?")
            raise ApiError(
                409,
                f"You already have job {job_id} queued or running. Wait for it to finish, "
                f"or stop it with submit_tpu.cancel('{job_id}').",
            )
        budget = self.settings.chip_hour_budget
        if budget > 0:
            used = self.usage_hours(user)
            if used >= budget:
                raise ApiError(
                    403,
                    f"You have used {used:.2f} of your {budget:g} chip-hours. "
                    f"Ask a TA if you need more.",
                )

    def preflight(self, user: str, spec: dict) -> dict:
        req = self.validate_request(user, spec)
        self.check_can_submit(user)
        return {
            "ok": True,
            "dataset": req.dataset,
            "max_snapshot_mb": self.settings.max_snapshot_mb,
            "max_job_hours": self.settings.max_job_hours,
        }

    def _check_snapshot(self, path: str, script: str) -> None:
        try:
            with tarfile.open(path, "r:gz") as tar:
                found = False
                for member in tar:
                    if member.name in (".", "./"):
                        continue
                    clean = names.clean_relpath(member.name.removeprefix("./"))
                    if clean is None:
                        raise ApiError(400, f"Snapshot contains an unsafe path: {member.name!r}.")
                    if not (member.isfile() or member.isdir()):
                        raise ApiError(400, f"Snapshot may only contain files and folders: {member.name!r}.")
                    if clean == script and member.isfile():
                        found = True
        except (tarfile.TarError, EOFError, OSError) as e:
            raise ApiError(400, f"Snapshot is not a readable .tar.gz: {e}") from None
        if not found:
            raise ApiError(400, f"{script} is not in the snapshot.")

    # -------------------------------------------------------------- user actions

    def _user_lock(self, user: str) -> threading.Lock:
        with self._locks_guard:
            return self._user_locks[user]

    def submit(self, user: str, req: JobRequest, snapshot_path: str) -> dict:
        self._check_snapshot(snapshot_path, req.script)
        with self._user_lock(user):
            self.check_can_submit(user)
            job_id = names.new_job_id()
            while self.get_meta(user, job_id) is not None:
                job_id = names.new_job_id()
            name = names.job_name(user, job_id)
            prefix = names.job_prefix(user, job_id)
            self.store.put_file(prefix + "snapshot.tar.gz", snapshot_path)

            token = secrets.token_urlsafe(32)
            manifest = build_job(
                self.settings, user=user, job_id=job_id, script=req.script, args=req.args,
                dataset_uri=req.dataset_uri, token_sha=names.token_sha256(token),
            )
            meta = {
                "id": job_id,
                "user": user,
                "name": name,
                "state": state.QUEUED,
                "script": req.script,
                "args": req.args,
                "dataset": req.dataset,
                "resume_from": req.resume_from,
                "env_keys": sorted(req.env),
                "created_at": state.now_iso(),
                "started_at": None,
                "finished_at": None,
                "detail": "",
            }
            self._save_meta(meta)
            try:
                created = self.cluster.create_job(manifest)
                self.cluster.create_secret(build_secret(
                    self.settings, name=name, token=token, env=req.env,
                    owner_uid=created["metadata"]["uid"],
                ))
            except Exception as e:
                try:
                    self.cluster.delete_job(name)
                finally:
                    meta["state"] = state.FAILED
                    meta["detail"] = f"Could not create the Kubernetes Job: {e}"
                    self._save_meta(meta)
                raise ApiError(500, meta["detail"]) from e
            return self.view(meta)

    def _queue_position(self, name: str) -> int | None:
        queued = [
            j for j in self.cluster.list_jobs(self._job_selector())
            if state.derive(j, [])["state"] == state.QUEUED
        ]
        queued.sort(key=lambda j: (j["metadata"].get("creationTimestamp") or "", j["metadata"]["name"]))
        for i, j in enumerate(queued):
            if j["metadata"]["name"] == name:
                return i + 1
        return None

    def _refresh(self, meta: dict) -> dict | None:
        """Update meta from the live Job. Returns the Job, or None if it is gone."""
        job = self.cluster.get_job(meta["name"])
        if job is None:
            created = state.parse_iso(meta.get("created_at")) or 0
            if meta.get("state") not in state.TERMINAL and self.clock() - created > _ACTIVE_LOST_GRACE:
                self._mark_lost(meta)
                self._save_meta(meta)
            return None
        pods = self.cluster.list_pods(f"job-name={meta['name']}")
        if self._apply(meta, state.derive(job, pods)):
            self._save_meta(meta)
        return job

    def status(self, user: str, job_id: str) -> dict:
        meta = self._require_meta(user, job_id)
        if meta.get("state") not in state.TERMINAL:
            self._refresh(meta)
        out = self.view(meta)
        if out["state"] == state.QUEUED:
            out["position"] = self._queue_position(meta["name"])
        return out

    def list_jobs(self, user: str, limit: int = 50) -> list[dict]:
        out = []
        for meta in self.list_metas(user)[:limit]:
            if meta.get("state") not in state.TERMINAL:
                self._refresh(meta)
            out.append(self.view(meta))
        return out

    def me(self, user: str) -> dict:
        metas = self.list_metas(user)
        budget = self.settings.chip_hour_budget
        return {
            "user": user,
            "chip_hours_used": round(sum(self.chip_hours(m) for m in metas), 3),
            "chip_hour_budget": budget if budget > 0 else None,
            "max_job_hours": self.settings.max_job_hours,
            "max_active_jobs": self.settings.max_active_jobs,
            "datasets": sorted(self.settings.datasets),
            "default_dataset": self.settings.default_dataset or None,
        }

    def all_active(self) -> list[dict]:
        """Every live managed Job, for TAs."""
        out = []
        for job in self.cluster.list_jobs(self._job_selector()):
            ann = (job.get("metadata") or {}).get("annotations") or {}
            d = state.derive(job, [])
            if d["state"] in state.TERMINAL:
                continue
            out.append({
                "id": ann.get(names.ANN_JOB_ID),
                "user": ann.get(names.ANN_USER),
                "name": job["metadata"]["name"],
                "state": d["state"],
                "script": ann.get(names.ANN_SCRIPT),
                "created_at": job["metadata"].get("creationTimestamp"),
                "started_at": d["started_at"],
            })
        return sorted(out, key=lambda j: j["created_at"] or "")

    def cancel(self, user: str, job_id: str, by: str | None = None) -> dict:
        with self._user_lock(user):
            meta = self._require_meta(user, job_id)
            if meta.get("state") in state.TERMINAL:
                return self.view(meta)
            self._refresh(meta)
            if meta.get("state") in state.TERMINAL:
                return self.view(meta)
            self.cluster.delete_job(meta["name"])
            meta["state"] = state.CANCELLED
            if meta.get("started_at"):
                meta["finished_at"] = state.now_iso()
            meta["detail"] = f"Cancelled by {by or user}."
            self._save_meta(meta)
            return self.view(meta)

    def logs(self, user: str, job_id: str, offset: int = 0, pod: str | None = None) -> dict:
        meta = self._require_meta(user, job_id)
        job = self._refresh(meta) if meta.get("state") not in state.TERMINAL else self.cluster.get_job(meta["name"])
        view = self.view(meta)
        if view["state"] == state.QUEUED:
            view["position"] = self._queue_position(meta["name"])
        done = view["state"] in state.TERMINAL
        if job is not None:
            pods = self.cluster.list_pods(f"job-name={meta['name']}")
            pods.sort(key=lambda p: p["metadata"].get("creationTimestamp") or "")
            if pods:
                latest = pods[-1]["metadata"]["name"]
                text = self.cluster.pod_log(latest)
                if pod != latest:
                    offset = 0
                offset = min(max(offset, 0), len(text))
                return {"source": "pod", "pod": latest, "offset": len(text),
                        "data": text[offset:], "job": view, "done": done}
            return {"source": "none", "pod": None, "offset": 0, "data": "", "job": view,
                    "done": done}
        raw = self.store.get_bytes(names.job_prefix(user, job_id) + "files/job.log") or b""
        text = raw.decode("utf-8", "replace")
        if pod is not None:
            offset = 0
        offset = min(max(offset, 0), len(text))
        return {"source": "stored", "pod": None, "offset": len(text), "data": text[offset:],
                "job": view, "done": True}

    def files(self, user: str, job_id: str) -> list[dict]:
        self._require_meta(user, job_id)
        prefix = names.job_prefix(user, job_id) + "files/"
        return [{"path": key[len(prefix):], "size": size} for key, size in self.store.list(prefix)]

    def open_file(self, user: str, job_id: str, path: str) -> tuple[BinaryIO, int]:
        clean = names.clean_relpath(path)
        if clean is None:
            raise ApiError(400, f"Bad path {path!r}.")
        opened = self.store.open_read(names.job_prefix(user, job_id) + "files/" + clean)
        if opened is None:
            raise ApiError(404, f"No file {clean} in job {job_id}.")
        return opened

    # ---------------------------------------------------------- TPU job actions

    def authenticate_job(self, name: str, token: str) -> tuple[str, str]:
        if not token:
            raise ApiError(401, "Missing job token.")
        now = self.clock()
        cached = self._job_auth_cache.get(name)
        if cached and cached[0] > now:
            _, user, job_id, sha = cached
        else:
            job = self.cluster.get_job(name)
            if job is None:
                raise ApiError(401, "Unknown job.")
            ann = (job.get("metadata") or {}).get("annotations") or {}
            user, job_id, sha = ann.get(names.ANN_USER), ann.get(names.ANN_JOB_ID), ann.get(names.ANN_TOKEN_SHA)
            if not (user and job_id and sha):
                raise ApiError(401, "Job is not managed by tpu-submit.")
            self._job_auth_cache[name] = (now + 60, user, job_id, sha)
        if not hmac.compare_digest(sha, names.token_sha256(token)):
            raise ApiError(401, "Bad job token.")
        return user, job_id

    def open_snapshot(self, user: str, job_id: str) -> tuple[BinaryIO, int]:
        opened = self.store.open_read(names.job_prefix(user, job_id) + "snapshot.tar.gz")
        if opened is None:
            raise ApiError(404, "Snapshot missing.")
        return opened

    def inputs(self, user: str, job_id: str) -> dict:
        """Files a starting pod should download before running the script.

        After a disruption the Job's own expts/ wins. On a fresh Job submitted with
        resume_from, the earlier Job's expts/ is the starting point.
        """
        meta = self._require_meta(user, job_id)
        own = self.files(user, job_id)
        if any(f["path"].startswith("expts/") for f in own):
            return {"files": [{**f, "source": job_id} for f in own]}
        out = [{**f, "source": job_id} for f in own if f["path"] == "job.log"]
        resume_from = meta.get("resume_from")
        if resume_from:
            out += [
                {**f, "source": resume_from}
                for f in self.files(user, resume_from) if f["path"].startswith("expts/")
            ]
        return {"files": out}

    def open_input(self, user: str, job_id: str, path: str, source: str) -> tuple[BinaryIO, int]:
        meta = self._require_meta(user, job_id)
        if source not in (job_id, meta.get("resume_from") or job_id):
            raise ApiError(403, "That job is not an input of this one.")
        return self.open_file(user, source, path)

    def open_upload(self, user: str, job_id: str, path: str) -> Writer:
        clean = names.clean_relpath(path)
        if clean is None:
            raise ApiError(400, f"Bad path {path!r}.")
        return self.store.open_write(names.job_prefix(user, job_id) + "files/" + clean)

    def delete_file(self, user: str, job_id: str, path: str) -> None:
        clean = names.clean_relpath(path)
        if clean is None:
            raise ApiError(400, f"Bad path {path!r}.")
        self.store.delete(names.job_prefix(user, job_id) + "files/" + clean)

    # ------------------------------------------------------------- reconcile

    def reconcile(self) -> None:
        """Copy Job state into meta.json, so timestamps survive the Job's TTL."""
        selector = self._job_selector()
        jobs = self.cluster.list_jobs(selector)
        pods_by_job: dict[str, list[dict]] = defaultdict(list)
        for pod in self.cluster.list_pods(selector):
            labels = (pod.get("metadata") or {}).get("labels") or {}
            job_name = labels.get("batch.kubernetes.io/job-name") or labels.get("job-name")
            if job_name:
                pods_by_job[job_name].append(pod)

        live: set[tuple[str, str]] = set()
        for job in jobs:
            md = job.get("metadata") or {}
            ann = md.get("annotations") or {}
            user, job_id = ann.get(names.ANN_USER), ann.get(names.ANN_JOB_ID)
            if not user or not job_id:
                continue
            live.add((user, job_id))
            derived = state.derive(job, pods_by_job.get(md["name"], []))
            sig = (derived["state"], derived["started_at"], derived["finished_at"])
            if self._reconcile_sigs.get(md["name"]) == sig:
                continue
            meta = self.get_meta(user, job_id)
            if meta is not None and self._apply(meta, derived):
                self._save_meta(meta)
            self._reconcile_sigs[md["name"]] = sig

        for user, job_id in self._reconcile_live - live:
            self._reconcile_sigs.pop(names.job_name(user, job_id), None)
            meta = self.get_meta(user, job_id)
            if meta is not None and meta.get("state") not in state.TERMINAL:
                self._mark_lost(meta)
                self._save_meta(meta)
        self._reconcile_live = live
