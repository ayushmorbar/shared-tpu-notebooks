"""HTTP API.

Students (JupyterHub token in "Authorization: token ..."):

    GET    /api/me                         usage, budget, datasets
    POST   /api/preflight                  validate a spec without uploading anything
    PUT    /api/uploads                    stream a snapshot .tar.gz, returns {"upload_id"}
    POST   /api/jobs                       spec + upload_id, creates the Job
    GET    /api/jobs                       my jobs (admins: ?user=NAME, or ?all=1 for every live job)
    GET    /api/jobs/<id>                  status, with place in line while queued
    DELETE /api/jobs/<id>                  cancel
    GET    /api/jobs/<id>/logs             ?offset=N&pod=NAME, incremental
    GET    /api/jobs/<id>/files            uploaded files
    GET    /api/jobs/<id>/files/<path>     download one file

TPU jobs ("X-TPU-Job: <job name>" and "Authorization: Bearer <job token>"):

    GET    /internal/snapshot
    GET    /internal/inputs
    GET    /internal/inputs/<source>/<path>
    PUT    /internal/files/<path>
    DELETE /internal/files/<path>
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tornado.web
from tornado.ioloop import IOLoop, PeriodicCallback

from . import names
from .config import Settings
from .core import ApiError, Core

log = logging.getLogger("tpu_submit")

_UPLOAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_UPLOAD_MAX_AGE = 3600
_CHUNK = 1024 * 1024


class Base(tornado.web.RequestHandler):
    def initialize(self, core: Core, auth, settings: Settings, executor: ThreadPoolExecutor):
        self.core = core
        self.auth = auth
        self.cfg = settings
        self.executor = executor

    def run(self, fn, *args):
        return IOLoop.current().run_in_executor(self.executor, fn, *args)

    def check_xsrf_cookie(self):
        pass

    def set_default_headers(self):
        self.set_header("Cache-Control", "no-store")

    def write_error(self, status_code, **kwargs):
        message = self._reason
        exc = kwargs.get("exc_info", (None, None, None))[1]
        if isinstance(exc, ApiError):
            message = exc.message
        elif isinstance(exc, tornado.web.HTTPError) and exc.log_message:
            message = exc.log_message % exc.args if exc.args else exc.log_message
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({"error": message}))

    def send_error(self, status_code=500, **kwargs):
        exc = kwargs.get("exc_info", (None, None, None))[1]
        if isinstance(exc, ApiError):
            status_code = exc.status
        super().send_error(status_code, **kwargs)

    def log_exception(self, typ, value, tb):
        if isinstance(value, ApiError):
            if value.status >= 500:
                log.error("%s %s: %s", self.request.method, self.request.path, value.message)
            return
        super().log_exception(typ, value, tb)

    def send_json(self, obj, status=200):
        self.set_status(status)
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps(obj))

    def body_json(self) -> dict:
        try:
            data = json.loads(self.request.body or b"{}")
        except ValueError:
            raise ApiError(400, "Body must be JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "Body must be a JSON object.")
        return data

    async def stream_file(self, opened):
        fobj, size = opened
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Content-Length", str(size))
        try:
            while True:
                chunk = await self.run(fobj.read, _CHUNK)
                if not chunk:
                    break
                self.write(chunk)
                await self.flush()
        finally:
            fobj.close()
        self.finish()


class UserBase(Base):
    """Resolves the calling student. Admins may act for another user with ?user=."""

    user: str
    caller: dict

    async def authenticate(self) -> str:
        header = self.request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() not in ("token", "bearer") or not token.strip():
            raise ApiError(401, "Missing JupyterHub token. Run this from your hub notebook.")
        caller = await self.run(self.auth.user_for_token, token.strip())
        if caller is None:
            raise ApiError(403, "This JupyterHub token cannot use the TPU submission service.")
        self.caller = caller
        target = self.get_query_argument("user", None)
        if target and target != caller["name"]:
            if not caller.get("admin"):
                raise ApiError(403, "Only admins can act for another user.")
            self.user = target
        else:
            self.user = caller["name"]
        return self.user


class MeHandler(UserBase):
    async def get(self):
        await self.authenticate()
        self.send_json(await self.run(self.core.me, self.user))


class PreflightHandler(UserBase):
    async def post(self):
        await self.authenticate()
        spec = self.body_json()
        self.send_json(await self.run(self.core.preflight, self.user, spec))


def _spool_user_dir(settings: Settings, user: str) -> Path:
    d = Path(settings.spool_dir) / names.user_hash(user)
    d.mkdir(parents=True, exist_ok=True)
    return d


@tornado.web.stream_request_body
class UploadHandler(UserBase):
    async def prepare(self):
        self._fh = None
        self._size = 0
        self._limit = self.cfg.max_snapshot_mb * 1024 * 1024
        if self.request.method != "PUT":
            return
        try:
            await self.authenticate()
        except ApiError as e:
            self.send_error(e.status, exc_info=(type(e), e, None))
            return
        length = int(self.request.headers.get("Content-Length") or 0)
        if length > self._limit:
            self.send_error(413, exc_info=(ApiError, ApiError(
                413, f"Snapshot is {length / 2**20:.1f} MB; the limit is {self.cfg.max_snapshot_mb} MB."
            ), None))
            return
        self.request.connection.set_max_body_size(self._limit + 1)
        self._upload_id = secrets.token_urlsafe(18)
        self._path = _spool_user_dir(self.cfg, self.user) / f"{self._upload_id}.tar.gz"
        self._fh = open(self._path, "wb")

    def data_received(self, chunk: bytes):
        if self._fh is None:
            return
        self._size += len(chunk)
        if self._size > self._limit:
            self._fh.close()
            self._fh = None
            self._path.unlink(missing_ok=True)
            raise ApiError(413, f"Snapshot is over the {self.cfg.max_snapshot_mb} MB limit.")
        self._fh.write(chunk)

    async def put(self):
        if self._fh is None:
            raise ApiError(400, "Upload failed.")
        self._fh.close()
        self._fh = None
        self.send_json({"upload_id": self._upload_id, "size": self._size})

    def on_connection_close(self):
        if getattr(self, "_fh", None) is not None:
            self._fh.close()
            self._fh = None
            self._path.unlink(missing_ok=True)


class JobsHandler(UserBase):
    async def get(self):
        await self.authenticate()
        if self.get_query_argument("all", None) == "1":
            if not self.caller.get("admin"):
                raise ApiError(403, "Only admins can list every job.")
            self.send_json({"jobs": await self.run(self.core.all_active)})
            return
        self.send_json({"jobs": await self.run(self.core.list_jobs, self.user)})

    async def post(self):
        await self.authenticate()
        spec = self.body_json()
        upload_id = spec.pop("upload_id", None)
        if not isinstance(upload_id, str) or not _UPLOAD_ID_RE.match(upload_id):
            raise ApiError(400, "upload_id is missing. Upload the snapshot first.")
        path = _spool_user_dir(self.cfg, self.user) / f"{upload_id}.tar.gz"
        if not path.is_file():
            raise ApiError(400, "That upload does not exist or has expired. Submit again.")
        try:
            req = await self.run(self.core.validate_request, self.user, spec)
            job = await self.run(self.core.submit, self.user, req, str(path))
        finally:
            path.unlink(missing_ok=True)
        log.info("submitted %s for %s (by %s)", job["id"], self.user, self.caller["name"])
        self.send_json(job, status=201)


class JobHandler(UserBase):
    async def get(self, job_id):
        await self.authenticate()
        self.send_json(await self.run(self.core.status, self.user, job_id))

    async def delete(self, job_id):
        await self.authenticate()
        job = await self.run(self.core.cancel, self.user, job_id, self.caller["name"])
        log.info("cancel %s for %s (by %s)", job_id, self.user, self.caller["name"])
        self.send_json(job)


class LogsHandler(UserBase):
    async def get(self, job_id):
        await self.authenticate()
        try:
            offset = int(self.get_query_argument("offset", "0"))
        except ValueError:
            raise ApiError(400, "offset must be an integer.") from None
        pod = self.get_query_argument("pod", None) or None
        self.send_json(await self.run(self.core.logs, self.user, job_id, offset, pod))


class FilesHandler(UserBase):
    async def get(self, job_id):
        await self.authenticate()
        self.send_json({"files": await self.run(self.core.files, self.user, job_id)})


class FileHandler(UserBase):
    async def get(self, job_id, path):
        await self.authenticate()
        await self.run(self.core._require_meta, self.user, job_id)
        await self.stream_file(await self.run(self.core.open_file, self.user, job_id, path))


class JobBase(Base):
    """Requests from a running TPU job, authenticated by its per-job token."""

    async def authenticate(self) -> tuple[str, str]:
        name = self.request.headers.get("X-TPU-Job", "")
        scheme, _, token = self.request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer":
            raise ApiError(401, "Missing job token.")
        self.user, self.job_id = await self.run(self.core.authenticate_job, name, token.strip())
        return self.user, self.job_id


class SnapshotHandler(JobBase):
    async def get(self):
        await self.authenticate()
        await self.stream_file(await self.run(self.core.open_snapshot, self.user, self.job_id))


class InputsHandler(JobBase):
    async def get(self):
        await self.authenticate()
        self.send_json(await self.run(self.core.inputs, self.user, self.job_id))


class InputFileHandler(JobBase):
    async def get(self, source, path):
        await self.authenticate()
        await self.stream_file(
            await self.run(self.core.open_input, self.user, self.job_id, path, source)
        )


@tornado.web.stream_request_body
class JobFileHandler(JobBase):
    """Streams straight into the bucket, so the service needs no disk for checkpoints.

    The object is committed only if exactly Content-Length bytes arrived.
    """

    async def prepare(self):
        self._writer = None
        self._buf = bytearray()
        self._size = 0
        try:
            await self.authenticate()
            if self.request.method != "PUT":
                return
            limit = self.cfg.max_file_mb * 1024 * 1024
            try:
                self._expected = int(self.request.headers["Content-Length"])
            except (KeyError, ValueError):
                raise ApiError(411, "Content-Length is required.") from None
            if self._expected > limit:
                raise ApiError(413, f"File is over the {self.cfg.max_file_mb} MB limit.")
            self.request.connection.set_max_body_size(limit + 1)
            path = self.path_args[0]
            self._writer = await self.run(self.core.open_upload, self.user, self.job_id, path)
        except ApiError as e:
            self.send_error(e.status, exc_info=(type(e), e, None))

    async def data_received(self, chunk: bytes):
        if self._writer is None:
            return
        self._size += len(chunk)
        if self._size > self._expected:
            self._abort()
            raise ApiError(400, "Body is longer than Content-Length.")
        self._buf += chunk
        if len(self._buf) >= 4 * _CHUNK:
            data, self._buf = bytes(self._buf), bytearray()
            await self.run(self._writer.write, data)

    def _abort(self):
        if self._writer is not None:
            writer, self._writer = self._writer, None
            self.executor.submit(writer.abort)

    async def put(self, path):
        if self._writer is None:
            raise ApiError(400, "Upload failed.")
        if self._size != self._expected:
            self._abort()
            raise ApiError(400, f"Received {self._size} of {self._expected} bytes.")
        writer, self._writer = self._writer, None
        try:
            if self._buf:
                await self.run(writer.write, bytes(self._buf))
            await self.run(writer.commit)
        except Exception:
            self.executor.submit(writer.abort)
            raise
        self.send_json({"ok": True, "size": self._size})

    async def delete(self, path):
        await self.run(self.core.delete_file, self.user, self.job_id, path)
        self.send_json({"ok": True})

    def on_connection_close(self):
        self._abort()


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.finish("ok")


def make_app(core: Core, auth, settings: Settings, executor: ThreadPoolExecutor | None = None):
    executor = executor or ThreadPoolExecutor(max_workers=32)
    Path(settings.spool_dir).mkdir(parents=True, exist_ok=True)
    kw = dict(core=core, auth=auth, settings=settings, executor=executor)
    job_id = r"([0-9]{8}-[0-9]{6}-[0-9a-f]{4})"
    return tornado.web.Application([
        (r"/healthz", HealthHandler),
        (r"/api/me", MeHandler, kw),
        (r"/api/preflight", PreflightHandler, kw),
        (r"/api/uploads", UploadHandler, kw),
        (r"/api/jobs", JobsHandler, kw),
        (rf"/api/jobs/{job_id}", JobHandler, kw),
        (rf"/api/jobs/{job_id}/logs", LogsHandler, kw),
        (rf"/api/jobs/{job_id}/files", FilesHandler, kw),
        (rf"/api/jobs/{job_id}/files/(.+)", FileHandler, kw),
        (r"/internal/snapshot", SnapshotHandler, kw),
        (r"/internal/inputs", InputsHandler, kw),
        (rf"/internal/inputs/{job_id}/(.+)", InputFileHandler, kw),
        (r"/internal/files/(.+)", JobFileHandler, kw),
    ])


def clean_spool(settings: Settings, max_age: float = _UPLOAD_MAX_AGE) -> None:
    cutoff = time.time() - max_age
    for p in Path(settings.spool_dir).rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except FileNotFoundError:
            pass


def start_background(core: Core, settings: Settings, executor: ThreadPoolExecutor) -> PeriodicCallback:
    running = False

    async def tick():
        nonlocal running
        if running:
            return
        running = True
        try:
            await IOLoop.current().run_in_executor(executor, core.reconcile)
            await IOLoop.current().run_in_executor(executor, clean_spool, settings)
        except Exception:
            log.exception("reconcile failed")
        finally:
            running = False

    pc = PeriodicCallback(tick, settings.reconcile_interval * 1000)
    pc.start()
    IOLoop.current().add_callback(tick)
    return pc
