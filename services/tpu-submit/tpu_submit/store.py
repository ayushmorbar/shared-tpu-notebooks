"""Object storage for snapshots, logs, and checkpoints.

GCSStore is the real one. LocalStore keeps the same layout on disk and exists for tests
and local development; it is never used in the cluster.
"""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path
from typing import BinaryIO, Protocol


class Writer(Protocol):
    """A streamed upload. Nothing is visible under the key until commit()."""

    def write(self, data: bytes) -> None: ...
    def commit(self) -> None: ...
    def abort(self) -> None: ...


class Store(Protocol):
    def put_file(self, key: str, path: str) -> None: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def open_read(self, key: str) -> tuple[BinaryIO, int] | None: ...
    def open_write(self, key: str) -> Writer: ...
    def list(self, prefix: str) -> list[tuple[str, int]]: ...
    def delete(self, key: str) -> None: ...


class _LocalWriter:
    def __init__(self, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._dst = dst
        self._tmp = dst.with_name(f"{dst.name}.{secrets.token_hex(4)}.part")
        self._fh = self._tmp.open("wb")

    def write(self, data: bytes) -> None:
        self._fh.write(data)

    def commit(self) -> None:
        self._fh.close()
        os.replace(self._tmp, self._dst)

    def abort(self) -> None:
        self._fh.close()
        self._tmp.unlink(missing_ok=True)


class _GCSWriter:
    def __init__(self, blob):
        self._w = blob.open("wb", chunk_size=8 * 1024 * 1024, ignore_flush=True)

    def write(self, data: bytes) -> None:
        self._w.write(data)

    def commit(self) -> None:
        self._w.close()

    def abort(self) -> None:
        # BlobWriter.close() finalizes the object with whatever was buffered, and
        # IOBase.__del__ calls close(). Closing the buffer first turns that into a no-op,
        # so an interrupted upload never replaces a good checkpoint.
        buf = getattr(self._w, "_buffer", None)
        if buf is not None:
            buf.close()


class LocalStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"key escapes store: {key!r}")
        return p

    def put_file(self, key: str, path: str) -> None:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dst)

    def put_bytes(self, key: str, data: bytes) -> None:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)

    def get_bytes(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.is_file() else None

    def open_read(self, key: str) -> tuple[BinaryIO, int] | None:
        p = self._path(key)
        if not p.is_file():
            return None
        return p.open("rb"), p.stat().st_size

    def open_write(self, key: str) -> _LocalWriter:
        return _LocalWriter(self._path(key))

    def list(self, prefix: str) -> list[tuple[str, int]]:
        base = self._path(prefix.rstrip("/")) if prefix.strip("/") else self.root
        if not base.exists():
            return []
        out = []
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.endswith(".part"):
                out.append((p.relative_to(self.root).as_posix(), p.stat().st_size))
        return out

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_file():
            p.unlink()


class GCSStore:
    def __init__(self, bucket: str):
        from google.cloud import storage
        from google.api_core.exceptions import NotFound

        self._NotFound = NotFound
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    def put_file(self, key: str, path: str) -> None:
        self._bucket.blob(key).upload_from_filename(path, timeout=600)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._bucket.blob(key).upload_from_string(data)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self._bucket.blob(key).download_as_bytes()
        except self._NotFound:
            return None

    def open_read(self, key: str) -> tuple[BinaryIO, int] | None:
        blob = self._bucket.get_blob(key)
        if blob is None:
            return None
        return blob.open("rb", chunk_size=8 * 1024 * 1024), int(blob.size or 0)

    def open_write(self, key: str) -> _GCSWriter:
        return _GCSWriter(self._bucket.blob(key))

    def list(self, prefix: str) -> list[tuple[str, int]]:
        return [
            (b.name, int(b.size or 0))
            for b in self._client.list_blobs(self._bucket, prefix=prefix)
        ]

    def delete(self, key: str) -> None:
        try:
            self._bucket.blob(key).delete()
        except self._NotFound:
            pass
