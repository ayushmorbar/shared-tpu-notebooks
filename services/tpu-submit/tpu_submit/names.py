"""Job ids, Kubernetes-safe names, and object-store paths.

Usernames come from JupyterHub (the local part of the IAP email). They can contain dots
and plus signs, which Kubernetes names and label values do not allow, so every
Kubernetes-facing name goes through a slug, and lookups use a hash label instead of the
slug. Two students whose names slug to the same string still get distinct hashes.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone
from pathlib import PurePosixPath

MANAGED_BY = "tpu-submit"
LABEL_MANAGED_BY = "app.kubernetes.io/managed-by"
LABEL_USER_HASH = "tpu-submit/user-hash"
LABEL_JOB_ID = "tpu-submit/job-id"
ANN_USER = "tpu-submit/user"
ANN_JOB_ID = "tpu-submit/job-id"
ANN_TOKEN_SHA = "tpu-submit/token-sha256"
ANN_SCRIPT = "tpu-submit/script"

JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_USER_DIR_RE = re.compile(r"^[a-z0-9][a-z0-9._+@-]{0,127}$")


def new_job_id(now: datetime | None = None) -> str:
    """Sortable, unique-enough id: 20261010-221530-a1b2."""
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def user_slug(user: str) -> str:
    slug = _SLUG_RE.sub("-", user.lower()).strip("-")[:28].rstrip("-")
    return slug or "user"


def user_hash(user: str) -> str:
    return hashlib.sha256(user.encode()).hexdigest()[:20]


def job_name(user: str, job_id: str) -> str:
    """Kubernetes Job name. At most 53 characters, under the 63 a Job allows."""
    return f"tpu-{user_slug(user)}-{job_id}"


def user_dir(user: str) -> str:
    """Folder name for a user in the bucket. Readable when the name is ordinary."""
    if _USER_DIR_RE.match(user) and ".." not in user:
        return user
    return f"h-{user_hash(user)}"


def job_prefix(user: str, job_id: str) -> str:
    return f"users/{user_dir(user)}/jobs/{job_id}/"


def clean_relpath(path: str, max_len: int = 512) -> str | None:
    """Return a normalised relative POSIX path, or None if it escapes or is malformed."""
    if not path or len(path) > max_len or "\\" in path or "\x00" in path:
        return None
    p = PurePosixPath(path)
    if p.is_absolute():
        return None
    parts = p.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return None
    return "/".join(parts)


def token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
