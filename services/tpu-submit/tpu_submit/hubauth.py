"""Resolve a JupyterHub API token to a user, the same way JupyterHub's HubAuth does.

GET /hub/api/user with the caller's token returns the token owner and the token's
scopes. Any token may identify itself there. Access is then limited to tokens holding
access:services (or access:services!service=<name>), which the hub config grants to
every user's server token.
"""

from __future__ import annotations

import hashlib
import threading
import time

import requests


class HubAuth:
    def __init__(self, api_url: str, service_name: str, require_access_scope: bool = True,
                 cache_seconds: int = 60):
        self.api_url = api_url.rstrip("/")
        self.access_scopes = {"access:services", f"access:services!service={service_name}"}
        self.require_access_scope = require_access_scope
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, dict | None]] = {}
        self._lock = threading.Lock()
        self._session = requests.Session()

    def user_for_token(self, token: str) -> dict | None:
        """Return {"name", "admin"} for a valid token with access, else None."""
        if not token:
            return None
        key = hashlib.sha256(token.encode()).hexdigest()
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        model = self._fetch(token)
        with self._lock:
            if len(self._cache) > 10000:
                self._cache.clear()
            self._cache[key] = (now + self.cache_seconds, model)
        return model

    def _fetch(self, token: str) -> dict | None:
        resp = self._session.get(
            f"{self.api_url}/user",
            headers={"Authorization": f"token {token}"},
            timeout=10,
        )
        if resp.status_code in (401, 403, 404):
            return None
        resp.raise_for_status()
        model = resp.json()
        if model.get("kind", "user") != "user" or not model.get("name"):
            return None
        scopes = model.get("scopes")
        if self.require_access_scope and scopes is not None:
            if not self.access_scopes.intersection(scopes):
                return None
        admin = bool(model.get("admin")) or "admin:users" in (scopes or ())
        return {"name": model["name"], "admin": admin}
