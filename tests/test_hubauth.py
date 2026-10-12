"""Token -> user resolution against a stubbed JupyterHub /hub/api/user."""

from __future__ import annotations

import pytest

from tpu_submit.hubauth import HubAuth


class Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.fixture
def hub(monkeypatch):
    replies = {}
    calls = []
    auth = HubAuth("http://hub:8081/hub/api", "tpu-submit")

    def get(url, headers, timeout):
        token = headers["Authorization"].split()[1]
        calls.append(token)
        return replies.get(token, Resp(403))

    monkeypatch.setattr(auth._session, "get", get)
    return auth, replies, calls


def test_server_token_with_service_scope(hub):
    auth, replies, _ = hub
    replies["t"] = Resp(200, {"kind": "user", "name": "alice", "scopes": [
        "users:activity!user=alice", "access:services!service=tpu-submit"]})
    assert auth.user_for_token("t") == {"name": "alice", "admin": False}


def test_token_without_service_scope_is_refused(hub):
    auth, replies, _ = hub
    replies["t"] = Resp(200, {"kind": "user", "name": "alice", "scopes": ["read:users:name!user=alice"]})
    assert auth.user_for_token("t") is None


def test_service_tokens_and_bad_tokens_are_refused(hub):
    auth, replies, _ = hub
    replies["svc"] = Resp(200, {"kind": "service", "name": "cull", "scopes": ["access:services"]})
    assert auth.user_for_token("svc") is None
    assert auth.user_for_token("nope") is None
    assert auth.user_for_token("") is None


def test_admin_from_scope(hub):
    auth, replies, _ = hub
    replies["t"] = Resp(200, {"kind": "user", "name": "prof", "scopes": ["admin:users", "access:services"]})
    assert auth.user_for_token("t") == {"name": "prof", "admin": True}


def test_results_are_cached(hub):
    auth, replies, calls = hub
    replies["t"] = Resp(200, {"kind": "user", "name": "alice", "scopes": ["access:services"]})
    for _ in range(3):
        auth.user_for_token("t")
    assert calls == ["t"]
