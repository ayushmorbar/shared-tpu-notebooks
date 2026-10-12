"""Turn a Kubernetes Job (and its pods) into the state a student sees."""

from __future__ import annotations

from datetime import datetime, timezone

QUEUED = "queued"
STARTING = "starting"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"
# The Job disappeared while it was still active, e.g. a TA deleted it by hand.
LOST = "lost"

TERMINAL = frozenset({SUCCEEDED, FAILED, TIMED_OUT, CANCELLED, LOST})


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def derive(job: dict, pods: list[dict]) -> dict:
    """Return {"state", "started_at", "finished_at", "detail"} for one Job."""
    status = job.get("status") or {}
    spec = job.get("spec") or {}
    started = status.get("startTime")
    for cond in status.get("conditions") or []:
        if cond.get("status") != "True":
            continue
        if cond.get("type") == "Complete":
            return {
                "state": SUCCEEDED,
                "started_at": started,
                "finished_at": status.get("completionTime") or cond.get("lastTransitionTime"),
                "detail": "",
            }
        if cond.get("type") == "Failed":
            reason = cond.get("reason") or ""
            return {
                "state": TIMED_OUT if reason == "DeadlineExceeded" else FAILED,
                "started_at": started,
                "finished_at": cond.get("lastTransitionTime"),
                "detail": cond.get("message") or reason,
            }
    if spec.get("suspend"):
        return {"state": QUEUED, "started_at": None, "finished_at": None, "detail": ""}
    phases = {(p.get("status") or {}).get("phase") for p in pods}
    state = RUNNING if "Running" in phases else STARTING
    return {"state": state, "started_at": started, "finished_at": None, "detail": ""}
