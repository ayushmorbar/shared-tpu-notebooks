#!/usr/bin/env bash
# ==============================================================================
# Script: 13_smoke_submit.sh
# Description: One real job through the TPU submission service, end to end.
#
# `make smoke` proves a chip can run a pod. This proves the student path: upload a
# script folder, queue, run on a v5e chip through the job wrapper, stream the log,
# upload expts/, and download it again. It uses the same submit_tpu.py students use.
#
# It talks to the service through a port-forward, as you, with a JupyterHub API token:
#   1. Open https://<your hub>/hub/token, request a token (default permissions), copy it.
#   2. HUB_TOKEN=... make smoke-submit
# Set SMOKE_DATASET=<name> to also check that dataset unpacks on the chip.
# ==============================================================================

source "$(dirname "$0")/common.sh"
check_prereqs kubectl python3 curl
ensure_k8s_context

if [[ -z "${HUB_TOKEN:-}" ]]; then
  log_error "HUB_TOKEN is not set."
  echo "       Create one at https://<your hub>/hub/token, then: HUB_TOKEN=... make smoke-submit" >&2
  exit 1
fi
python3 -c "import requests" 2>/dev/null || { log_error "python3 needs requests (pip install requests)"; exit 1; }

PORT="${SMOKE_PORT:-18080}"
WORK=$(mktemp -d)
kubectl -n "${NAMESPACE}" port-forward svc/tpu-submit "${PORT}:8080" >"${WORK}/pf.log" 2>&1 &
PF=$!
trap 'kill ${PF} 2>/dev/null; rm -rf "${WORK}"' EXIT
for _ in $(seq 1 30); do
  curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null || { cat "${WORK}/pf.log"; log_error "service not reachable"; exit 1; }

log_header "Submitting a smoke job through tpu-submit"
mkdir -p "${WORK}/smoke"
cat > "${WORK}/smoke/hello.py" <<'PY'
import os, pathlib
import jax
dev = jax.devices()[0]
print("devices:", jax.devices())
assert dev.platform == "tpu", f"expected a TPU, got {dev.platform}"
root = os.environ.get("DATA_ROOT")
if root:
    print("DATA_ROOT:", root, sorted(os.listdir(root))[:10])
out = pathlib.Path(os.environ["OUTPUT_DIR"]) / "smoke"
out.mkdir(parents=True, exist_ok=True)
(out / "ok.txt").write_text(str(dev))
print("smoke ok")
PY

TPU_SUBMIT_URL="http://127.0.0.1:${PORT}" JUPYTERHUB_API_TOKEN="${HUB_TOKEN}" \
PYTHONPATH="${REPO_ROOT}/notebooks" SMOKE_WORK="${WORK}" python3 - <<'PY'
import os, sys
from pathlib import Path
import submit_tpu

work = Path(os.environ["SMOKE_WORK"])
print("service sees:", submit_tpu.usage())
job = submit_tpu.submit(str(work / "smoke" / "hello.py"),
                        dataset=os.environ.get("SMOKE_DATASET") or None)
info = submit_tpu.watch(job, poll=5)
if info["state"] != "succeeded":
    sys.exit(f"smoke job ended {info['state']}: {info.get('detail')}")
dest = submit_tpu.pull(job, dest=work / "out")
ok = dest / "expts" / "smoke" / "ok.txt"
if not ok.exists():
    sys.exit("job succeeded but expts/smoke/ok.txt was not uploaded")
print("downloaded:", ok.read_text())
PY
log_success "submission service smoke test passed"
