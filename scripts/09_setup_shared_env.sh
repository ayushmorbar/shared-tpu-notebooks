#!/usr/bin/env bash
# ==============================================================================
# Script: 09_setup_shared_env.sh
# Description: Initialize or update a shared course Python virtual environment
#              directly on the central Tartan GCS store (/tartan/11785/envs/<ENV_NAME>).
#
# Zero Docker Rebuilds:
# Instead of building and hosting heavy container images in Artifact Registry,
# packages are installed into the central shared volume from requirements.txt.
# All 300+ students instantly receive new libraries and kernels without restarting
# their servers or rebuilding Docker.
#
# Usage:
#   bash scripts/09_setup_shared_env.sh
#   ENV_NAME=capstone-env REQ_FILE=requirements.txt bash scripts/09_setup_shared_env.sh
# ==============================================================================
source "$(dirname "$0")/common.sh"
require_project
check_prereqs kubectl

ENV_NAME="${ENV:-${ENV_NAME:-${SHARED_ENV_NAME:-idlf26jax}}}"
REQ_FILE="${REQ:-${REQ_FILE:-${REQUIREMENTS_FILE:-requirements.txt}}}"
ENV_BUILDER_CPU="${ENV_BUILDER_CPU:-2}"
ENV_BUILDER_MEMORY="${ENV_BUILDER_MEMORY:-8Gi}"
ENV_BUILDER_STORAGE="${ENV_BUILDER_STORAGE:-5Gi}"

# Strip stray outer quotes if passed from makefile or shell
ENV_NAME="${ENV_NAME%\"}"
ENV_NAME="${ENV_NAME#\"}"
ENV_NAME="${ENV_NAME%\'}"
ENV_NAME="${ENV_NAME#\'}"
REQ_FILE="${REQ_FILE%\"}"
REQ_FILE="${REQ_FILE#\"}"
REQ_FILE="${REQ_FILE%\'}"
REQ_FILE="${REQ_FILE#\'}"
ENV_BUILDER_CPU="${ENV_BUILDER_CPU%\"}"
ENV_BUILDER_CPU="${ENV_BUILDER_CPU#\"}"
ENV_BUILDER_MEMORY="${ENV_BUILDER_MEMORY%\"}"
ENV_BUILDER_MEMORY="${ENV_BUILDER_MEMORY#\"}"
ENV_BUILDER_STORAGE="${ENV_BUILDER_STORAGE%\"}"
ENV_BUILDER_STORAGE="${ENV_BUILDER_STORAGE#\"}"

# Resolve path to requirements.txt (relative to repo root or cwd)
if [[ ! -f "${REQ_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/${REQ_FILE}" ]]; then
    REQ_FILE="${REPO_ROOT}/${REQ_FILE}"
  else
    echo "Error: Requirements file '${REQ_FILE}' not found." >&2
    exit 1
  fi
fi

SAFE_ENV_NAME="$(echo "${ENV_NAME}" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -cd 'a-z0-9-')"
JOB_ID="$(date +%s)"
JOB_NAME="setup-env-${SAFE_ENV_NAME}-${JOB_ID}"
CM_NAME="env-req-${SAFE_ENV_NAME}-${JOB_ID}"
SHARED_ENV_PATH="${TARTAN_MOUNT_PATH}/envs/${ENV_NAME}"
KERNEL_NAME="kernel-${SAFE_ENV_NAME}"

log_header "Provisioning Zero-Rebuild Shared Python Environment"
log_info "Environment Name : ${ENV_NAME}"
log_info "Target Directory : ${SHARED_ENV_PATH}"
log_info "Requirements File: ${REQ_FILE}"
log_info "Builder Resources: CPU=${ENV_BUILDER_CPU}, RAM=${ENV_BUILDER_MEMORY}, Storage=${ENV_BUILDER_STORAGE}"
log_info "Base Image       : ${SINGLEUSER_IMAGE}:${SINGLEUSER_IMAGE_TAG}"

ensure_k8s_context

# Clean up resources on exit
cleanup() {
  kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" >/dev/null 2>&1 || true
  kubectl -n "${NAMESPACE}" delete configmap "${CM_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 1. Create temporary ConfigMap for requirements.txt
log_info "Staging requirements file as ConfigMap '${CM_NAME}'..."
kubectl -n "${NAMESPACE}" create configmap "${CM_NAME}" \
  --from-file="requirements.txt=${REQ_FILE}" \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Launch batch Job to provision venv on Tartan
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
  annotations:
    gke-gcsfuse/volumes: "true"
spec:
  ttlSecondsAfterFinished: 600
  backoffLimit: 1
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      serviceAccountName: student
      restartPolicy: Never
      containers:
        - name: env-builder
          image: ${SINGLEUSER_IMAGE}:${SINGLEUSER_IMAGE_TAG}
          securityContext:
            runAsUser: 1000
            runAsGroup: 1000
          resources:
            requests:
              cpu: "${ENV_BUILDER_CPU}"
              memory: "${ENV_BUILDER_MEMORY}"
              ephemeral-storage: "${ENV_BUILDER_STORAGE}"
            limits:
              cpu: "$((ENV_BUILDER_CPU * 2))"
              memory: "${ENV_BUILDER_MEMORY}"
              ephemeral-storage: "${ENV_BUILDER_STORAGE}"
          command:
            - "bash"
            - "-c"
            - |
              set -ex
              echo "==> Initializing Tartan shared directories..."
              mkdir -p "${TARTAN_MOUNT_PATH}/envs" "${TARTAN_MOUNT_PATH}/share/jupyter/kernels"

              if [ ! -d "${SHARED_ENV_PATH}" ]; then
                echo "==> Creating virtual environment at ${SHARED_ENV_PATH}..."
                python3 -m venv "${SHARED_ENV_PATH}" --system-site-packages --copies
              fi

              echo "==> Upgrading pip..."
              "${SHARED_ENV_PATH}/bin/pip" install --upgrade pip

              echo "==> Installing packages from requirements.txt..."
              "${SHARED_ENV_PATH}/bin/pip" install --no-cache-dir -r /etc/requirements/requirements.txt

              echo "==> Registering system-wide Jupyter Kernel..."
              SITE_PACKAGES=\$("${SHARED_ENV_PATH}/bin/python" -c "import site; print(site.getsitepackages()[0])")
              PY_VER=\$("${SHARED_ENV_PATH}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
              "${SHARED_ENV_PATH}/bin/python" -m ipykernel install \
                --prefix="${TARTAN_MOUNT_PATH}" \
                --name="${KERNEL_NAME}" \
                --display-name="Python \${PY_VER} (CMU 11-785: ${ENV_NAME})" \
                --env PYTHONPATH "\${SITE_PACKAGES}:/opt/course"

              echo "✓ Shared environment '${ENV_NAME}' (Python \${PY_VER}) successfully initialized!"
          volumeMounts:
            - name: course-tartan-rw
              mountPath: ${TARTAN_MOUNT_PATH}
            - name: requirements-vol
              mountPath: /etc/requirements
      volumes:
        - name: course-tartan-rw
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes:
              bucketName: ${TARTAN_BUCKET:-${OCEAN_BUCKET:-cmu-idl-ocean}}
              mountOptions: "implicit-dirs,file-cache:max-size-mb:20000,metadata-cache:ttl-secs:60,uid=1000,gid=1000,dir-mode=0777,file-mode=0777"
        - name: requirements-vol
          configMap:
            name: ${CM_NAME}
EOF

log_info "Waiting for builder pod to schedule and initialize..."
POD_NAME=""
for i in {1..60}; do
  POD_NAME=$(kubectl -n "${NAMESPACE}" get pods -l job-name="${JOB_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -n "${POD_NAME}" ]]; then
    break
  fi
  sleep 2
done

if [[ -z "${POD_NAME}" ]]; then
  log_warn "Pod for job '${JOB_NAME}' not found after 120s. Job diagnostics:"
  kubectl -n "${NAMESPACE}" describe job "${JOB_NAME}" || true
  exit 1
fi

log_info "Builder Pod: ${POD_NAME}"
log_info "Streaming build and package installation logs in real time..."
kubectl -n "${NAMESPACE}" logs -f "${POD_NAME}" -c env-builder --pod-running-timeout=10m || true

if ! kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" --timeout=30s >/dev/null 2>&1; then
  log_warn "Job did not report complete. Fetching diagnostic pod events:"
  kubectl -n "${NAMESPACE}" describe pod "${POD_NAME}" | tail -n 25
  exit 1
fi

log_success "Shared course environment '${ENV_NAME}' is active and ready on ${SHARED_ENV_PATH}."
log_info "Jupyter Kernel registered: 'kernel-${SAFE_ENV_NAME}' (prefix: ${TARTAN_MOUNT_PATH})."
