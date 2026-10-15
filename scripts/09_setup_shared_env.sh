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

ENV_NAME="${ENV_NAME:-${SHARED_ENV_NAME:-idl-shared-env}}"
REQ_FILE="${REQ_FILE:-${REQUIREMENTS_FILE:-requirements.txt}}"

# Resolve path to requirements.txt (relative to repo root or cwd)
if [[ ! -f "${REQ_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/${REQ_FILE}" ]]; then
    REQ_FILE="${REPO_ROOT}/${REQ_FILE}"
  elif [[ -f "${REPO_ROOT}/docker/requirements.txt" ]]; then
    REQ_FILE="${REPO_ROOT}/docker/requirements.txt"
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
DISPLAY_NAME="Python 3.11 (CMU 11-785: ${ENV_NAME})"

log_header "Provisioning Zero-Rebuild Shared Python Environment"
log_info "Environment Name : ${ENV_NAME}"
log_info "Target Directory : ${SHARED_ENV_PATH}"
log_info "Requirements File: ${REQ_FILE}"
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
          command:
            - "bash"
            - "-c"
            - |
              set -ex
              echo "==> Initializing Tartan shared directories..."
              mkdir -p "${TARTAN_MOUNT_PATH}/envs" "${TARTAN_MOUNT_PATH}/share/jupyter/kernels"

              if [ ! -d "${SHARED_ENV_PATH}" ]; then
                echo "==> Creating virtual environment at ${SHARED_ENV_PATH}..."
                python3 -m venv "${SHARED_ENV_PATH}" --system-site-packages
              fi

              echo "==> Upgrading pip..."
              "${SHARED_ENV_PATH}/bin/pip" install --upgrade pip

              echo "==> Installing packages from requirements.txt..."
              "${SHARED_ENV_PATH}/bin/pip" install -r /etc/requirements/requirements.txt

              echo "==> Registering system-wide Jupyter Kernel..."
              "${SHARED_ENV_PATH}/bin/python" -m ipykernel install \
                --prefix="${TARTAN_MOUNT_PATH}" \
                --name="${KERNEL_NAME}" \
                --display-name="${DISPLAY_NAME}"

              echo "✓ Shared environment '${ENV_NAME}' successfully initialized!"
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
              mountOptions: "implicit-dirs,file-cache:max-size-mb:20000,metadata-cache:ttl-secs:60"
        - name: requirements-vol
          configMap:
            name: ${CM_NAME}
EOF

log_info "Waiting for environment builder job to complete..."
kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" --timeout=1200s 2>/dev/null || {
  log_warn "Job did not complete within timeout. Streaming logs for diagnosis:"
  kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" -c env-builder --tail=80
  exit 1
}

log_success "Shared course environment '${ENV_NAME}' is active and ready on ${SHARED_ENV_PATH}."
log_info "Jupyter Kernel registered: '${DISPLAY_NAME}'."
