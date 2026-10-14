#!/usr/bin/env bash
# ==============================================================================
# Script: 09_setup_shared_env.sh
# Description: Initialize or update the shared course Python virtual environment
#              directly on the central Tartan GCS store (/tartan/11785/envs/idl-shared-env).
#
# Zero Docker Rebuilds:
# Instead of building and hosting heavy container images in Artifact Registry,
# packages are installed into the central shared volume. All 300+ students instantly
# receive new libraries and kernels without restarting their servers or rebuilding Docker.
# ==============================================================================
source "$(dirname "$0")/common.sh"
require_project
check_prereqs kubectl

JOB_NAME="setup-shared-env-$(date +%s)"
SHARED_ENV_PATH="${TARTAN_MOUNT_PATH}/envs/idl-shared-env"

log_header "Provisioning Zero-Rebuild Shared Python Environment"
log_info "Target Directory: ${SHARED_ENV_PATH}"
log_info "Using Base Image: ${SINGLEUSER_IMAGE}:${SINGLEUSER_IMAGE_TAG}"

ensure_k8s_context

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

              echo "==> Upgrading pip and installing core course packages..."
              "${SHARED_ENV_PATH}/bin/pip" install --upgrade pip
              "${SHARED_ENV_PATH}/bin/pip" install \
                jax[tpu] \
                torch torchvision torchaudio \
                augmax \
                kaggle wandb \
                optax flax \
                timm scikit-learn \
                kubernetes tqdm PyYAML

              echo "==> Registering system-wide Jupyter Kernel..."
              "${SHARED_ENV_PATH}/bin/python" -m ipykernel install \
                --prefix="${TARTAN_MOUNT_PATH}" \
                --name="idl-tpu-kernel" \
                --display-name="Python 3.11 (CMU 11-785 TPU Shared Env)"

              echo "✓ Shared environment successfully initialized!"
          volumeMounts:
            - name: course-tartan-rw
              mountPath: ${TARTAN_MOUNT_PATH}
      volumes:
        - name: course-tartan-rw
          csi:
            driver: gcsfuse.csi.storage.gke.io
            volumeAttributes:
              bucketName: ${TARTAN_BUCKET:-${OCEAN_BUCKET:-cmu-idl-ocean}}
              mountOptions: "implicit-dirs,file-cache:max-size-mb:20000,metadata-cache:ttl-secs:60"
EOF

log_info "Waiting for environment builder job to complete..."
kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" --timeout=1200s 2>/dev/null || {
  log_warn "Job did not complete within timeout. Streaming logs for diagnosis:"
  kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" -c env-builder --tail=50
  exit 1
}

log_success "Shared course environment is active and ready on ${SHARED_ENV_PATH}."
kubectl -n "${NAMESPACE}" delete "job/${JOB_NAME}" >/dev/null 2>&1 || true
