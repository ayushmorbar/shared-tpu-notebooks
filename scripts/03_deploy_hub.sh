#!/usr/bin/env bash
# ==============================================================================
# Script: 03_deploy_hub.sh
# Description: Deploy the TPU submission service, then JupyterHub.
#
# The submission service is the security boundary of the classroom. Students have no
# Kubernetes permissions and their notebooks carry no Kubernetes token. They send a
# script folder to tpu-submit with their JupyterHub token, and tpu-submit builds every
# Job from one fixed template: one chip, the course image, the queue, a deadline, and
# a read-only mount of /tartan/11785. A student cannot pick a node, an image, or a
# service account, and cannot see or touch anyone else's jobs, logs, or files.
# ==============================================================================
source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud kubectl helm

RELEASE="${RELEASE:-hub}"
CHART_VERSION="${CHART_VERSION:-4.4.1}"

ensure_k8s_context

# Earlier versions gave every notebook a RoleBinding that could create, list, and
# delete any Job in the namespace and read every pod's logs. Remove those RBAC rules.
log_header "Removing direct Kubernetes Job RBAC from student notebooks"
kubectl -n "${NAMESPACE}" delete rolebinding student-tpu-submit --ignore-not-found
kubectl -n "${NAMESPACE}" delete role student-tpu-submit --ignore-not-found

log_header "Deploying the TPU submission service"
if [[ -n "${DEFAULT_DATASET}" && ",${DATASETS}," != *",${DEFAULT_DATASET}="* && -z "${TARTAN_BUCKET}" ]]; then
  log_error "DEFAULT_DATASET=${DEFAULT_DATASET} is not one of DATASETS (${DATASETS:-empty})"
  exit 1
fi
if [[ -z "${DATASETS}" ]]; then
  log_info "DATASETS is empty: jobs can still read datasets from ${TARTAN_MOUNT_PATH}/datasets/ or upload tarballs via scripts/12_setup_storage.sh."
fi
if ! gcloud storage buckets describe "gs://${STUDENT_BUCKET}" --project="${PROJECT}" >/dev/null 2>&1; then
  log_error "gs://${STUDENT_BUCKET} does not exist. Run make storage first."
  exit 1
fi
log_info "service image: ${SUBMIT_IMAGE}"
log_info "job image:     ${TPU_JOB_IMAGE}"
log_info "tartan bucket: ${TARTAN_BUCKET}"
log_info "runs bucket:   ${STUDENT_BUCKET}"
sed -e "s|__NAMESPACE__|${NAMESPACE}|g" \
    -e "s|__SUBMIT_IMAGE__|${SUBMIT_IMAGE}|g" \
    -e "s|__TPU_JOB_IMAGE__|${TPU_JOB_IMAGE}|g" \
    -e "s|__STUDENT_BUCKET__|${STUDENT_BUCKET}|g" \
    -e "s|__TARTAN_BUCKET__|${TARTAN_BUCKET}|g" \
    -e "s|__SHARED_ENV_NAME__|${SHARED_ENV_NAME}|g" \
    -e "s|__DATASETS__|${DATASETS}|g" \
    -e "s|__DEFAULT_DATASET__|${DEFAULT_DATASET}|g" \
    -e "s|__TPU_ACCELERATOR__|${TPU_ACCELERATOR}|g" \
    -e "s|__TPU_TOPOLOGY__|${TPU_TOPOLOGY}|g" \
    -e "s|__MAX_JOB_HOURS__|${MAX_JOB_HOURS}|g" \
    -e "s|__CHIP_HOUR_BUDGET__|${CHIP_HOUR_BUDGET}|g" \
    -e "s|__JOB_DISK__|${JOB_DISK}|g" \
    "${REPO_ROOT}/k8s/tpu-submit.yaml" | kubectl apply -f -
# Settings live in the pod spec, so a config change rolls the pod by itself; an image
# rebuild under the same tag does not, hence the explicit restart.
kubectl -n "${NAMESPACE}" rollout restart deployment/tpu-submit
kubectl -n "${NAMESPACE}" rollout status deployment/tpu-submit --timeout=10m

echo "==> helm repo"
helm repo add jupyterhub https://hub.jupyter.org/helm-chart/ >/dev/null 2>&1 || true
helm repo update >/dev/null

VALUES_TEMPLATE="${REPO_ROOT}/k8s/jupyterhub-values.yaml"
VALUES=$(mktemp)
trap 'rm -f "${VALUES}"' EXIT
sed -e "s|__TARTAN_BUCKET__|${TARTAN_BUCKET}|g" \
    -e "s|__OCEAN_BUCKET__|${TARTAN_BUCKET}|g" \
    -e "s|__PROJECT__|${PROJECT}|g" \
    -e "s|__SHARED_ENV_NAME__|${SHARED_ENV_NAME}|g" \
    -e "s|__CULL_ADMIN_USERS__|${CULL_ADMIN_USERS:-false}|g" \
    "${VALUES_TEMPLATE}" > "${VALUES}"

ADMIN_USERS="${ADMIN_USERS:-instructor}"
ADMIN_LIST=()
for u in $(echo "${ADMIN_USERS}" | tr ',' ' '); do
  uname=$(echo "${u}" | sed -e 's/^user://' -e 's/@.*//')
  [[ -n "${uname}" ]] && ADMIN_LIST+=("${uname}")
done
ADMIN_STR=$(IFS=,; echo "${ADMIN_LIST[*]}")
echo "==> configured JupyterHub admin users: ${ADMIN_STR}"

echo "==> installing JupyterHub ${CHART_VERSION} into ${NAMESPACE}"
helm upgrade --install "${RELEASE}" jupyterhub/jupyterhub \
  --namespace "${NAMESPACE}" \
  --version "${CHART_VERSION}" \
  --values "${VALUES}" \
  --set singleuser.image.name="${SINGLEUSER_IMAGE}" \
  --set singleuser.image.tag="${SINGLEUSER_IMAGE_TAG}" \
  --set "hub.config.Authenticator.admin_users={${ADMIN_STR}}" \
  --set-file "singleuser.extraFiles.platform_config\.py.stringData=${REPO_ROOT}/notebooks/platform_config.py" \
  --set-file "singleuser.extraFiles.submit_tpu\.py.stringData=${REPO_ROOT}/notebooks/submit_tpu.py" \
  --set-file "singleuser.extraFiles.student\.env\.template.stringData=${REPO_ROOT}/notebooks/student.env.template" \
  --set-file "singleuser.extraFiles.hw0_tpu_hello\.ipynb.stringData=${REPO_ROOT}/notebooks/hw0_tpu_hello.ipynb" \
  --set-file "singleuser.extraFiles.tpu_platform_guide\.ipynb.stringData=${REPO_ROOT}/notebooks/00_tpu_platform_guide.ipynb" \
  --timeout 20m \
  --wait

echo
kubectl -n "${NAMESPACE}" get pods
echo

# proxy-public is a ClusterIP now, so there is no external address to wait for.
# The hub is reached through the Ingress instead. Run 08_setup_iap.sh once per cluster.
if kubectl -n "${NAMESPACE}" get ingress hub-ingress >/dev/null 2>&1; then
  DOMAIN=$(kubectl -n "${NAMESPACE}" get managedcertificate hub-cert \
             -o jsonpath='{.spec.domains[0]}' 2>/dev/null || true)
  STATUS=$(kubectl -n "${NAMESPACE}" get managedcertificate hub-cert \
             -o jsonpath='{.status.certificateStatus}' 2>/dev/null || true)
  echo "hub: https://${DOMAIN}   (certificate: ${STATUS:-unknown})"
else
  echo "No ingress yet. Put the hub behind HTTPS and Google sign-in with:"
  echo "  bash $(dirname "$0")/08_setup_iap.sh"
fi
