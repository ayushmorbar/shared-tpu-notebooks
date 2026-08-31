#!/usr/bin/env bash
# ==============================================================================
# Script: 03_deploy_hub.sh
# Description: Install JupyterHub and give student notebooks permission to submit TPU Jobs.
#
# The RBAC below is the security boundary of the classroom. A student notebook can
# create, watch and delete Jobs in its own namespace and read their logs. It cannot
# ==============================================================================
# touch another namespace, cannot create a Job with a node selector of its choosing
# outside what the pod template allows, and cannot edit the ClusterQueue that caps
# the pool. Student code is untrusted; treat it that way.
source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud kubectl helm

RELEASE="${RELEASE:-hub}"
CHART_VERSION="${CHART_VERSION:-4.4.1}"

ensure_k8s_context

log_header "Configuring Student RBAC in Namespace '${NAMESPACE}'"
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: student
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: student-tpu-submit
  namespace: ${NAMESPACE}
rules:
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete"]
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  # Read-only on their own queued work, so a student can see their place in line.
  - apiGroups: ["kueue.x-k8s.io"]
    resources: ["workloads", "localqueues"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: student-tpu-submit
  namespace: ${NAMESPACE}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: student-tpu-submit
subjects:
  - kind: ServiceAccount
    name: student
    namespace: ${NAMESPACE}
EOF

echo "==> helm repo"
helm repo add jupyterhub https://hub.jupyter.org/helm-chart/ >/dev/null 2>&1 || true
helm repo update >/dev/null

# The singleuser NetworkPolicy has to name the API server's real endpoint, because
# Dataplane V2 applies policy after the Service DNAT and the ClusterIP never matches.
# Resolve it via modern discovery.k8s.io/v1 EndpointSlices with fallback to v1 Endpoints.
APISERVER=$(kubectl get endpointslices -n default -l kubernetes.io/service-name=kubernetes \
              -o jsonpath='{.items[0].endpoints[0].addresses[0]}' 2>/dev/null \
            || kubectl get endpoints kubernetes -n default \
              -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || true)
if [[ -z "${APISERVER}" ]]; then
  echo "could not resolve the API server endpoint; student notebooks will not be able" >&2
  echo "to submit TPU jobs. Check: kubectl get endpointslices -n default" >&2
  exit 1
fi
echo "==> API server endpoint ${APISERVER} (not the ClusterIP -- see the values file)"

VALUES=$(mktemp)
trap 'rm -f "${VALUES}"' EXIT
sed "s|__APISERVER_ENDPOINT__|${APISERVER}|" \
  "$(dirname "$0")/../k8s/jupyterhub-values.yaml" > "${VALUES}"

IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT}/course-images/scipy-notebook"
IMAGE_TAG="latest"

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
  --set singleuser.image.name="${IMAGE_NAME}" \
  --set singleuser.image.tag="${IMAGE_TAG}" \
  --set "hub.config.Authenticator.admin_users={${ADMIN_STR}}" \
  --set-file "singleuser.extraFiles.submit_tpu\.py.stringData=$(dirname "$0")/../notebooks/submit_tpu.py" \
  --set-file "singleuser.extraFiles.hw0_tpu_hello\.ipynb.stringData=$(dirname "$0")/../notebooks/hw0_tpu_hello.ipynb" \
  --timeout 20m \
  --wait

echo
kubectl -n "${NAMESPACE}" get pods
echo

# proxy-public is a ClusterIP now, so there is no external address to wait for. The
# earlier version of this script polled for a LoadBalancer IP for ten minutes on every
# deploy, and in this org that IP never arrives: an L4 load balancer defaults its source
# range to 0.0.0.0/0, which the org policy forbids.
#
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
