#!/usr/bin/env bash
# ==============================================================================
# Script: 02_create_cluster.sh
# Description: Stand up the classroom substrate: Autopilot cluster, Kueue,
#              StorageClasses, PriorityClasses, and ResourceQuotas.
# Idempotent -- safe to re-run.
#
# Why Autopilot instead of Standard:
# Autopilot provisions a ct5lp-hightpu-1t node per TPU pod from the two
# nodeSelector labels and scales back to zero when the pod ends. Standard would
# need a node pool per topology plus the cluster autoscaler, which hard-fails
# on GCE_STOCKOUT during atomic resizes, whereas Kueue queues until capacity
# becomes available.
#
# Billing:
# A TPU pod is node-billed on Autopilot (24 vCPU / 48 GiB node for as long as
# the pod lives). That is why student TPU work is a short Job, not an open
# notebook.
# ==============================================================================
source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud kubectl

KUEUE_VERSION="${KUEUE_VERSION:-v0.19.1}"

log_header "Provisioning GKE Autopilot Cluster '${CLUSTER}' in '${REGION}'"
if ! gcloud container clusters describe "${CLUSTER}" --region="${REGION}" \
       --project="${PROJECT}" >/dev/null 2>&1; then
  log_info "Creating Autopilot cluster (Rapid release channel)..."
  gcloud container clusters create-auto "${CLUSTER}" \
    --project="${PROJECT}" --region="${REGION}" \
    --release-channel=rapid \
    --network=default --subnetwork=default
else
  log_success "Cluster '${CLUSTER}' already exists."
fi

ensure_k8s_context
K=(kubectl --context="${GKE_CTX}")

# Guard against runaway log ingestion costs ($0.50/GiB). A student writing
# `while True: print("hello")` can silently ingest terabytes of logs overnight.
# This exclusion filter drops container stdout/stderr from student namespaces
# before it reaches Cloud Logging billing. Logs from kube-system, the hub pod,
# and other infrastructure namespaces are preserved.
log_header "Configuring Log Exclusion Filter for Student Namespaces"
if ! gcloud logging sinks describe "_Default" --project="${PROJECT}" >/dev/null 2>&1; then
  log_warn "Could not verify default sink; skipping log exclusion."
else
  if gcloud logging sinks update "_Default" \
    --project="${PROJECT}" \
    --add-exclusion="name=student-notebook-noise,filter=resource.labels.namespace_name=~\"^(${NAMESPACE:?}|${NAMESPACE:?}-)\" AND resource.labels.pod_name=~\"^jupyter-\"" \
    2>/dev/null; then
    log_success "Log exclusion filter active for namespace '${NAMESPACE}'."
  else
    log_info "Log exclusion filter already exists or updated."
  fi
fi

log_header "Installing Kueue ${KUEUE_VERSION}"
"${K[@]}" apply --server-side -f \
  "https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/manifests.yaml"

log_info "Waiting for the Kueue controller to be available..."
"${K[@]}" -n kueue-system wait --for=condition=Available deploy/kueue-controller-manager --timeout=600s

# The webhook takes a few seconds past Available before it will accept CRs. Applying
# a ClusterQueue too early fails with 'no endpoints available for service'.
log_info "Waiting for the Kueue webhook to answer..."
for i in $(seq 1 60); do
  "${K[@]}" get clusterqueue >/dev/null 2>&1 && break
  sleep 5
done

# Priority classes must exist before the hub is installed. 03_deploy_hub.sh references
# jupyterhub-core and student-notebook, and Kubernetes rejects a pod whose
# priorityClassName does not resolve. Applying these by hand during development and
# forgetting to wire them in here is what broke `make hub` for a user.
log_header "Applying PriorityClasses and ResourceFlavors"
"${K[@]}" apply -f "$(dirname "$0")/../k8s/priority-classes.yaml"
"${K[@]}" apply -f "$(dirname "$0")/../k8s/kueue-tpu-queues.yaml"

log_header "Configuring StorageClass (standard-rwo-retain)"
"${K[@]}" apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard-rwo-retain
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-balanced
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF

NFLAVORS=2
PER=$(( POOL_CHIPS / NFLAVORS ))
log_header "Configuring Kueue ClusterQueue & Namespace '${NAMESPACE}'"
log_info "Capacity: ${POOL_CHIPS} chips across ${NFLAVORS} flavors (${PER} on-demand, ${PER} flex)."

"${K[@]}" apply -f - <<EOF
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: shared-tpu-pool
spec:
  cohortName: classroom
  namespaceSelector: {}
  queueingStrategy: BestEffortFIFO
  resourceGroups:
    - coveredResources: ["google.com/tpu"]
      flavors:
        - name: v5e-ondemand
          resources:
            - name: "google.com/tpu"
              nominalQuota: ${PER}
        - name: v5e-flex
          resources:
            - name: "google.com/tpu"
              nominalQuota: ${PER}
---
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: class-quota
  namespace: ${NAMESPACE}
spec:
  hard:
    count/jobs.batch: "60"
    count/pods: "100"
    count/persistentvolumeclaims: "40"
    # Sized for 25 student home volumes x 32Gi (800Gi) with 1Ti quota headroom
    requests.storage: "1Ti"
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: tpu
  namespace: ${NAMESPACE}
spec:
  clusterQueue: shared-tpu-pool
EOF

# ==============================================================================
# MULTI-SECTION REFERENCE (For large cohorts requiring section-split queues)
# ==============================================================================
# If scaling beyond 50 students requires multiplexing namespaces again, revert
# the above single-namespace block and uncomment the section loop below:
#
# SECTIONS="${SECTIONS:-a b c d}"
# NSEC=$(echo "${SECTIONS}" | wc -w)
# NFLAVORS=2
# PER=$(( POOL_CHIPS / NSEC / NFLAVORS ))
# log_info "${NSEC} sections x ${NFLAVORS} flavors x ${PER} chips = ${POOL_CHIPS} in the cohort"
#
# for S in ${SECTIONS}; do
#   "${K[@]}" apply -f - <<EOF
# apiVersion: kueue.x-k8s.io/v1beta2
# kind: ClusterQueue
# metadata:
#   name: section-${S}
# spec:
#   cohortName: classroom
#   namespaceSelector: {}
#   queueingStrategy: BestEffortFIFO
#   resourceGroups:
#     - coveredResources: ["google.com/tpu"]
#       flavors:
#         - name: v5e-ondemand
#           resources:
#             - name: "google.com/tpu"
#               nominalQuota: ${PER}
#         - name: v5e-flex
#           resources:
#             - name: "google.com/tpu"
#               nominalQuota: ${PER}
# ---
# apiVersion: v1
# kind: Namespace
# metadata:
#   name: class-sec-${S}
# ---
# apiVersion: v1
# kind: ResourceQuota
# metadata:
#   name: section-quota
#   namespace: class-sec-${S}
# spec:
#   hard:
#     count/jobs.batch: "250"
#     count/pods: "500"
#     count/persistentvolumeclaims: "100"
#     requests.storage: "2Ti"
# ---
# apiVersion: kueue.x-k8s.io/v1beta2
# kind: LocalQueue
# metadata:
#   name: tpu
#   namespace: class-sec-${S}
# spec:
#   clusterQueue: section-${S}
# EOF
# done

log_header "Cluster & Queue Status"
"${K[@]}" get clusterqueue
echo
"${K[@]}" get localqueue -A
echo
log_success "Substrate ready. Shared pool has ${POOL_CHIPS} chips (${PER} on-demand, ${PER} flex)."
