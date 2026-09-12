#!/usr/bin/env bash
# ==============================================================================
# 11_expand_pvcs.sh: Online Volume Expansion for Student Notebook PVCs
# ==============================================================================
# Expands existing student PVCs (e.g. 10Gi -> 32Gi) online without deleting
# data or stopping running notebook pods.
#
# Why this is needed:
#   Updating `capacity: 32Gi` in `jupyterhub-values.yaml` only applies to newly
#   created student volumes. Students who already have active PVCs remain at
#   10Gi unless expanded. Because the `standard-rwo-retain` StorageClass has
#   `allowVolumeExpansion: true`, Kubernetes dynamically resizes the underlying
#   GCP persistent disk online.
#
# Usage:
#   bash scripts/11_expand_pvcs.sh [--dry-run]
#   bash scripts/11_expand_pvcs.sh --execute
#   TARGET_SIZE=32Gi bash scripts/11_expand_pvcs.sh --execute
# ==============================================================================
source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud kubectl

TARGET_SIZE="${TARGET_SIZE:-32Gi}"
DRY_RUN=1

if [[ "${1:-}" == "--execute" ]]; then
  DRY_RUN=0
  log_info "Running in EXECUTE mode. Volumes will be resized to ${TARGET_SIZE}."
else
  log_info "Running in DRY-RUN mode. Pass --execute to actually expand volumes."
fi

ensure_k8s_context
K=(kubectl --context="${GKE_CTX}")

log_header "Scanning Student PVCs in Namespace '${NAMESPACE}'"

# Query PVCs in the student namespace
PVCS=$("${K[@]}" -n "${NAMESPACE}" get pvc -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.resources.requests.storage}{"\n"}{end}')

if [[ -z "${PVCS}" ]]; then
  log_info "No PVCs found in namespace '${NAMESPACE}'."
  exit 0
fi

RESIZED_COUNT=0
ALREADY_TARGET=0

while IFS=$'\t' read -r PVC_NAME CURRENT_SIZE; do
  [[ -z "${PVC_NAME}" ]] && continue

  if [[ "${CURRENT_SIZE}" == "${TARGET_SIZE}" ]]; then
    log_success "${PVC_NAME} is already at ${TARGET_SIZE}."
    ALREADY_TARGET=$((ALREADY_TARGET + 1))
    continue
  fi

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log_warn "[DRY-RUN] Would expand ${PVC_NAME}: ${CURRENT_SIZE} -> ${TARGET_SIZE}"
  else
    log_info "Expanding ${PVC_NAME}: ${CURRENT_SIZE} -> ${TARGET_SIZE}..."
    "${K[@]}" -n "${NAMESPACE}" patch pvc "${PVC_NAME}" --type='merge' \
      -p "{\"spec\":{\"resources\":{\"requests\":{\"storage\":\"${TARGET_SIZE}\"}}}}"
    log_success "Successfully patched ${PVC_NAME} to request ${TARGET_SIZE}."
  fi
  RESIZED_COUNT=$((RESIZED_COUNT + 1))
done <<< "${PVCS}"

echo
if [[ "${DRY_RUN}" -eq 1 ]]; then
  log_info "Dry-run complete: ${RESIZED_COUNT} volume(s) need expansion, ${ALREADY_TARGET} already at ${TARGET_SIZE}."
  echo "Run with '--execute' to apply changes: bash scripts/11_expand_pvcs.sh --execute"
else
  log_success "Execution complete: ${RESIZED_COUNT} volume(s) expanded to ${TARGET_SIZE}."
  log_info "Note: GKE CSI dynamically resizes the filesystem online as pods write."
fi
