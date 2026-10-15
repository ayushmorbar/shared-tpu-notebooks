#!/usr/bin/env bash
# ==============================================================================
# Optional: Put Course Infrastructure Credentials into a Kubernetes Secret
# ==============================================================================
# Usage:
#   WANDB_API_KEY=xxx KAGGLE_USERNAME=yyy KAGGLE_API_TOKEN=zzz bash scripts/12_create_secrets.sh
#
# NOTE ON STUDENT PRIVACY:
# Student credentials (WANDB_API_KEY, KAGGLE_USERNAME, KAGGLE_API_TOKEN, HF_TOKEN)
# are strictly PER-STUDENT and configured in each student's personal JupyterHub
# home directory (/home/jovyan/.env) via student.env.template.
#
# This script is intended only for course staff who wish to provision a cluster-wide
# baseline secret (course-secrets) in namespace cmu-idl without publishing keys.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

require_project
ensure_k8s_context

SECRET="${SECRET_NAME:-course-secrets}"

WANDB_KEY="${WANDB_API_KEY:-}"
KAGGLE_USER="${KAGGLE_USERNAME:-}"
KAGGLE_TOKEN="${KAGGLE_API_TOKEN:-}"
HUGGINGFACE_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

if [[ -z "${WANDB_KEY}" && -z "${KAGGLE_TOKEN}" && -z "${HUGGINGFACE_TOKEN}" ]]; then
  log_warn "No credentials supplied via environment or config.env."
  echo "       Set at least one of WANDB_API_KEY, KAGGLE_API_TOKEN, or HF_TOKEN." >&2
  echo "       Example: WANDB_API_KEY=xxx bash $(basename "$0")" >&2
  exit 1
fi

log_header "Creating Kubernetes Secret '${SECRET}' in namespace '${NAMESPACE}'"

ARGS=()
[[ -n "${WANDB_KEY}" ]]          && ARGS+=(--from-literal=wandb-api-key="${WANDB_KEY}")
[[ -n "${KAGGLE_USER}" ]]         && ARGS+=(--from-literal=kaggle-username="${KAGGLE_USER}")
[[ -n "${KAGGLE_TOKEN}" ]]        && ARGS+=(--from-literal=kaggle-api-token="${KAGGLE_TOKEN}")
[[ -n "${HUGGINGFACE_TOKEN}" ]]   && ARGS+=(--from-literal=hf-token="${HUGGINGFACE_TOKEN}")

kubectl --context="${GKE_CTX}" -n "${NAMESPACE}" create secret generic "${SECRET}" \
  "${ARGS[@]}" --dry-run=client -o yaml \
  | kubectl --context="${GKE_CTX}" apply -f - >/dev/null

log_success "secret/${SECRET} created/updated successfully in ${NAMESPACE}:"
kubectl --context="${GKE_CTX}" -n "${NAMESPACE}" get secret "${SECRET}" \
  -o go-template='{{range $k, $v := .data}}  {{$k}} ({{len $v}} bytes base64){{"\n"}}{{end}}'

echo
log_info "Values are securely stored in Kubernetes and never logged in cleartext."
log_info "Training pods read personal keys from /home/jovyan/.env, with secretKeyRef as fallback."
