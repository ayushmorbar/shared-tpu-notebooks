#!/usr/bin/env bash
# ==============================================================================
# shared-tpu-notebooks: Common Bash Utility Library
# ==============================================================================
# Sourced by infrastructure automation scripts in scripts/ to provide:
#   1. Automatic configuration loading from config.env (if present)
#   2. Standardized variable defaults across all scripts
#   3. GKE cluster context management and credential retrieval
#   4. Reusable colorized status logging
#   5. Prerequisite verification (gcloud, kubectl, helm)
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Configuration Discovery & Loading
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source config.env if present in the repository root
if [[ -f "${REPO_ROOT}/config.env" ]]; then
  # Sourcing without overriding already exported environment variables
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/config.env"
  set +a
fi

# ------------------------------------------------------------------------------
# 2. Standard Variables & Defaults
# ------------------------------------------------------------------------------
PROJECT="${PROJECT:-${GCP_PROJECT:-}}"
REGION="${REGION:-us-west4}"
CLUSTER="${CLUSTER:-tpu-notebooks}"
NAMESPACE="${NAMESPACE:-${NS:-cmu-idl}}"

STUDENTS="${STUDENTS:-40}"
POOL_CHIPS="${POOL_CHIPS:-32}"
WARM="${WARM:-1}"
CHIP_HR="${CHIP_HR:-1.35}"

TPU_ACCELERATOR="${TPU_ACCELERATOR:-tpu-v5-lite-podslice}"
TPU_TOPOLOGY="${TPU_TOPOLOGY:-1x1}"
TPU_IMAGE="${TPU_IMAGE:-us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest}"

DOMAIN="${DOMAIN:-}"
STUDENT_GROUP="${STUDENT_GROUP:-}"
TA_GROUP="${TA_GROUP:-}"
ADMIN_USERS="${ADMIN_USERS:-}"
TEST_ACCOUNTS="${TEST_ACCOUNTS:-}"

# GKE Cluster Context Identifier
GKE_CTX="gke_${PROJECT}_${REGION}_${CLUSTER}"

# ------------------------------------------------------------------------------
# 3. Formatted Logging Utilities
# ------------------------------------------------------------------------------
# ANSI Color Codes (disabled if not on a terminal)
if [[ -t 1 ]]; then
  C_RESET="\033[0m"
  C_BOLD="\033[1m"
  C_BLUE="\033[1;34m"
  C_GREEN="\033[1;32m"
  C_YELLOW="\033[1;33m"
  C_RED="\033[1;31m"
  C_CYAN="\033[1;36m"
else
  C_RESET=""
  C_BOLD=""
  C_BLUE=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
  C_CYAN=""
fi

log_header() {
  echo -e "\n${C_BOLD}${C_BLUE}==>${C_RESET} ${C_BOLD}$1${C_RESET}"
}

log_info() {
  echo -e "    ${C_CYAN}•${C_RESET} $1"
}

log_success() {
  echo -e "    ${C_GREEN}✓${C_RESET} $1"
}

log_warn() {
  echo -e "    ${C_YELLOW}⚠ Warning:${C_RESET} $1" >&2
}

log_error() {
  echo -e "    ${C_RED}✖ Error:${C_RESET} $1" >&2
}

# ------------------------------------------------------------------------------
# 4. Prerequisites & Environment Verification
# ------------------------------------------------------------------------------
require_project() {
  if [[ -z "${PROJECT}" ]]; then
    log_error "PROJECT is not set."
    echo "       Set it via config.env or export PROJECT=my-gcp-project-id" >&2
    echo "       Example: cp config.env.example config.env" >&2
    exit 1
  fi
}

check_prereqs() {
  local missing=()
  for tool in "$@"; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      missing+=("${tool}")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    log_error "Missing required tool(s): ${missing[*]}"
    echo "       Please install the required tools and ensure they are in your PATH." >&2
    exit 1
  fi
}

# ------------------------------------------------------------------------------
# 5. GKE Context & Kubernetes Client Management
# ------------------------------------------------------------------------------
ensure_k8s_context() {
  require_project
  check_prereqs gcloud kubectl

  if ! kubectl config get-contexts -o name 2>/dev/null | grep -qx "${GKE_CTX}"; then
    log_info "Fetching GKE cluster credentials for '${CLUSTER}' in '${REGION}'..."
    gcloud container clusters get-credentials "${CLUSTER}" \
      --region="${REGION}" --project="${PROJECT}" >/dev/null 2>&1
  fi
}
