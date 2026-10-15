#!/usr/bin/env bash
# ==============================================================================
# Script: 01_build_image.sh
# Description: Build and push the TPU job and submission service images.
#
#   tpu-job     docker/tpu-job/      what student training scripts run in on the chip
#   tpu-submit  services/tpu-submit/ the submission service
#
# Note: Student CPU notebooks use the upstream quay.io/jupyter/scipy-notebook image
# paired with the zero-rebuild shared Python environment on /tartan/11785/envs/
# (provisioned via `make shared-env` / scripts/09_setup_shared_env.sh).
#
# Usage:
#   make image                       both tpu-job and tpu-submit
#   bash scripts/01_build_image.sh tpu-job
# ==============================================================================

source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud docker

TARGETS=("${@:-all}")
[[ "${TARGETS[0]}" == "all" ]] && TARGETS=(tpu-job tpu-submit)

log_header "Building & Pushing Course Images: ${TARGETS[*]}"
log_info "Registry: ${COURSE_REGISTRY}"

if ! gcloud artifacts repositories describe course-images --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1; then
  log_info "Creating Artifact Registry repository 'course-images' in ${REGION}..."

  # Cleanup policy: automatically delete untagged images older than 30 days.
  # Without this, stale image layers accumulate indefinitely and bill for storage.
  CLEANUP_POLICY=$(mktemp)
  trap 'rm -f "${CLEANUP_POLICY}"' EXIT
  cat > "${CLEANUP_POLICY}" <<'POLICY'
[
  {
    "name": "delete-untagged",
    "action": {"type": "Delete"},
    "condition": {
      "tagState": "untagged",
      "olderThan": "30d"
    }
  },
  {
    "name": "keep-recent-tagged",
    "action": {"type": "Keep"},
    "mostRecentVersions": {
      "keepCount": 5
    }
  }
]
POLICY

  gcloud artifacts repositories create course-images \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Docker repository for course images" \
    --project="${PROJECT}"

  echo "    applying cleanup policies..."
  gcloud artifacts repositories set-cleanup-policies course-images \
    --location="${REGION}" \
    --project="${PROJECT}" \
    --policy="${CLEANUP_POLICY}" \
    --no-dry-run
fi

echo "==> configuring docker auth for Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

build_push() {
  local image="$1"; shift
  echo "==> building ${image}"
  docker build -t "${image}" "$@"
  echo "==> pushing ${image}"
  docker push "${image}"
}

for target in "${TARGETS[@]}"; do
  case "${target}" in
    notebook)
      log_info "Notebook pods use ${SINGLEUSER_IMAGE}:${SINGLEUSER_IMAGE_TAG} + Tartan shared-env."
      log_info "To install or update notebook Python packages, run: make shared-env" ;;
    tpu-job)
      # TPU nodes are amd64; build for it even from an arm64 laptop.
      build_push "${TPU_JOB_IMAGE}" --platform linux/amd64 \
        --build-arg "BASE_IMAGE=${TPU_IMAGE}" "${REPO_ROOT}/docker/tpu-job" ;;
    tpu-submit)
      build_push "${SUBMIT_IMAGE}" --platform linux/amd64 "${REPO_ROOT}/services/tpu-submit" ;;
    *)
      log_error "unknown image '${target}' (expected tpu-job, tpu-submit, or all)"
      exit 1 ;;
  esac
done

echo
echo "Done. 03_deploy_hub.sh points the hub and the submission service at these images."
echo "A new tpu-job image is picked up by the next job; a new tpu-submit image needs:"
echo "    kubectl -n ${NAMESPACE} rollout restart deployment/tpu-submit"
