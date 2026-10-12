#!/usr/bin/env bash
# ==============================================================================
# Script: 12_setup_storage.sh
# Description: Buckets and identities for the TPU submission service.
#
# Two buckets, two Kubernetes service accounts, no Google service accounts and no keys:
#
#   STUDENT_BUCKET  snapshots, job logs, checkpoints, results. Read/write for the
#                   tpu-submit service only. Students never touch it directly; they
#                   go through the service, which keeps each student in their own
#                   users/<name>/ folder.
#   DATASET_BUCKET  one archive per assignment. Read-only for tpu-job, the account
#                   student code runs as on the chip. The data is the same for
#                   everyone, so one shared read-only identity is fine.
#
# Permissions are granted to the Kubernetes service accounts directly through Workload
# Identity Federation ("principal://.../subject/ns/NS/sa/NAME"), so no Google service
# account has to be created, annotated, or kept in sync. Autopilot clusters always have
# Workload Identity on.
#
# Usage:
#   make storage                                          buckets + bindings
#   bash scripts/12_setup_storage.sh upload hw3p2 ~/hw3p2_data
#       tar a dataset folder (or copy a .tar) to gs://DATASET_BUCKET/hw3p2.tar and print
#       the DATASETS line for config.env
# ==============================================================================

source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud

project_number() {
  gcloud projects describe "${PROJECT}" --format='value(projectNumber)'
}

principal() {
  local ksa="$1"
  echo "principal://iam.googleapis.com/projects/$(project_number)/locations/global/workloadIdentityPools/${PROJECT}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${ksa}"
}

ensure_bucket() {
  local bucket="$1"
  if gcloud storage buckets describe "gs://${bucket}" --project="${PROJECT}" >/dev/null 2>&1; then
    log_info "gs://${bucket} exists"
    return
  fi
  log_info "creating gs://${bucket} in ${REGION}"
  # Same region as the cluster: no egress charge between the chips and the bucket.
  # Uniform access + public access prevention: IAM is the only way in.
  gcloud storage buckets create "gs://${bucket}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --uniform-bucket-level-access \
    --public-access-prevention
}

setup() {
  log_header "Storage for the TPU submission service"
  gcloud services enable storage.googleapis.com --project="${PROJECT}" >/dev/null

  ensure_bucket "${STUDENT_BUCKET}"
  ensure_bucket "${DATASET_BUCKET}"

  # Jobs overwrite their checkpoint every epoch. With the default 7-day soft delete,
  # every overwritten copy keeps billing for a week, which for a class of 300 is
  # several times the live data. Old runs expire after RUNS_RETENTION_DAYS.
  log_info "student bucket: no soft delete, delete objects after ${RUNS_RETENTION_DAYS} days"
  gcloud storage buckets update "gs://${STUDENT_BUCKET}" --clear-soft-delete >/dev/null
  LIFECYCLE=$(mktemp)
  trap 'rm -f "${LIFECYCLE}"' EXIT
  cat > "${LIFECYCLE}" <<EOF
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": ${RUNS_RETENTION_DAYS}}}]}
EOF
  gcloud storage buckets update "gs://${STUDENT_BUCKET}" --lifecycle-file="${LIFECYCLE}" >/dev/null

  log_info "tpu-submit -> objectAdmin on gs://${STUDENT_BUCKET}"
  gcloud storage buckets add-iam-policy-binding "gs://${STUDENT_BUCKET}" \
    --member="$(principal tpu-submit)" --role=roles/storage.objectAdmin >/dev/null

  log_info "tpu-job -> objectViewer on gs://${DATASET_BUCKET}"
  gcloud storage buckets add-iam-policy-binding "gs://${DATASET_BUCKET}" \
    --member="$(principal tpu-job)" --role=roles/storage.objectViewer >/dev/null

  log_success "buckets and bindings ready"
  echo
  echo "IAM changes can take a few minutes to reach the GKE metadata server."
  if [[ -z "${DATASETS}" ]]; then
    echo
    log_warn "DATASETS is empty, so jobs get no data. Upload an archive with:"
    echo "    bash scripts/12_setup_storage.sh upload hw3p2 /path/to/hw3p2_data"
  fi
}

upload() {
  local name="${1:-}" src="${2:-}"
  if [[ -z "${name}" || -z "${src}" ]]; then
    echo "usage: $0 upload NAME PATH   (PATH is a folder or a .tar/.tar.gz archive)" >&2
    exit 1
  fi
  if [[ ! "${name}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    log_error "dataset name must be lowercase letters, digits, - and _"
    exit 1
  fi
  local dest
  if [[ -d "${src}" ]]; then
    # Uncompressed: filterbank .npy files barely compress, and gzip would make the
    # download on the chip CPU-bound. The job unpacks the archive as it streams in.
    dest="gs://${DATASET_BUCKET}/${name}.tar"
    log_header "Uploading ${src} as ${dest}"
    log_info "the job sets DATA_ROOT to the archive's single top-level folder, here $(basename "${src}")/"
    tar -C "$(dirname "${src}")" -cf - "$(basename "${src}")" | gcloud storage cp - "${dest}"
  elif [[ -f "${src}" ]]; then
    case "${src}" in
      *.tar) dest="gs://${DATASET_BUCKET}/${name}.tar" ;;
      *.tar.gz|*.tgz) dest="gs://${DATASET_BUCKET}/${name}.tar.gz" ;;
      *) log_error "expected a folder, .tar, or .tar.gz"; exit 1 ;;
    esac
    log_header "Uploading ${src} as ${dest}"
    gcloud storage cp "${src}" "${dest}"
  else
    log_error "${src} does not exist"
    exit 1
  fi
  log_success "uploaded"
  gcloud storage ls -l "${dest}"
  echo
  echo "Add this to config.env, then run make hub to restart the service with it:"
  echo "    DATASETS=\"${name}=${dest}\"     # comma-separate several: a=gs://...,b=gs://..."
  echo "    DEFAULT_DATASET=\"${name}\""
  echo
  echo "Set JOB_DISK to at least twice the unpacked size so checkpoints still fit."
}

case "${1:-setup}" in
  setup) setup ;;
  upload) shift; upload "$@" ;;
  *) echo "usage: $0 [setup | upload NAME PATH]" >&2; exit 1 ;;
esac
