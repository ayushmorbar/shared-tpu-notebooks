#!/usr/bin/env bash
# Put JupyterHub behind HTTPS and Identity-Aware Proxy.
#
# After this, the hub has a real URL and Google sign-in. No port-forward, no tunnel to
# babysit, and nothing that dies when the proxy pod moves.
#
# WHY THIS WORKS IN AN ORG THAT BLOCKS L4 LOAD BALANCERS
#
# A Service of type LoadBalancer defaults its source range to 0.0.0.0/0, and some orgs
# forbid a firewall rule that broad. A GKE Ingress never asks for one: Google Front Ends
# terminate the connection and only need 130.211.0.0/22 and 35.191.0.0/16 opened.
#
# TWO STEPS THIS SCRIPT CANNOT DO FOR YOU
#
#   1. The OAuth consent screen has to be configured once, by hand, in the console. IAP
#      refuses to serve until it exists. The script prints the link.
#   2. A Google-managed certificate takes 15 to 60 minutes to reach ACTIVE. The script
#      polls and tells you where it is rather than pretending it is instant.
source "$(dirname "$0")/common.sh"
require_project
check_prereqs gcloud kubectl

IP_NAME="${IP_NAME:-tpu-notebooks-ip}"

ensure_k8s_context
K=(kubectl --context="${GKE_CTX}")

echo "==> APIs"
gcloud services enable iap.googleapis.com compute.googleapis.com --project="${PROJECT}" >/dev/null

echo "==> static IP ${IP_NAME}"
if ! gcloud compute addresses describe "${IP_NAME}" --global --project="${PROJECT}" >/dev/null 2>&1; then
  gcloud compute addresses create "${IP_NAME}" --global --project="${PROJECT}" >/dev/null
fi
IP=$(gcloud compute addresses describe "${IP_NAME}" --global --project="${PROJECT}" --format="value(address)")

# nip.io resolves <anything>.<ip>.nip.io to that IP, so the managed certificate can
# validate without us owning a DNS zone.
if [[ -z "${DOMAIN}" ]]; then
  DOMAIN="${IP}.nip.io"
  echo "    No DOMAIN specified in Makefile. Using ${IP}  ->  ${DOMAIN}"
else
  echo "    Using custom domain: ${DOMAIN}"
fi

echo "==> BackendConfig, ManagedCertificate, Ingress"
sed -e "s|__DOMAIN__|${DOMAIN}|g" -e "s|__NAMESPACE__|${NAMESPACE}|g" \
  "$(dirname "$0")/../k8s/ingress-iap.yaml" | "${K[@]}" apply -f -

echo "==> granting IAP access"

bind_iam_member() {
  local member="$1"
  local role="roles/iap.httpsResourceAccessor"
  echo "    granting to ${member}"
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="${member}" \
    --role="${role}" \
    --condition=None >/dev/null 2>&1 \
    && echo "      ok: ${member}" \
    || echo "      warning: could not bind ${member}; grant ${role} manually if needed"
}

# 1. Student Group
if [[ -n "${STUDENT_GROUP}" ]]; then
  for grp in ${STUDENT_GROUP}; do
    [[ "${grp}" != group:* && "${grp}" != user:* ]] && grp="group:${grp}"
    bind_iam_member "${grp}"
  done
fi

# 2. TA Group
if [[ -n "${TA_GROUP}" ]]; then
  for grp in ${TA_GROUP}; do
    [[ "${grp}" != group:* && "${grp}" != user:* ]] && grp="group:${grp}"
    bind_iam_member "${grp}"
  done
fi

# 3. Admin Users
if [[ -n "${ADMIN_USERS}" ]]; then
  for adm in ${ADMIN_USERS}; do
    [[ "${adm}" != user:* && "${adm}" != group:* ]] && adm="user:${adm}"
    bind_iam_member "${adm}"
  done
fi

# 4. Individual Test / Manual Accounts
if [[ -n "${TEST_ACCOUNTS}" ]]; then
  for acct in ${TEST_ACCOUNTS}; do
    [[ "${acct}" != user:* && "${acct}" != group:* ]] && acct="user:${acct}"
    bind_iam_member "${acct}"
  done
fi

# 5. Fallback if nothing configured at all
if [[ -z "${STUDENT_GROUP}" ]] && [[ -z "${TA_GROUP}" ]] && [[ -z "${ADMIN_USERS}" ]] && [[ -z "${TEST_ACCOUNTS}" ]]; then
  FALLBACK_USER="user:$(gcloud config get-value account 2>/dev/null)"
  echo "    no IAP identities configured. granting to fallback account ${FALLBACK_USER}"
  bind_iam_member "${FALLBACK_USER}"
fi

cat <<EOF

==> ONE MANUAL STEP, do it now or IAP will refuse every request

    Configure the OAuth consent screen (internal), once per project:
    https://console.cloud.google.com/auth/branding?project=${PROJECT}

EOF

echo "==> waiting for the certificate (15-60 min is normal)"
for i in $(seq 1 80); do
  STATUS=$("${K[@]}" -n "${NAMESPACE}" get managedcertificate hub-cert \
             -o jsonpath='{.status.certificateStatus}' 2>/dev/null || true)
  LBIP=$("${K[@]}" -n "${NAMESPACE}" get ingress hub-ingress \
           -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
  printf '\r  cert=%-18s ingress-ip=%-16s %ds' "${STATUS:-pending}" "${LBIP:-pending}" $((i*30))
  [[ "${STATUS}" == "Active" ]] && break
  sleep 30
done
echo

STATUS=$("${K[@]}" -n "${NAMESPACE}" get managedcertificate hub-cert \
           -o jsonpath='{.status.certificateStatus}' 2>/dev/null || true)

if [[ "${STATUS}" == "Active" ]]; then
  cat <<EOF

    https://${DOMAIN}

Sign in with a Google account in this org. IAP checks it before JupyterHub sees the
request, and the hub reads the verified identity from the X-Goog-Authenticated-User-Email
header, so there's no second login.
EOF
else
  cat <<EOF

Certificate is still "${STATUS:-pending}". That's normal for up to an hour. Check with:

  kubectl --context=${GKE_CTX} -n ${NAMESPACE} describe managedcertificate hub-cert

FailedNotVisible means the domain doesn't resolve to ${IP} yet. If it's still failing
after an hour, nip.io may be rejected by the CA; register a real domain, point an A
record at ${IP}, and re-run with DOMAIN=your.domain.
EOF
fi
