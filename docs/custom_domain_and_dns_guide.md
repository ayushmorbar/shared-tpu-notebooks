# 🌐 Custom Domain & DNS Mapping Guide

This guide details how to map static IP addresses and custom subdomains (e.g. `jupyter.cs.cmu.edu`, `11785.tpu.cmu.edu`) to the **CMU 11-785 Cloud TPU Hub** using Google Cloud Ingress, Google-managed SSL certificates, and Identity-Aware Proxy (IAP).

---

## 1. Network & Ingress Architecture

```
                          Internet Traffic (Students & TAs)
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   GCP Global Anycast Static IPv4 Address  │
                  │              (tpu-notebooks-ip)           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │        Google Cloud HTTPS Load Balancer   │
                  │   - Google Front End (GFE) Termination   │
                  │   - Google Managed SSL Certificate (Auto) │
                  │   - Identity-Aware Proxy (Google SSO)     │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │    Network Endpoint Groups (NEGs)         │
                  │    Direct routing to proxy-public:80      │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │            JupyterHub Pod               │
                  │      (Validates IAP JWT Assertion)        │
                  └───────────────────────────────────────────┘
```

---

## 2. Step-by-Step DNS Mapping Approaches

### Approach 1: Zero-Config Automated Subdomain (`nip.io`) — Default for Staging

If you do not have a custom domain or are conducting an initial deployment:
```bash
make iap
```
- **How it works:** `scripts/08_setup_iap.sh` reserves a GCP Global Static IP (e.g., `34.149.20.100`) and automatically points to `34.149.20.100.nip.io`.
- `nip.io` is a public wildcard DNS service that resolves `<IP>.nip.io` directly to that IP address.
- Google Managed Certificates provisions an active SSL certificate against this domain within 15–30 minutes with zero manual DNS configuration!

---

### Approach 2: Custom University Subdomain (`jupyter.cs.cmu.edu`) — Production Recommended

For production course deployments where students access the hub through an official CMU URL:

#### Step A: Discover the Reserved Static IP
Run:
```bash
make ip
```
Output:
```text
==================================================
GCP Global Static IP : 34.149.20.100
Default nip.io URL   : https://34.149.20.100.nip.io
DNS Configuration    : Create A-Record: jupyter.cs.cmu.edu -> 34.149.20.100
==================================================
```

#### Step B: Create the DNS A-Record
In your department DNS manager, university registrar, or Cloudflare/Route53:
- **Type:** `A`
- **Host / Name:** `jupyter` (or `11785.tpu`)
- **Target / Value:** `<YOUR_STATIC_IP>` (e.g., `34.149.20.100`)
- **TTL:** 300 seconds (5 minutes)

#### Step C: Apply Custom Domain to JupyterHub
Once the DNS record is published, configure Ingress and Google Managed Certificate directly via Makefile:
```bash
make domain DOMAIN=jupyter.cs.cmu.edu
```
- This updates `k8s/ingress-iap.yaml` with the domain.
- Google Cloud automatically begins provisioning a new Google-managed SSL certificate for `jupyter.cs.cmu.edu`.
- Check certificate status:
  ```bash
  kubectl -n cmu-idl get managedcertificate hub-cert -o yaml
  ```
  Status transitions: `PROVISIONING` $\to$ `ACTIVE` (typically 15–45 minutes).

---

### Approach 3: Fully Automated Google Cloud DNS Sync (`make dns-record`)

If your domain's DNS zone is managed directly in Google Cloud DNS:

```bash
make dns-record DOMAIN=jupyter.cs.cmu.edu ZONE=cmu-dns-zone
```

**What this command does automatically:**
1. Discovers the reserved `tpu-notebooks-ip`.
2. Starts an atomic Google Cloud DNS transaction on managed zone `cmu-dns-zone`.
3. Adds/updates the `A` record mapping `jupyter.cs.cmu.edu.` to the static IP.
4. Executes and publishes the transaction in Google Cloud DNS.
5. Invokes `08_setup_iap.sh` to bind the new domain to `hub-ingress` and `hub-cert`.

---

## 3. Google OAuth & Identity-Aware Proxy (IAP) Configuration

> [!IMPORTANT]
> Google Identity-Aware Proxy requires an **Internal OAuth Consent Screen** before it will serve traffic.

1. Navigate to **Google Cloud Console** $\to$ **APIs & Services** $\to$ **OAuth consent screen**.
2. Select **Internal** (restricted to users in your Google Workspace / CMU organization).
3. App name: `CMU 11-785 TPU Hub`.
4. User support email: `your_email@andrew.cmu.edu`.
5. Authorized domain: `cmu.edu` (and your domain if distinct).
6. Save and submit.

Once configured, users navigate to `https://jupyter.cs.cmu.edu`, sign in with their `@andrew.cmu.edu` account, and are seamlessly authenticated into JupyterHub!

---

## 4. Operational Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **Certificate stuck in `PROVISIONING` > 60m** | DNS A-record does not yet point to the Static IP, or CAA record blocks Google CA | Verify DNS: `dig +short A jupyter.cs.cmu.edu`. Ensure output matches `make ip`. Check CAA records: `dig +short CAA cmu.edu`. |
| **HTTP 502 Bad Gateway** | GFE health check failing on backend | Verify `BackendConfig` in `k8s/ingress-iap.yaml` points to `/hub/health` on port 8000. Check NEG status: `kubectl get svc proxy-public -n cmu-idl -o yaml`. |
| **HTTP 403 "You don't have access"** | Authenticated user lacks `roles/iap.httpsResourceAccessor` | Re-run `make iap` with user in `ADMIN_USERS`, or grant the role via `gcloud iap web add-iam-policy-binding`. |
