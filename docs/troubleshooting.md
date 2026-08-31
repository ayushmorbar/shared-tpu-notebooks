# 🔧 Troubleshooting Guide

This document lists the most common errors encountered when operating the `shared-tpu-notebooks` cluster, updated for 2026 Autopilot behaviors.

## 1. IAP: "You don't have access"

**Symptom:**
You browse to `https://jupyter.yourdomain.com` and Google shows a white screen saying you do not have access.

**Root Cause:**
Your email address is not bound to the `roles/iap.httpsResourceAccessor` IAM role.

**Fix:**
Run the IAP setup script again with your email in the `ADMIN_USERS` environment variable in `config.env`.
```bash
make iap
```

## 2. Notebooks fail to spawn (Stuck in Pending)

**Symptom:**
Students click "Start" and the progress bar stalls indefinitely.

**Root Cause:**
You have hit the GCP Regional `CPUS` quota limit. GKE Autopilot cannot provision more Spot CPU VMs.

**Fix:**
Verify your quota:
```bash
gcloud compute project-info describe --project=YOUR_PROJECT | grep -A 5 "CPUS"
```
You need exactly 2 vCPU of quota per concurrent student. Request a quota increase in the GCP Console.

## 3. TPU Jobs hang in the queue for 5+ minutes

**Symptom:**
The cell output shows:
```text
submitted tpu-studentname to queue 'tpu'
```
But the `admitted` message never appears.

**Root Cause:**
1. **Zero Flex Capacity:** The DWS Flex pool in your zone is empty, and standard on-demand capacity is heavily congested.
2. **Kueue Cohort Full:** 300 students submitted a job at exactly the same time, and Kueue is processing them (this is normal behavior, just wait).

**Fix:**
Check Kueue cluster queues:
```bash
kubectl get clusterqueues
kubectl get workloads -n cmu-idl
```
If a workload says "Quota Reserved" but the Pod is pending, GKE is currently booting the TPU VM. Wait 2-4 minutes.

## 4. `ApiException: 403 Forbidden` in Notebook

**Symptom:**
Running `submit_tpu.run()` yields an HTTP 403 error from the Python Kubernetes client.

**Root Cause:**
The student notebook ServiceAccount does not have RBAC permissions to create Jobs.

**Fix:**
Ensure the `student-tpu-rbac` Role and RoleBinding are correctly applied. Re-run:
```bash
make hub
```

## 5. Webhook Rejection for Privilege

**Symptom:**
JupyterHub pods fail to start with `denied by autogke-disallow-privilege: container block-cloud-metadata is privileged`.

**Root Cause:**
You modified `jupyterhub-values.yaml` and re-enabled `blockWithIptables`. GKE Autopilot strictly forbids `NET_ADMIN` privileges.

**Fix:**
In `jupyterhub-values.yaml`, ensure:
```yaml
singleuser:
  cloudMetadata:
    blockWithIptables: false
```
Our NetworkPolicy already blocks the metadata server via Cilium Dataplane V2, so iptables are not needed.
