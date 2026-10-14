# 🔧 Troubleshooting Guide: CMU 11-785 Tartan TPU Hub

This document lists the most common errors encountered when operating the **CMU 11-785 Cloud TPU Hub** and their validated solutions.

---

## 1. Identity-Aware Proxy: "You don't have access"

**Symptom:**
You browse to `https://jupyter.yourdomain.com` and Google displays an HTTP 403 error saying: *"You do not have access to this application."*

**Root Cause:**
Your Google/Andrew account has not been granted the IAM role `roles/iap.httpsResourceAccessor`.

**Fix:**
Add your email to `ADMIN_USERS` or `STUDENT_GROUP` in `config.env` and re-run:
```bash
make iap
```
Or grant the role directly via `gcloud`:
```bash
gcloud iap web add-iam-policy-binding \
  --resource-type=backend-services \
  --service=YOUR_BACKEND_SERVICE \
  --member='user:your_email@andrew.cmu.edu' \
  --role='roles/iap.httpsResourceAccessor'
```

---

## 2. Notebooks Fail to Spawn (Stuck in Pending)

**Symptom:**
Students click "Start" and the JupyterHub progress bar stalls indefinitely.

**Root Cause:**
1. **Regional CPU Quota Exhaustion**: GKE Autopilot cannot provision additional Spot CPU nodes.
2. **Start Timeout**: Autopilot node spin-up took longer than the default timeout.

**Fix:**
1. Verify quota in GCP Console or via CLI:
   ```bash
   gcloud compute project-info describe --project=YOUR_PROJECT | grep -A 5 "CPUS"
   ```
2. In `k8s/jupyterhub-values.yaml`, ensure `startTimeout: 600` is set so students are not prematurely timed out while Autopilot spins up a new Spot VM.

---

## 3. TPU Jobs Hang in Queue for 5+ Minutes

**Symptom:**
In the notebook cell, output shows:
```text
submitted tpu-hw2p2-student to queue 'tpu'
```
But `[Admitted]` never appears.

**Root Cause:**
1. **DWS Flex Stockout**: TPU v5e chips in the specified zone are temporarily congested.
2. **Kueue Concurrency Limit**: The active TPU workload cohort has reached its chip cap (e.g., 32 chips).

**Fix:**
Inspect Kueue queues and workloads:
```bash
kubectl get clusterqueues
kubectl get workloads -n cmu-idl
```
- If the workload status says `Admitted: true`, GKE is currently booting the TPU node. TPU VM cold-starts take **2–4 minutes** (and up to 16 minutes in peak load). Heartbeats will keep streaming in the notebook.
- If `Admitted: false`, jobs will automatically start as preceding student jobs finish.

---

## 4. Tartan Storage Mount (`/tartan/11785/` or `/ocean/`) Missing

**Symptom:**
In a student notebook, `ls /tartan/11785` returns:
```text
ls: cannot access '/tartan/11785': No such file or directory
```

**Root Cause:**
1. The GCS FUSE CSI driver failed to mount `gs://cmu-idl-ocean` because the GCS bucket does not exist or has incorrect Workload Identity permissions.
2. `jupyterhub-values.yaml` did not have `__OCEAN_BUCKET__` substituted at deploy time.

**Fix:**
1. Verify the GCS bucket exists:
   ```bash
   gcloud storage buckets describe gs://cmu-idl-ocean --project=YOUR_PROJECT
   ```
2. Verify the singleuser pod mount status:
   ```bash
   kubectl -n cmu-idl describe pod jupyter-<student>
   ```
   Check for events referencing `gcsfuse.csi.storage.gke.io`.
3. Re-deploy the Hub using `bash scripts/03_deploy_hub.sh` or `make hub`.

---

## 5. Student Cannot See "Tier 1: Capstone Grant" Spawner Profile

**Symptom:**
A capstone student only sees the default "Course default (CPU notebook)" profile and cannot select the 4 vCPU / 32 GiB / 100 GiB scratch profile.

**Root Cause:**
The student's JupyterHub user is not a member of the group `project-grant-tier1`.

**Fix:**
1. Teaching assistants can add the student to the group via the JupyterHub Admin panel (`/hub/admin`).
2. Alternatively, use `hub_admin.py` to award the grant directly:
   ```bash
   python scripts/hub_admin.py grant <student_username> --tier 1
   ```
   This expands their PVC to 100 GiB and registers compute limits in `cmu-user-overrides`.

---

## 6. Shared Environment Kernel Missing from Dropdown

**Symptom:**
The JupyterLab kernel selector does not display **`Python 3.11 (CMU 11-785 TPU Shared Env)`**.

**Root Cause:**
The shared kernel metadata in `gs://cmu-idl-ocean/share/jupyter/kernels/` is either not initialized or not readable.

**Fix:**
1. Verify the kernel exists on the shared Tartan mount:
   ```bash
   ls -la /tartan/11785/share/jupyter/kernels/
   ```
2. If missing, register the kernel specs into Tartan storage:
   ```bash
   python -m ipykernel install --prefix=/tartan/11785 --name idl-tpu-kernel --display-name "Python 3.11 (CMU 11-785 TPU Shared Env)"
   ```
3. Restart the student's notebook session.

---

## 7. `ApiException: 403 Forbidden` in Notebook

**Symptom:**
Running `submit_tpu.train()` yields:
```text
kubernetes.client.exceptions.ApiException: (403) Reason: Forbidden
```

**Root Cause:**
The student notebook `ServiceAccount` (`student`) lacks RBAC permissions to create batch Jobs.

**Fix:**
Re-apply the RBAC definitions by running:
```bash
make hub
```
Verify the RoleBinding exists:
```bash
kubectl -n cmu-idl get rolebinding student-tpu-submit
```

---

## 8. Webhook Rejection: "container block-cloud-metadata is privileged"

**Symptom:**
JupyterHub singleuser pods fail to create with:
```text
denied by autogke-disallow-privilege: container block-cloud-metadata is privileged
```

**Root Cause:**
`blockWithIptables` was re-enabled in `jupyterhub-values.yaml`. GKE Autopilot strictly forbids `NET_ADMIN` privileges.

**Fix:**
Ensure `singleuser.cloudMetadata.blockWithIptables: false` in `k8s/jupyterhub-values.yaml`. Egress blocking to `169.254.169.254` is handled natively by Dataplane V2 NetworkPolicies.
