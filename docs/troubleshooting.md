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
You need about 8 vCPU of *node* quota per concurrent student: each default notebook requests 2 vCPU / 16 GiB, and Autopilot packs one of those per ~32 GiB node after DaemonSets. For 15 concurrent students, this requires ~120 vCPU (only 8% of the standard 1500 vCPU GCP project quota). If scaling to 100+ concurrent students, request a quota increase in the GCP Console.

## 3. TPU Jobs hang in the queue for 5+ minutes

**Symptom:**
`submit_tpu.watch()` keeps printing:
```text
[20261010-221530-a1b2] queued, #14 in line
```
or stays at `starting` for several minutes.

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

## 4. `submit_tpu.submit()` errors

Every refusal from the service is a `SubmitError` whose message says what to do. The ones that point at the deployment rather than the student:

| Message | Cause | Fix |
| :--- | :--- | :--- |
| `Could not reach the TPU submission service` | The tpu-submit pod is down or restarting, or the notebook's NetworkPolicy does not allow it. | `kubectl -n cmu-idl get pods -l app.kubernetes.io/name=tpu-submit`; check its logs; re-run `make hub`. |
| `This JupyterHub token cannot use the TPU submission service. (HTTP 403)` | The notebook's token lacks `access:services!service=tpu-submit`. Servers started before the hub config change keep their old token. | Have the student restart their server (File → Hub Control Panel → Stop, then Start). |
| `TPU_SUBMIT_URL is not set` | The server predates the hub config change, or the code is running outside the hub. | Restart the server. |
| `Could not create the Kubernetes Job: ... exceeded quota` (HTTP 500) | The namespace ResourceQuota `count/jobs.batch` is full. | Raise `STUDENTS` and re-run `make cluster`, or lower `JOB_TTL_SECONDS`. |
| `You already have job ... queued or running` (HTTP 409) | One active job per student, by design. | `submit_tpu.cancel()` or wait. |

The service log has one line per submit and cancel:
```bash
kubectl -n cmu-idl logs deploy/tpu-submit --tail=100
```

## 5. A job ends `failed` before training starts

Open the log with `submit_tpu.logs(job)`. Lines starting with `[tpu ...]` come from the job wrapper:

- `setup failed before your script ran: dataset archive gs://... does not exist`, or a 403 from Cloud Storage: the dataset path in `DATASETS` is wrong, or `make storage` has not granted `tpu-job` read access. IAM changes can take a few minutes to apply.
- `No space left on device` while unpacking the dataset: raise `JOB_DISK` (and keep it within Autopilot's ephemeral-storage limit for TPU pods).
- A pod that stays `Pending` with `CreateContainerConfigError` for more than a minute means its per-job Secret is missing; cancel and resubmit.

## 6. Webhook Rejection for Privilege

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
