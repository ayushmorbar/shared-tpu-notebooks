# shared-tpu-notebooks

Give several hundred students access to Cloud TPUs from Jupyter notebooks without allocating a dedicated chip for each student.

In a benchmark test run, **300 students shared 64 v5e chips**. The entire queue drained in **12.4 minutes** and cost **$11.93**.

---

## 1. How It Works

GKE assigns one pod per Cloud TPU node, and that pod uses all chips on that node. Two pods cannot share a single physical TPU chip. Consequently, attaching a TPU directly to a student's notebook holds an expensive node for the entire duration the browser tab remains open. For a class of 300 students, that would require 300 dedicated chips (~$405/hr on-demand).

This repository decouples the notebook editor from accelerator execution:

```
[ Student Browser ]
       │  (HTTPS / Google Org Sign-in via Cloud IAP)
       ▼
[ JupyterHub Proxy & Hub Pods ] (cmu-idl namespace)
       │  (Spawns on demand)
       ▼
[ CPU-Only Notebook Pod ] (2 vCPU, 8 GiB RAM, Spot VM, gVisor sandboxed)
       │  (Student writes code & runs submit_tpu.run(...))
       ▼
[ Kueue Admission Controller ] (ClusterQueue: shared-tpu-pool, 32-128 chips)
       │  (Admits job when chip is available: on-demand or DWS flex)
       ▼
[ TPU Batch Job ] (GKE Autopilot provisions ct5lp-hightpu-1t: 1x1 v5e)
```

1. **The Notebook:** Each student gets a sandboxed JupyterLab pod running on low-cost Spot CPU nodes (~$0.02/hr). Students write, debug, and read assignments here.
2. **The Accelerator:** TPU work leaves the notebook as an ephemeral Kubernetes Job. [Kueue](https://kueue.sigs.k8s.io/) manages admission into a shared pool of v5e chips.
3. **The Client Helper:** `notebooks/submit_tpu.py` submits the job, tracks its lifecycle, streams logs, and deletes the job upon completion.

```python
import submit_tpu

print(submit_tpu.run('''
import jax
print(jax.devices())
'''))
```

```
submitted tpu-ada-04ab51c3 to queue 'tpu'
  admitted after 5s in queue
  Succeeded after 15s total
devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

A student holds a chip **only while their code runs** (typically 30–60 seconds).

---

## 2. Tech Stack & Components (2026 Edition)

- **Kubernetes Engine:** GKE Autopilot (Rapid Channel, version 1.36+).
- **Admission Controller:** [Kueue v0.19.1](https://github.com/kubernetes-sigs/kueue) with Dynamic Workload Scheduler (DWS) flex and on-demand ResourceFlavors.
- **JupyterHub:** [Zero-to-JupyterHub Helm Chart v4.4.1](https://hub.jupyter.org/helm-chart/) (JupyterHub App v5.5.1).
- **Ingress & Security:** Google Cloud Identity-Aware Proxy (IAP), Google Front End (GFE) HTTPS Ingress (`spec.ingressClassName: "gce"`), Google-Managed SSL Certificates, and gVisor container sandboxing.
- **Accelerator Architecture:** Google Cloud TPU v5e (`ct5lp-hightpu-1t` / `v5litepod-1`, topology `1x1`).
- **Endpoint Discovery:** Modern Kubernetes `discovery.k8s.io/v1 EndpointSlice` resolution for in-cluster API server NetworkPolicies.

---

## 3. Quick Start Guide

### Prerequisites
- A GCP Project with billing enabled.
- Command-line tools: `gcloud`, `kubectl`, `helm`, `make`, and `docker` (if building custom images).
- Authenticated `gcloud` session: `gcloud auth login` and `gcloud auth application-default login`.

### Step-by-Step Deployment

```bash
# 1. Zero-cost Preflight (validates regional TPU_LITE_PODSLICE_V5 quotas and flex zones)
make preflight PROJECT=my-project REGION=us-west4

# 2. Build & Push Custom Course Image (with JAX, PyTorch, Kaggle, W&B, and Kubernetes client)
make image PROJECT=my-project REGION=us-west4

# 3. Create Autopilot Cluster, Kueue Queues, PriorityClasses & StorageClasses (~12 min)
make cluster PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 4. Deploy JupyterHub & RBAC
make hub PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 5. Run Smoke Test (submits 1 real TPU job on 1 real v5e chip)
make smoke PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 6. Secure with HTTPS & Google Identity-Aware Proxy (IAP)
make iap PROJECT=my-project REGION=us-west4 NS=cmu-idl
```

> [!IMPORTANT]
> **One-Time Manual Step for IAP:**
> Configure the **OAuth consent screen** (Internal) in the Google Cloud Console:
> [https://console.cloud.google.com/auth/branding?project=my-project](https://console.cloud.google.com/auth/branding)
> Without this screen configured, IAP will refuse incoming web requests.

Once the Google-managed SSL certificate transitions to `Active` (typically 10–20 minutes), navigate to the output URL (e.g. `https://<ip>.nip.io` or your custom domain), sign in with an authorized Google account, and open `hw0_tpu_hello.ipynb`.

---

## 4. Operation & Testing Workflows

### Holding Warm TPU Nodes (Zero Cold-Start)
Autopilot scales idle TPU nodes to zero after ~2 minutes. To eliminate the 2–4 minute node provision time for scheduled labs or office hours:

```bash
# Hold N chips warm ready (default 1 chip = $1.35/hr)
make warm-on PROJECT=my-project WARM=1

# Check warm pool status and live hourly burn rate
bash scripts/07_warm_pool.sh status

# Release warm nodes when done
make warm-off PROJECT=my-project
```

*Note: An automatic guardrail CronJob (`tpu-warm-pool-auto-off`) scales the warm pool to 0 daily at 22:00 UTC.*

### High-Concurrency Scale Testing
Simulate 100 students submitting TPU jobs concurrently through a 32-chip pool:

```bash
make scale PROJECT=my-project STUDENTS=100 POOL_CHIPS=32 NS=cmu-idl
make report
```

### End-of-Term Cleanup & Teardown
Because student volumes use `reclaimPolicy: Retain` to protect student work from accidental deletion, use the cleanup utility to release persistent disks:

```bash
# Preview retained PVCs and disks (Dry Run)
bash scripts/10_cleanup_pvcs.sh --dry-run PROJECT=my-project

# Permanently delete PVCs and GCP persistent disks
bash scripts/10_cleanup_pvcs.sh --execute PROJECT=my-project

# Destroy cluster, queued resources, and static IPs
make teardown PROJECT=my-project
```

---

## 5. Speed Benchmarks

| Condition | Pod Scheduling Time | Total Job Completion |
| :--- | :---: | :---: |
| **Warm Pool Enabled (`make warm-on`)** | **+0 s** | **21 s** |
| **Cold Pool (Node scale-up required)** | **120 s – 240 s** | **157 s – 261 s** |

*Hardware Initialization:* Every TPU container execution requires ~10s for JAX/XLA runtime and `libtpu` device discovery.

---

## 6. Cost Architecture & Budgeting

Rates based on `us-west4` on-demand pricing:

| Resource Item | Rate | Billed When |
| :--- | :---: | :--- |
| Cloud TPU v5e (`ct5lp-hightpu-1t`) | $1.20 / chip-hour | Job executes |
| GKE Autopilot TPU Accelerator Premium | $0.15 / chip-hour | Job executes |
| Autopilot Spot Pod vCPU | ~$0.015 / vCPU-hour | Notebook pod running |
| Autopilot Spot Pod Memory | ~$0.002 / GiB-hour | Notebook pod running |
| GKE Cluster Management Fee | $0.10 / hour | Cluster exists |

### 100-Student Burst Cost Matrix (1-Hour Session)

| Runs per Student | Concurrent Chips Active | TPU Cost | Notebook Pods (Spot) + Cluster | Total Cost |
| :---: | :---: | :---: | :---: | :---: |
| 2 runs | 6 chips | $8.00 | $5.50 | **$13.50** |
| 5 runs | 15 chips | $20.00 | $5.50 | **$25.50** |
| 10 runs | 29 chips | $40.00 | $5.50 | **$45.50** |
| 20 runs | 59 chips | $80.00 | $5.50 | **$85.50** |

---

## 7. Configuration Reference

All options can be passed as `Makefile` arguments or shell environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROJECT` | *Required* | Google Cloud Project ID. |
| `REGION` | `us-west4` | GCP Region for GKE cluster and TPUs. |
| `CLUSTER` | `tpu-notebooks` | GKE cluster name. |
| `NS` | `cmu-idl` | Kubernetes namespace for course workloads. |
| `POOL_CHIPS` | `32` | Total v5e capacity shared across queues. |
| `WARM` | `1` | Number of warm placeholder chips to hold. |
| `DOMAIN` | `<IP>.nip.io` | Custom domain name for IAP HTTPS Ingress. |
| `STUDENT_GROUP` | *Optional* | Google Group email for enrolled students (e.g. `group:students@cmu.edu`). |
| `TA_GROUP` | *Optional* | Google Group email for course TAs (e.g. `group:tas@cmu.edu`). |
| `ADMIN_USERS` | *Optional* | Space/comma-separated admin emails (e.g. `user:instructor@cmu.edu`). |

---

## 8. Repository File Structure

```
shared-tpu-notebooks/
├── Makefile                     # Central build, deploy, test, and teardown interface
├── README.md                    # Complete architectural and operational manual
├── docker/
│   ├── Dockerfile               # Scipy-notebook image extended with JAX TPU stack
│   └── requirements.txt         # Pinned Python ML dependencies (JAX, PyTorch, W&B)
├── k8s/
│   ├── ingress-iap.yaml         # L7 Ingress, BackendConfig, and ManagedCertificate
│   ├── jupyterhub-values.yaml   # Zero-to-JupyterHub Helm configuration & security profiles
│   ├── kueue-tpu-queues.yaml    # ResourceFlavors & ClusterQueue specifications
│   ├── priority-classes.yaml    # Preemption hierarchy (Core > Notebook > Job > Warm)
│   ├── student-tpu-job.yaml     # Declarative batch Job template for TPU runs
│   └── tpu-warm-pool.yaml       # Priority -10 placeholder deployment & auto-off CronJob
├── notebooks/
│   ├── hw0_tpu_hello.ipynb      # Sample assignment: Attention FLOPs & roofline analysis
│   └── submit_tpu.py            # Client library for submitting TPU jobs from notebooks
└── scripts/
    ├── 00_preflight.sh          # Regional quota and accelerator discovery (Read-only)
    ├── 01_build_image.sh        # Artifact Registry repo creation and Docker build/push
    ├── 01_spray_v5e.sh          # Live probing of DWS Flex capacity across zones
    ├── 02_create_cluster.sh     # GKE Autopilot cluster creation and Kueue setup
    ├── 03_deploy_hub.sh         # Helm deployment of JupyterHub with RBAC & NetworkPolicy
    ├── 04_scale_test.py         # Concurrent batch load testing harness
    ├── 05_report.py             # Performance percentiles and exact billing reporter
    ├── 06_hub_spawn_test.py     # Concurrent JupyterHub spawn latency benchmark
    ├── 07_warm_pool.sh          # CLI tool to manage warm TPU node placeholders
    ├── 08_setup_iap.sh          # Cloud IAP, static IP, and IAM binding automation
    ├── 10_cleanup_pvcs.sh       # End-of-term PVC and retained persistent disk cleanup
    └── 99_teardown.sh           # Full infrastructure teardown with zero-cost verification
```

---

## 9. License

Apache 2.0. See [`LICENSE`](LICENSE) for details.
