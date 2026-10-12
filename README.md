# shared-tpu-notebooks

Give several hundred students access to Cloud TPUs from Jupyter notebooks without allocating a dedicated chip for each student.

In a measured benchmark run, **300 students shared 64 v5e chips**. The entire queue drained in **12.4 minutes** and cost **$11.93** (compared to $73.58 for dedicated chips).

---

## 1. How It Works

GKE schedules one pod per Cloud TPU node, and that pod uses all chips on that node. Two pods cannot share a single physical TPU chip. Consequently, attaching a TPU directly to a student's notebook holds an expensive node for the entire duration the browser tab remains open. For a class of 300 students, that would require 300 dedicated chips (~$405/hr on-demand).

This repository decouples the notebook editor from accelerator execution:

```
[ Student Browser ]
       │  (HTTPS / Google Org Sign-in via Cloud IAP)
       ▼
[ JupyterHub Proxy & Hub Pods ] (cmu-idl namespace)
       │  (Spawns on demand)
       ▼
[ CPU-Only Notebook Pod ] (2 vCPU, 16 GiB RAM, 32 GiB home disk, Spot VM, gVisor sandboxed)
       │  (Student writes train.py & runs submit_tpu.submit("train.py"))
       ▼
[ tpu-submit service ] (the only thing that can create Jobs; one fixed Job template)
       │  (Stores the script folder in the job bucket, creates a queued Job)
       ▼
[ Kueue Admission Controller ] (ClusterQueue: shared-tpu-pool, 32-128 chips)
       │  (Admits job when chip is available: on-demand or DWS flex)
       ▼
[ TPU Batch Job ] (GKE Autopilot provisions ct5lp-hightpu-1t: 1x1 v5e)
       (unpacks the dataset to local disk, runs the script, uploads logs & checkpoints)
```

1. **The Notebook:** Each student gets a sandboxed JupyterLab pod running on low-cost Spot CPU nodes (~$0.015/vCPU-hr). Students write, debug, unit-test, and read assignments here.
2. **The Submission Service:** `services/tpu-submit` receives scripts from notebooks, authenticated by the student's JupyterHub token, and builds every Job from one template: one chip, the course image, the queue, a time limit. One active job per student, optional chip-hour budget. Students have no Kubernetes permissions at all.
3. **The Accelerator:** [Kueue](https://kueue.sigs.k8s.io/) admits Jobs into a shared pool of v5e chips. On the chip, a wrapper copies the student's folder and the dataset onto local disk, runs the script, and uploads `job.log` and the `expts/` folder through the service every few minutes and at exit. A job pushed off its chip restarts and resumes from the last uploaded checkpoint.
4. **The Client Helper:** `notebooks/submit_tpu.py` works like `sbatch`/`squeue`/`scancel`: submit, watch, status, cancel, and pull results back into the notebook.

```python
import submit_tpu

job = submit_tpu.submit("train.py", args=["--config", "config.yaml"],
                        env={"WANDB_API_KEY": WANDB_API_KEY})
submit_tpu.watch(job)    # live log; stopping the cell does not stop the job
submit_tpu.pull(job)     # copy expts/ (checkpoints, results.csv) and job.log here
```

```
packed 50 files from /home/jovyan/hw3p2 (0.4 MB)
submitted job 20261010-221530-a1b2: python train.py --config config.yaml
[20261010-221530-a1b2] queued, #3 in line
[20261010-221530-a1b2] starting
[tpu 22:19:02] job 20261010-221530-a1b2 starting on tpu-ada-20261010-221530-a1b2-x7k2p
[tpu 22:19:04] unpacked your files (0.4 MB)
[tpu 22:19:05] downloading dataset gs://.../hw3p2.tar (11.20 GB) to local disk
...
```

A student holds a chip **only while their job runs**. Short HW0-style runs take 30–60 seconds; HW3-style training runs are bounded by `MAX_JOB_HOURS`.

---

## 2. Tech Stack & Components (2026 Edition)

- **Kubernetes Engine:** GKE Autopilot (Rapid Channel, version 1.36+).
- **Admission Controller:** [Kueue v0.19.1](https://github.com/kubernetes-sigs/kueue) with Dynamic Workload Scheduler (DWS) flex and on-demand ResourceFlavors in a unified cohort (`BestEffortFIFO`).
- **JupyterHub:** [Zero-to-JupyterHub Helm Chart v4.4.1](https://hub.jupyter.org/helm-chart/) (JupyterHub App v5.5.1).
- **Ingress & Security:** Google Cloud Identity-Aware Proxy (IAP), Google Front End (GFE) HTTPS Ingress (`spec.ingressClassName: "gce"`), Google-Managed SSL Certificates, and gVisor container sandboxing.
- **Accelerator Architecture:** Google Cloud TPU v5e (`ct5lp-hightpu-1t` / `v5litepod-1`, topology `1x1`).
- **Job Submission:** `tpu-submit`, a small Tornado service authenticated by JupyterHub tokens, with Cloud Storage buckets reached through Workload Identity Federation (no Google service accounts or keys).

---

## 3. Quick Start Guide

### Prerequisites
- A GCP Project with billing enabled.
- Command-line tools: `gcloud`, `kubectl`, `helm`, `make`, and `docker` (if building custom images).
- Authenticated `gcloud` session: `gcloud auth login` and `gcloud auth application-default login`.

> [!TIP]
> For a detailed, step-by-step walkthrough, please read the [Getting Started Guide](docs/getting_started.md). For architectural details, see the [Architecture Document](docs/architecture.md).

### Step-by-Step Deployment

```bash
# 1. Zero-cost Preflight (validates regional TPU_LITE_PODSLICE_V5 quotas and flex zones)
make preflight PROJECT=my-project REGION=us-west4

# 2. Build & Push the notebook, TPU job, and tpu-submit images
make image PROJECT=my-project REGION=us-west4

# 3. Create Autopilot Cluster, Kueue Queues, PriorityClasses & StorageClasses (~12 min)
make cluster PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 4. Create the job and dataset buckets, then upload an assignment's dataset once
make storage PROJECT=my-project REGION=us-west4 NS=cmu-idl
bash scripts/12_setup_storage.sh upload hw3p2 /path/to/hw3p2_data   # prints DATASETS= for config.env

# 5. Deploy the TPU submission service & JupyterHub
make hub PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 6. Run Smoke Test (submits 1 real TPU job on 1 real v5e chip)
make smoke PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 7. Secure with HTTPS & Google Identity-Aware Proxy (IAP)
make iap PROJECT=my-project REGION=us-west4 NS=cmu-idl

# 8. The student path end to end, with a token from https://<hub>/hub/token
HUB_TOKEN=... make smoke-submit PROJECT=my-project NS=cmu-idl
```

`make test` runs the service, client, and TPU-wrapper tests locally (needs `pip install -r requirements-dev.txt`, no cloud access).

> [!IMPORTANT]
> **One-Time Manual Step for IAP:**
> Configure the **OAuth consent screen** (Internal) in the Google Cloud Console:
> [https://console.cloud.google.com/auth/branding?project=my-project](https://console.cloud.google.com/auth/branding)
> Without this screen configured, IAP will refuse incoming web requests.

Once the Google-managed SSL certificate transitions to `Active` (typically 10–20 minutes), navigate to the output URL (e.g. `https://<ip>.nip.io` or your custom domain), sign in with an authorized Google account, and open `hw0_tpu_hello.ipynb`.

---

## 4. Speed & Cold Start Analysis

A **warm pool** contains a TPU node that an earlier job left in place. A **cold pool** contains no active nodes, requiring GKE Autopilot to provision one from Compute Engine.

| Condition | Pod Scheduled | Job Finished | Notes |
| :--- | :---: | :---: | :--- |
| **Warm Pool Enabled (`make warm-on`)** | **+0 s** | **21 s** | Instant scheduling; node is ready. |
| **Cold Pool (Autopilot Node scale-up)** | **120 s – 240 s** | **157 s – 261 s** | 2–4 min node build dominates latency. |

*Hardware Initialization:* Every TPU container execution requires ~10s for JAX/XLA runtime and `libtpu` device discovery.

### Understanding DWS Flex Zones vs. Cold Starts

The repository configures two Kueue ResourceFlavors: `v5e-ondemand` and `v5e-flex`. The flex flavor targets nodes labeled `cloud.google.com/gke-flex-start: "true"`, which only exist where Google Cloud provides a v5e Dynamic Workload Scheduler (DWS) Flex pool.

Measured on 2026-08-25 by submitting real `FLEX_START` requests:

| Zone | Result | Impact |
| :--- | :--- | :--- |
| **`us-west4-a`** | **Accepted** | Full flex pool active; lowest cost and fast cold starts. |
| `us-central1-a` | Rejected (Code 3: No flex pool) | All jobs fall back to on-demand capacity. |
| `us-central1-b/c` | v5e not offered | 0 accelerator types available. |
| `europe-west4-b` | Rejected (Code 3: No flex pool) | All jobs fall back to on-demand capacity. |

Where no flex pool exists, the flex queue quota cannot place nodes and all work runs on on-demand capacity. Confirm during a run:

```bash
kubectl get nodes -l cloud.google.com/gke-flex-start=true
```

Always check by submitting a probe request rather than reading dashboard pool sizes:
```bash
gcloud alpha compute tpus queued-resources create probe --node-id=probe \
  --zone=us-west4-a --accelerator-type=v5litepod-1 --runtime-version=v2-alpha-tpuv5-lite \
  --provisioning-model=FLEX_START --max-run-duration=3600s
```

---

## 5. Cost Architecture & Budgeting

Rates based on `us-west4` on-demand pricing:

| Resource Item | Rate | Billed When |
| :--- | :---: | :--- |
| Cloud TPU v5e (`ct5lp-hightpu-1t`) | $1.20 / chip-hour | Job executes |
| GKE Autopilot TPU Accelerator Premium | $0.15 / chip-hour | Job executes |
| Autopilot Spot Pod vCPU | ~$0.015 / vCPU-hour | Notebook pod running |
| Autopilot Spot Pod Memory | ~$0.002 / GiB-hour | Notebook pod running |
| GKE Cluster Management Fee | $0.10 / hour | Cluster exists |

### Unit Costs
- **One TPU run costs ~$0.040** (mean duration of 106s). A short 15s debug run costs **$0.006**.
- **One open notebook costs ~$0.06/hr** on Spot (2 vCPU, 16 GiB) vs ~$0.289/hr on-demand (4 vCPU, 16 GiB).

### 100-Student Burst Cost Matrix (1-Hour Session)

| Runs per Student | Concurrent Chips Active | TPU Cost | Notebook Pods (Spot) + Cluster | Total Cost |
| :---: | :---: | :---: | :---: | :---: |
| 2 runs | 6 chips | $8.00 | $6.50 | **$14.50** |
| 5 runs | 15 chips | $20.00 | $6.50 | **$26.50** |
| 10 runs | 29 chips | $40.00 | $6.50 | **$46.50** |
| 20 runs | 59 chips | $80.00 | $6.50 | **$86.50** |

### Where the Cost Goes
A notebook left open for hours without running code can cost more than several TPU executions. Three levers keep costs minimal:
1. **Spot VMs for Notebooks:** Cuts CPU pod cost by 60–90%.
2. **2 vCPU / 16 GiB Default Profile:** Headroom for interactive JAX next to a memmap'd disk cache, without promoting onto a memory-optimized SKU.
3. **Automated Idle Culling:** Inactivity culler in `k8s/jupyterhub-values.yaml` terminates idle sessions after 60 minutes (`maxAge: 28800` / 8h ceiling).

---

## 6. Capacity & Node Packing Math

| Quota Metric | Default Limit | Source / Purpose |
| :--- | :---: | :--- |
| Concurrent Open Notebooks | ~187 | Regional `CPUS` quota (1500 default), memory-bound at 16 GiB |
| Concurrent TPU Chips (On-Demand) | 512 | `TPU_LITE_PODSLICE_V5` quota |
| Concurrent TPU Chips (Spot) | 1536 | `PREEMPTIBLE_TPU_LITE_PODSLICE_V5` |

### Why Quota Math Requires Node-Packing Analysis
Autopilot provisions an 8 vCPU / ~32 GiB node with ~7.9 vCPU allocatable. GKE system DaemonSets (CSI, logging, monitoring) consume ~2.6 vCPU and several GiB of RAM. CPU would still fit two 2 vCPU notebooks, but **16 GiB RAM fits only one notebook per node**.
- A quota of 1500 vCPU provisions 187 nodes $\rightarrow$ **~187 concurrent notebooks**.
- *Caution:* Dividing 1500 by 2 vCPU = 750 ignores both system pods and the memory ceiling, and overstates capacity by 4x.

### Multi-Region MultiKueue for Global Scheduling
If regional TPU inventory stocks out during high-demand events, [MultiKueue](https://kueue.sigs.k8s.io/docs/concepts/multikueue/) can span the queue across clusters in multiple regions, dispatching jobs to wherever chips become available.

---

## 7. Scaling to 500 Students

Scaling to 500 students requires only routine configuration adjustments:

1. **Raise Regional `CPUS` Quota:** 500 notebooks at 1 per node (16 GiB RAM) require 500 nodes. Request 2400+ vCPU in your region for DaemonSet overhead (free quota increase).
2. **TPU Quotas Need No Change:** Default regional quota of 512 on-demand and 1536 Spot chips easily handles 500 students.
3. **Storage Sizing:** 500 students × 32 GiB = 16 TB of `pd-balanced` storage (~$1,600/month across the term). The 32 GiB home volume is what lets a disk-cache of a 112x112 train split (~16 GiB) finish without filling the PVC. Reclaim storage at term end using `scripts/10_cleanup_pvcs.sh`.

```bash
# Size the pool and sections (e.g. 500 students, 128 chips, 6 sections)
POOL_CHIPS=128 SECTIONS="a b c d e f" make cluster PROJECT=my-project
```

### 300 vs 500 Student Comparison

| Metric | Measured (300 Students) | Projected (500 Students) |
| :--- | :---: | :---: |
| Shared Chip Pool | 64 chips | 128 chips |
| Wall Clock Queue Drain | 12.4 min (3.0 min burst) | ~3.0 min |
| Total Chip-Hours | 8.84 | ~14.7 |
| Total TPU Burst Cost | $11.93 | ~$19.80 |
| **Cost Per Student** | **$0.040** | **$0.040** |

---

## 8. Measured Benchmark Results

Conducted in `us-west4` with 4 lab sections sharing a single Kueue cohort:

| Metric | Value |
| :--- | :---: |
| Total Students | 300 |
| Completed Workloads | 300 (100% success, 0 failed) |
| Maximum Concurrent TPU Chips | 64 |
| Total Wall Clock Duration | 12.4 min |
| Queue Wait Time (p50 / p95) | 270 s / 487 s |
| Node Build & Image Pull (p50 / p95) | 23 s / 74 s |
| Job Execution Duration (p50) | 52 s |
| Total Chip-Hours Consumed | 8.84 |
| Concurrent Hub Spawns (50 users) | 50/50 Ready (p50: 137s, p95: 191s) |

---

## 9. Configuration & JupyterHub Profiles

### Makefile & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROJECT` | *Required* | Google Cloud Project ID. |
| `REGION` | `us-west4` | GCP Region for GKE cluster and TPUs. |
| `CLUSTER` | `tpu-notebooks` | GKE cluster name. |
| `NS` | `cmu-idl` | Kubernetes namespace for course workloads. |
| `POOL_CHIPS` | `32` | Total v5e capacity shared across queues. |
| `WARM` | `1` | Number of warm placeholder chips to hold. |
| `DOMAIN` | `<IP>.nip.io` | Custom domain name for IAP HTTPS Ingress. |
| `STUDENT_GROUP` | *Optional* | Google Group for enrolled students (`group:students@cmu.edu`). |
| `TA_GROUP` | *Optional* | Google Group for course TAs (`group:tas@cmu.edu`). |
| `ADMIN_USERS` | *Optional* | Admin user emails (`user:instructor@cmu.edu`). |
| `STUDENT_BUCKET` | `<PROJECT>-tpu-runs` | Job snapshots, logs, and checkpoints. Only tpu-submit can use it. |
| `DATASET_BUCKET` | `<PROJECT>-course-data` | Dataset archives. TPU jobs can read it, nothing can write it from the cluster. |
| `DATASETS` / `DEFAULT_DATASET` | *Empty* | `name=gs://...` archives students can pick with `submit(dataset=...)`. |
| `MAX_JOB_HOURS` | `8` | Run-time limit per job (queue time not counted). |
| `CHIP_HOUR_BUDGET` | `0` (off) | Total chip-hours per student. |
| `JOB_DISK` | `100Gi` | Local disk per job for the unpacked dataset and checkpoints. |

### JupyterHub Profile Options (`k8s/jupyterhub-values.yaml`)

| Profile Name | Resource Allocation | Intended Audience |
| :--- | :--- | :--- |
| **Course default — CPU notebook** | 2 vCPU, 16 GiB RAM, 32 GiB home disk (Spot VM, gVisor) | All Students (interactive coding & TPU submission) |
| **Large CPU notebook** | 8 vCPU, 32 GiB RAM (Spot VM, gVisor) | Students (dataset preprocessing, tokenization) |
| **Pinned TPU v5e notebook** | 1 v5e chip attached directly ($1.35/hr) | Staff Only (`allowed_groups: [staff]`) |

---

## 10. Operations, Warm Pool & Maintenance

### Managing Warm TPU Node Placeholders
```bash
# Hold 1 warm chip ready ($1.35/hr)
make warm-on PROJECT=my-project WARM=1

# Check status and running burn rate
bash scripts/07_warm_pool.sh status

# Release placeholders
make warm-off PROJECT=my-project
```
*Guardrail:* The `tpu-warm-pool-auto-off` CronJob automatically releases warm nodes daily at 22:00 UTC.

### Storage Reclamation & Full Teardown
```bash
# Preview retained PVCs & persistent disks
make clean-pvcs-dry-run PROJECT=my-project

# Permanently delete retained PVCs & GCP disks
make clean-pvcs PROJECT=my-project

# Destroy cluster, queued resources, and static IPs
make teardown PROJECT=my-project
```

---

## 11. Adapting for Other TPU Generations (e.g. v6e / v4)

To switch from TPU v5e (`v5litepod-1` / `ct5lp-hightpu-1t`) to another generation:

| Target File | Required Update |
| :--- | :--- |
| [`k8s/student-tpu-job.yaml`](k8s/student-tpu-job.yaml) | Update `nodeSelector` accelerator and topology labels. |
| `config.env` | Set `TPU_ACCELERATOR` / `TPU_TOPOLOGY`, then `make hub`; tpu-submit builds student Jobs with them. |
| [`k8s/kueue-tpu-queues.yaml`](k8s/kueue-tpu-queues.yaml) | Update `nodeLabels` on all ResourceFlavors. |
| [`k8s/tpu-warm-pool.yaml`](k8s/tpu-warm-pool.yaml) | Update `nodeSelector` labels on the placeholder Deployment. |
| [`scripts/05_report.py`](scripts/05_report.py) | Update `CHIP_HR` rate for accurate billing reports. |

Find all references in one command:
```bash
grep -rn "tpu-v5-lite-podslice" k8s/ notebooks/
```

Confirm regional quota for the target generation via `make preflight` before modifying manifests.

---

## 12. Repository Layout

```
shared-tpu-notebooks/
├── Makefile                     # Central build, deploy, test, and teardown interface
├── README.md                    # Complete architectural and operational manual
├── docker/
│   ├── Dockerfile               # Scipy-notebook image extended with JAX TPU stack
│   ├── requirements.txt         # Pinned Python ML dependencies (JAX, PyTorch, W&B, HW libs)
│   └── tpu-job/                 # Image student scripts run in on the chip
│       ├── Dockerfile           # FROM Google's JAX TPU image + HW3 deps (CPU torch, tokenizers...)
│       └── tpu_entrypoint.py    # Fetch files & dataset, run the script, upload logs & expts/
├── services/tpu-submit/         # Submission service: auth, Job template, rules, file relay
├── tests/                       # make test: unit + end-to-end tests with a fake cluster
├── k8s/
│   ├── ingress-iap.yaml         # L7 Ingress, BackendConfig, and ManagedCertificate
│   ├── jupyterhub-values.yaml   # Zero-to-JupyterHub Helm configuration & security profiles
│   ├── kueue-tpu-queues.yaml    # ResourceFlavors & ClusterQueue specifications
│   ├── priority-classes.yaml    # Preemption hierarchy (Core > Notebook > Job > Warm)
│   ├── student-tpu-job.yaml     # Raw Job template used by `make smoke` and the scale test
│   ├── tpu-submit.yaml          # tpu-submit Deployment, RBAC, NetworkPolicy, service accounts
│   └── tpu-warm-pool.yaml       # Priority -10 placeholder deployment & auto-off CronJob
├── notebooks/
│   ├── hw0_tpu_hello.ipynb      # Sample assignment: Attention FLOPs & roofline analysis
│   └── submit_tpu.py            # Student client: submit, watch, status, cancel, pull
└── scripts/
    ├── 00_preflight.sh          # Regional quota and accelerator discovery (Read-only)
    ├── 01_build_image.sh        # Artifact Registry repo creation and Docker build/push
    ├── 01_spray_v5e.sh          # Live probing of DWS Flex capacity across zones
    ├── 02_create_cluster.sh     # GKE Autopilot cluster creation and Kueue setup
    ├── 03_deploy_hub.sh         # Deploy tpu-submit, then JupyterHub via Helm
    ├── 04_scale_test.py         # Concurrent batch load testing harness
    ├── 05_report.py             # Performance percentiles and exact billing reporter
    ├── 06_hub_spawn_test.py     # Concurrent JupyterHub spawn latency benchmark
    ├── 07_warm_pool.sh          # CLI tool to manage warm TPU node placeholders
    ├── 08_setup_iap.sh          # Cloud IAP, static IP, and IAM binding automation
    ├── 10_cleanup_pvcs.sh       # End-of-term PVC and retained persistent disk cleanup
    ├── 12_setup_storage.sh      # Job & dataset buckets, Workload Identity bindings, dataset upload
    ├── 13_smoke_submit.sh       # End-to-end submit through tpu-submit with submit_tpu.py
    └── 99_teardown.sh           # Full infrastructure teardown with zero-cost verification
```

---

## 13. License

Apache 2.0. See [`LICENSE`](LICENSE) for details.
