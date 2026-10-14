# 💰 Cost Management & Financial Model

This document outlines the cost structure, financial engineering, and automated guardrails for operating the **CMU 11-785 Cloud TPU Hub** on Google Cloud Platform.

---

## 1. Executive Summary: 87% Cost Reduction

Traditional academic computing deployments allocate dedicated GPU workstations to students, resulting in massive idle expenditure. By combining **ephemeral Spot CPU notebooks**, **queued Cloud TPU v5e batch scaling via Kueue**, and **Tartan shared storage**, our platform delivers state-of-the-art deep learning infrastructure at **$70.59 per student for the entire semester** (87% less than pinned GPU clusters).

| Platform / Architecture | 4-Month Total Cost (300 Students) | Cost per Student | Primary Cost Driver |
| :--- | :--- | :--- | :--- |
| **Traditional Pinned GPU Cluster** (NVIDIA A100 / H100) | $160,000 – $210,000 | $530 – $700 | Idle GPUs holding allocations 24/7 |
| **PSC Bridges-2 / Bridges-3** (NSF Allocation) | $0 direct (High proposal overhead) | $0 direct | 3–6 month proposal cycle; limited elasticity |
| **CMU 11-785 Tartan TPU Hub** (Our Setup) | **$21,178** | **$70.59** | Pay-per-second ephemeral TPU compute + Spot CPUs |

---

## 2. October 2026 Comprehensive Financial Model (300 Students, 4 Months)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     CMU 11-785 4-Month Term Budget Breakdown                     │
├──────────────────────────────────────────────────────┬─────────────┬─────────────┤
│ Component                                            │ 4-Month Sub │ Per Student │
├──────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ 1. Spot CPU Notebooks (2 vCPU, 16 GiB, ~4 hr/day)   │  $2,592.00  │    $8.64    │
│ 2. Ephemeral TPU v5e Pool (600 chip-hrs/day, $1.35) │ $12,960.00  │   $43.20    │
│ 3. Persistent Storage (300 × 32 GiB SSD PVCs)        │  $1,920.00  │    $6.40    │
│ 4. Tartan Central Store (gs://cmu-idl-ocean, 20 TB)  │  $1,600.00  │    $5.33    │
│ 5. Capstone Grants (30 teams × Tier 1, 100 GiB PVC)  │  $1,440.00  │    $4.80    │
│ 6. GKE Autopilot Cluster Management ($0.10/hr)       │    $288.00  │    $0.96    │
│ 7. Cloud NAT & Egress (Model Hub, Weights & Biases) │    $378.00  │    $1.26    │
├──────────────────────────────────────────────────────┼─────────────┼─────────────┤
│ TOTAL ESTIMATED SEMESTER EXPENDITURE                 │ $21,178.00  │   $70.59    │
└──────────────────────────────────────────────────────┴─────────────┴─────────────┘
```

---

## 3. Cost Profile by Grant Tier

| Grant Tier | Storage Quota | Compute Spec | TPU Allocation | Cost per Student / Term |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: Coursework** | 32 GiB SSD | 2 vCPU / 16 GiB (Spot) | Queued v5e via `submit_tpu` | ~$58.24 |
| **Tier 1: Capstone Team** | 100 GiB SSD | 4 vCPU / 32 GiB (Spot) | Queued v5e + Cohort priority | ~$84.40 |
| **Tier 2: Research Fellow**| 200 GiB SSD | 8 vCPU / 64 GiB (Spot) | Queued v5e + Pinned slice | ~$142.10 |
| **Maintainer Workstation** | 100 GiB SSD | 16 vCPU / 64 GiB (On-Demand)| Unrestricted v5e slice | Course Overhead |

---

## 4. Why Ephemeral TPU Execution Saves Money

1. **Pay-Per-Second Execution**:
   When a student trains for 40 epochs on HW2P2 (~32 minutes), they bill exactly $0.72 of TPU time. While analyzing error curves or editing hyperparameters, their TPU consumption is **$0.00**.
2. **Dynamic Workload Scheduler (DWS) Flex**:
   GKE Autopilot provisions TPU nodes dynamically and tears them down the moment the pod terminates.
3. **Storage Caching Line-Rate Savings**:
   Instead of each student repeatedly downloading 18.5 GB datasets over the internet (generating substantial network egress fees), datasets are hosted in-region (`us-west4`) in `gs://cmu-idl-ocean`. GCS FUSE with local NVMe caching (`file-cache:max-size-mb:50000`) streams data at line-rate (>1.2 GB/s) with **zero cross-region network egress charges**.

---

## 5. Automated Cost-Saving Guardrails

### 1. Multi-Tier Idle Culler
Students who close browser tabs without stopping notebooks do not accumulate unnecessary node charges:
```yaml
cull:
  enabled: true
  timeout: 3600    # Cull pod if idle for 60 minutes
  every: 300       # Scan active kernels every 5 minutes
  maxAge: 28800    # Hard kill after 8 hours to stop zombie sessions
```

### 2. Warm Pool Auto-Shutdown Cron
To avoid morning cold-starts during live lectures, instructors can hold 1–2 warm nodes via `make warm-on WARM=1`. To prevent accidental overnight billing ($32.40/day per idle node), `k8s/tpu-warm-pool.yaml` includes a Kubernetes CronJob that automatically executes:
```bash
kubectl scale deployment tpu-warm-pool --replicas=0
```
every day at **22:00 UTC**.

### 3. Log Ingestion Exclusion Filter
Autopilot cluster logs can silently accumulate costs ($0.50/GiB) if a student runs an infinite print loop. `scripts/02_create_cluster.sh` installs a Cloud Logging exclusion filter that automatically drops high-volume stdout/stderr noise from student containers before it reaches Google Cloud Logging billing.

---

## 6. End-of-Term Storage Reclamation

Persistent Volume Claims bill continuously until deleted. To stop all storage billing at the end of the semester:

```bash
# 1. Preview retained PVCs without deleting
make clean-pvcs-dry-run

# 2. Permanently delete all student PVCs and underlying persistent disks
make clean-pvcs

# 3. Destroy cluster and stop all GCP billing
make teardown
```
