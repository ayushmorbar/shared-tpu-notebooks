# CMU 11-785: Tartan 3-Tier Storage Hierarchy & Live Multi-User Environments Architecture

**Platform Maintainer:** Ayush Morbar  
**Target Environment:** Google Kubernetes Engine (GKE Autopilot) & Cloud TPU v5e/v6e  
**Course Reference:** CMU 11-785: Introduction to Deep Learning (Fall 2026 / Spring 2027) & Capstone Grants  

---

## 1. Executive Summary & HPC Mapping (Bridges-2 vs. Tartan Hub)

On traditional supercomputing clusters like **PSC Bridges-2**, courses rely on a two-tier storage and software model:
* **Ocean (`/ocean/projects/...`)**: Central high-capacity parallel storage shared by all users in a grant. Course datasets, assignment starter files, and shared software environments reside here. Read-only for students, read-write for course staff.
* **Jet / Scratch (`$SCRATCH` / `/local`)**: Ultra-fast node-local scratch storage for job execution and caching.
* **Home (`/jet/home/...`)**: Private persistent user home directory.
* **Shared Software**: `module load anaconda3` allows any user to load shared Conda/virtual environments maintained by instructors.

We map this supercomputing storage paradigm directly onto **Google Cloud Platform (GCP) and GKE Autopilot** as the **Tartan 3-Tier Storage Hierarchy**:

```
+---------------------------------------------------------------------------------------------------+
|                                 CMU 11-785 TPU HUB STORAGE TOPOLOGY                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ 1. Central Tartan Store ]  -> Mount: /tartan/11785 (Dual-mounted to /ocean for compat)         |
|  ├── /tartan/11785/datasets/      <- Pre-processed .npy caches (HW1, HW2, HW3, HW4)               |
|  ├── /tartan/11785/assignments/   <- Released starter notebooks and assignment scripts            |
|  ├── /tartan/11785/envs/          <- Shared Python & Conda environments (Admin-managed)           |
|  └── /tartan/11785/projects/      <- Dedicated directories for approved Capstone Grant teams      |
|                                                                                                   |
|  [ 2. Student Private Home ]  -> Mount: /home/jovyan (Read-Write, 32 GiB Standard-RWO PVC)       |
|  ├── /home/jovyan/.env            <- Student personal secrets (WandB, Kaggle, HF tokens)          |
|  ├── /home/jovyan/notebooks/      <- Student active coursework and modified code                  |
|  └── /home/jovyan/checkpoints/    <- Locally synced model checkpoints and submission.csv          |
|                                                                                                   |
|  [ 3. Local Node Scratch ]    -> Mount: /tmp (Local NVMe SSD, emptyDir, Line-Rate IOPS)           |
|  └── High-throughput temporary cache for TPU jobs; bypasses network storage bottlenecks.          |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Shared Storage Implementation Options on GKE

### Option A: GKE Cloud Storage FUSE CSI Driver with NVMe Caching (Standard)
* **Mechanism:** GKE Autopilot natively supports the `gcsfuse.csi.storage.gke.io` CSI driver. A single GCS bucket (`gs://cmu-idl-ocean`) is dual-mounted into all student and staff pods at both `/tartan/11785` and `/ocean`.
* **High-Speed Caching:** Configured with local NVMe disk caching (`file-cache:max-size-mb:50000`) so datasets stream at local SSD bandwidth ($>1.2\text{ GB/s}$) on subsequent training epochs.
* **Cost:** Zero idle compute cost; you pay only standard Cloud Storage storage rates ($0.020/GB/month).
* **Permissions:** Configured via Google Cloud Workload Identity:
  * Student ServiceAccount (`student`): Granted `roles/storage.objectViewer` on the bucket (Read-Only).
  * Staff ServiceAccount (`staff`): Granted `roles/storage.objectAdmin` on the bucket (Read-Write).

---

## 3. Live Shared Environments (`/tartan/11785/envs/`): Zero Docker Rebuilds

Instead of forcing students to run `pip install` or rebuilding heavy Docker container images every time a new library is needed, admins manage shared environments directly inside `/tartan/11785/envs/idl-shared-env`.

### How Admins Create & Update Packages in 15 Seconds
1. An instructor or admin (`amorbar`) launches the **Admin Workstation** notebook.
2. In the terminal, create or update the shared environment:
   ```bash
   # Create a shared virtual environment on the Tartan volume
   python3 -m venv /tartan/11785/envs/idl-shared-env --system-site-packages

   # Install new course packages (instantly visible to all 300 students!)
   /tartan/11785/envs/idl-shared-env/bin/pip install \
       flax==0.11.0 \
       jax[tpu]==0.4.35 \
       augmax \
       torchvision \
       timm \
       wandb

   # Register as a system-wide Jupyter Kernel
   /tartan/11785/envs/idl-shared-env/bin/python -m ipykernel install \
       --prefix=/tartan/11785 \
       --name=idl-tpu-kernel \
       --display-name="Python 3.11 (CMU 11-785 TPU Shared Env)"
   ```

### How Students Use the Shared Environment
* **Inside JupyterLab:** The kernel automatically appears in the top-right Kernel selector dropdown:
  * Select `Python 3.11 (CMU 11-785 TPU Shared Env)` $\rightarrow$ notebook instantly has access to all packages.
* **In Terminal / Scripts:**
  ```bash
  source /tartan/11785/envs/idl-shared-env/bin/activate
  python train.py
  ```
* **Developer Experience Benefit:** When staff adds a package (e.g., `timm` or `scikit-image`), all students receive it **instantaneously without restarting their servers or rebuilding Docker images**.

---

## 4. Spawner Profiles & Student Capstone Grant Tiers

To eliminate accidental cloud cost leakage while providing maximum flexibility:

| Profile | Target Users | vCPU | RAM | Disk | Hardware | Visibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Course Default (CPU)** | All Enrolled Students | 2 | 16 GiB | 32 GiB | Spot CPU ($0.04/hr) | Visible to Everyone |
| **Tier 1: Capstone Project Grant** | Capstone Teams (`project-grant-tier1`) | 4 | 32 GiB | 100 GiB | Spot CPU | Visible to Grant Recipients |
| **Admin CPU Workstation** | `amorbar`, Staff | 16 | 64 GiB | 100 GiB | Dedicated CPU | Visible ONLY to Admins |
| **Pinned TPU v5e-1** | Staff, `tpu-access` | 8 | 32 GiB | 32 GiB | 1x TPU v5e ($1.35/hr) | Hidden from General Students |
| **Pinned TPU v5e-4 Mesh** | Staff, `tpu-access` | 16 | 64 GiB | 100 GiB | 4x TPU v5e ($5.40/hr) | Hidden from General Students |

### Programmatic Grant Management (`scripts/hub_admin.py`)
Instructors award student grants and create project directories with zero CLI commands:

```python
# In an Admin Workstation notebook:
import hub_admin

# 1. Award Tier 1 Capstone Grant (100 GiB disk, 4 vCPU, 32G RAM):
hub_admin.grant_tier("student123", tier=1)

# 2. Create structured team project folder:
hub_admin.create_project_directory("capstone-team-alpha")
```

Or via CLI:
```bash
python scripts/hub_admin.py grant student123 --tier 1
python scripts/hub_admin.py project capstone-team-alpha
```

---

## 5. Cost Analysis & Financial Safeguards

| Workload Mode | Hourly Rate | Idle Risk | Safeguard Mechanism |
| :--- | :--- | :--- | :--- |
| **Student CPU Notebook** | ~$0.04 / hr | Low | Sits on Spot VMs; 1-hour idle cull (`timeout: 3600`). |
| **Tier 1 Capstone CPU** | ~$0.08 / hr | Low | Spot VMs; awarded strictly via `project-grant-tier1`. |
| **Admin CPU Workstation** | ~$0.38 / hr | Low | Restricted to `admin_users` (`amorbar`, `instructor`). |
| **Pinned TPU v5e-1** | $1.35 / hr | High ($32.40/day) | Strict 30-minute auto-culling (`cull_timeout: 1800`), hidden from students. |
| **Pinned TPU v5e-4 Podslice** | $5.40 / hr | Very High ($129.60/day) | Multi-chip mesh restricted to staff; 30-minute auto-cull. |
| **Queued TPU Jobs (Kueue)** | $1.35 / hr | Zero | **Pay-per-second execution.** The chip is allocated only while the job runs and freed instantly upon completion. |

---

## 6. How the Dual Mount is Configured in `jupyterhub-values.yaml`

```yaml
singleuser:
  extraVolumes:
    - name: course-tartan
      csi:
        driver: gcsfuse.csi.storage.gke.io
        volumeAttributes:
          bucketName: __OCEAN_BUCKET__
          mountOptions: "implicit-dirs,file-cache:max-size-mb:50000,metadata-cache:stat-cache-max-size-mb:2048,metadata-cache:ttl-secs:86400,ro"
  extraVolumeMounts:
    - name: course-tartan
      mountPath: /tartan/11785
    - name: course-tartan
      mountPath: /ocean
```
