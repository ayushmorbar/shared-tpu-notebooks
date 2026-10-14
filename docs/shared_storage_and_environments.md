# CMU 11-785: Shared Storage ("Ocean") & Multi-User Environments Architecture

**Platform Maintainer:** Ayush Morbar  
**Target Environment:** Google Kubernetes Engine (GKE Autopilot) & Cloud TPU v5e  
**Course Reference:** Pittsburgh Supercomputing Center (PSC) Bridges-2 Storage Architecture  

---

## 1. Executive Summary & HPC Mapping (Bridges-2 vs. GKE)

On supercomputing clusters like **PSC Bridges-2**, courses rely on a two-tier storage and software model:
* **Ocean (`/ocean/projects/...`)**: Central high-capacity parallel storage shared by all users in a grant. Course datasets, assignment starter files, and shared software environments reside here. Read-only for students, read-write for course staff.
* **Jet / Scratch (`$SCRATCH` / `/local`)**: Ultra-fast node-local scratch storage for job execution and caching.
* **Home (`/jet/home/...`)**: Private persistent user home directory.
* **Shared Software**: `module load anaconda3` allows any user to load shared Conda/virtual environments maintained by instructors.

We map this architecture directly onto **Google Cloud Platform (GCP) and GKE Autopilot**:

```
+---------------------------------------------------------------------------------------------------+
|                                 CMU 11-785 TPU HUB STORAGE TOPOLOGY                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ 1. Central "Ocean" Storage ] -> Mount: /ocean (Read-Only for Students, RW for Staff)           |
|  ├── /ocean/datasets/          <- Pre-processed .npy caches (HW1, HW2, HW3, HW4)                  |
|  ├── /ocean/assignments/       <- Released starter notebooks and assignment scripts               |
|  └── /ocean/envs/              <- Shared Python & Conda environments (Admin-managed)              |
|                                                                                                   |
|  [ 2. Student Private Home ]   -> Mount: /home/jovyan (Read-Write, 32 GiB Standard-RWO PVC)       |
|  ├── /home/jovyan/.env         <- Student personal secrets (WandB, Kaggle, HF)                   |
|  ├── /home/jovyan/notebooks/   <- Student active coursework and modified code                     |
|  └── /home/jovyan/checkpoints/ <- Locally synced model checkpoints and submission.csv             |
|                                                                                                   |
|  [ 3. Local "Jet" Scratch ]    -> Mount: /tmp (Local NVMe SSD, emptyDir, Line-Rate IOPS)          |
|  └── High-throughput temporary cache for TPU jobs; bypasses network storage bottlenecks.          |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Shared Storage Implementation Options on GKE

### Option A: GKE Cloud Storage FUSE CSI Driver (Recommended)
* **Mechanism:** GKE Autopilot natively supports the `gcsfuse.csi.storage.gke.io` CSI driver. A single GCS bucket (e.g. `gs://hw2p2-cached-data` or `gs://cmu-idl-ocean`) is mounted into all student and staff pods at `/ocean`.
* **Cost:** Zero idle compute cost; you pay only standard Cloud Storage storage rates ($0.020/GB/month).
* **Permissions:** Configured via Google Cloud Workload Identity:
  * Student ServiceAccount (`student`): Granted `roles/storage.objectViewer` on the bucket (Read-Only).
  * Staff ServiceAccount (`staff`): Granted `roles/storage.objectAdmin` on the bucket (Read-Write).

### Option B: Google Cloud Filestore (NFS ReadWriteMany PVC)
* **Mechanism:** A managed NFS share (`standard` tier) mounted via Kubernetes `ReadWriteMany` PVC to `/ocean`.
* **Advantage:** Full POSIX compatibility, native hard links, and lower metadata latency for deeply nested Python virtual environments.

---

## 3. Shared Environments Architecture ("Module Load" for JupyterHub)

Instead of forcing students to run `pip install` or rebuilding heavy Docker container images every time a new library is needed, admins manage shared environments directly inside `/ocean/envs/`.

### How Admins Create & Update Packages
1. An instructor or admin (`amorbar`) launches the **Admin Workstation** notebook.
2. In the terminal, create or update the shared environment:
   ```bash
   # Create a shared virtual environment on the Ocean volume
   python3 -m venv /ocean/envs/idl-shared-env --system-site-packages

   # Install new course packages (instantly visible to all 300 students!)
   /ocean/envs/idl-shared-env/bin/pip install \
       flax==0.11.0 \
       jax[tpu]==0.4.35 \
       augmax \
       torchvision \
       timm \
       wandb

   # Register as a system-wide Jupyter Kernel
   /ocean/envs/idl-shared-env/bin/python -m ipykernel install \
       --prefix=/ocean/envs/idl-shared-env \
       --name=idl-shared-env \
       --display-name="CMU 11-785 Shared TPU Env (JAX/Flax/Torch)"
   ```

### How Students Use the Shared Environment
* **Inside JupyterLab:** The kernel automatically appears in the top-right Kernel selector dropdown:
  * Select `CMU 11-785 Shared TPU Env (JAX/Flax/Torch)` $\rightarrow$ notebook instantly has access to all packages.
* **In Terminal / Scripts:**
  ```bash
  source /ocean/envs/idl-shared-env/bin/activate
  python train.py
  ```
* **Developer Experience Benefit:** When staff adds a package (e.g., `timm` or `scikit-image`), all students receive it **instantaneously without restarting their servers or rebuilding Docker images**.

---

## 4. Admin Resource Profiles & Web Console Toggling

### Profile Separation & Visibility Rules
To eliminate accidental cloud cost leakage while providing maximum flexibility:

| Profile | Target Users | vCPU | RAM | Disk | Hardware | Visibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Course Default (CPU)** | All Students | 2 | 16 GiB | 32 GiB | Spot CPU | Visible to Everyone |
| **Admin CPU Workstation** | `amorbar`, Staff | 16 | 64 GiB | 100 GiB | Dedicated CPU | Visible ONLY to Admins |
| **Pinned TPU v5e-1** | Staff, `tpu-access` | 4 | 16 GiB | 32 GiB | 1x TPU v5e ($1.35/hr) | Hidden from Students |
| **Pinned TPU v5e-4 Mesh** | Staff, `tpu-access` | 16 | 64 GiB | 100 GiB | 4x TPU v5e ($5.40/hr) | Hidden from Students |

### Zero-CLI Web Toggle for Admin Users
Course administrators can grant interactive TPU or high-compute access to any specific student (e.g., for thesis research, debugging, or special projects) **directly through the JupyterHub web interface**:

1. Navigate to `https://<JUPYTERHUB_URL>/hub/admin`.
2. Under **Groups**, create or select the group `tpu-access`.
3. Add the target student's username and click **Save**.
4. The student restarts their server from `/hub/home` $\rightarrow$ the **Pinned TPU v5e** profile is immediately unlocked in their spawner menu!
5. When finished, removing the user from `tpu-access` locks the profile again. **Zero kubectl commands required.**

---

## 5. Cost Analysis & Financial Safeguards

| Workload Mode | Hourly Rate | Idle Risk | Safeguard Mechanism |
| :--- | :--- | :--- | :--- |
| **Student CPU Notebook** | ~$0.04 / hr | Low | Sits on Spot VMs; 1-hour idle cull (`timeout: 3600`). |
| **Admin CPU Workstation** | ~$0.38 / hr | Low | Restricted to `admin_users` (`amorbar`, `instructor`). |
| **Pinned TPU v5e-1** | $1.35 / hr | High ($32.40/day) | Strict 30-minute auto-culling (`cull_timeout: 1800`), hidden from students. |
| **Pinned TPU v5e-4 Podslice** | $5.40 / hr | Very High ($129.60/day) | Multi-chip mesh restricted to staff; 30-minute auto-cull. |
| **Queued TPU Jobs (Kueue)** | $1.35 / hr | Zero | **Pay-per-second execution.** The chip is allocated only while the job runs and freed instantly upon completion. |

---

## 6. How to Enable the `/ocean` Mount in `jupyterhub-values.yaml`

To mount the shared GCS bucket as `/ocean` in every singleuser pod:

```yaml
singleuser:
  extraVolumes:
    - name: course-ocean
      csi:
        driver: gcsfuse.csi.storage.gke.io
        volumeAttributes:
          bucketName: hw2p2-cached-data
          mountOptions: "implicit-dirs"
  extraVolumeMounts:
    - name: course-ocean
      mountPath: /ocean
```
