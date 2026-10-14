# 🚀 Getting Started Guide: CMU 11-785 Tartan TPU Hub

Welcome to the **CMU 11-785 Cloud TPU Hub** deployment guide. This document walks you through spinning up the complete GKE Autopilot cluster, deploying JupyterHub with Tartan shared storage, and configuring Identity-Aware Proxy (IAP) for Google SSO.

---

## 1. Prerequisites Checklist

Before running deployment scripts, verify you have:
1. A **Google Cloud Project** with active billing enabled.
2. The **Google Cloud CLI** (`gcloud`) installed and authenticated:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
3. **Kubernetes tools**: `kubectl` and `helm` installed on your path.
4. An **Internal OAuth Consent Screen** created in Google Cloud Console (`APIs & Services > OAuth consent screen`) if using IAP authentication.

---

## 2. Configuration Setup

Copy the example configuration into your local environment:
```bash
cp config.env.example config.env
```

Open `config.env` and populate the fields:
- `PROJECT`: Your GCP project ID (e.g. `deep-learning-with-tpu-util`).
- `REGION`: TPU-enabled region (e.g. `us-west4`).
- `OCEAN_BUCKET`: Course shared GCS bucket name (e.g. `cmu-idl-ocean`).
- `ADMIN_USERS`: Space-separated list of TA/instructor emails (e.g. `user:amorbar@andrew.cmu.edu`).

---

## 3. Verify Quotas (Free)

Before creating resources, run the quota preflight check to ensure your region has available Cloud TPU v5e quota:
```bash
make preflight PROJECT=my-project-id
```

---

## 4. The 4-Step Deployment Flow

Run each step in order. All scripts are idempotent and include comprehensive error logging:

### Step 1: Build the Course Docker Image
Builds and pushes the base JupyterHub environment image (JAX, PyTorch, Kubernetes client) to Artifact Registry:
```bash
make image
```

### Step 2: Provision the GKE Autopilot Cluster
Creates the Autopilot cluster, configures Kueue 0.8+, sets up PriorityClasses, and registers the `standard-rwo-retain` StorageClass (~12 minutes):
```bash
make cluster
```

### Step 3: Deploy JupyterHub & Tartan Storage Mounts
Installs JupyterHub via Helm with the Tartan 3-Tier storage hierarchy (GCS FUSE dual-mount at `/tartan/11785` and `/ocean`), student RBAC, and dynamic profile filters:
```bash
make hub
```

### Step 4: Secure with Google Identity-Aware Proxy (IAP)
Creates an external HTTPS Load Balancer with Google-managed SSL certificates and zero-trust SSO authentication:
```bash
make iap
```

---

## 5. Verify Operation with a Smoke Test

To verify that the cluster can schedule and run jobs on a real Cloud TPU v5e chip, run the smoke test:
```bash
make smoke
```
This submits a real JAX verification job via Kueue. If it prints:
```text
devices: [TpuDevice(id=0, process_index=0, slice_index=0)]
```
Your Cloud TPU v5e cluster is fully operational!

---

## 6. Administrative Management & Student Grants

Once deployed, instructors manage student storage and grants using `scripts/hub_admin.py` or the `Makefile` shortcuts:

```bash
# View all student PVCs and storage allocations
make admin-list

# Award a project grant tier (e.g. Tier 1: 100 GiB PVC, 4 vCPU, 32 GiB RAM)
make admin-grant USER=student123 TIER=1

# Scaffold a shared project workspace in Tartan storage
make admin-project TEAM=capstone-alpha
```

For advanced customization and shared environment updates, consult the [Developer Guide](developer_guide.md) and [Architecture Guide](architecture.md).
