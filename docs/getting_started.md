# 🚀 Getting Started Guide

Welcome to the `shared-tpu-notebooks` environment! This guide is designed for newcomers to help you spin up the entire cluster and deploy JupyterHub smoothly.

## Prerequisites Checklist

Before running any `make` commands, ensure you have:
1. A **Google Cloud Project** with an active billing account.
2. The **Google Cloud CLI** (`gcloud`) installed and updated to the latest 2026 version.
3. **Kubernetes tools**: `kubectl` and `helm` installed.
4. **Authentication** established:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

## 1. Configure Your Environment

Copy the example configuration file:
```bash
cp config.env.example config.env
```

Open `config.env` and populate the fields. Crucially, set `PROJECT` to your GCP Project ID and `REGION` to a TPU-enabled region (e.g., `us-west4`).

## 2. Verify Quotas (Free)

Before provisioning resources, verify that your region has the required quotas for v5e TPUs (`TPU_LITE_PODSLICE_V5`).

```bash
make preflight PROJECT=my-project-id
```

## 3. The 5-Step Deployment

Run these steps in order. Each script has detailed error handling and will abort safely if an issue occurs.

### Step A: Build the Images
Three images: the CPU notebook (JAX, PyTorch, and the HW libraries for local unit tests), the TPU job image students' scripts run in, and the tpu-submit service.
```bash
make image
```

### Step B: Provision the GKE Cluster
Creates a GKE Autopilot cluster, configures Kueue, and sets up StorageClasses. (Takes ~12 minutes).
```bash
make cluster
```

### Step C: Create the Buckets
One bucket for job files (snapshots, logs, checkpoints) that only the tpu-submit service can use, and one for datasets that only TPU jobs can read. Then upload each assignment's dataset once:
```bash
make storage
bash scripts/12_setup_storage.sh upload hw3p2 /path/to/hw3p2_data
```
Paste the `DATASETS=` and `DEFAULT_DATASET=` lines it prints into `config.env`.

### Step D: Deploy the Submission Service and JupyterHub
Deploys tpu-submit, then installs the Zero-to-JupyterHub Helm chart with our custom security profiles. Re-run it after changing any `config.env` value in the TPU Job Submission section.
```bash
make hub
```

### Step E: Secure with IAP (HTTPS)
Replaces insecure port-forwarding with Google SSO Identity-Aware Proxy.
```bash
make iap
```

> [!CAUTION]
> **OAuth Consent Screen**
> For IAP to work, you *must* configure an Internal OAuth consent screen in your Google Cloud Console under `APIs & Services > OAuth consent screen`.

## 4. Run a Smoke Test

To verify that the cluster can actually schedule a job to a TPU, run the smoke test. This submits a real JAX script via a Kubernetes Job.

```bash
make smoke
```

If it prints `devices: [TpuDevice(id=0, ...)]`, your TPU pool is fully operational!

Then test the path students use, through the submission service. Sign in to the hub, open `https://<your hub>/hub/token`, create a token, and run:

```bash
HUB_TOKEN=<token> make smoke-submit
HUB_TOKEN=<token> SMOKE_DATASET=hw3p2 make smoke-submit   # also checks the dataset unpacks
```

It submits a one-file script with the same `submit_tpu.py` students use, watches it, and downloads its `expts/` folder.

---

## Modifying the Flow

If you are a course TA or Instructor looking to modify how jobs run or how the Hub spawns, please see the [Developer Guide](developer_guide.md).
