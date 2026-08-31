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

## 3. The 4-Step Deployment

Run these steps in order. Each script has detailed error handling and will abort safely if an issue occurs.

### Step A: Build the Custom Docker Image
The course image contains JAX, PyTorch, and the Kubernetes client.
```bash
make image
```

### Step B: Provision the GKE Cluster
Creates a GKE Autopilot cluster, configures Kueue, and sets up StorageClasses. (Takes ~12 minutes).
```bash
make cluster
```

### Step C: Deploy JupyterHub
Installs the Zero-to-JupyterHub Helm chart with our custom security profiles.
```bash
make hub
```

### Step D: Secure with IAP (HTTPS)
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

---

## Modifying the Flow

If you are a course TA or Instructor looking to modify how jobs run or how the Hub spawns, please see the [Developer Guide](developer_guide.md).
