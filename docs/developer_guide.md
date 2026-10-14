# 🛠️ Developer & Maintainer Guide

This guide is for teaching assistants, course maintainers, and system administrators who operate, customize, and extend the **CMU 11-785 Cloud TPU Hub** platform.

---

## 1. Directory Structure & Key Components

```
shared-tpu-notebooks/
├── Makefile                       # Entrypoint for infrastructure automation and admin CLI
├── config.env.example             # Template for GCP project, region, cluster, and Tartan bucket
├── docker/
│   ├── Dockerfile                 # Custom Jupyter course container (JAX, PyTorch, K8s client)
│   └── requirements.txt           # Base Python packages bundled into the image
├── docs/                          # Comprehensive architectural and operational documentation
├── k8s/
│   ├── jupyterhub-values.yaml     # Production Helm values (RBAC, IAP, Tartan mounts, profiles)
│   ├── jupyterhub-values-dev.yaml # Dev overlay (disables CSRF for local port-forwarding)
│   ├── kueue-tpu-queues.yaml      # Multi-flavor Kueue queues (Flex and On-Demand TPU v5e)
│   ├── priority-classes.yaml      # Kubernetes priority tiers (jupyterhub-core > student-notebook > job)
│   ├── student-tpu-job.yaml       # Template Job manifest submitted by submit_tpu.py
│   └── tpu-warm-pool.yaml         # Placeholder deployment & auto-scale-down cron
├── notebooks/
│   ├── submit_tpu.py              # User-facing client library for queuing batch TPU jobs
│   ├── hw2p2_train.py             # Complete ResNet-34 ElasticFace training script for TPU v5e
│   ├── train_hw2p2.py             # 2-minute infrastructure verification & smoke script
│   ├── F26_HW2P2_JAX_TPU_Hub.ipynb# Production homework notebook with Tartan paths & submit_tpu
│   └── student.env.template       # Isolated student credentials template (.env)
└── scripts/
    ├── common.sh                  # Shared logging, environment discovery, and k8s context
    ├── hub_admin.py               # Administrative CLI and interactive widget for storage & grants
    ├── 00_preflight.sh            # Regional accelerator quota verification
    ├── 01_build_image.sh          # Container build & push to Artifact Registry
    ├── 02_create_cluster.sh       # GKE Autopilot provisioning & Kueue setup
    ├── 03_deploy_hub.sh           # JupyterHub deployment with Tartan GCS FUSE mounts
    ├── 08_setup_iap.sh            # HTTPS Ingress & Google Identity-Aware Proxy setup
    └── 99_teardown.sh             # Tear down cluster and release all cloud resources
```

---

## 2. Administrative Grant & Storage Management (`hub_admin.py`)

Teaching assistants and maintainers manage student allocations directly via `scripts/hub_admin.py` or the `Makefile` shortcuts:

### CLI Usage:
```bash
# 1. Discover all student PVCs and active storage capacities
make admin-list
# Or directly:
python scripts/hub_admin.py list

# 2. Award a project grant tier to a student
# (Expands PVC online and applies vCPU/RAM overrides on next spawn)
make admin-grant USER=student123 TIER=1
# Or directly:
python scripts/hub_admin.py grant student123 --tier 1

# 3. Scaffold a structured shared project directory in Tartan storage
make admin-project TEAM=capstone-alpha
# Or directly:
python scripts/hub_admin.py project capstone-alpha
```

### Interactive JupyterLab Widget:
Maintainers running an Admin notebook can render the interactive point-and-click management widget:
```python
import hub_admin
hub_admin.render_admin_widget()
```
This renders dropdowns to select any student, pick their grant tier (or custom storage/vCPU/RAM limits), and apply updates immediately without touching terminal commands.

---

## 3. Maintaining the Tartan Shared Environment (Zero Rebuilds)

To add or update Python packages for the entire course **without rebuilding or redeploying the Docker image**:

1. Log into the **Admin / Power User CPU Workstation** or an administrative pod with write access to `gs://cmu-idl-ocean`.
2. Activate or target the shared environment directory:
   ```bash
   conda run -p /tartan/11785/envs/idl-shared-env pip install --upgrade <package_name>
   ```
3. Because `jupyterhub-values.yaml` injects:
   ```yaml
   JUPYTER_PATH: "/tartan/11785/share/jupyter:/ocean/share/jupyter:/opt/conda/share/jupyter"
   PYTHONPATH: "/opt/course:/tartan/11785/envs/idl-shared-env/lib/python3.11/site-packages"
   ```
   All 300 student notebook pods discover the new package instantly upon import or kernel restart.

---

## 4. Modifying the Job Submission Client (`submit_tpu.py`)

When modifying `submit_tpu.train()` or `submit_tpu.run()`:

1. **Type Hints & Docstrings**: Maintain comprehensive type annotations and Google-style docstrings.
2. **Billing & Accounting Labels**: Ensure every submitted Job carries the standard CMU billing labels:
   ```python
   metadata.labels = {
       "cmu-course": "11-785",
       "cmu-student": student,
       "cmu-assignment": assignment_clean,
       "cmu-grant-tier": grant_tier,
       "kueue.x-k8s.io/queue-name": queue,
   }
   ```
   These labels enable automated cost auditing and student capstone budget tracking in BigQuery.
3. **Security Constraints**: **Never** add privileged flags (`privileged: true`, `NET_ADMIN`) or root user contexts. Autopilot admission controllers will reject the submission immediately.

---

## 5. Spawner Profile Management & Group Filtering

JupyterHub spawner profiles are defined in `k8s/jupyterhub-values.yaml`:

- **Visibility Filtering (`filter_profile_list`)**: Exposes profiles conditionally:
  - Profiles without `allowed_groups` are visible to all students.
  - Profiles with `allowed_groups` (e.g., `project-grant-tier1`, `tpu-access`) are only shown to users in those groups or admins.
- **Adding Students to Groups**:
  In JupyterHub, instructors can assign groups via the Admin panel (`/hub/admin`) or configure group membership through Google Workspace / IAM directory sync.

---

## 6. Upgrading the JupyterHub Helm Chart

To upgrade the underlying JupyterHub version:
1. Update `CHART_VERSION` in `scripts/03_deploy_hub.sh`.
2. Test the upgrade with `--dry-run --debug` first.
3. Review chart release notes for breaking changes in `singleuser.networkPolicy` and `hub.config`.
4. Apply using `make hub`.
