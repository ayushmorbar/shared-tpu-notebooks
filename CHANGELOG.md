# Changelog

All notable changes to the `shared-tpu-notebooks` platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 2026-10-12

### Added
- **Dedicated Training Path (`submit_tpu.train`)**:
  - Standalone training script mounting via Kubernetes ConfigMap (`{name}-src` -> `/src/entrypoint.py`).
  - Fast local NVMe scratch disk attachment (`emptyDir` on `/tmp`) for high-throughput dataset caching.
  - Live container log streaming (`read_namespaced_pod_log(follow=True)`) during multi-hour runs.
  - 30-second live queue heartbeats (`[Queue Update]`, `[Node Build]`, `[Starting]`) while waiting for Kueue admission and GKE Autopilot TPU node provisioning.
  - Job retention post-execution for TA and student inspection.
  - Automated post-execution userspace artifact delivery: downloads `submission.csv` to `/home/jovyan/submission.csv` and syncs model checkpoints to `/home/jovyan/model_checkpoints/`.
- **Per-Student Private Credential Isolation**:
  - Created [`notebooks/student.env.template`](notebooks/student.env.template) for personal credentials stored on each student's private 32 GiB Persistent Volume Claim (`/home/jovyan/.env`).
  - Dynamic discovery supporting Kaggle (`KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`, `KAGGLE_KEY`), Weights & Biases (`WANDB_API_KEY`), Hugging Face (`HF_TOKEN`), and arbitrary custom third-party keys (`extra_env`).
  - Zero key leakage guarantee: only key presence, length, and truncated SHA-256 prefixes (`sha256[:8]`) are logged.
- **Dual-Script Separation (Staff vs. Student)**:
  - Added [`notebooks/train_hw2p2.py`](notebooks/train_hw2p2.py): Willis Zhang's benchmark test harness proving GCS line-rate copy (18.5 GiB in 78–117s), in-RAM loading (`VmRSS ~24 GB`), secret verification, and Orbax resume in ~2 minutes with synthetic labels.
  - Added [`notebooks/hw2p2_train.py`](notebooks/hw2p2_train.py): Complete student homework training script implementing ResNet-34 (< 30M parameter budget, ~21.8M params), ElasticFace margin head ($s=40.0, m=0.30, \text{std}=0.05$), verification EER via `scipy.optimize.brentq`, ROC AUC, Top-1/Top-5 accuracy, test set inference, and `submission.csv` generation.
- **October 2026 Tech Stack Alignment**:
  - Flax NNX 0.11 syntax: explicit model passing in optimizer updates (`optimizer.update(model, grads)`).
  - Orbax Checkpoint modern preservation policy: `ocp.CheckpointManagerOptions(preserve_max_to_keep=1, max_to_keep=2, enable_async_checkpointing=True)`.
- **Documentation & Operational Manuals**:
  - Added Section 12 on Long Training Runs, real HW2P2 benchmarks, and 3 critical operational caveats to [`README.md`](README.md).
  - Added beginner-friendly [`docs/student_guide.md`](docs/student_guide.md) explaining CPU notebook vs TPU pod, credential setup, and Gradescope submission.
  - Added [`SECURITY.md`](SECURITY.md) covering student privacy, Workload Identity, sandboxing, and vulnerability reporting.
- **Infrastructure Automation**:
  - Added [`scripts/12_create_secrets.sh`](scripts/12_create_secrets.sh) sourcing `common.sh` and targeting namespace `cmu-idl` for optional course baseline secrets.
  - Added `make secrets` target to [`Makefile`](Makefile).

### Changed
- Standardized default ServiceAccount in `submit_tpu.train()` to `student` with parameter override (`service_account_name`) and `TPU_SERVICE_ACCOUNT` environment fallback.
- Standardized environment configuration template strictly on [`config.env.example`](config.env.example); removed personal student keys from course-level template to enforce privacy.
- Updated all student submission references, validation checks, and zip packaging cells in [`notebooks/F26_HW2P2_JAX_TPU_Hub.ipynb`](notebooks/F26_HW2P2_JAX_TPU_Hub.ipynb) to target **Gradescope** (`HW2P2_final_submission.zip`).

### Removed
- Removed duplicate `config.example.env` in favor of standard `config.env.example`.
- Removed duplicate `scripts/11_create_secrets.sh` to prevent conflict with `scripts/11_expand_pvcs.sh`.

---

## [1.2.0] - 2026-09-13

### Added
- CMU 11-785 Tartan branding for JupyterHub UI.
- JupyterHub single-user spawner timeout extension to 600s (`c.KubeSpawner.start_timeout = 600`).
- Online volume expansion script (`scripts/11_expand_pvcs.sh`) for resizing student persistent disks to 32 GiB.
- Storage quota tuning in `scripts/02_create_cluster.sh`.

### Fixed
- Fixed storage conflict between JupyterHub internal database (`hub.db.pvc`) and student persistent volumes.
- Fixed namespace inconsistencies across automation scripts, standardizing on `cmu-idl`.

---

## [1.1.0] - 2026-08-30

### Added
- Identity-Aware Proxy (IAP) automation via `scripts/08_setup_iap.sh`.
- Warm TPU node placeholder deployment (`scripts/07_warm_pool.sh`) using PriorityClass -10.
- Live DWS Flex capacity probe (`scripts/01_spray_v5e.sh`).
- Preflight quota checker (`scripts/00_preflight.sh`).

---

## [1.0.0] - 2026-08-25

### Initial Release
- Core decoupled JupyterLab on Spot CPU + Cloud TPU v5e execution substrate via Kueue.
- Baseline `submit_tpu.run()` client for executing inline strings on Cloud TPU v5e chips.
- Sample hello world assignment (`notebooks/hw0_tpu_hello.ipynb`).
