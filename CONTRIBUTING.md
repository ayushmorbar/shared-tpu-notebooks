# 🤝 Contributing to shared-tpu-notebooks

Thank you for your interest in contributing to **shared-tpu-notebooks**! This repository provides the infrastructure, orchestration, and client tools that enable hundreds of students to share Cloud TPUs from JupyterLab notebooks via Google Kubernetes Engine (GKE Autopilot) and Kueue.

This project is actively maintained and managed by **Ayush Morbar** ([@ayushmorbar](https://github.com/ayushmorbar)).

---

## 1. Architectural Philosophy & Design Principles

When modifying existing components or adding new coursework, adhere to these fundamental principles:

1. **Decoupled Architecture:**
   - **Never** attach dedicated TPUs directly to interactive notebook pods. Notebooks run on cost-effective CPU Spot instances.
   - Heavy computation leaves the notebook as a Kubernetes Batch Job queued through Kueue.
2. **Assignment Agnosticism (Zero Hardcoding):**
   - The platform is **not confined to a single homework**. It is designed to host any assignment (HW1, HW2, HW3, HW4, capstone projects, custom research).
   - All dataset paths, checkpoint locations, submission targets, and hyperparameters must be customizable via environment variables and function parameters.
3. **Student Privacy & Credential Isolation:**
   - Personal credentials (WandB, Kaggle, Hugging Face, custom keys) belong strictly in each student's persistent volume (`/home/jovyan/.env`).
   - Cleartext keys must **never** be logged to stdout/stderr, committed to Git, or shared across students.
4. **Beginner-Friendly Developer Experience (DX):**
   - Code must be clean, modular, and extensively documented with inline comments and Google-style docstrings.
   - Avoid cryptic abstractions. A student or new TA should immediately understand how each component works.

---

## 2. How to Add a New Homework Assignment

Adding a new assignment (e.g., `hw3_speech`) is straightforward and does not require modifying core infrastructure:

### Step 1: Stage the Pre-Processed Dataset in Google Cloud Storage
Stage your unified `.npy` image caches, audio features, or tokenized datasets in your course GCS bucket:
```bash
gcloud storage cp -r /path/to/hw3_cache gs://course-data-bucket/hw3/cache
```

### Step 2: Create the Standalone Training Script
Create `notebooks/hw3_train.py` following the established patterns:
- Ingest data from `os.environ.get("DATA_URI")` to local NVMe scratch (`/tmp`).
- For TPU nodes, load into physical RAM (`mmap_mode=None`) for zero-disk-I/O training.
- Use Flax NNX 0.11+ syntax (`optimizer.update(model, grads)`).
- Manage checkpoints via Orbax `CheckpointManager` with preservation policies.
- Generate predictions and save to `os.environ.get("SUBMISSION_URI")`.
- Call `prove_secret()` to safely verify any student credentials.

### Step 3: Provide the Student Notebook
Create `notebooks/F26_HW3_JAX_TPU_Hub.ipynb`:
- Enforce memory-mapped reads (`mmap_mode='r'`) for local CPU exploration so RAM stays $< 1.5\text{ GiB}$.
- Submit to Cloud TPU via `submit_tpu.train()`:
  ```python
  import submit_tpu

  submit_tpu.train(
      "hw3_train.py",
      assignment="hw3",
      epochs=30,
      download_submission=True,
  )
  ```
- Package final submission zip for Gradescope.

---

## 3. Engineering & Code Standards

### Bash Scripts (`scripts/`)
- **Strict Mode:** Every bash script must start with:
  ```bash
  set -euo pipefail
  ```
- **Source Common Library:** Always source `scripts/common.sh`:
  ```bash
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source "${SCRIPT_DIR}/common.sh"
  ```
- **Monotonic Script Numbering:** Scripts are numbered by lifecycle phase (e.g., `00_preflight.sh`, `01_build_image.sh`, `02_create_cluster.sh`, `03_deploy_hub.sh`, `10_cleanup_pvcs.sh`, `11_expand_pvcs.sh`, `12_create_secrets.sh`, `99_teardown.sh`).
- **Standardized Logging:** Use `log_header`, `log_info`, `log_success`, `log_warn`, and `log_error` instead of raw `echo`.

### Python Code (`notebooks/`, `scripts/`)
- **Type Annotations:** All functions must have comprehensive type hints (`from __future__ import annotations`).
- **Non-Root Execution:** Never configure containers with `run_as_user: 0` or request Linux capabilities. GKE Autopilot admission controllers will reject privileged manifests.
- **Flax NNX & Orbax Standards:** Adhere to modern October 2026 APIs.

---

## 4. Development Workflow

1. **Fork and Branch:** Create a feature or fix branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Local Validation:**
   - Verify Python syntax: `python -m py_compile notebooks/*.py`
   - Validate notebook JSON: parse with `json.load`
   - Test bash scripts with `bash -n scripts/*.sh`
3. **Commit Messages:** Follow Conventional Commits:
   - `feat: add assignment parameter to submit_tpu.train`
   - `fix: resolve namespace default in secrets creation`
   - `docs: update student onboarding guide`
4. **Pull Requests:** Open a PR against `main` on [ayushmorbar/shared-tpu-notebooks](https://github.com/ayushmorbar/shared-tpu-notebooks). Ensure `CHANGELOG.md` is updated.

---

## 5. Maintainer

- **Ayush Morbar** — Project Lead & Infrastructure Maintainer ([@ayushmorbar](https://github.com/ayushmorbar))
