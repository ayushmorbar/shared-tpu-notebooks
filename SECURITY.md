# Security Policy: CMU 11-785 Cloud TPU Hub

Security and privacy are core architectural priorities of the `shared-tpu-notebooks` platform. This document outlines the security architecture, student credential isolation policies, runtime guardrails, and procedures for reporting vulnerabilities.

---

## 1. Student Credential Isolation & Privacy Policy

In large university courses, hundreds of students share a cluster while using personal third-party accounts (Weights & Biases, Kaggle, Hugging Face).

### 🔒 Architectural Guarantees:
1. **Zero Shared Personal Keys:**
   - Student credentials are **never placed in shared configuration files** (`config.env`, `config.env.example`) or cluster-wide Secrets.
   - Each student stores their personal keys in their own persistent userspace: `/home/jovyan/.env` or `~/.kaggle/kaggle.json`.
   - The student's home volume is backed by a dedicated Kubernetes `PersistentVolumeClaim` (ReadWriteOnce) attached only to their personal notebook pod. Classmates cannot read other students' volumes.
2. **Masked Verification Guarantee (No Log Leakage):**
   - Both the client dispatcher (`submit_tpu.py`) and training scripts (`hw2p2_train.py`, `train_hw2p2.py`) never print cleartext credential values to standard output or standard error.
   - Credentials are verified using a length check and the first 8 characters of their SHA-256 digest:
     ```
     [Secret] WANDB_API_KEY: present, len=40, sha256[:8]=9f8a2b1c
     ```
   - Cleartext keys never enter Kubernetes Job logs, Cloud Logging, or notebook outputs.
3. **Transient Injection:**
   - Personal credentials are forwarded directly as environment variables only to that student's specific TPU Job pod (`tpu-train-<student>`), scoped strictly to their namespace (`cmu-idl`).

---

## 2. Infrastructure & Kubernetes Security Guardrails

### 🛡️ Container Sandboxing & Least Privilege:
Every student notebook pod and Cloud TPU execution pod enforces strict Kubernetes security contexts:
- **Non-Root Execution:** All containers run as unprivileged user `UID 1000` / `GID 1000` (`run_as_non_root: true`).
- **Privilege Escalation Blocked:** `allow_privilege_escalation: false` is enforced.
- **Dropped Linux Capabilities:** All Linux capabilities are dropped (`capabilities: { drop: ["ALL"] }`).
- **GKE Autopilot Admission:** GKE Autopilot's admission controller automatically rejects any pod attempting to request privileged access or host-level mounting.

### 🔑 Workload Identity Federation (Keyless Google Cloud Access):
- **No Static Service Account JSON Keys:** Neither notebook pods nor TPU pods store or download static GCP service account keys.
- **Workload Identity:** Pods run as the Kubernetes ServiceAccount `student`, which binds to the Google Service Account via IAM Workload Identity Federation.
- Access to Google Cloud Storage (`gs://hw2p2-cached-data/`) is granted via short-lived, automatically rotated OAuth2 tokens issued by the Kubernetes metadata server.

### 🌐 Perimeter Defense & Authentication:
- **Cloud Identity-Aware Proxy (IAP):** JupyterHub is placed behind Google Cloud IAP with managed TLS certificates. Only authenticated users in the university Google Workspace domain (`@andrew.cmu.edu` or authorized roster group) can access the proxy.
- **Network Policies:** Traffic between student pods is restricted by Kubernetes NetworkPolicies. Student pods cannot access cluster infrastructure services or other students' notebook pods.

---

## 3. Reporting a Security Vulnerability

If you discover a security vulnerability or credential leak within this codebase or the CMU 11-785 Cloud TPU Hub infrastructure:

1. **Do NOT open a public GitHub issue.**
2. Email the course infrastructure administrators and instructors immediately:
   - Ayush Morbar: `amorbar@andrew.cmu.edu`
   - Course Staff: `deeplearning-tas@cs.cmu.edu`
3. Include in your report:
   - A detailed description of the vulnerability.
   - Steps to reproduce or proof-of-concept manifests.
   - Impact assessment (e.g., credential exposure, unauthorized resource consumption).
4. The course infrastructure team will acknowledge receipt within 24 hours and coordinate a private patch and security advisory.
