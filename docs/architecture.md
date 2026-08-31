# 🏛️ Architecture: Ephemeral TPU Sharing

This document details the architectural design of the `shared-tpu-notebooks` environment as of the 2026 GKE Autopilot standards.

## The Core Problem

In standard Cloud environments, a Jupyter notebook running on an accelerator node permanently holds that accelerator for as long as the session is active. A Cloud TPU v5e (`ct5lp-hightpu-1t`) costs roughly $1.35/hour. If 300 students keep a tab open to read assignments and occasionally run code, it would cost **$405/hour**, which is prohibitively expensive for a course budget.

## The Solution: Decoupled Job Execution

Instead of providing a TPU to the notebook directly, this architecture decouples the notebook from the hardware.

1. **CPU Notebooks:** Students are given a low-cost Spot CPU notebook (2 vCPU, 8 GiB).
2. **Kueue Scheduling:** Code meant for the TPU is submitted as an ephemeral Kubernetes `Job` via `submit_tpu.py`.
3. **Queue Execution:** `Kueue` (the admission controller) queues these jobs against a shared pool of TPU resources (e.g., 32 or 64 chips).
4. **Scale-to-Zero:** GKE Autopilot provisions TPU nodes from the Dynamic Workload Scheduler (DWS) Flex Pool only when jobs exist, and destroys them after.

---

## 🏗️ System Diagram

The following Mermaid diagram illustrates the data and execution flow when a student submits a TPU job.

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant Hub as JupyterHub (CPU Pod)
    participant Kueue as Kueue Admission Controller
    participant GKE as GKE Autopilot & DWS
    participant TPU as TPU v5e Node (ct5lp)

    Student->>Hub: Run cell (submit_tpu.run)
    Hub->>Kueue: Submit V1Job (tpu-studentname)
    Note right of Kueue: Suspend=True<br/>Queued in 'tpu' Cohort
    Kueue->>GKE: Request Node (if pool empty/busy)
    GKE->>TPU: Provision Node (FLEX_START)
    Kueue->>Hub: Unsuspend Job (Admitted)
    Hub->>TPU: Schedule Pod
    TPU-->>Hub: Stream Execution Logs
    Hub-->>Student: Display Output in Cell
    Hub->>Kueue: Delete Job (TTL)
```

## Security Posture (CIS & NIST Guidelines)

Following **NIST SP 800-190** (Application Container Security Guide) and the **CIS GKE Benchmark v1.5.0** (2026):

- **Least Privilege (RBAC):** Students are mapped to a `ServiceAccount` (`student`) restricted to creating/reading Jobs *only* in the `cmu-idl` namespace. They cannot list other students' pods.
- **Identity-Aware Proxy (IAP):** Replaces basic auth. All traffic is zero-trust, authenticated by Google SSO via the GFE (Google Front End).
- **Network Isolation:** `jupyterhub-values.yaml` enforces egress NetworkPolicies. Students can only communicate with the Kubernetes API Server Endpoint (`443`). Egress to `169.254.169.254` (metadata) and internal pod ranges is blocked.
- **Container Sandboxing:** Notebook pods utilize **gVisor** (`runtimeClassName: gvisor`) to strictly isolate untrusted python code from the GKE host kernel. TPU jobs bypass this solely because hardware drivers require host kernel access, but those jobs run transiently as non-root users.

## Queuing Architecture (Multi-Flavor)

```mermaid
graph TD
    A[ClusterQueue: tpu] --> B[ResourceFlavor: v5e-flex]
    A --> C[ResourceFlavor: v5e-ondemand]
    B -- Node Label --> D[cloud.google.com/gke-flex-start: true]
    C -- Node Label --> E[No DWS Label]
    
    style B fill:#d4edda,stroke:#28a745,color:#000
    style C fill:#fff3cd,stroke:#ffc107,color:#000
```

Kueue is configured to attempt `FLEX_START` (highly available but zone-specific) first. If the flex pool is exhausted or unsupported in the zone, it falls back to standard on-demand capacity, ensuring seamless class execution.
