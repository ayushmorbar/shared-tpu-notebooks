"""JupyterHub Admin Storage & Compute Resource Management Utility.

CMU 11-785: Introduction to Deep Learning
Platform Maintainer: Ayush Morbar

This module provides administrative functions for managing student persistent volumes (PVCs)
and custom compute/accelerator resources directly from a Jupyter notebook or administrative
CLI without having to manually run repetitive kubectl commands.

Features:
1. Inspect student PVC capacity and volume status across the cluster.
2. Dynamically expand student PVC storage online (e.g. 32Gi -> 64Gi -> 100Gi) using the GKE CSI driver.
3. Interactive JupyterLab widget (ipywidgets) for one-click storage resizing.
4. Role-based resource profile helper for instructor and power-user configurations.
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException
    HAS_K8S = True
except ImportError:
    HAS_K8S = False


def _init_k8s() -> tuple[client.CoreV1Api, str]:
    """Initialize Kubernetes API client and detect namespace."""
    if not HAS_K8S:
        raise RuntimeError("kubernetes Python package is required. Run: pip install kubernetes")
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    core = client.CoreV1Api()
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(ns_path):
        with open(ns_path, encoding="utf-8") as f:
            ns = f.read().strip()
    else:
        ns = os.environ.get("COURSE_NAMESPACE", "cmu-idl")
    return core, ns


def list_student_storage(namespace: str | None = None) -> list[dict[str, Any]]:
    """List all student PersistentVolumeClaims in the cluster.

    Args:
        namespace (str, optional): Target namespace. Defaults to course namespace (cmu-idl).

    Returns:
        list[dict]: List of student PVC records with user, name, size, phase, and storage class.
    """
    core, default_ns = _init_k8s()
    ns = namespace or default_ns

    pvcs = core.list_namespaced_persistent_volume_claim(namespace=ns)
    records = []
    for pvc in pvcs.items:
        name = pvc.metadata.name
        # Match JupyterHub singleuser PVC naming pattern: claim-{user} or claim-jovyan-{user}
        if name.startswith("claim-"):
            username = name.replace("claim-jovyan-", "").replace("claim-", "")
            capacity = pvc.spec.resources.requests.get("storage", "unknown")
            actual_capacity = pvc.status.capacity.get("storage", capacity) if pvc.status.capacity else capacity
            records.append({
                "username": username,
                "pvc_name": name,
                "requested_storage": capacity,
                "actual_storage": actual_capacity,
                "phase": pvc.status.phase,
                "storage_class": pvc.spec.storage_class_name,
            })
    return sorted(records, key=lambda x: x["username"])


def expand_student_storage(username: str, new_size: str, namespace: str | None = None) -> bool:
    """Dynamically expand a student's persistent volume claim online.

    Kubernetes and GKE CSI driver support online volume expansion without pod termination.
    Note: Storage can only be increased (e.g. 32Gi -> 64Gi), not decreased, to prevent data loss.

    Args:
        username (str): Student username (e.g. 'amorbar' or 'student123').
        new_size (str): Target storage capacity (e.g. '64Gi', '100Gi', '128Gi').
        namespace (str, optional): Target namespace. Defaults to course namespace.

    Returns:
        bool: True if patch succeeded.
    """
    core, default_ns = _init_k8s()
    ns = namespace or default_ns

    # Normalize username and find matching PVC
    clean_user = username.replace("_", "-").lower()
    candidate_names = [
        f"claim-jovyan-{clean_user}",
        f"claim-{clean_user}",
    ]

    target_pvc = None
    target_name = None
    for name in candidate_names:
        try:
            pvc = core.read_namespaced_persistent_volume_claim(name=name, namespace=ns)
            target_pvc = pvc
            target_name = name
            break
        except ApiException as e:
            if e.status != 404:
                raise

    if not target_pvc:
        # Fallback: search all PVCs matching the user
        pvcs = list_student_storage(ns)
        for r in pvcs:
            if clean_user in r["username"]:
                target_name = r["pvc_name"]
                break

    if not target_name:
        raise FileNotFoundError(f"PersistentVolumeClaim for user '{username}' not found in namespace '{ns}'.")

    # Patch the PVC spec with the new storage request
    patch_body = {
        "spec": {
            "resources": {
                "requests": {
                    "storage": new_size
                }
            }
        }
    }

    try:
        core.patch_namespaced_persistent_volume_claim(name=target_name, namespace=ns, body=patch_body)
        print(f"✓ Successfully expanded PVC '{target_name}' for user '{username}' to {new_size} in namespace '{ns}'.")
        print("  GKE CSI Driver will resize the underlying ext4/xfs filesystem automatically.")
        return True
    except ApiException as e:
        print(f"✗ Failed to patch PVC '{target_name}': {e}", file=sys.stderr)
        return False


def set_user_compute(
    username: str,
    cpu_limit: int | None = None,
    mem_limit: str | None = None,
    namespace: str | None = None,
) -> bool:
    """Dynamically set custom compute overrides (vCPU and RAM) for a student or admin.

    Stored in the 'cmu-user-overrides' ConfigMap and read by KubeSpawner on next pod spawn.

    Args:
        username (str): Target user account (e.g. 'amorbar' or 'student123').
        cpu_limit (int, optional): Max vCPU limit (e.g. 4, 8, 16).
        mem_limit (str, optional): Max RAM limit (e.g. '32G', '64G').
        namespace (str, optional): Course namespace.
    """
    core, default_ns = _init_k8s()
    ns = namespace or default_ns
    clean_user = username.replace("_", "-").lower()

    cm_name = "cmu-user-overrides"
    import json

    try:
        cm = core.read_namespaced_config_map(name=cm_name, namespace=ns)
        overrides = json.loads(cm.data.get("overrides.json", "{}"))
    except ApiException as e:
        if e.status == 404:
            overrides = {}
            cm = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name=cm_name, namespace=ns),
                data={"overrides.json": "{}"},
            )
            core.create_namespaced_config_map(namespace=ns, body=cm)
        else:
            raise

    if clean_user not in overrides:
        overrides[clean_user] = {}

    if cpu_limit is not None:
        overrides[clean_user]["cpu_limit"] = cpu_limit
        overrides[clean_user]["cpu_guarantee"] = max(1, cpu_limit // 2)
    if mem_limit is not None:
        overrides[clean_user]["mem_limit"] = mem_limit
        overrides[clean_user]["mem_guarantee"] = "8G"

    cm.data["overrides.json"] = json.dumps(overrides, indent=2)
    try:
        core.replace_namespaced_config_map(name=cm_name, namespace=ns, body=cm)
        print(f"✓ Successfully updated compute quota for '{clean_user}': {overrides[clean_user]}")
        return True
    except ApiException as e:
        print(f"✗ Failed to update compute overrides: {e}", file=sys.stderr)
        return False


GRANT_TIERS: dict[int, dict[str, Any]] = {
    0: {
        "name": "Tier 0: Course Default Baseline",
        "storage": "32Gi",
        "cpu_limit": 2,
        "mem_limit": "16G",
        "description": "Standard coursework profile for homework EDA and queued training.",
    },
    1: {
        "name": "Tier 1: Capstone / Project Team Grant",
        "storage": "100Gi",
        "cpu_limit": 4,
        "mem_limit": "32G",
        "description": "Approved capstone teams: expanded PVC disk, high-compute CPU, and cohort queue priority.",
    },
    2: {
        "name": "Tier 2: Research Fellow / Competitive Grant",
        "storage": "200Gi",
        "cpu_limit": 8,
        "mem_limit": "64G",
        "description": "Faculty-approved fellows: high-capacity storage, extreme CPU, and pinned/priority TPU access.",
    },
}


def grant_tier(username: str, tier: int = 1, namespace: str | None = None) -> bool:
    """Award a standardized compute and storage grant tier to a student.

    Args:
        username (str): Student username (e.g. 'amorbar' or 'student123').
        tier (int): Grant tier level (0: default baseline, 1: capstone grant, 2: research fellow).
        namespace (str, optional): Target Kubernetes namespace.

    Returns:
        bool: True if both storage expansion and compute override updates succeeded.
    """
    if tier not in GRANT_TIERS:
        raise ValueError(f"Unknown grant tier {tier}. Supported tiers: {list(GRANT_TIERS.keys())}")

    config_tier = GRANT_TIERS[tier]
    print(f"Applying {config_tier['name']} for student '{username}'...")
    print(f"  Storage: {config_tier['storage']} | vCPU: {config_tier['cpu_limit']} | RAM: {config_tier['mem_limit']}")
    s_ok = expand_student_storage(username, config_tier["storage"], namespace=namespace)
    c_ok = set_user_compute(username, cpu_limit=config_tier["cpu_limit"], mem_limit=config_tier["mem_limit"], namespace=namespace)
    if s_ok and c_ok:
        print(f"✓ Grant Tier {tier} successfully active for '{username}'. (Effective on next spawner launch)")
        return True
    return False


def create_project_directory(team_id: str, base_path: str = "/tartan/11785/projects") -> str:
    """Create a structured shared project directory for a capstone or research team.

    Args:
        team_id (str): Unique team identifier (e.g. 'capstone-alpha').
        base_path (str): Base filesystem path for course projects. Defaults to /tartan/11785/projects.

    Returns:
        str: Absolute path to the created project directory.
    """
    clean_team = team_id.replace(" ", "_").lower()
    project_dir = os.path.join(base_path, clean_team)
    subdirs = ["checkpoints", "data", "code", "submissions"]
    try:
        os.makedirs(project_dir, exist_ok=True)
        for s in subdirs:
            os.makedirs(os.path.join(project_dir, s), exist_ok=True)
        print(f"✓ Project directory created at: {project_dir}")
        for s in subdirs:
            print(f"  ├── {s}/")
        return project_dir
    except OSError as e:
        print(f"Notice: Could not write directly to {base_path} ({e}).")
        print("Ensure you are running from the Admin Workstation with Read-Write mount to /tartan/11785.")
        return project_dir


def render_admin_widget():
    """Render an interactive JupyterLab widget for point-and-click resource management.

    Usage in an admin Jupyter notebook:
        import hub_admin
        hub_admin.render_admin_widget()
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets and IPython are required for the interactive widget. Run: pip install ipywidgets")
        return

    records = list_student_storage()
    if not records:
        print("No student PVCs discovered in current namespace.")
        return

    user_options = [(f"{r['username']} ({r['actual_storage']})", r['username']) for r in records]
    user_dropdown = widgets.Dropdown(
        options=user_options,
        description="User:",
        disabled=False,
    )
    tier_dropdown = widgets.Dropdown(
        options=[
            ("Custom Selection", -1),
            ("Tier 0: Default Baseline (32Gi / 2 vCPU / 16G RAM)", 0),
            ("Tier 1: Capstone Team Grant (100Gi / 4 vCPU / 32G RAM)", 1),
            ("Tier 2: Research Fellow Grant (200Gi / 8 vCPU / 64G RAM)", 2),
        ],
        value=1,
        description="Grant Tier:",
        disabled=False,
    )
    size_input = widgets.Dropdown(
        options=["32Gi", "48Gi", "64Gi", "80Gi", "100Gi", "128Gi", "200Gi"],
        value="100Gi",
        description="Disk Size:",
        disabled=False,
    )
    cpu_input = widgets.Dropdown(
        options=[("2 vCPU (Default)", 2), ("4 vCPU (Tier 1)", 4), ("8 vCPU (Tier 2)", 8), ("16 vCPU (Workstation)", 16)],
        value=4,
        description="vCPU Limit:",
        disabled=False,
    )
    ram_input = widgets.Dropdown(
        options=["16G (Default)", "32G (Tier 1)", "64G (Tier 2)"],
        value="32G (Tier 1)",
        description="RAM Limit:",
        disabled=False,
    )

    def on_tier_change(change):
        selected_tier = change["new"]
        if selected_tier in GRANT_TIERS:
            cfg = GRANT_TIERS[selected_tier]
            size_input.value = cfg["storage"]
            cpu_input.value = cfg["cpu_limit"]
            for opt in ram_input.options:
                if cfg["mem_limit"] in opt:
                    ram_input.value = opt
                    break

    tier_dropdown.observe(on_tier_change, names="value")

    apply_btn = widgets.Button(
        description="Award Grant / Apply Resources",
        button_style="danger",
        tooltip="Click to expand disk quota and set compute overrides",
        icon="graduation-cap",
    )
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_user = user_dropdown.value
            selected_tier = tier_dropdown.value
            if selected_tier in GRANT_TIERS:
                print(f"Applying Grant Tier {selected_tier} for '{selected_user}'...")
                grant_tier(selected_user, tier=selected_tier)
            else:
                target_size = size_input.value
                target_cpu = cpu_input.value
                target_ram = ram_input.value.split()[0]
                print(f"Applying custom resource allocations for '{selected_user}'...")
                expand_student_storage(selected_user, target_size)
                set_user_compute(selected_user, cpu_limit=target_cpu, mem_limit=target_ram)
            print("✓ Changes saved! Active immediately on the student's next notebook spawn.")

    apply_btn.on_click(on_button_clicked)

    form = widgets.VBox([
        widgets.HTML("<h3>🎓 CMU 11-785 JupyterHub Grant & Resource Manager</h3><p>Award student capstone project grants or tune PVC storage, vCPUs, and RAM dynamically.</p>"),
        user_dropdown,
        tier_dropdown,
        size_input,
        cpu_input,
        ram_input,
        apply_btn,
        output
    ])
    display(form)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CMU 11-785 JupyterHub Storage Admin & Grant CLI")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_p = subparsers.add_parser("list", help="List all student PVCs")
    list_p.add_argument("--namespace", default=None, help="Kubernetes namespace")

    # Expand command
    expand_p = subparsers.add_parser("expand", help="Expand a student's PVC capacity")
    expand_p.add_argument("username", help="Student username")
    expand_p.add_argument("size", help="New storage size (e.g. 64Gi, 100Gi)")
    expand_p.add_argument("--namespace", default=None, help="Kubernetes namespace")

    # Grant command
    grant_p = subparsers.add_parser("grant", help="Award a project grant tier to a student")
    grant_p.add_argument("username", help="Student username")
    grant_p.add_argument("--tier", type=int, default=1, choices=[0, 1, 2], help="Grant tier (0: baseline, 1: capstone team, 2: fellow)")
    grant_p.add_argument("--namespace", default=None, help="Kubernetes namespace")

    # Project dir command
    proj_p = subparsers.add_parser("project", help="Create a shared project directory for a capstone team")
    proj_p.add_argument("team_id", help="Capstone team identifier (e.g. capstone-alpha)")
    proj_p.add_argument("--base-path", default="/tartan/11785/projects", help="Base path for course projects")

    args = parser.parse_args()
    if args.command == "list":
        rows = list_student_storage(args.namespace)
        print(f"{'Username':<20} {'PVC Name':<30} {'Requested':<10} {'Actual':<10} {'Status':<10}")
        print("-" * 85)
        for r in rows:
            print(f"{r['username']:<20} {r['pvc_name']:<30} {r['requested_storage']:<10} {r['actual_storage']:<10} {r['phase']:<10}")
    elif args.command == "expand":
        expand_student_storage(args.username, args.size, args.namespace)
    elif args.command == "grant":
        grant_tier(args.username, tier=args.tier, namespace=args.namespace)
    elif args.command == "project":
        create_project_directory(args.team_id, base_path=args.base_path)
    else:
        parser.print_help()
