# ==============================================================================
# shared-tpu-notebooks Makefile
# ==============================================================================
-include config.env

PROJECT ?=
REGION  ?= us-west4
CLUSTER ?= tpu-notebooks
NS      ?= cmu-idl
WARM    ?= 1

# Course Scale and Configuration
STUDENTS      ?= 15
POOL_CHIPS    ?= 32
DOMAIN        ?= # e.g. jupyter.cs.cmu.edu
STUDENT_GROUP ?= # e.g. group:idl-11785-students@cmu.edu
TA_GROUP      ?= # e.g. group:idl-11785-tas@cmu.edu
ADMIN_USERS   ?= # e.g. user:bradley@cmu.edu user:ayush@cmu.edu
TEST_ACCOUNTS ?= # e.g. user:test1@cmu.edu user:devansh@cmu.edu

# Hardware, Shared Environment & Storage
TPU_ACCELERATOR      ?= tpu-v5-lite-podslice
TPU_TOPOLOGY         ?= 1x1
TPU_IMAGE            ?= us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest
SINGLEUSER_IMAGE     ?= quay.io/jupyter/scipy-notebook
SINGLEUSER_IMAGE_TAG ?= 2024-11-11
SHARED_ENV_NAME      ?= idlf26jax
REQUIREMENTS_FILE    ?= requirements.txt
ENV_BUILDER_CPU      ?= 2
ENV_BUILDER_MEMORY   ?= 8Gi
ENV_BUILDER_STORAGE  ?= 5Gi
TARTAN_BUCKET        ?= $(PROJECT)-tartan
OCEAN_BUCKET         ?= $(TARTAN_BUCKET)

export PROJECT REGION CLUSTER NS NAMESPACE=$(NS) STUDENTS POOL_CHIPS WARM DOMAIN STUDENT_GROUP TA_GROUP ADMIN_USERS TEST_ACCOUNTS TPU_ACCELERATOR TPU_TOPOLOGY TPU_IMAGE SINGLEUSER_IMAGE SINGLEUSER_IMAGE_TAG SHARED_ENV_NAME REQUIREMENTS_FILE ENV_BUILDER_CPU ENV_BUILDER_MEMORY ENV_BUILDER_STORAGE TARTAN_BUCKET OCEAN_BUCKET

.PHONY: check preflight cluster image storage shared-env hub iap ip domain dns-record warm-on warm-off demo smoke smoke-submit test scale report expand-pvcs expand-pvcs-dry-run clean-pvcs clean-pvcs-dry-run secrets teardown venv admin-list admin-grant admin-project help

# Ensure the PROJECT variable is set before proceeding.
check:
ifndef PROJECT
	$(error PROJECT is not set. Set it in config.env or run: make $(MAKECMDGOALS) PROJECT=my-gcp-project)
endif

# Run the regional quota and availability preflight script.
preflight: check
	bash scripts/00_preflight.sh

# Build and push the TPU job and submission service images (docker/tpu-job, services/tpu-submit).
# Student CPU notebooks use upstream scipy-notebook + `make shared-env` (zero Docker rebuilds).
image: check
	bash scripts/01_build_image.sh

# Provision or update a zero-rebuild shared Python environment on Tartan GCS store.
# Usage:
#   make shared-env                                (defaults to idlf26jax and requirements.txt)
#   make shared-env ENV=capstone-nlp REQ=reqs.txt
shared-env: check
	ENV="$(ENV)" REQ="$(REQ)" bash scripts/09_setup_shared_env.sh

# Provision the GKE Autopilot cluster, Kueue configurations, and StorageClasses.
cluster: check
	bash scripts/02_create_cluster.sh

# Buckets for Tartan shared storage and TPU job runs, plus Workload Identity bindings.
storage: check
	bash scripts/12_setup_storage.sh

# Deploy the TPU submission service and JupyterHub.
hub: check
	bash scripts/03_deploy_hub.sh

# One real job on one real chip. This is the proof the substrate works, and it is
# the thing to run before any scale test.
smoke: check
	sed -e 's/__STUDENT__/smoke-000/g' -e 's/__NAMESPACE__/$(NS)/g' -e 's/__QUEUE__/tpu/g' \
	  k8s/student-tpu-job.yaml | kubectl apply -f -
	kubectl -n $(NS) wait --for=condition=complete job/smoke-000 --timeout=900s
	kubectl -n $(NS) logs -l job-name=smoke-000 --tail=-1
	kubectl -n $(NS) delete job smoke-000

# The student path end to end: submit_tpu.py -> tpu-submit -> Kueue -> chip -> bucket.
# Needs a JupyterHub API token: HUB_TOKEN=... make smoke-submit
smoke-submit: check
	bash scripts/13_smoke_submit.sh

# Unit and end-to-end tests for the service, client, and TPU wrapper. No cloud calls.
test:
	python3 -m pytest tests -q

# HTTPS + Google sign-in via Identity-Aware Proxy. Run once per cluster. Replaces
# port-forwarding, which binds to a single proxy pod and dies when that pod moves.
iap: check
	bash scripts/08_setup_iap.sh

# Display the reserved GCP Global Static IP and active ingress mapping
ip: check
	@IP=$$(gcloud compute addresses describe tpu-notebooks-ip --global --project=$(PROJECT) --format="value(address)" 2>/dev/null || echo ""); \
	if [ -z "$$IP" ]; then \
	  echo "Static IP 'tpu-notebooks-ip' is not yet reserved. Run 'make iap' to provision."; \
	else \
	  echo "=================================================="; \
	  echo "GCP Global Static IP : $$IP"; \
	  echo "Default nip.io URL   : https://$$IP.nip.io"; \
	  if [ -n "$(DOMAIN)" ]; then \
	    echo "Custom Subdomain     : https://$(DOMAIN)"; \
	    echo "DNS Configuration    : Create A-Record: $(DOMAIN) -> $$IP"; \
	  fi; \
	  echo "=================================================="; \
	fi

# Reconfigure Ingress and ManagedCertificate with a custom university subdomain
domain: check
ifndef DOMAIN
	$(error DOMAIN is not set. Usage: make domain DOMAIN=jupyter.cs.cmu.edu)
endif
	@echo "==> Configuring custom domain: $(DOMAIN)..."
	DOMAIN=$(DOMAIN) bash scripts/08_setup_iap.sh

# Automatically add an A-record to Google Cloud DNS if managing DNS in GCP
dns-record: check
ifndef DOMAIN
	$(error DOMAIN is not set. Usage: make dns-record DOMAIN=jupyter.cs.cmu.edu ZONE=cmu-dns-zone)
endif
ifndef ZONE
	$(error ZONE is not set. Usage: make dns-record DOMAIN=jupyter.cs.cmu.edu ZONE=cmu-dns-zone)
endif
	@IP=$$(gcloud compute addresses describe tpu-notebooks-ip --global --project=$(PROJECT) --format="value(address)" 2>/dev/null || echo ""); \
	if [ -z "$$IP" ]; then \
	  echo "Error: Static IP 'tpu-notebooks-ip' does not exist. Run 'make iap' first." >&2; exit 1; \
	fi; \
	echo "==> Mapping $(DOMAIN) -> $$IP in Cloud DNS Managed Zone '$(ZONE)'..."; \
	gcloud dns record-sets transaction start --zone=$(ZONE) --project=$(PROJECT) 2>/dev/null || true; \
	gcloud dns record-sets transaction add $$IP --name="$(DOMAIN)." --ttl=300 --type=A --zone=$(ZONE) --project=$(PROJECT); \
	gcloud dns record-sets transaction execute --zone=$(ZONE) --project=$(PROJECT); \
	echo "✓ Cloud DNS A-record successfully published!"; \
	echo "==> Applying domain to Ingress and ManagedCertificate..."; \
	DOMAIN=$(DOMAIN) bash scripts/08_setup_iap.sh

# Hold warm TPU nodes so the first job of the day skips the 2-4 min node build.
# Warm chips bill continuously, so turn them off when you are done.
warm-on: check
	bash scripts/07_warm_pool.sh on $(WARM)

# Release any warm placeholder TPU chips to stop billing.
warm-off: check
	bash scripts/07_warm_pool.sh off

# Full automated demo: Create cluster, storage, images, shared-env, hub, and smoke test.
demo: cluster storage image shared-env hub smoke
	@echo
	@echo "Put the hub behind HTTPS and Google sign-in:  make iap PROJECT=$(PROJECT)"

# Create a python virtual environment for local scripts.
venv:
	python3 -m venv .venv && ./.venv/bin/pip install --quiet --upgrade pip

# Run the concurrency scale test.
scale: check venv
	./.venv/bin/python scripts/04_scale_test.py --students $(STUDENTS) --chips $(POOL_CHIPS) --namespace $(NS)
	./.venv/bin/python scripts/05_report.py

# Online expansion of existing student PVCs (e.g. 10Gi -> 32Gi)
expand-pvcs-dry-run: check
	bash scripts/11_expand_pvcs.sh --dry-run

expand-pvcs: check
	bash scripts/11_expand_pvcs.sh --execute

# Optional: Create baseline Kubernetes Secret for cluster infrastructure
secrets: check
	bash scripts/12_create_secrets.sh

# Clean up retained PVCs and underlying GCP persistent disks at end of term
clean-pvcs-dry-run: check
	bash scripts/10_cleanup_pvcs.sh --dry-run

clean-pvcs: check
	bash scripts/10_cleanup_pvcs.sh --execute

# Admin & Student Grant Management (scripts/hub_admin.py)
admin-list: check
	python3 scripts/hub_admin.py list --namespace $(NS)

admin-grant: check
ifndef USER
	$(error USER is not set. Usage: make admin-grant USER=student_id TIER=1)
endif
	python3 scripts/hub_admin.py grant $(USER) --tier $(or $(TIER),1) --namespace $(NS)

admin-project: check
ifndef TEAM
	$(error TEAM is not set. Usage: make admin-project TEAM=team-name)
endif
	python3 scripts/hub_admin.py project $(TEAM)

# Fully destroy the cluster, queued resources, static IPs, and stop all billing.
teardown: check
	bash scripts/99_teardown.sh

help:
	@echo "shared-tpu-notebooks Makefile commands:"
	@echo "  make preflight PROJECT=...   Check regional quota and accelerator types (free)"
	@echo "  make cluster PROJECT=...     Create GKE Autopilot cluster, Kueue, & quotas (~12m)"
	@echo "  make storage PROJECT=...     Create Tartan & student-runs buckets + Workload Identity"
	@echo "  make image PROJECT=...       Build & push tpu-job and tpu-submit images"
	@echo "  make shared-env [ENV=..]     Provision/update zero-rebuild shared Python env on Tartan"
	@echo "  make hub PROJECT=...         Deploy tpu-submit service & JupyterHub Helm chart"
	@echo "  make smoke PROJECT=...       Submit 1 verification TPU job on 1 v5e chip (raw YAML)"
	@echo "  make smoke-submit PROJECT=.. Submit hello.py through the tpu-submit service"
	@echo "  make test                    Unit & e2e tests for tpu-submit, client, and TPU wrapper"
	@echo "  make iap PROJECT=...         Configure HTTPS Ingress & Google Identity-Aware Proxy"
	@echo "  make ip PROJECT=...          Show reserved static IP and active domain mapping"
	@echo "  make domain DOMAIN=...       Reconfigure Ingress/SSL cert for custom university subdomain"
	@echo "  make dns-record DOMAIN=.. ZONE=.. Auto-create Cloud DNS A-record and apply to Ingress"
	@echo "  make warm-on PROJECT=...     Hold WARM placeholder chips ready (default WARM=1)"
	@echo "  make warm-off PROJECT=...    Release warm placeholder chips"
	@echo "  make scale PROJECT=...       Run concurrency scale test (STUDENTS=15 POOL_CHIPS=32)"
	@echo "  make report                  Generate analytics report from latest test run"
	@echo "  make expand-pvcs-dry-run     Preview existing student PVCs to be resized (10Gi->32Gi)"
	@echo "  make expand-pvcs             Dynamically resize existing PVCs to 32Gi online"
	@echo "  make admin-list              List all student PVCs and storage usage via hub_admin.py"
	@echo "  make admin-grant USER=.. TIER=..  Award Grant Tier (1: Capstone, 2: Fellow) to student"
	@echo "  make admin-project TEAM=..   Scaffold shared team directory in Tartan storage"
	@echo "  make clean-pvcs-dry-run      Preview retained PVCs & persistent disks"
	@echo "  make clean-pvcs              Permanently delete retained PVCs & persistent disks"
	@echo "  make teardown PROJECT=...    Destroy cluster, queued resources, and static IP"
