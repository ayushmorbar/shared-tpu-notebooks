"""The few Kubernetes calls the service makes, returned as plain camelCase dicts."""

from __future__ import annotations

from typing import Protocol


class Cluster(Protocol):
    def create_job(self, manifest: dict) -> dict: ...
    def create_secret(self, manifest: dict) -> None: ...
    def get_job(self, name: str) -> dict | None: ...
    def list_jobs(self, selector: str) -> list[dict]: ...
    def delete_job(self, name: str) -> None: ...
    def list_pods(self, selector: str) -> list[dict]: ...
    def pod_log(self, pod: str) -> str: ...


class KubeCluster:
    def __init__(self, namespace: str):
        from kubernetes import client, config
        from kubernetes.client.exceptions import ApiException

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._ApiException = ApiException
        self._api = client.ApiClient()
        self._batch = client.BatchV1Api(self._api)
        self._core = client.CoreV1Api(self._api)
        self.namespace = namespace

    def _dict(self, obj) -> dict:
        return self._api.sanitize_for_serialization(obj)

    def create_job(self, manifest: dict) -> dict:
        return self._dict(self._batch.create_namespaced_job(self.namespace, manifest))

    def create_secret(self, manifest: dict) -> None:
        self._core.create_namespaced_secret(self.namespace, manifest)

    def get_job(self, name: str) -> dict | None:
        try:
            return self._dict(self._batch.read_namespaced_job(name, self.namespace))
        except self._ApiException as e:
            if e.status == 404:
                return None
            raise

    def list_jobs(self, selector: str) -> list[dict]:
        resp = self._batch.list_namespaced_job(self.namespace, label_selector=selector)
        return [self._dict(j) for j in resp.items]

    def delete_job(self, name: str) -> None:
        try:
            self._batch.delete_namespaced_job(
                name, self.namespace, propagation_policy="Background"
            )
        except self._ApiException as e:
            if e.status != 404:
                raise

    def list_pods(self, selector: str) -> list[dict]:
        resp = self._core.list_namespaced_pod(self.namespace, label_selector=selector)
        return [self._dict(p) for p in resp.items]

    def pod_log(self, pod: str) -> str:
        try:
            return self._core.read_namespaced_pod_log(pod, self.namespace, container="job")
        except self._ApiException as e:
            # 400 while the container has not started yet; 404 once the pod is gone.
            if e.status in (400, 404):
                return ""
            raise
