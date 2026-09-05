"""
Kubernetes Controller
Manages deployment replicas via the Kubernetes Python client.
Falls back to simulation mode if cluster is unavailable.
"""
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class KubernetesController:
    """
    Wraps kubectl / k8s API for scaling deployments.
    Simulation mode when minikube is unavailable.
    """

    def __init__(self, namespace: str = "default",
                 deployment_name: str = "sla-microservice",
                 simulate: bool = False):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.simulate = simulate
        self._sim_replicas = 2
        self._pending_scale: Optional[Tuple[int, float]] = None   # (target, start_time)
        self._client = None
        self._apps_v1 = None

        if not simulate:
            try:
                from kubernetes import client, config
                config.load_kube_config()
                self._client = client
                self._apps_v1 = client.AppsV1Api()
                logger.info("Kubernetes client initialized.")
            except Exception as e:
                logger.warning(f"K8s unavailable ({e}), running in simulation mode.")
                self.simulate = True

    def get_replicas(self) -> int:
        if self.simulate:
            return self._sim_replicas
        try:
            dep = self._apps_v1.read_namespaced_deployment(
                self.deployment_name, self.namespace
            )
            return dep.spec.replicas or 1
        except Exception as e:
            logger.error(f"Failed to get replicas: {e}")
            return self._sim_replicas

    def scale_to(self, target: int) -> dict:
        current = self.get_replicas()
        target = max(1, min(target, 10))
        if target == current:
            return {"success": True, "from": current, "to": target, "changed": False}

        if self.simulate:
            self._sim_replicas = target
            logger.info(f"[SIM] Scaled {self.deployment_name}: {current} → {target}")
            return {"success": True, "from": current, "to": target, "changed": True, "mode": "simulation"}

        try:
            body = {"spec": {"replicas": target}}
            self._apps_v1.patch_namespaced_deployment_scale(
                self.deployment_name, self.namespace, body
            )
            logger.info(f"[K8S] Scaled {self.deployment_name}: {current} → {target}")
            return {"success": True, "from": current, "to": target, "changed": True, "mode": "kubernetes"}
        except Exception as e:
            logger.error(f"K8s scale failed: {e}")
            self._sim_replicas = target   # fallback
            return {"success": False, "error": str(e), "from": current, "to": target}

    def scale_by(self, delta: int) -> dict:
        current = self.get_replicas()
        return self.scale_to(current + delta)

    def get_pod_status(self) -> list:
        if self.simulate:
            return [
                {"name": f"pod-{i}", "status": "Running", "ready": True,
                 "cpu": "120m", "memory": "256Mi"}
                for i in range(self._sim_replicas)
            ]
        try:
            from kubernetes import client
            v1 = client.CoreV1Api()
            pods = v1.list_namespaced_pod(
                self.namespace,
                label_selector=f"app={self.deployment_name}"
            )
            return [
                {
                    "name": p.metadata.name,
                    "status": p.status.phase,
                    "ready": all(c.ready for c in (p.status.container_statuses or [])),
                    "node": p.spec.node_name
                }
                for p in pods.items
            ]
        except Exception as e:
            logger.error(f"Pod list failed: {e}")
            return []

    def get_deployment_info(self) -> dict:
        replicas = self.get_replicas()
        pods = self.get_pod_status()
        ready = sum(1 for p in pods if p.get("ready"))
        return {
            "deployment": self.deployment_name,
            "namespace": self.namespace,
            "replicas": replicas,
            "ready_pods": ready,
            "pods": pods,
            "mode": "simulation" if self.simulate else "kubernetes"
        }
