"""
Cloud Environment Simulation
Simulates realistic cloud workload behavior for DRL training and live operation.
Models: request queuing, latency under load, pod startup delay, cost model.
"""
import time
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class WorkloadProfile:
    name: str
    base_rps: float
    burst_probability: float
    burst_multiplier: float
    noise_std: float


WORKLOAD_PROFILES = {
    "login":    WorkloadProfile("login",    50,  0.05, 3.0, 5.0),
    "payment":  WorkloadProfile("payment",  20,  0.02, 5.0, 2.0),
    "checkout": WorkloadProfile("checkout", 30,  0.08, 4.0, 3.0),
    "browse":   WorkloadProfile("browse",  150,  0.10, 2.0, 15.0),
    "error":    WorkloadProfile("error",    10,  0.30, 8.0, 5.0),
}


class CloudEnvironment:
    """
    Simulates a cloud microservice under variable load.
    Used for DRL offline pre-training and as a digital twin during live operation.
    """

    POD_COST_PER_HOUR = 0.048   # USD per pod-hour (e.g., t3.small)
    POD_STARTUP_SECONDS = 15    # cold-start delay
    MAX_RPS_PER_POD = 80        # saturation point per pod
    BASE_LATENCY_MS = 15        # idle latency
    QUEUE_MAX = 1000

    def __init__(self, min_pods: int = 1, max_pods: int = 10):
        self.min_pods = min_pods
        self.max_pods = max_pods
        self.current_pods = 2
        self.pending_pods = 0       # pods starting up
        self.pod_start_times = []   # timestamps of pod startups
        self.queue = 0
        self.time_step = 0
        self.active_profiles: Dict[str, float] = {}   # profile_name -> weight
        self._episode_start = time.time()

    def set_workload(self, profiles: Dict[str, float]):
        """Set active workload mix. profiles = {type: rps_weight}"""
        self.active_profiles = profiles

    def _generate_request_rate(self) -> float:
        """Generate realistic request rate with diurnal pattern + bursts."""
        t = self.time_step
        # Diurnal pattern (simulated over 1000 steps = 1 day)
        diurnal = 1.0 + 0.4 * math.sin(2 * math.pi * t / 1000 - math.pi / 2)
        total_rps = 0.0
        for pname, weight in self.active_profiles.items():
            p = WORKLOAD_PROFILES.get(pname)
            if p:
                rps = p.base_rps * weight * diurnal
                rps += np.random.normal(0, p.noise_std)
                if random.random() < p.burst_probability:
                    rps *= p.burst_multiplier
                total_rps += max(0, rps)
        return max(1.0, total_rps)

    def _compute_response_time(self, rps: float, pods: int) -> Tuple[float, float, int]:
        """M/M/c queueing model for latency estimation."""
        capacity = pods * self.MAX_RPS_PER_POD
        utilization = min(rps / max(capacity, 1), 0.99)

        # M/M/1 approximation
        service_rate = self.MAX_RPS_PER_POD
        queue_wait = (utilization / (service_rate * max(pods, 1) * (1 - utilization + 1e-6))) * 1000

        # Base latency grows with utilization
        processing = self.BASE_LATENCY_MS * (1 + 2.0 * utilization ** 3)
        p50 = processing + queue_wait * 0.3
        p99 = processing + queue_wait * 2.5 + np.random.exponential(5)

        # Update queue depth
        overflow = max(0, rps - capacity)
        self.queue = min(int(self.queue * 0.9 + overflow * 0.5), self.QUEUE_MAX)

        return float(np.clip(p50, self.BASE_LATENCY_MS, 5000)),\
               float(np.clip(p99, self.BASE_LATENCY_MS * 2, 10000)),\
               self.queue

    def _compute_error_rate(self, rps: float, pods: int) -> float:
        capacity = pods * self.MAX_RPS_PER_POD
        overload = max(0, rps - capacity) / max(rps, 1)
        base_error = 0.001
        return float(np.clip(base_error + overload * 0.3 + random.gauss(0, 0.001), 0, 1))

    def _compute_cpu_memory(self, rps: float, pods: int) -> Tuple[float, float]:
        capacity = pods * self.MAX_RPS_PER_POD
        load = rps / max(capacity, 1)
        cpu = float(np.clip(0.05 + 0.80 * load + np.random.normal(0, 0.03), 0, 1))
        mem = float(np.clip(0.20 + 0.50 * load + np.random.normal(0, 0.02), 0, 1))
        return cpu, mem

    def step(self) -> dict:
        """Advance one time step and return full state."""
        self.time_step += 1

        # Handle pod startup completions
        now = time.time()
        completed = [t for t in self.pod_start_times if now - t >= self.POD_STARTUP_SECONDS]
        if completed:
            self.current_pods += len(completed)
            self.pod_start_times = [t for t in self.pod_start_times if now - t < self.POD_STARTUP_SECONDS]

        rps = self._generate_request_rate()
        p50, p99, queue = self._compute_response_time(rps, self.current_pods)
        error_rate = self._compute_error_rate(rps, self.current_pods)
        cpu, mem = self._compute_cpu_memory(rps, self.current_pods)
        throughput = min(rps, self.current_pods * self.MAX_RPS_PER_POD)
        cost = (self.current_pods + len(self.pod_start_times)) * self.POD_COST_PER_HOUR

        return {
            "timestamp": now,
            "cpu_util": cpu,
            "memory_util": mem,
            "request_rate": rps,
            "response_time_p50": p50,
            "response_time_p99": p99,
            "error_rate": error_rate,
            "active_pods": self.current_pods,
            "pending_pods": len(self.pod_start_times),
            "queue_depth": queue,
            "throughput": throughput,
            "cost_per_hour": cost,
        }

    def scale(self, delta: int) -> bool:
        """Apply scaling action. Returns True if action was taken."""
        target = self.current_pods + delta
        target = int(np.clip(target, self.min_pods, self.max_pods))
        delta = target - self.current_pods

        if delta > 0:
            # Scale out: pods start with delay
            for _ in range(delta):
                self.pod_start_times.append(time.time())
            return True
        elif delta < 0:
            # Scale in: immediate
            self.current_pods = max(self.min_pods, self.current_pods + delta)
            return True
        return False

    def reset(self):
        self.current_pods = 2
        self.pod_start_times = []
        self.queue = 0
        self.time_step = 0
        return self.step()

    def compute_reward(self, state: dict, sla_config) -> Tuple[float, bool, str]:
        """
        Multi-objective reward:
        R = α * SLA_score - β * cost_score - γ * instability_penalty
        """
        violation = False
        violation_type = None

        # SLA violations
        rt_violation = state["response_time_p99"] > sla_config.p99_latency_ms
        err_violation = state["error_rate"] * 100 > sla_config.error_rate_pct
        tput_violation = state["throughput"] < sla_config.min_throughput_rps

        if rt_violation:
            violation, violation_type = True, "latency"
        elif err_violation:
            violation, violation_type = True, "error_rate"
        elif tput_violation:
            violation, violation_type = True, "throughput"

        # Normalized scores
        rt_score = 1.0 - min(state["response_time_p99"] / sla_config.p99_latency_ms, 2.0) / 2.0
        err_score = 1.0 - min(state["error_rate"] * 100 / sla_config.error_rate_pct, 2.0) / 2.0
        sla_score = (rt_score + err_score) / 2.0

        # Cost score (lower pods = better)
        max_cost = sla_config.max_pods * self.POD_COST_PER_HOUR
        cost_score = state["cost_per_hour"] / max(max_cost, 1)

        # Weighted reward
        w = sla_config.cost_sla_weight
        reward = w * sla_score - (1 - w) * cost_score
        if violation:
            reward -= 2.0   # hard penalty

        return float(reward), violation, violation_type
