"""
Metrics Store — time-series ring buffer for all system metrics.
Thread-safe, supports WebSocket broadcasting.
"""
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class MetricPoint:
    timestamp: float
    cpu_util: float           # 0-1
    memory_util: float        # 0-1
    request_rate: float       # req/s
    response_time_p50: float  # ms
    response_time_p99: float  # ms
    error_rate: float         # 0-1
    active_pods: int
    queue_depth: int
    throughput: float         # req/s served
    cost_per_hour: float      # USD
    sla_violated: bool
    violation_type: Optional[str] = None


@dataclass
class ScalingDecision:
    timestamp: float
    action: str               # scale_up, scale_down, hold
    delta: int                # +/- pods
    from_pods: int
    to_pods: int
    reason: str
    agent: str                # which agent triggered
    shap_values: Dict = field(default_factory=dict)
    predicted_workload: float = 0.0
    confidence: float = 0.0
    is_safe_override: bool = False


@dataclass
class SLAConfig:
    p99_latency_ms: float = 200.0
    availability_pct: float = 99.9
    error_rate_pct: float = 0.1
    min_throughput_rps: float = 100.0
    max_cost_per_hour: float = 10.0
    # Pareto weight: 0=pure cost saving, 1=pure SLA compliance
    cost_sla_weight: float = 0.7
    min_pods: int = 1
    max_pods: int = 10


class MetricsStore:
    """Thread-safe ring-buffer metrics store with statistics."""

    def __init__(self, maxlen: int = 3600):
        self._lock = threading.RLock()
        self.metrics: deque[MetricPoint] = deque(maxlen=maxlen)
        self.decisions: deque[ScalingDecision] = deque(maxlen=500)
        self.sla_config = SLAConfig()
        self._subscribers = []  # WebSocket callbacks

    def push_metric(self, point: MetricPoint):
        with self._lock:
            self.metrics.append(point)
        self._notify({"type": "metric", "data": asdict(point)})

    def push_decision(self, decision: ScalingDecision):
        with self._lock:
            self.decisions.append(decision)
        self._notify({"type": "decision", "data": asdict(decision)})

    def get_recent(self, n: int = 60) -> List[MetricPoint]:
        with self._lock:
            return list(self.metrics)[-n:]

    def get_state_vector(self, window: int = 60) -> np.ndarray:
        """Build normalized state vector for DRL agent."""
        with self._lock:
            recent = list(self.metrics)[-window:]
        if not recent:
            return np.zeros(9)
        latest = recent[-1]
        avg_rt = np.mean([m.response_time_p99 for m in recent])
        trend = self._compute_trend([m.request_rate for m in recent])
        violation_rate = sum(1 for m in recent if m.sla_violated) / max(len(recent), 1)

        return np.array([
            latest.cpu_util,
            latest.memory_util,
            min(latest.request_rate / 1000.0, 1.0),
            min(latest.response_time_p99 / 1000.0, 1.0),
            latest.error_rate,
            latest.active_pods / self.sla_config.max_pods,
            min(latest.queue_depth / 500.0, 1.0),
            violation_rate,
            trend
        ], dtype=np.float32)

    def get_window_tensor(self, window: int = 60) -> np.ndarray:
        """Return (window, features) array for Transformer encoder."""
        with self._lock:
            recent = list(self.metrics)[-window:]
        if len(recent) < window:
            pad = [MetricPoint(0,0,0,0,0,0,0,1,0,0,0,False)] * (window - len(recent))
            recent = pad + recent
        rows = []
        for m in recent:
            rows.append([
                m.cpu_util, m.memory_util,
                min(m.request_rate / 1000.0, 1.0),
                min(m.response_time_p99 / 1000.0, 1.0),
                m.error_rate,
                m.active_pods / self.sla_config.max_pods,
                min(m.queue_depth / 500.0, 1.0),
                float(m.sla_violated)
            ])
        return np.array(rows, dtype=np.float32)

    def _compute_trend(self, values: list) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(np.clip(slope / 100.0, -1.0, 1.0))

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def unsubscribe(self, callback):
        self._subscribers = [s for s in self._subscribers if s != callback]

    def _notify(self, data):
        for cb in self._subscribers:
            try:
                cb(data)
            except Exception:
                pass

    def get_dashboard_snapshot(self) -> dict:
        with self._lock:
            metrics_list = [asdict(m) for m in list(self.metrics)[-120:]]
            decisions_list = [asdict(d) for d in list(self.decisions)[-50:]]
        return {
            "metrics": metrics_list,
            "decisions": decisions_list,
            "sla_config": asdict(self.sla_config),
            "summary": self._compute_summary()
        }

    def _compute_summary(self) -> dict:
        with self._lock:
            recent = list(self.metrics)[-60:]
        if not recent:
            return {}
        violations = [m for m in recent if m.sla_violated]
        return {
            "avg_response_time": float(np.mean([m.response_time_p99 for m in recent])),
            "avg_cpu": float(np.mean([m.cpu_util for m in recent])),
            "avg_pods": float(np.mean([m.active_pods for m in recent])),
            "sla_compliance_pct": round((1 - len(violations)/max(len(recent),1))*100, 2),
            "total_decisions": len(self.decisions),
            "current_cost_per_hour": recent[-1].cost_per_hour if recent else 0,
        }


# Global singleton
store = MetricsStore()
