"""
Advanced Monitoring:
- Isolation Forest anomaly detection
- ADWIN concept drift detection
- SLA violation classification
"""
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) algorithm for concept drift detection.
    Detects changes in the mean of a stream using adaptive window statistics.
    """
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.n = 0
        self.drift_detected = False
        self.drift_count = 0

    def add(self, value: float) -> bool:
        self.window.append(value)
        self.total += value
        self.n += 1
        self.drift_detected = self._detect()
        if self.drift_detected:
            self.drift_count += 1
        return self.drift_detected

    def _detect(self) -> bool:
        if self.n < 30:
            return False
        win = list(self.window)
        mean = np.mean(win)
        std = np.std(win) + 1e-8
        # Scan cut-points
        for i in range(10, self.n - 10, 5):
            left = win[:i]
            right = win[i:]
            m_left = np.mean(left)
            m_right = np.mean(right)
            n_left, n_right = len(left), len(right)
            eps_cut = std * np.sqrt((1/n_left + 1/n_right) * np.log(4 * self.n / self.delta))
            if abs(m_left - m_right) >= eps_cut:
                # Drift found — shrink window to right portion
                for _ in range(i):
                    old = self.window.popleft()
                    self.total -= old
                self.n = len(self.window)
                return True
        return False

    def status(self) -> dict:
        if self.n < 5:
            return {"drift": False, "drift_count": 0, "window_size": 0, "mean": 0.0}
        return {
            "drift": self.drift_detected,
            "drift_count": self.drift_count,
            "window_size": self.n,
            "mean": float(np.mean(list(self.window)[-50:]) if self.window else 0)
        }


class AnomalyDetector:
    """
    Isolation Forest for detecting traffic anomalies.
    Trained online with rolling refit every N samples.
    """
    def __init__(self, window: int = 200, contamination: float = 0.05,
                 refit_every: int = 50):
        self.window = window
        self.refit_every = refit_every
        self.contamination = contamination
        self.buffer = deque(maxlen=window)
        self.model: Optional[IsolationForest] = None
        self.step = 0
        self.last_score = 0.0
        self.is_anomaly = False
        self.anomaly_count = 0

    def update(self, features: np.ndarray) -> Tuple[bool, float]:
        """features: 1D array of current metrics."""
        self.buffer.append(features.copy())
        self.step += 1

        if len(self.buffer) < 30:
            return False, 0.0

        if self.step % self.refit_every == 0:
            X = np.array(self.buffer)
            self.model = IsolationForest(
                n_estimators=50, contamination=self.contamination,
                random_state=42, n_jobs=-1
            )
            self.model.fit(X)

        if self.model is None:
            return False, 0.0

        score = float(self.model.decision_function([features])[0])
        pred = int(self.model.predict([features])[0])
        self.last_score = score
        self.is_anomaly = pred == -1
        if self.is_anomaly:
            self.anomaly_count += 1

        return self.is_anomaly, score

    def status(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "score": self.last_score,
            "anomaly_count": self.anomaly_count,
            "model_fitted": self.model is not None
        }


class SLAMonitor:
    """
    Real-time SLA violation tracking with severity scoring.
    """
    def __init__(self):
        self.violations = deque(maxlen=1000)
        self.current_window_violations = 0
        self.window_start = time.time()
        self.window_seconds = 60

        # Per-metric drift detectors
        self.drift_detectors = {
            "latency": ADWINDriftDetector(delta=0.002),
            "request_rate": ADWINDriftDetector(delta=0.002),
            "error_rate": ADWINDriftDetector(delta=0.005),
        }
        self.anomaly_detector = AnomalyDetector()
        self.total_checks = 0
        self.total_violations = 0

    def check(self, metrics: dict, sla_config) -> dict:
        self.total_checks += 1
        now = time.time()
        violated = False
        violation_types = []
        severity = 0.0

        # Check each SLA dimension
        if metrics["response_time_p99"] > sla_config.p99_latency_ms:
            violated = True
            overshoot = (metrics["response_time_p99"] - sla_config.p99_latency_ms) / sla_config.p99_latency_ms
            severity = max(severity, overshoot)
            violation_types.append(f"latency({metrics['response_time_p99']:.0f}ms)")

        if metrics["error_rate"] * 100 > sla_config.error_rate_pct:
            violated = True
            violation_types.append(f"error_rate({metrics['error_rate']*100:.2f}%)")
            severity = max(severity, 0.5)

        if metrics.get("throughput", 999) < sla_config.min_throughput_rps:
            violated = True
            violation_types.append(f"throughput({metrics.get('throughput',0):.0f}rps)")
            severity = max(severity, 0.3)

        # Drift detection
        drifts = {}
        self.drift_detectors["latency"].add(metrics["response_time_p99"])
        self.drift_detectors["request_rate"].add(metrics["request_rate"])
        self.drift_detectors["error_rate"].add(metrics["error_rate"])
        for k, d in self.drift_detectors.items():
            drifts[k] = d.status()

        # Anomaly detection
        feat = np.array([
            metrics["cpu_util"], metrics["memory_util"],
            min(metrics["request_rate"] / 1000.0, 1.0),
            min(metrics["response_time_p99"] / 1000.0, 1.0),
            metrics["error_rate"]
        ], dtype=np.float32)
        is_anomaly, anomaly_score = self.anomaly_detector.update(feat)

        if violated:
            self.total_violations += 1
            self.current_window_violations += 1
            self.violations.append({
                "timestamp": now, "types": violation_types, "severity": severity
            })

        # Reset window
        if now - self.window_start > self.window_seconds:
            self.current_window_violations = 0
            self.window_start = now

        violation_rate = self.current_window_violations / max(
            (now - self.window_start), 1) / self.window_seconds

        return {
            "violated": violated,
            "violation_types": violation_types,
            "severity": round(severity, 3),
            "violation_rate": round(min(violation_rate, 1.0), 4),
            "total_violations": self.total_violations,
            "compliance_pct": round((1 - self.total_violations / max(self.total_checks, 1)) * 100, 2),
            "drifts": drifts,
            "anomaly": {"is_anomaly": is_anomaly, "score": round(anomaly_score, 4)},
        }

    def get_recent_violations(self, n: int = 20) -> list:
        return list(self.violations)[-n:]
