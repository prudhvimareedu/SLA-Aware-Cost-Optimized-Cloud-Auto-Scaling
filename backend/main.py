"""
SLA-Aware Auto-Scaler — FastAPI Backend
========================================
Endpoints:
  POST /api/requests/send      — simulate requests
  GET  /api/metrics/dashboard  — snapshot
  POST /api/sla/config         — update SLA config
  GET  /api/k8s/status         — cluster info
  POST /api/training/start     — trigger offline training
  GET  /api/training/status    — training progress
  WS   /ws/metrics             — live stream
"""

import asyncio
import json
import logging
import sys
import os
import time
import random
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Path setup
sys.path.insert(0, str(Path(__file__).parent))

from monitoring.metrics_store import MetricsStore, MetricPoint, ScalingDecision, SLAConfig, store as _store
from monitoring.cloud_env import CloudEnvironment, WORKLOAD_PROFILES
from monitoring.sla_monitor import SLAMonitor
from drl.agent import SLADRLController
from k8s.controller import KubernetesController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Global singletons
# ──────────────────────────────────────────────
metrics_store: MetricsStore = _store
cloud_env = CloudEnvironment()
sla_monitor = SLAMonitor()
drl_agent = SLADRLController()
k8s_ctrl = KubernetesController(simulate=True)

# Active workload mix (adjusted by request simulator)
_active_workload: Dict[str, float] = {"browse": 1.0}
_window_buffer: list = []
_loop_running = False
_training_status = {
    "phase": "not_started",
    "progress": 0.0,
    "episode": 0,
    "message": "Training not started",
    "is_trained": False
}

# WebSocket connections
_ws_clients: List[WebSocket] = []


# ──────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop_running
    _loop_running = True
    # Start background simulation loop
    asyncio.create_task(simulation_loop())
    yield
    _loop_running = False


app = FastAPI(title="SLA Auto-Scaler API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True
)


# ──────────────────────────────────────────────
# Background simulation loop
# ──────────────────────────────────────────────

async def simulation_loop():
    """Runs every second: step env, check SLA, maybe scale, push metrics."""
    global _window_buffer, _active_workload
    cloud_env.set_workload(_active_workload)
    prev_win = None
    prev_raw = None
    prev_action = 2

    while _loop_running:
        try:
            cloud_env.set_workload(_active_workload)
            state = cloud_env.step()

            # SLA check
            sla_result = sla_monitor.check(state, metrics_store.sla_config)
            state["sla_violated"] = sla_result["violated"]

            # Build window for transformer
            vec8 = np.array([
                state["cpu_util"], state["memory_util"],
                min(state["request_rate"] / 1000.0, 1.0),
                min(state["response_time_p99"] / 1000.0, 1.0),
                state["error_rate"],
                state["active_pods"] / 10.0,
                min(state["queue_depth"] / 500.0, 1.0),
                float(state["sla_violated"])
            ], dtype=np.float32)
            _window_buffer.append(vec8)
            if len(_window_buffer) > 60:
                _window_buffer.pop(0)

            win = np.zeros((60, 8), dtype=np.float32)
            buf = _window_buffer[-60:]
            win[-len(buf):] = np.array(buf)

            raw_state = metrics_store.get_state_vector()

            # DRL decision every 5 steps
            if cloud_env.time_step % 5 == 0:
                decision = drl_agent.decide(
                    win, raw_state,
                    sla_result["violation_rate"],
                    deterministic=True
                )
                current_pods = cloud_env.current_pods
                delta = decision["delta"]

                if delta != 0:
                    cloud_env.scale(delta)
                    k8s_ctrl.scale_by(delta)
                    new_pods = cloud_env.current_pods

                    # Build explanation
                    shap = decision.get("shap_values", {})
                    top_reason = max(shap, key=lambda k: abs(shap.get(k, 0)), default="load")

                    sd = ScalingDecision(
                        timestamp=time.time(),
                        action=decision["action_name"],
                        delta=delta,
                        from_pods=current_pods,
                        to_pods=new_pods,
                        reason=f"Primary driver: {top_reason} (SHAP={shap.get(top_reason, 0):.3f})",
                        agent="coordinator" if not decision.get("is_safe_override") else "safe_rl",
                        shap_values=shap,
                        predicted_workload=decision.get("predicted_rps_next_10s", 0),
                        confidence=decision.get("confidence", 0),
                        is_safe_override=decision.get("is_safe_override", False)
                    )
                    metrics_store.push_decision(sd)

                # Online learning update
                if prev_win is not None and drl_agent.is_trained:
                    reward, _, _ = cloud_env.compute_reward(state, metrics_store.sla_config)
                    drl_agent.online_update((
                        prev_win.copy(), prev_raw.copy(),
                        prev_action, reward,
                        win.copy(), raw_state.copy(), False
                    ))
                prev_win = win.copy()
                prev_raw = raw_state.copy()
                prev_action = decision.get("action_idx", 2)

            # Push metric point
            mp = MetricPoint(
                timestamp=state["timestamp"],
                cpu_util=state["cpu_util"],
                memory_util=state["memory_util"],
                request_rate=state["request_rate"],
                response_time_p50=state["response_time_p50"],
                response_time_p99=state["response_time_p99"],
                error_rate=state["error_rate"],
                active_pods=state["active_pods"],
                queue_depth=state["queue_depth"],
                throughput=state["throughput"],
                cost_per_hour=state["cost_per_hour"],
                sla_violated=state["sla_violated"],
                violation_type=", ".join(sla_result["violation_types"]) if sla_result["violation_types"] else None
            )
            metrics_store.push_metric(mp)

            # Broadcast to WebSocket clients
            payload = {
                "type": "tick",
                "metric": _metric_to_dict(mp),
                "sla_status": sla_result,
                "anomaly": sla_result.get("anomaly", {}),
                "drift": sla_result.get("drifts", {}),
                "pods": cloud_env.current_pods,
                "training": _training_status,
                "timestamp": time.time()
            }
            await broadcast(payload)

        except Exception as e:
            logger.error(f"Simulation loop error: {e}", exc_info=True)

        await asyncio.sleep(1.0)


def _metric_to_dict(mp: MetricPoint) -> dict:
    return {
        "timestamp": mp.timestamp,
        "cpu_util": round(mp.cpu_util * 100, 1),
        "memory_util": round(mp.memory_util * 100, 1),
        "request_rate": round(mp.request_rate, 1),
        "response_time_p50": round(mp.response_time_p50, 1),
        "response_time_p99": round(mp.response_time_p99, 1),
        "error_rate": round(mp.error_rate * 100, 3),
        "active_pods": mp.active_pods,
        "queue_depth": mp.queue_depth,
        "throughput": round(mp.throughput, 1),
        "cost_per_hour": round(mp.cost_per_hour, 4),
        "sla_violated": mp.sla_violated
    }


async def broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────

class RequestPayload(BaseModel):
    type: str = "browse"        # login | payment | checkout | browse | error
    rps: float = 50.0
    duration_seconds: int = 30
    burst: bool = False

class SLAPayload(BaseModel):
    p99_latency_ms: float = 200.0
    availability_pct: float = 99.9
    error_rate_pct: float = 0.1
    min_throughput_rps: float = 100.0
    max_cost_per_hour: float = 10.0
    cost_sla_weight: float = 0.7
    min_pods: int = 1
    max_pods: int = 10


@app.post("/api/requests/send")
async def send_requests(payload: RequestPayload, background_tasks: BackgroundTasks):
    global _active_workload
    if payload.type not in WORKLOAD_PROFILES:
        raise HTTPException(400, f"Unknown type {payload.type}")

    multiplier = 3.0 if payload.burst else 1.0
    _active_workload[payload.type] = (payload.rps / 100.0) * multiplier

    # Auto-decay after duration
    async def decay():
        await asyncio.sleep(payload.duration_seconds)
        _active_workload.pop(payload.type, None)
        if not _active_workload:
            _active_workload["browse"] = 0.5

    background_tasks.add_task(decay)
    return {"status": "ok", "workload": _active_workload, "burst": payload.burst}


@app.post("/api/sla/config")
async def update_sla(payload: SLAPayload):
    cfg = metrics_store.sla_config
    cfg.p99_latency_ms = payload.p99_latency_ms
    cfg.availability_pct = payload.availability_pct
    cfg.error_rate_pct = payload.error_rate_pct
    cfg.min_throughput_rps = payload.min_throughput_rps
    cfg.max_cost_per_hour = payload.max_cost_per_hour
    cfg.cost_sla_weight = payload.cost_sla_weight
    cfg.min_pods = payload.min_pods
    cfg.max_pods = payload.max_pods
    cloud_env.min_pods = payload.min_pods
    cloud_env.max_pods = payload.max_pods
    return {"status": "ok", "config": payload.dict()}


@app.get("/api/metrics/dashboard")
async def get_dashboard():
    snap = metrics_store.get_dashboard_snapshot()
    snap["k8s"] = k8s_ctrl.get_deployment_info()
    snap["anomaly"] = sla_monitor.anomaly_detector.status()
    snap["drift"] = {k: v.status() for k, v in sla_monitor.drift_detectors.items()}
    snap["training"] = _training_status
    return snap


@app.get("/api/metrics/recent")
async def get_recent(n: int = 60):
    from dataclasses import asdict
    return [asdict(m) for m in metrics_store.get_recent(n)]


@app.get("/api/k8s/status")
async def get_k8s_status():
    return k8s_ctrl.get_deployment_info()


@app.post("/api/training/start")
async def start_training(background_tasks: BackgroundTasks,
                          episodes: int = 200, steps: int = 200):
    if _training_status["phase"] in ("running_offline", "running_online"):
        return {"status": "already_running"}
    background_tasks.add_task(_run_training, episodes, steps)
    return {"status": "started", "episodes": episodes}


@app.get("/api/training/status")
async def get_training_status():
    return _training_status


@app.post("/api/scaling/manual")
async def manual_scale(delta: int):
    result = k8s_ctrl.scale_by(delta)
    cloud_env.scale(delta)
    return result


@app.get("/api/decisions/recent")
async def get_decisions(n: int = 30):
    from dataclasses import asdict
    with metrics_store._lock:
        return [asdict(d) for d in list(metrics_store.decisions)[-n:]]


@app.get("/api/workload/profiles")
async def get_profiles():
    return list(WORKLOAD_PROFILES.keys())


@app.get("/health")
async def health():
    return {"status": "ok", "trained": drl_agent.is_trained, "pods": cloud_env.current_pods}


# ──────────────────────────────────────────────
# WebSocket
# ──────────────────────────────────────────────

@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WS client connected. Total: {len(_ws_clients)}")
    try:
        # Send initial snapshot
        snap = metrics_store.get_dashboard_snapshot()
        await websocket.send_json({"type": "init", "data": snap})
        while True:
            # Keep alive ping
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        logger.info("WS client disconnected.")
    except Exception:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ──────────────────────────────────────────────
# Training background task
# ──────────────────────────────────────────────

def _run_training(episodes: int, steps: int):
    global _training_status

    def progress_cb(prog, ep, phase):
        _training_status.update({
            "phase": f"running_{phase}",
            "progress": round(prog * 100, 1),
            "episode": ep,
            "message": f"{'Offline pre-training' if phase=='offline' else 'Online fine-tuning'} — episode {ep}/{episodes} ({prog*100:.1f}%)",
            "is_trained": False
        })

    try:
        _training_status["phase"] = "running_offline"
        _training_status["message"] = "Starting offline pre-training..."
        drl_agent.pretrain_offline(
            n_episodes=episodes, steps_per_episode=steps,
            progress_callback=progress_cb
        )
        _training_status.update({
            "phase": "complete",
            "progress": 100.0,
            "message": "Training complete! Agent is live.",
            "is_trained": True
        })
        drl_agent.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sla_agent.pt"))
    except Exception as e:
        _training_status.update({
            "phase": "error", "message": str(e)
        })
        logger.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")