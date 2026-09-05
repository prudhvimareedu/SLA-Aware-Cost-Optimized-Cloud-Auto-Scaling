# SLA-Aware Cloud Auto-Scaling using Deep Reinforcement Learning

> **Research-grade, production-ready** auto-scaling system with Multi-Agent DRL,
> Transformer encoders, Safe RL, and real-time explainability.

---

## Novel Contributions

| Feature                           | Description                                         | Status    |
| --------------------------------- | --------------------------------------------------- | --------- |
| **Transformer State Encoder**     | 4-head self-attention over 60s metric windows       | completed |
| **Multi-Agent Decomposition**     | Separate CPU/MEM/NET agents + coordinator           | completed |
| **Safe RL (Lagrangian)**          | Hard SLA constraints during exploration             | completed |
| **Mixed Training**                | Behavioral Cloning pre-train → PPO online fine-tune | completed |
| **Predictive Pre-scaling**        | LSTM forecasts workload 10s ahead                   | completed |
| **ADWIN Drift Detection**         | Detects workload distribution shifts                | completed |
| **Isolation Forest Anomaly**      | Online anomaly detection with rolling refit         | completed |
| **SHAP Explainability**           | Perturbation-based feature attribution per decision | completed |
| **Prioritized Experience Replay** | TD-error weighted sampling                          | completed |
| **Cost-SLA Pareto Slider**        | Operator-adjustable α/β tradeoff in real-time       | completed |

---

## Architecture

```
Request Traffic → CloudEnvironment (M/M/c Queueing Model)
                         ↓
              MetricsStore (Ring Buffer, Thread-safe)
                         ↓
         ┌───────────────────────────────┐
         │   Transformer State Encoder   │
         │   (Multi-head Self-Attention) │
         └──────────────┬────────────────┘
                ┌───────┼───────┐
                ↓       ↓       ↓
           Agent_CPU Agent_MEM Agent_NET
                └───────┼───────┘
                        ↓
              Coordinator + Safe RL
              (Lagrangian Constraint)
                        ↓
               Scaling Decision
               + SHAP Explanation
                        ↓
         Kubernetes Controller (or simulation)
```

---

## Quick Start

```bash
# Option 1: Script (recommended)
bash scripts/run.sh

# Option 2: Manual
cd backend
pip install -r requirements.txt
python main.py

cd ../frontend
npm install && npm start
```

### With Docker Compose

```bash
docker-compose up --build
```

### With Kubernetes (minikube)

```bash
minikube start --cpus=4 --memory=4096
kubectl apply -f k8s-manifests/deployment.yaml
bash scripts/run.sh
```

---

## 📱 Frontend Tabs

| Tab            | Description                                                       |
| -------------- | ----------------------------------------------------------------- |
| **Dashboard**  | Live metrics: P99 latency, pods, CPU, RPS, error rate, cost       |
| **Simulator**  | Fire login/payment/checkout/browse/error requests with burst mode |
| **SLA Config** | Set latency/error/throughput thresholds + Pareto slider           |
| **Decisions**  | Full scaling timeline with SHAP bars, agent, confidence           |
| **Training**   | Start training, watch progress, see architecture                  |

---

## 🔌 API Reference

| Method | Endpoint                 | Description              |
| ------ | ------------------------ | ------------------------ |
| POST   | `/api/requests/send`     | Simulate request type    |
| POST   | `/api/sla/config`        | Update SLA thresholds    |
| GET    | `/api/metrics/dashboard` | Full snapshot            |
| POST   | `/api/training/start`    | Trigger offline training |
| GET    | `/api/training/status`   | Training progress        |
| POST   | `/api/scaling/manual`    | Manual scale delta       |
| WS     | `/ws/metrics`            | Live metric stream       |
| GET    | `/docs`                  | Swagger UI               |

---

## 📐 DRL Technical Details

### State Space (9 features)

- CPU utilization, Memory utilization
- Request rate (normalized), P99 latency (normalized)
- Error rate, Active pods (normalized), Queue depth
- SLA violation rate, Workload trend (slope)

### Action Space (5 discrete)

- Scale In Aggressive (-3), Scale In (-1), Hold (0)
- Scale Out (+1), Scale Out Aggressive (+3)

### Reward Function

```
R = α × SLA_score − β × cost_score − γ × violation_penalty
```

- α, β controlled by Pareto slider (operator-adjustable)
- γ = 2.0 (fixed hard penalty for violations)

### Training Pipeline

1. **Offline (BC + DQN)**: 150 episodes × 200 steps on simulated env
2. **Online (PPO)**: Continues learning from real traffic
3. **Continual**: ADWIN triggers selective replay on drift

---

## 📊 Monitoring

| Metric            | Dashboard Location                |
| ----------------- | --------------------------------- |
| P99 Latency       | Chart 1 (with SLA threshold line) |
| Active Pods       | Chart 2 (step function)           |
| CPU + Memory      | Chart 3                           |
| RPS + Throughput  | Chart 4                           |
| Error Rate + Cost | Chart 5                           |
| Anomaly Status    | Bottom left card                  |
| Drift Detectors   | Bottom middle card                |
| Scaling Timeline  | Bottom right + Decisions tab      |
