#!/bin/bash
# ============================================================
# SLA-Aware Auto-Scaler — Setup & Run Script
# ============================================================
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner() { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════${NC}"; echo -e "${CYAN}${BOLD}  $1${NC}"; echo -e "${CYAN}${BOLD}══════════════════════════════════════════${NC}\n"; }
ok() { echo -e "${GREEN}✔ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
info() { echo -e "  $1"; }

banner "SLA-Aware Cloud Auto-Scaler"
info "Deep Reinforcement Learning | Multi-Agent | Safe RL"
info "Transformer Encoder | SHAP Explainability | ADWIN Drift"

# ── 1. Check Python ───────────────────────────────────────
banner "Checking Python"
if ! command -v python3 &>/dev/null; then
  echo -e "${RED}Python 3 not found. Install Python 3.9+${NC}"; exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "Python $PY_VER found"

# ── 2. Backend setup ──────────────────────────────────────
banner "Setting up Backend"
cd backend
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Backend dependencies installed"

# Create __init__.py files
touch monitoring/__init__.py drl/__init__.py k8s/__init__.py api/__init__.py
ok "Package structure ready"

# ── 3. Kubernetes check ───────────────────────────────────
banner "Kubernetes Check"
if command -v minikube &>/dev/null; then
  if minikube status 2>/dev/null | grep -q "Running"; then
    ok "Minikube is running"
    kubectl apply -f ../k8s-manifests/deployment.yaml 2>/dev/null && ok "K8s manifests applied"
    export K8S_SIMULATE=false
  else
    warn "Minikube installed but not running. Starting simulation mode."
    warn "To start minikube: minikube start --cpus=4 --memory=4096"
    export K8S_SIMULATE=true
  fi
else
  warn "Minikube not found — running in K8s simulation mode"
  warn "Install: https://minikube.sigs.k8s.io/docs/start/"
  export K8S_SIMULATE=true
fi

# ── 4. Start backend ──────────────────────────────────────
banner "Starting Backend (FastAPI)"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
ok "Backend started (PID: $BACKEND_PID)"
sleep 3

# Health check
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
  ok "Backend health check passed"
else
  warn "Backend may still be starting..."
fi

# ── 5. Frontend setup ─────────────────────────────────────
cd ../frontend
banner "Setting up Frontend"
if ! command -v node &>/dev/null; then
  warn "Node.js not found. Install Node.js 18+ from https://nodejs.org"
  warn "Frontend will not start. Backend is running at http://localhost:8000"
else
  npm install --silent
  ok "Frontend dependencies installed"
  npm start &
  FRONTEND_PID=$!
  ok "Frontend started (PID: $FRONTEND_PID)"
fi

# ── 6. Summary ────────────────────────────────────────────
banner "System Ready 🚀"
echo -e "${GREEN}${BOLD}Services:${NC}"
echo -e "  🌐 Frontend:  ${CYAN}http://localhost:3000${NC}"
echo -e "  🔧 Backend:   ${CYAN}http://localhost:8000${NC}"
echo -e "  📡 API Docs:  ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  🔌 WebSocket: ${CYAN}ws://localhost:8000/ws/metrics${NC}"
echo ""
echo -e "${YELLOW}${BOLD}Quick Start:${NC}"
echo -e "  1. Open ${CYAN}http://localhost:3000${NC}"
echo -e "  2. Go to 'Training' tab → Click 'START TRAINING'"
echo -e "  3. Go to 'Simulator' tab → Send requests"
echo -e "  4. Watch Dashboard for live metrics & scaling decisions"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"

# Keep alive
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
