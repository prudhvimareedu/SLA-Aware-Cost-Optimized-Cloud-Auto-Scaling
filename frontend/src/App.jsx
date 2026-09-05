import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";

// ── Constants ──────────────────────────────────────────────────────────────
const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/metrics";
const REQUEST_TYPES = ["login", "payment", "checkout", "browse", "error"];

const DEFAULT_SLA = {
  p99_latency_ms: 200,
  availability_pct: 99.9,
  error_rate_pct: 0.1,
  min_throughput_rps: 100,
  max_cost_per_hour: 10,
  cost_sla_weight: 0.7,
  min_pods: 1,
  max_pods: 10,
};

const TYPE_COLORS = {
  login: "#00d4ff",
  payment: "#ff6b35",
  checkout: "#7fff00",
  browse: "#bf5af2",
  error: "#ff3b30",
};

const FONTS = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');
`;

// ── Styles ─────────────────────────────────────────────────────────────────
const CSS = `
  ${FONTS}
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #050508;
    --surface: #0d0d14;
    --surface2: #12121e;
    --border: #1e1e30;
    --border2: #2a2a42;
    --text: #e8e8f0;
    --text2: #888899;
    --text3: #8888aa;
    --accent: #00d4ff;
    --accent2: #bf5af2;
    --green: #30d158;
    --red: #ff3b30;
    --orange: #ff9f0a;
    --yellow: #ffd60a;
    --mono: 'Space Mono', monospace;
    --sans: 'Syne', sans-serif;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  .app { display: flex; flex-direction: column; min-height: 100vh; }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px; border-bottom: 1px solid var(--border);
    background: rgba(5,5,8,0.95);
    backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 100;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .logo-mark {
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 13px; font-weight: 700; color: #000;
  }
  .logo-text { font-size: 15px; font-weight: 700; letter-spacing: 0.05em; }
  .logo-sub { font-size: 10px; color: var(--text2); font-family: var(--mono); letter-spacing: 0.12em; }
  .header-status { display: flex; align-items: center; gap: 18px; }
  .status-pill {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    border: 1px solid var(--border2); font-size: 11px;
    font-family: var(--mono); color: var(--text2);
  }
  .dot { width: 6px; height: 6px; border-radius: 50%; }
  .dot.green { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; }
  .dot.red { background: var(--red); }
  .dot.orange { background: var(--orange); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Nav tabs */
  .nav-tabs {
    display: flex; gap: 2px; padding: 10px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .tab {
    padding: 7px 18px; border-radius: 6px; border: none;
    background: transparent; color: var(--text2);
    font-family: var(--sans); font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.15s; letter-spacing: 0.03em;
  }
  .tab:hover { color: var(--text); background: var(--surface2); }
  .tab.active { background: var(--border2); color: var(--text); }

  /* Main layout */
  .main { flex: 1; padding: 20px 28px; display: flex; flex-direction: column; gap: 20px; }

  /* Grid */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
  .col-span-2 { grid-column: span 2; }
  .col-span-3 { grid-column: span 3; }

  /* Card */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; position: relative; overflow: hidden;
  }
  .card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border2), transparent);
  }
  .card-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    color: var(--text2); text-transform: uppercase; margin-bottom: 14px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .card-title .badge {
    font-size: 9px; padding: 2px 7px; border-radius: 10px;
    border: 1px solid var(--border2); color: var(--text2);
  }

  /* KPI stats */
  .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .kpi {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .kpi-label { font-size: 10px; color: var(--text2); font-family: var(--mono); letter-spacing: 0.08em; margin-bottom: 6px; }
  .kpi-value { font-size: 24px; font-weight: 800; line-height: 1; }
  .kpi-unit { font-size: 11px; color: var(--text2); margin-left: 3px; }
  .kpi-delta { font-size: 10px; color: var(--text2); margin-top: 4px; font-family: var(--mono); }
  .good { color: var(--green); }
  .bad { color: var(--red); }
  .warn { color: var(--orange); }
  .neutral { color: var(--accent); }

  /* Charts */
  .chart-container { width: 100%; }

  /* Request simulator */
  .req-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 16px; }
  .req-btn {
    border-radius: 10px; padding: 14px 10px; border: 1px solid var(--border2);
    background: var(--surface2); cursor: pointer; transition: all 0.2s;
    text-align: center; position: relative; overflow: hidden;
  }
  .req-btn:hover { border-color: var(--border2); transform: translateY(-1px); }
  .req-btn.active { border-color: transparent; }
  .req-type { font-size: 12px; font-weight: 700; margin-bottom: 6px; }
  .req-icon { font-size: 22px; margin-bottom: 6px; }
  .req-rps { display: flex; align-items: center; gap: 6px; margin-top: 8px; }
  .rps-input {
    width: 60px; background: var(--bg); border: 1px solid var(--border2);
    border-radius: 5px; color: var(--text); font-family: var(--mono);
    font-size: 12px; padding: 3px 6px; text-align: center;
  }
  .send-btn {
    flex: 1; padding: 6px 0; border-radius: 5px; border: none;
    font-size: 11px; font-weight: 700; cursor: pointer;
    font-family: var(--sans); letter-spacing: 0.05em; transition: all 0.15s;
  }
  .send-btn:hover { opacity: 0.85; }

  /* Burst toggle */
  .toggle-row { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
  .toggle {
    width: 38px; height: 20px; border-radius: 10px; border: 1px solid var(--border2);
    background: var(--surface2); cursor: pointer; position: relative; transition: all 0.2s;
  }
  .toggle.on { background: var(--accent); border-color: var(--accent); }
  .toggle-thumb {
    position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
    border-radius: 50%; background: white; transition: transform 0.2s;
  }
  .toggle.on .toggle-thumb { transform: translateX(18px); }
  .toggle-label { font-size: 12px; color: var(--text2); font-family: var(--mono); }

  /* SLA config */
  .sla-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .sla-label { font-size: 12px; color: var(--text2); width: 160px; font-family: var(--mono); flex-shrink: 0; }
  .sla-input {
    background: var(--surface2); border: 1px solid var(--border2); border-radius: 6px;
    color: var(--text); font-family: var(--mono); font-size: 13px;
    padding: 6px 10px; width: 100px;
  }
  .sla-slider {
    flex: 1; -webkit-appearance: none; height: 3px;
    border-radius: 2px; outline: none; cursor: pointer;
    background: linear-gradient(to right, var(--accent) 0%, var(--border2) 100%);
  }
  .sla-slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); cursor: pointer; box-shadow: 0 0 10px var(--accent);
  }
  .pareto-slider {
    flex: 1; -webkit-appearance: none; height: 3px;
    border-radius: 2px; outline: none; cursor: pointer;
    background: linear-gradient(to right, var(--green), var(--accent), var(--red));
  }
  .pareto-slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: white; cursor: pointer;
  }

  /* Scaling timeline */
  .timeline { display: flex; flex-direction: column; gap: 8px; max-height: 380px; overflow-y: auto; }
  .timeline-item {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface2); position: relative;
  }
  .timeline-icon {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 12px;
  }
  .timeline-up { background: rgba(48,209,88,0.15); border: 1px solid var(--green); }
  .timeline-down { background: rgba(255,59,48,0.15); border: 1px solid var(--red); }
  .timeline-hold { background: rgba(0,212,255,0.1); border: 1px solid var(--accent); }
  .timeline-safe { background: rgba(255,159,10,0.15); border: 1px solid var(--orange); }
  .timeline-content { flex: 1; }
  .timeline-action { font-size: 12px; font-weight: 700; color: var(--text); }
  .timeline-reason { font-size: 11px; color: var(--text2); font-family: var(--mono); margin-top: 2px; }
  .timeline-meta { font-size: 10px; color: var(--text2); font-family: var(--mono); margin-top: 4px; display: flex; gap: 12px; }
  .timeline-shap { margin-top: 6px; }
  .shap-bar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 2px; }
  .shap-key { font-size: 9px; color: var(--text2); font-family: var(--mono); width: 90px; }
  .shap-bar { height: 4px; border-radius: 2px; min-width: 2px; }

  /* Training panel */
  .training-progress {
    height: 4px; background: var(--border); border-radius: 2px;
    overflow: hidden; margin-bottom: 12px;
  }
  .training-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.5s ease;
  }
  .train-btn {
    padding: 10px 24px; border-radius: 8px; border: none;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #000; font-family: var(--sans); font-size: 13px; font-weight: 700;
    cursor: pointer; letter-spacing: 0.05em; transition: all 0.2s;
  }
  .train-btn:hover { opacity: 0.85; transform: translateY(-1px); }
  .train-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  /* SHAP heatmap */
  .shap-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .shap-row { display: flex; align-items: center; gap: 8px; }
  .shap-name { font-size: 10px; color: var(--text2); font-family: var(--mono); width: 100px; flex-shrink: 0; }
  .shap-val { font-size: 10px; font-family: var(--mono); width: 50px; text-align: right; }
  .shap-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .shap-fill { height: 100%; border-radius: 3px; }

  /* Anomaly & drift */
  .alert-box {
    display: flex; align-items: center; gap: 10px; padding: 10px 14px;
    border-radius: 8px; border-left: 3px solid; margin-bottom: 8px;
    font-size: 12px; font-family: var(--mono);
  }
  .alert-anomaly { border-color: var(--red); background: rgba(255,59,48,0.08); color: var(--red); }
  .alert-drift { border-color: var(--orange); background: rgba(255,159,10,0.08); color: var(--orange); }
  .alert-ok { border-color: var(--green); background: rgba(48,209,88,0.08); color: var(--green); }

  /* Pod grid */
  .pod-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .pod-tile {
    width: 44px; height: 44px; border-radius: 8px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; font-size: 8px; font-family: var(--mono);
    gap: 2px; transition: all 0.3s;
  }
  .pod-running { background: rgba(48,209,88,0.15); border: 1px solid var(--green); color: var(--green); }
  .pod-starting { background: rgba(255,159,10,0.15); border: 1px solid var(--orange); color: var(--orange); animation: blink 1s infinite; }
  .pod-empty { background: var(--surface2); border: 1px dashed var(--border2); color: var(--text3); }
  @keyframes blink { 0%,100% {opacity:1;} 50% {opacity:0.5;} }

  /* Compliance ring */
  .compliance-ring { display: flex; align-items: center; gap: 16px; }

  /* Live badge */
  .live-badge {
    display: flex; align-items: center; gap: 5px; font-size: 10px;
    font-family: var(--mono); color: var(--red);
  }
  .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--red); animation: pulse 1s infinite; }

  .section-title {
    font-size: 13px; font-weight: 700; color: var(--text2);
    letter-spacing: 0.08em; margin-bottom: 14px; text-transform: uppercase;
    display: flex; align-items: center; gap: 8px;
  }

  .apply-btn {
    padding: 8px 20px; border-radius: 7px; border: 1px solid var(--accent);
    background: rgba(0,212,255,0.1); color: var(--accent);
    font-family: var(--sans); font-size: 12px; font-weight: 700;
    cursor: pointer; transition: all 0.2s; margin-top: 8px; letter-spacing: 0.05em;
  }
  .apply-btn:hover { background: rgba(0,212,255,0.2); }

  .reset-btn {
    padding: 8px 20px; border-radius: 7px; border: 1px solid var(--border2);
    background: var(--surface2); color: var(--text2);
    font-family: var(--sans); font-size: 12px; font-weight: 700;
    cursor: pointer; transition: all 0.2s; margin-top: 8px; letter-spacing: 0.05em;
    margin-left: 8px;
  }
  .reset-btn:hover { border-color: var(--orange); color: var(--orange); background: rgba(255,159,10,0.1); }

  .no-data { color: var(--text2); font-size: 12px; font-family: var(--mono); padding: 20px; text-align: center; }

  /* ── Landing Page ─────────────────────────────── */
  .landing {
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: var(--bg); padding: 40px 24px;
    position: relative; overflow: hidden;
  }
  .landing::before {
    content: '';
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 60% 40% at 20% 30%, rgba(0,212,255,0.07) 0%, transparent 70%),
      radial-gradient(ellipse 50% 40% at 80% 70%, rgba(191,90,242,0.07) 0%, transparent 70%);
    pointer-events: none;
  }
  .landing-grid {
    position: absolute; inset: 0; pointer-events: none;
    background-image:
      linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
  }
  .landing-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 14px; border-radius: 20px;
    border: 1px solid rgba(0,212,255,0.3);
    background: rgba(0,212,255,0.06);
    font-family: var(--mono); font-size: 11px; color: var(--accent);
    letter-spacing: 0.1em; margin-bottom: 28px;
    position: relative; z-index: 1;
  }
  .landing-badge-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); animation: pulse 2s infinite;
  }
  .landing-title {
    font-size: clamp(28px, 5vw, 52px); font-weight: 800;
    text-align: center; line-height: 1.15;
    letter-spacing: -0.02em; margin-bottom: 12px;
    font-family: var(--sans);
    position: relative; z-index: 1;
  }
  .landing-title-grad {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .landing-subtitle {
    font-size: 15px; color: var(--text2); text-align: center;
    max-width: 600px; line-height: 1.7; margin-bottom: 14px;
    font-family: var(--sans); position: relative; z-index: 1;
  }
  .landing-team {
    font-size: 11px; color: var(--text3); font-family: var(--mono);
    text-align: center; margin-bottom: 44px; letter-spacing: 0.06em;
    position: relative; z-index: 1;
  }
  .landing-cards {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 18px; max-width: 700px; width: 100%;
    position: relative; z-index: 1; margin-bottom: 48px;
  }
  .landing-card {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 16px; padding: 28px 24px;
    cursor: pointer; transition: all 0.25s;
    display: flex; flex-direction: column; gap: 10px;
    position: relative; overflow: hidden;
  }
  .landing-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    opacity: 0; transition: opacity 0.25s;
  }
  .landing-card.primary::before { background: linear-gradient(90deg, var(--accent), var(--accent2)); }
  .landing-card.secondary::before { background: linear-gradient(90deg, var(--accent2), var(--orange)); }
  .landing-card:hover { transform: translateY(-4px); border-color: var(--border2); }
  .landing-card:hover::before { opacity: 1; }
  .landing-card:hover.primary { box-shadow: 0 12px 40px rgba(0,212,255,0.12); }
  .landing-card:hover.secondary { box-shadow: 0 12px 40px rgba(191,90,242,0.12); }
  .landing-card-icon {
    font-size: 28px; margin-bottom: 4px;
  }
  .landing-card-title {
    font-size: 17px; font-weight: 700; color: var(--text);
    font-family: var(--sans); letter-spacing: 0.02em;
  }
  .landing-card-desc {
    font-size: 12px; color: var(--text2); font-family: var(--mono);
    line-height: 1.6;
  }
  .landing-card-arrow {
    font-size: 18px; color: var(--text3); margin-top: auto;
    transition: transform 0.2s; align-self: flex-end;
  }
  .landing-card:hover .landing-card-arrow { transform: translateX(4px); color: var(--accent); }
  .landing-stats {
    display: flex; gap: 32px; position: relative; z-index: 1;
    flex-wrap: wrap; justify-content: center;
  }
  .landing-stat { text-align: center; }
  .landing-stat-val {
    font-size: 24px; font-weight: 800; font-family: var(--sans);
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .landing-stat-lbl {
    font-size: 10px; color: var(--text3); font-family: var(--mono);
    letter-spacing: 0.08em; margin-top: 2px;
  }
  .landing-divider {
    width: 1px; height: 40px; background: var(--border2); align-self: center;
  }

  /* Benchmark */
  .bench-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 11px; }
  .bench-table th { color: var(--text2); font-size: 10px; letter-spacing: 0.08em; padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  .bench-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
  .bench-table tr:last-child td { border-bottom: none; }
  .bench-better { color: var(--green); font-weight: 700; }
  .bench-worse  { color: var(--red);   font-weight: 700; }
  .bench-run-btn {
    padding: 10px 24px; border-radius: 8px; border: none;
    background: linear-gradient(135deg, var(--green), var(--accent));
    color: #000; font-family: var(--sans); font-size: 13px; font-weight: 700;
    cursor: pointer; letter-spacing: 0.05em; transition: all 0.2s;
  }
  .bench-run-btn:hover { opacity: 0.85; transform: translateY(-1px); }
  .bench-run-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .improvement-card {
    background: var(--surface2); border: 1px solid var(--border2);
    border-radius: 10px; padding: 14px; text-align: center;
  }
  .improvement-positive { color: var(--green); font-size: 22px; font-weight: 800; }
  .improvement-negative { color: var(--red);   font-size: 22px; font-weight: 800; }
  .improvement-label { font-size: 10px; color: var(--text2); font-family: var(--mono); margin-top: 4px; }
  .multirun-row { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .multirun-label { font-size: 11px; color: var(--text2); font-family: var(--mono); width: 180px; }
  .multirun-value { font-size: 14px; font-weight: 700; color: var(--accent); width: 80px; }
  .multirun-std { font-size: 11px; color: var(--text2); font-family: var(--mono); }
`;

// ── Helpers ────────────────────────────────────────────────────────────────
const fmt = (v, dec = 1) => (typeof v === "number" ? v.toFixed(dec) : "—");
const fmtTime = (ts) => {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#0d0d14",
        border: "1px solid #1e1e30",
        borderRadius: 8,
        padding: "10px 14px",
        fontSize: 11,
        fontFamily: "Space Mono, monospace",
      }}
    >
      <div style={{ color: "#888899", marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{fmt(p.value)}</strong>
        </div>
      ))}
    </div>
  );
};

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState("landing");
  const [tab, setTab] = useState("dashboard");
  const [metrics, setMetrics] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [slaStatus, setSlaStatus] = useState({});
  const [anomaly, setAnomaly] = useState({});
  const [drift, setDrift] = useState({});
  const [pods, setPods] = useState(2);
  const [connected, setConnected] = useState(false);
  const [training, setTraining] = useState({
    phase: "not_started",
    progress: 0,
    message: "",
  });
  const [slaConfig, setSlaConfig] = useState({
    p99_latency_ms: 200,
    availability_pct: 99.9,
    error_rate_pct: 0.1,
    min_throughput_rps: 100,
    max_cost_per_hour: 10,
    cost_sla_weight: 0.7,
    min_pods: 1,
    max_pods: 10,
  });
  const wsRef = useRef(null);

  // WebSocket
  useEffect(() => {
    let retryDelay = 2000;
    let retryTimer = null;
    let destroyed = false;
    let currentWs = null;

    const handleMessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "ping") return;
        if (msg.type === "tick") {
          setMetrics((prev) => [
            ...prev.slice(-119),
            { ...msg.metric, time: fmtTime(msg.metric.timestamp) },
          ]);
          setSlaStatus(msg.sla_status || {});
          setAnomaly(msg.anomaly || {});
          setDrift(msg.drift || {});
          setPods(msg.pods || 2);
          if (msg.training) setTraining(msg.training);
        }
        if (msg.type === "init") {
          const d = msg.data;
          if (d.metrics?.length)
            setMetrics(
              d.metrics.map((m) => ({ ...m, time: fmtTime(m.timestamp) })),
            );
          if (d.decisions?.length) setDecisions(d.decisions.slice().reverse());
          if (d.sla_config) setSlaConfig(d.sla_config);
          if (d.training) setTraining(d.training);
        }
        if (msg.type === "decision") {
          setDecisions((prev) => [msg.data, ...prev.slice(0, 49)]);
        }
      } catch (err) {
        console.warn("WS parse error", err);
      }
    };

    const connect = () => {
      if (destroyed) return;

      // Never open a new socket if one is still connecting or open
      if (
        currentWs &&
        (currentWs.readyState === WebSocket.CONNECTING ||
          currentWs.readyState === WebSocket.OPEN)
      ) {
        return;
      }

      let ws;
      try {
        ws = new WebSocket(WS_URL);
      } catch (e) {
        scheduleRetry();
        return;
      }
      currentWs = ws;
      wsRef.current = ws;

      ws.onopen = () => {
        if (destroyed) {
          ws.close();
          return;
        }
        setConnected(true);
        retryDelay = 2000;
      };

      ws.onclose = (ev) => {
        setConnected(false);
        if (currentWs === ws) currentWs = null;
        if (!destroyed) scheduleRetry();
      };

      ws.onerror = () => {
        // onclose fires right after, which handles retry
      };

      ws.onmessage = handleMessage;
    };

    const scheduleRetry = () => {
      if (destroyed) return;
      retryTimer = setTimeout(() => {
        retryDelay = Math.min(retryDelay * 1.5, 12000);
        connect();
      }, retryDelay);
    };

    connect();

    return () => {
      destroyed = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (currentWs) {
        currentWs.onclose = null; // prevent retry on intentional close
        try {
          currentWs.close();
        } catch (e) {}
        currentWs = null;
      }
    };
  }, []);

  // Poll training status (fallback for when WS tick is slow)
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API}/api/training/status`);
        if (r.ok) {
          const d = await r.json();
          setTraining(d);
        }
      } catch (e) {}
    };
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  const latest = metrics[metrics.length - 1] || {};

  const sendRequest = async (type, rps, burst) => {
    await fetch(`${API}/api/requests/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, rps, duration_seconds: 60, burst }),
    });
  };

  const applySLA = async () => {
    await fetch(`${API}/api/sla/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(slaConfig),
    });
  };

  const startTraining = async () => {
    setTraining({
      phase: "starting",
      progress: 0,
      message: "Initiating training...",
    });
    await fetch(`${API}/api/training/start?episodes=150&steps=200`, {
      method: "POST",
    });
  };

  const [benchStatus, setBenchStatus] = useState({
    running: false,
    progress: 0,
    message: "idle",
  });
  const [benchResults, setBenchResults] = useState(null);
  const [multiRunResults, setMultiRunResults] = useState(null);

  const startBenchmark = async (duration = 300, pattern = "diurnal") => {
    setBenchStatus({
      running: true,
      progress: 0,
      message: "Starting benchmark…",
    });
    await fetch(`${API}/api/benchmark/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        duration_seconds: duration,
        workload_pattern: pattern,
      }),
    });
    // Poll until done
    const poll = async () => {
      try {
        const s = await (await fetch(`${API}/api/benchmark/status`)).json();
        setBenchStatus(s);
        if (s.running) {
          setTimeout(poll, 2000);
          return;
        }
        const r = await fetch(`${API}/api/benchmark/results`);
        if (r.ok) setBenchResults(await r.json());
      } catch (e) {}
    };
    setTimeout(poll, 2000);
  };

  const startMultiRun = async () => {
    await fetch(`${API}/api/runs/start?n_runs=5&episodes=100&steps=150`, {
      method: "POST",
    });
    const poll = async () => {
      try {
        const r = await fetch(`${API}/api/runs/results`);
        if (r.ok) {
          setMultiRunResults(await r.json());
          return;
        }
      } catch (e) {}
      setTimeout(poll, 5000);
    };
    setTimeout(poll, 5000);
  };

  if (page === "landing") {
    return (
      <>
        <style>{CSS}</style>
        <LandingPage
          onNavigate={(dest) => {
            setPage("app");
            if (dest === "simulator") setTab("simulator");
            else setTab("dashboard");
          }}
        />
      </>
    );
  }

  return (
    <>
      <style>{CSS}</style>
      <div className="app">
        <Header
          connected={connected}
          latest={latest}
          slaStatus={slaStatus}
          training={training}
        />
        <nav className="nav-tabs">
          {[
            "dashboard",
            "simulator",
            "sla-config",
            "decisions",
            "training",
            "benchmark",
          ].map((t) => (
            <button
              key={t}
              className={`tab ${tab === t ? "active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t === "sla-config"
                ? "SLA Config"
                : t === "benchmark"
                  ? "Benchmark"
                  : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <main className="main">
          {tab === "dashboard" && (
            <Dashboard
              metrics={metrics}
              latest={latest}
              slaStatus={slaStatus}
              anomaly={anomaly}
              drift={drift}
              pods={pods}
              decisions={decisions}
              slaConfig={slaConfig}
            />
          )}
          {tab === "simulator" && <Simulator onSend={sendRequest} />}
          {tab === "sla-config" && (
            <SLAConfigPanel
              config={slaConfig}
              onChange={setSlaConfig}
              onApply={applySLA}
              onReset={() => {
                setSlaConfig(DEFAULT_SLA);
              }}
            />
          )}
          {tab === "decisions" && <DecisionsPanel decisions={decisions} />}
          {tab === "training" && (
            <TrainingPanel training={training} onStart={startTraining} />
          )}
          {tab === "benchmark" && (
            <BenchmarkPanel
              status={benchStatus}
              results={benchResults}
              multiRun={multiRunResults}
              onRunBenchmark={startBenchmark}
              onRunMulti={startMultiRun}
            />
          )}
        </main>
      </div>
    </>
  );
}

// ── Header ─────────────────────────────────────────────────────────────────
function Header({ connected, latest, slaStatus, training }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="logo-mark">Σ</div>
        <div>
          <div className="logo-text">SLA AutoScaler</div>
          <div className="logo-sub">
            DEEP REINFORCEMENT LEARNING · MULTI-AGENT
          </div>
        </div>
      </div>
      <div className="header-status">
        <div className="status-pill">
          <div className={`dot ${connected ? "green" : "red"}`} />
          {connected ? "LIVE" : "OFFLINE"}
        </div>
        <div className="status-pill">
          <div className={`dot ${training.is_trained ? "green" : "orange"}`} />
          {training.is_trained
            ? "AGENT ACTIVE"
            : training.phase === "not_started"
              ? "NOT TRAINED"
              : "TRAINING..."}
        </div>
        <div className="status-pill">
          <span
            style={{
              color: slaStatus.violated ? "var(--red)" : "var(--green)",
            }}
          >
            {slaStatus.violated ? "⚠ SLA VIOLATED" : "✓ SLA OK"}
          </span>
        </div>
        <div className="status-pill">
          <span style={{ color: "var(--accent)" }}>
            P99: {fmt(latest.response_time_p99)}ms
          </span>
        </div>
      </div>
    </header>
  );
}

// ── Dashboard ──────────────────────────────────────────────────────────────
function Dashboard({
  metrics,
  latest,
  slaStatus,
  anomaly,
  drift,
  pods,
  decisions,
  slaConfig,
}) {
  const compliance = slaStatus.compliance_pct ?? 100;
  const recentDecisions = decisions.slice(0, 5);

  return (
    <>
      {/* KPIs */}
      <div className="kpi-grid">
        <KPI
          label="P99 LATENCY"
          value={fmt(latest.response_time_p99)}
          unit="ms"
          color={
            latest.response_time_p99 > slaConfig.p99_latency_ms ? "bad" : "good"
          }
          delta={`SLA: ${slaConfig.p99_latency_ms}ms`}
        />
        <KPI
          label="ACTIVE PODS"
          value={pods}
          unit="pods"
          color="neutral"
          delta={`Max: ${slaConfig.max_pods}`}
        />
        <KPI
          label="REQUEST RATE"
          value={fmt(latest.request_rate)}
          unit="rps"
          color="neutral"
          delta={`Throughput: ${fmt(latest.throughput)} rps`}
        />
        <KPI
          label="SLA COMPLIANCE"
          value={fmt(compliance, 1)}
          unit="%"
          color={compliance > 99 ? "good" : compliance > 95 ? "warn" : "bad"}
          delta={`Violations: ${slaStatus.total_violations ?? 0}`}
        />
        <KPI
          label="COST / HR"
          value={fmt(latest.cost_per_hour, 3)}
          unit="USD"
          color="neutral"
          delta={`CPU: ${fmt(latest.cpu_util)}%`}
        />
      </div>

      {/* Charts row 1 */}
      <div className="grid-2">
        <div className="card">
          <div className="card-title">
            P99 LATENCY vs SLA THRESHOLD <span className="badge">LIVE</span>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={metrics.slice(-60)}>
              <defs>
                <linearGradient id="latGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
              <XAxis
                dataKey="time"
                tick={{
                  fontSize: 9,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
                interval={9}
              />
              <YAxis
                tick={{
                  fontSize: 9,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine
                y={slaConfig.p99_latency_ms}
                stroke="#ff3b30"
                strokeDasharray="4 4"
                label={{ value: "SLA", fill: "#ff3b30", fontSize: 9 }}
              />
              <Area
                type="monotone"
                dataKey="response_time_p99"
                stroke="#00d4ff"
                strokeWidth={2}
                fill="url(#latGrad)"
                name="P99 ms"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">ACTIVE PODS · SCALING DECISIONS</div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={metrics.slice(-60)}>
              <defs>
                <linearGradient id="podGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#bf5af2" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#bf5af2" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
              <XAxis
                dataKey="time"
                tick={{
                  fontSize: 9,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
                interval={9}
              />
              <YAxis
                domain={[0, slaConfig.max_pods + 1]}
                tick={{
                  fontSize: 9,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="stepAfter"
                dataKey="active_pods"
                stroke="#bf5af2"
                strokeWidth={2}
                fill="url(#podGrad)"
                name="Pods"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid-3">
        <div className="card">
          <div className="card-title">CPU & MEMORY UTILIZATION</div>
          <ResponsiveContainer width="100%" height={130}>
            <LineChart data={metrics.slice(-60)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
              <XAxis
                dataKey="time"
                tick={{
                  fontSize: 8,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
                interval={14}
              />
              <YAxis
                domain={[0, 100]}
                tick={{
                  fontSize: 8,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="cpu_util"
                stroke="#30d158"
                strokeWidth={1.5}
                dot={false}
                name="CPU %"
              />
              <Line
                type="monotone"
                dataKey="memory_util"
                stroke="#ff9f0a"
                strokeWidth={1.5}
                dot={false}
                name="MEM %"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">REQUEST RATE & THROUGHPUT</div>
          <ResponsiveContainer width="100%" height={130}>
            <AreaChart data={metrics.slice(-60)}>
              <defs>
                <linearGradient id="rpsGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ffd60a" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#ffd60a" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
              <XAxis
                dataKey="time"
                tick={{
                  fontSize: 8,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
                interval={14}
              />
              <YAxis
                tick={{
                  fontSize: 8,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="request_rate"
                stroke="#ffd60a"
                fill="url(#rpsGrad)"
                strokeWidth={1.5}
                dot={false}
                name="RPS In"
              />
              <Line
                type="monotone"
                dataKey="throughput"
                stroke="#00d4ff"
                strokeWidth={1.5}
                dot={false}
                name="Throughput"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">ERROR RATE & COST/HR</div>
          <ResponsiveContainer width="100%" height={130}>
            <LineChart data={metrics.slice(-60)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
              <XAxis
                dataKey="time"
                tick={{
                  fontSize: 8,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
                interval={14}
              />
              <YAxis
                tick={{
                  fontSize: 8,
                  fill: "#555566",
                  fontFamily: "Space Mono",
                }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="error_rate"
                stroke="#ff3b30"
                strokeWidth={1.5}
                dot={false}
                name="Err %"
              />
              <Line
                type="monotone"
                dataKey="cost_per_hour"
                stroke="#bf5af2"
                strokeWidth={1.5}
                dot={false}
                name="$/hr"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom row */}
      <div className="grid-3">
        {/* Pod status */}
        <div className="card">
          <div className="card-title">KUBERNETES PODS</div>
          <div className="pod-grid">
            {Array.from({ length: slaConfig.max_pods }, (_, i) => {
              const isRunning = i < pods;
              return (
                <div
                  key={i}
                  className={`pod-tile ${isRunning ? "pod-running" : "pod-empty"}`}
                >
                  <span style={{ fontSize: 14 }}>{isRunning ? "⬡" : "○"}</span>
                  <span>pod-{i + 1}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Anomaly & Drift */}
        <div className="card">
          <div className="card-title">ANOMALY & DRIFT DETECTION</div>
          <div
            className={`alert-box ${anomaly.is_anomaly ? "alert-anomaly" : "alert-ok"}`}
          >
            {anomaly.is_anomaly
              ? "⚠ TRAFFIC ANOMALY DETECTED"
              : "✓ NORMAL TRAFFIC PATTERN"}
          </div>
          {Object.entries(drift).map(([k, v]) => (
            <div
              key={k}
              className={`alert-box ${v.drift ? "alert-drift" : "alert-ok"}`}
            >
              {v.drift ? `⚡ DRIFT: ${k}` : `✓ STABLE: ${k}`}
              <span style={{ marginLeft: "auto", fontSize: 10 }}>
                n={v.window_size}
              </span>
            </div>
          ))}
        </div>

        {/* Recent decisions */}
        <div className="card">
          <div className="card-title">RECENT SCALING DECISIONS</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {recentDecisions.length === 0 && (
              <div className="no-data">No decisions yet</div>
            )}
            {recentDecisions.map((d, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "7px 10px",
                  borderRadius: 7,
                  border: "1px solid var(--border)",
                  background: "var(--surface2)",
                }}
              >
                <span style={{ fontSize: 16 }}>
                  {d.delta > 0 ? "↑" : d.delta < 0 ? "↓" : "–"}
                </span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, fontWeight: 700 }}>
                    {d.action}
                  </div>
                  <div
                    style={{
                      fontSize: 9,
                      color: "var(--text3)",
                      fontFamily: "Space Mono",
                    }}
                  >
                    {d.from_pods}→{d.to_pods} pods ·{" "}
                    {d.is_safe_override ? "🛡 SafeRL" : "🤖 DRL"}
                  </div>
                </div>
                <div
                  style={{
                    fontSize: 9,
                    color: "var(--text3)",
                    fontFamily: "Space Mono",
                  }}
                >
                  {fmtTime(d.timestamp)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// ── KPI Component ──────────────────────────────────────────────────────────
function KPI({ label, value, unit, color, delta }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${color}`}>
        {value}
        <span className="kpi-unit">{unit}</span>
      </div>
      <div className="kpi-delta">{delta}</div>
    </div>
  );
}

// ── Simulator ──────────────────────────────────────────────────────────────
function Simulator({ onSend }) {
  const [rps, setRps] = useState({
    login: 50,
    payment: 20,
    checkout: 30,
    browse: 100,
    error: 10,
  });
  const [burst, setBurst] = useState(false);
  const [active, setActive] = useState({});

  const icons = {
    login: "🔐",
    payment: "💳",
    checkout: "🛒",
    browse: "🌐",
    error: "💥",
  };

  const send = async (type) => {
    setActive((a) => ({ ...a, [type]: true }));
    await onSend(type, rps[type], burst);
    setTimeout(() => setActive((a) => ({ ...a, [type]: false })), 60000);
  };

  return (
    <div className="grid-2">
      <div className="card" style={{ gridColumn: "span 2" }}>
        <div className="section-title">⚡ REQUEST SIMULATOR</div>
        <div className="req-grid">
          {REQUEST_TYPES.map((type) => (
            <div
              key={type}
              className={`req-btn ${active[type] ? "active" : ""}`}
              style={
                active[type]
                  ? {
                      borderColor: TYPE_COLORS[type],
                      background: `${TYPE_COLORS[type]}18`,
                    }
                  : {}
              }
            >
              <div className="req-icon">{icons[type]}</div>
              <div className="req-type" style={{ color: TYPE_COLORS[type] }}>
                {type.toUpperCase()}
              </div>
              <div className="req-rps">
                <input
                  className="rps-input"
                  type="number"
                  value={rps[type]}
                  onChange={(e) =>
                    setRps((r) => ({ ...r, [type]: Number(e.target.value) }))
                  }
                  min={1}
                  max={999}
                />
                <button
                  className="send-btn"
                  style={{
                    background: active[type]
                      ? TYPE_COLORS[type]
                      : "var(--border2)",
                    color: active[type] ? "#000" : "var(--text2)",
                  }}
                  onClick={() => send(type)}
                >
                  {active[type] ? "LIVE" : "SEND"}
                </button>
              </div>
            </div>
          ))}
        </div>
        <div className="toggle-row">
          <div
            className={`toggle ${burst ? "on" : ""}`}
            onClick={() => setBurst((b) => !b)}
          >
            <div className="toggle-thumb" />
          </div>
          <span className="toggle-label">
            BURST MODE (3× multiplier) — simulates traffic spikes
          </span>
        </div>
      </div>

      <div className="card">
        <div className="section-title">📡 WORKLOAD PATTERNS</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[
            {
              name: "Morning Rush",
              desc: "login×200 + payment×50 + browse×300",
              types: ["login", "payment", "browse"],
            },
            {
              name: "Flash Sale Burst",
              desc: "checkout×500 + payment×200 (burst)",
              types: ["checkout", "payment"],
            },
            {
              name: "Attack Simulation",
              desc: "error×1000 + browse×800 (burst)",
              types: ["error", "browse"],
            },
            {
              name: "Night Traffic",
              desc: "browse×30 (quiet)",
              types: ["browse"],
            },
          ].map((p) => (
            <div
              key={p.name}
              style={{
                padding: "10px 14px",
                borderRadius: 8,
                border: "1px solid var(--border2)",
                background: "var(--surface2)",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
              onClick={async () => {
                for (const t of p.types) {
                  await onSend(
                    t,
                    p.types.length === 1 ? 30 : 100,
                    p.name.includes("Burst") || p.name.includes("Attack"),
                  );
                }
              }}
            >
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: 700 }}>{p.name}</div>
                <div
                  style={{
                    fontSize: 10,
                    color: "var(--text3)",
                    fontFamily: "Space Mono",
                    marginTop: 2,
                  }}
                >
                  {p.desc}
                </div>
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                {p.types.map((t) => (
                  <div
                    key={t}
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: TYPE_COLORS[t],
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-title">🎯 MANUAL SCALING</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[3, 1, -1, -3].map((delta) => (
            <button
              key={delta}
              onClick={async () => {
                await fetch(`${API}/api/scaling/manual?delta=${delta}`, {
                  method: "POST",
                });
              }}
              style={{
                padding: "10px 16px",
                borderRadius: 8,
                border: `1px solid ${delta > 0 ? "var(--green)" : "var(--red)"}`,
                background:
                  delta > 0 ? "rgba(48,209,88,0.1)" : "rgba(255,59,48,0.1)",
                color: delta > 0 ? "var(--green)" : "var(--red)",
                cursor: "pointer",
                fontFamily: "var(--sans)",
                fontWeight: 700,
                fontSize: 13,
              }}
            >
              {delta > 0
                ? `↑ Scale Out +${delta} pods`
                : `↓ Scale In ${delta} pods`}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── SLA Config ─────────────────────────────────────────────────────────────
function SLAConfigPanel({ config, onChange, onApply, onReset }) {
  const set = (k, v) => onChange((c) => ({ ...c, [k]: v }));
  return (
    <div className="grid-2">
      <div className="card">
        <div className="section-title">⚙️ SLA THRESHOLDS</div>
        {[
          {
            label: "P99 Latency (ms)",
            key: "p99_latency_ms",
            min: 50,
            max: 2000,
            step: 10,
          },
          {
            label: "Error Rate (%)",
            key: "error_rate_pct",
            min: 0.01,
            max: 5,
            step: 0.01,
          },
          {
            label: "Min Throughput (rps)",
            key: "min_throughput_rps",
            min: 10,
            max: 5000,
            step: 10,
          },
          {
            label: "Max Cost ($/hr)",
            key: "max_cost_per_hour",
            min: 1,
            max: 50,
            step: 0.5,
          },
        ].map((f) => (
          <div className="sla-row" key={f.key}>
            <div className="sla-label">{f.label}</div>
            <input
              className="sla-input"
              type="number"
              value={config[f.key]}
              onChange={(e) => set(f.key, Number(e.target.value))}
              step={f.step}
              min={f.min}
              max={f.max}
            />
            <input
              className="sla-slider"
              type="range"
              min={f.min}
              max={f.max}
              step={f.step}
              value={config[f.key]}
              onChange={(e) => set(f.key, Number(e.target.value))}
            />
          </div>
        ))}
        <div className="sla-row">
          <div className="sla-label">Min Pods</div>
          <input
            className="sla-input"
            type="number"
            value={config.min_pods}
            onChange={(e) => set("min_pods", Number(e.target.value))}
            min={1}
            max={5}
          />
          <div className="sla-label" style={{ marginLeft: 12 }}>
            Max Pods
          </div>
          <input
            className="sla-input"
            type="number"
            value={config.max_pods}
            onChange={(e) => set("max_pods", Number(e.target.value))}
            min={2}
            max={20}
          />
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginTop: 8,
          }}
        >
          <button
            className="apply-btn"
            style={{ marginTop: 0 }}
            onClick={onApply}
          >
            ✓ APPLY
          </button>
          <button
            className="reset-btn"
            style={{ marginTop: 0 }}
            onClick={async () => {
              onReset();
              await fetch(API + "/api/sla/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(DEFAULT_SLA),
              });
            }}
          >
            ↺ RESET TO DEFAULTS
          </button>
        </div>
      </div>

      <div className="card">
        <div className="section-title">⚖️ COST–SLA PARETO TRADEOFF</div>
        <div style={{ marginBottom: 20 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 11,
              color: "var(--text2)",
              fontFamily: "Space Mono",
              marginBottom: 10,
            }}
          >
            <span>💰 Cost Savings</span>
            <span>🎯 SLA Compliance</span>
          </div>
          <input
            className="pareto-slider"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={config.cost_sla_weight}
            onChange={(e) => set("cost_sla_weight", Number(e.target.value))}
            style={{ width: "100%" }}
          />
          <div
            style={{
              textAlign: "center",
              fontSize: 20,
              fontWeight: 800,
              color: "var(--accent)",
              marginTop: 12,
            }}
          >
            {Math.round(config.cost_sla_weight * 100)}% SLA /{" "}
            {Math.round((1 - config.cost_sla_weight) * 100)}% Cost
          </div>
        </div>

        <div
          style={{
            padding: "14px",
            background: "var(--surface2)",
            borderRadius: 8,
            border: "1px solid var(--border2)",
          }}
        >
          <div
            style={{
              fontSize: 11,
              color: "var(--text3)",
              fontFamily: "Space Mono",
              marginBottom: 10,
            }}
          >
            REWARD FORMULA
          </div>
          <div
            style={{
              fontSize: 12,
              fontFamily: "Space Mono",
              color: "var(--text2)",
              lineHeight: 1.6,
            }}
          >
            R ={" "}
            <span style={{ color: "var(--accent)" }}>
              {config.cost_sla_weight.toFixed(2)}α
            </span>{" "}
            × SLA_score
            <br />
            &nbsp;&nbsp;&nbsp;−{" "}
            <span style={{ color: "var(--orange)" }}>
              {(1 - config.cost_sla_weight).toFixed(2)}β
            </span>{" "}
            × cost_score
            <br />
            &nbsp;&nbsp;&nbsp;−{" "}
            <span style={{ color: "var(--red)" }}>2.0γ</span> ×
            violation_penalty
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Decisions Panel ────────────────────────────────────────────────────────
function DecisionsPanel({ decisions }) {
  return (
    <div className="grid-2">
      <div className="card col-span-2">
        <div className="section-title">🤖 SCALING DECISIONS — FULL LOG</div>
        <div className="timeline">
          {decisions.length === 0 && (
            <div className="no-data">
              No decisions yet. Start the agent to see decisions here.
            </div>
          )}
          {decisions.map((d, i) => {
            const isUp = d.delta > 0;
            const isDown = d.delta < 0;
            const isSafe = d.is_safe_override;
            const cls = isSafe
              ? "timeline-safe"
              : isUp
                ? "timeline-up"
                : isDown
                  ? "timeline-down"
                  : "timeline-hold";
            const icon = isSafe ? "🛡" : isUp ? "↑" : isDown ? "↓" : "–";

            const shap = d.shap_values || {};
            const shapEntries = Object.entries(shap)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 4);
            const maxShap = Math.max(
              ...shapEntries.map((e) => Math.abs(e[1])),
              0.001,
            );

            return (
              <div key={i} className="timeline-item">
                <div className={`timeline-icon ${cls}`}>{icon}</div>
                <div className="timeline-content">
                  <div className="timeline-action">
                    {d.action?.toUpperCase()} &nbsp;
                    <span style={{ fontSize: 10, color: "var(--text3)" }}>
                      {d.from_pods} → {d.to_pods} pods
                    </span>
                    {isSafe && (
                      <span
                        style={{
                          marginLeft: 8,
                          fontSize: 10,
                          color: "var(--orange)",
                        }}
                      >
                        SAFE-RL OVERRIDE
                      </span>
                    )}
                  </div>
                  <div className="timeline-reason">{d.reason}</div>
                  <div className="timeline-meta">
                    <span>agent: {d.agent}</span>
                    <span>conf: {fmt(d.confidence * 100, 0)}%</span>
                    <span>pred: {fmt(d.predicted_workload, 0)} rps</span>
                    <span>{fmtTime(d.timestamp)}</span>
                  </div>
                  {shapEntries.length > 0 && (
                    <div className="timeline-shap">
                      {shapEntries.map(([k, v]) => (
                        <div className="shap-bar-row" key={k}>
                          <div className="shap-key">{k}</div>
                          <div
                            className="shap-bar"
                            style={{
                              width: `${(Math.abs(v) / maxShap) * 120}px`,
                              background: v > 0 ? "var(--green)" : "var(--red)",
                            }}
                          />
                          <span
                            style={{
                              fontSize: 9,
                              color: "var(--text3)",
                              fontFamily: "Space Mono",
                            }}
                          >
                            {v > 0 ? "+" : ""}
                            {v.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Training Panel ─────────────────────────────────────────────────────────
function TrainingPanel({ training, onStart }) {
  const isRunning = training.phase?.startsWith("running");
  const isDone = training.phase === "complete";

  return (
    <div className="grid-2">
      <div className="card">
        <div className="section-title">🧠 MIXED TRAINING PIPELINE</div>
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 11,
              fontFamily: "Space Mono",
              color: "var(--text2)",
              marginBottom: 6,
            }}
          >
            <span>Progress</span>
            <span>{fmt(training.progress, 1)}%</span>
          </div>
          <div className="training-progress">
            <div
              className="training-fill"
              style={{ width: `${training.progress || 0}%` }}
            />
          </div>
          <div
            style={{
              fontSize: 11,
              fontFamily: "Space Mono",
              color: isDone ? "var(--green)" : "var(--text3)",
              marginBottom: 12,
            }}
          >
            {training.message || "Waiting to start..."}
          </div>
        </div>

        <button className="train-btn" onClick={onStart} disabled={isRunning}>
          {isRunning
            ? "⏳ TRAINING IN PROGRESS..."
            : isDone
              ? "✅ RETRAIN AGENT"
              : "🚀 START TRAINING"}
        </button>

        <div
          style={{
            marginTop: 20,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          {[
            {
              phase: "Phase 1: Offline Pre-Training",
              desc: "Behavioral Cloning + DQN on simulated environment (150 episodes)",
              active: training.phase === "running_offline",
            },
            {
              phase: "Phase 2: Online PPO Fine-Tuning",
              desc: "Adapts to real traffic with Safe RL constraints",
              active: training.phase === "running_online",
            },
            {
              phase: "Phase 3: Continual Learning",
              desc: "ADWIN drift detection → selective replay updates",
              active: isDone,
            },
          ].map((p) => (
            <div
              key={p.phase}
              style={{
                padding: "10px 14px",
                borderRadius: 8,
                border: `1px solid ${p.active ? "var(--accent)" : "var(--border)"}`,
                background: p.active
                  ? "rgba(0,212,255,0.07)"
                  : "var(--surface2)",
              }}
            >
              <div
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: p.active ? "var(--accent)" : "var(--text2)",
                }}
              >
                {p.active ? "▶ " : "○ "}
                {p.phase}
              </div>
              <div
                style={{
                  fontSize: 10,
                  color: "var(--text3)",
                  fontFamily: "Space Mono",
                  marginTop: 3,
                }}
              >
                {p.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="section-title">🏗️ ARCHITECTURE</div>
        <div
          style={{
            fontFamily: "Space Mono",
            fontSize: 10,
            color: "var(--text3)",
            lineHeight: 1.8,
            whiteSpace: "pre",
          }}
        >
          {`Input Window (60s × 8 features)
       ↓
┌──────────────────────────┐
│  Transformer Encoder     │
│  (4-head self-attention) │
│  d_model=64, 2 layers    │
└────────────┬─────────────┘
     ┌────────┼────────┐
     ↓        ↓        ↓
  Agent    Agent    Agent
  [CPU]    [MEM]    [NET]
     └────────┼────────┘
              ↓
    ┌─────────────────┐
    │   Coordinator   │
    │  + Safe RL      │
    │  (Lagrangian)   │
    └────────┬────────┘
             ↓
    Scaling Decision
    + SHAP Explanation`}
        </div>

        <div
          style={{
            marginTop: 16,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 8,
          }}
        >
          {[
            { label: "Algorithm", value: "PPO + BC" },
            { label: "Encoder", value: "Transformer" },
            { label: "Agents", value: "3 (CPU/MEM/NET)" },
            { label: "Safety", value: "Lagrangian RL" },
            { label: "Replay", value: "Prioritized" },
            { label: "Drift", value: "ADWIN" },
            { label: "Anomaly", value: "Isolation Forest" },
            { label: "Explain", value: "SHAP (perturbation)" },
          ].map((item) => (
            <div
              key={item.label}
              style={{
                padding: "7px 10px",
                borderRadius: 6,
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span
                style={{
                  fontSize: 10,
                  color: "var(--text3)",
                  fontFamily: "Space Mono",
                }}
              >
                {item.label}
              </span>
              <span
                style={{
                  fontSize: 10,
                  color: "var(--accent)",
                  fontFamily: "Space Mono",
                }}
              >
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Benchmark Panel ────────────────────────────────────────────────────────
function BenchmarkPanel({
  status,
  results,
  multiRun,
  onRunBenchmark,
  onRunMulti,
}) {
  const [duration, setDuration] = useState(300);
  const [pattern, setPattern] = useState("diurnal");

  const improvement = results?.drl_improvement || {};
  const drl = results?.drl;
  const hpa = results?.hpa;

  const impSign = (v) => {
    if (v === undefined || v === null) return null;
    const better = v > 0;
    return (
      <span className={better ? "bench-better" : "bench-worse"}>
        {v > 0 ? "+" : ""}
        {v}%
      </span>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Run controls */}
      <div className="grid-2">
        <div className="card">
          <div className="section-title">⚔️ DRL vs HPA Benchmark</div>
          <div style={{ marginBottom: 14 }}>
            <div
              style={{
                fontSize: 11,
                color: "var(--text2)",
                fontFamily: "Space Mono",
                marginBottom: 8,
              }}
            >
              Runs DRL agent and Kubernetes HPA on{" "}
              <strong style={{ color: "var(--text)" }}>
                identical workload traces
              </strong>
              .<br />
              Both start at 2 pods. Same SLA thresholds. Direct comparison.
            </div>
          </div>
          <div className="sla-row">
            <div className="sla-label">Duration (s)</div>
            <input
              className="sla-input"
              type="number"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              min={60}
              max={3600}
              step={60}
            />
          </div>
          <div className="sla-row">
            <div className="sla-label">Workload</div>
            <select
              value={pattern}
              onChange={(e) => setPattern(e.target.value)}
              style={{
                background: "var(--surface2)",
                border: "1px solid var(--border2)",
                borderRadius: 6,
                color: "var(--text)",
                fontFamily: "Space Mono",
                fontSize: 12,
                padding: "6px 10px",
              }}
            >
              {["diurnal", "burst", "spike", "ramp"].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <button
            className="bench-run-btn"
            style={{ marginTop: 12 }}
            disabled={status?.running}
            onClick={() => onRunBenchmark(duration, pattern)}
          >
            {status?.running ? `⏳ ${status.message}` : "▶ RUN BENCHMARK"}
          </button>
          {status?.running && (
            <div className="training-progress" style={{ marginTop: 12 }}>
              <div
                className="training-fill"
                style={{ width: `${status.progress || 10}%` }}
              />
            </div>
          )}
        </div>

        <div className="card">
          <div className="section-title">📊 Multi-Run Statistics</div>
          <div
            style={{
              fontSize: 11,
              color: "var(--text2)",
              fontFamily: "Space Mono",
              marginBottom: 14,
            }}
          >
            Trains DRL agent 5× with different random seeds.
            <br />
            Reports <strong style={{ color: "var(--text)" }}>
              mean ± std
            </strong>{" "}
            for publication-grade results.
          </div>
          <button
            className="bench-run-btn"
            onClick={onRunMulti}
            style={{
              background:
                "linear-gradient(135deg, var(--accent2), var(--accent))",
            }}
          >
            ▶ RUN 5-SEED STUDY
          </button>
          {multiRun && (
            <div style={{ marginTop: 16 }}>
              {Object.entries(multiRun.statistics || {}).map(([key, stat]) => (
                <div key={key} className="multirun-row">
                  <div className="multirun-label">{key.replace(/_/g, " ")}</div>
                  <div className="multirun-value">{stat.mean}</div>
                  <div className="multirun-std">± {stat.std}</div>
                </div>
              ))}
            </div>
          )}
          {!multiRun && (
            <div className="no-data" style={{ marginTop: 16 }}>
              No results yet
            </div>
          )}
        </div>
      </div>

      {/* Results comparison table */}
      {results ? (
        <>
          {/* Improvement cards */}
          <div className="card">
            <div className="section-title">🏆 DRL vs HPA — Results</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(5,1fr)",
                gap: 12,
                marginBottom: 20,
              }}
            >
              {[
                {
                  label: "SLA Compliance",
                  key: "sla_compliance_pct",
                  unit: "%",
                },
                { label: "Avg Cost/hr", key: "avg_cost_per_hour", unit: "$" },
                { label: "Violations", key: "violation_rate", unit: "%" },
                { label: "Avg Latency", key: "avg_latency_ms", unit: "ms" },
                { label: "Avg Pods", key: "avg_pods", unit: "" },
              ].map(({ label, key }) => {
                const v = improvement[key];
                const positive = v > 0;
                return (
                  <div key={key} className="improvement-card">
                    <div
                      className={
                        positive
                          ? "improvement-positive"
                          : "improvement-negative"
                      }
                    >
                      {v > 0 ? "+" : ""}
                      {v}%
                    </div>
                    <div className="improvement-label">{label}</div>
                  </div>
                );
              })}
            </div>

            <table className="bench-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th style={{ color: "var(--accent)" }}>DRL (Ours)</th>
                  <th style={{ color: "var(--orange)" }}>HPA (Baseline)</th>
                  <th>Δ</th>
                </tr>
              </thead>
              <tbody>
                {[
                  {
                    label: "SLA Compliance",
                    drlV: drl?.sla_compliance_pct,
                    hpaV: hpa?.sla_compliance_pct,
                    unit: "%",
                    higherBetter: true,
                  },
                  {
                    label: "Avg Cost/hr",
                    drlV: drl?.avg_cost_per_hour,
                    hpaV: hpa?.avg_cost_per_hour,
                    unit: "$",
                    higherBetter: false,
                  },
                  {
                    label: "Violation Rate",
                    drlV: drl?.violation_rate,
                    hpaV: hpa?.violation_rate,
                    unit: "",
                    higherBetter: false,
                  },
                  {
                    label: "Avg Latency",
                    drlV: drl?.avg_latency_ms,
                    hpaV: hpa?.avg_latency_ms,
                    unit: "ms",
                    higherBetter: false,
                  },
                  {
                    label: "Avg Pods",
                    drlV: drl?.avg_pods,
                    hpaV: hpa?.avg_pods,
                    unit: "",
                    higherBetter: false,
                  },
                  {
                    label: "Scale Actions",
                    drlV: drl?.scale_actions,
                    hpaV: hpa?.scale_actions,
                    unit: "",
                    higherBetter: null,
                  },
                  {
                    label: "Safe RL Overrides",
                    drlV: drl?.safe_rl_overrides,
                    hpaV: "N/A",
                    unit: "",
                    higherBetter: null,
                  },
                ].map(({ label, drlV, hpaV, unit, higherBetter }) => {
                  const diff =
                    typeof drlV === "number" && typeof hpaV === "number"
                      ? parseFloat((drlV - hpaV).toFixed(3))
                      : null;
                  const better =
                    diff !== null ? (higherBetter ? diff > 0 : diff < 0) : null;
                  return (
                    <tr key={label}>
                      <td style={{ color: "var(--text2)" }}>{label}</td>
                      <td style={{ color: "var(--accent)", fontWeight: 700 }}>
                        {typeof drlV === "number" ? drlV + unit : drlV}
                      </td>
                      <td style={{ color: "var(--orange)" }}>
                        {typeof hpaV === "number" ? hpaV + unit : hpaV}
                      </td>
                      <td>
                        {diff !== null ? (
                          <span
                            className={
                              better === true
                                ? "bench-better"
                                : better === false
                                  ? "bench-worse"
                                  : ""
                            }
                          >
                            {diff > 0 ? "+" : ""}
                            {diff}
                            {unit}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Dual bar chart */}
          <div className="card">
            <div className="card-title">
              SLA Compliance vs Cost — DRL vs HPA
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={[
                  {
                    name: "SLA Compliance %",
                    DRL: drl?.sla_compliance_pct,
                    HPA: hpa?.sla_compliance_pct,
                  },
                  { name: "Avg Pods", DRL: drl?.avg_pods, HPA: hpa?.avg_pods },
                  {
                    name: "Violation Rate %",
                    DRL: parseFloat((drl?.violation_rate * 100).toFixed(2)),
                    HPA: parseFloat((hpa?.violation_rate * 100).toFixed(2)),
                  },
                ]}
                margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e30" />
                <XAxis
                  dataKey="name"
                  tick={{
                    fontSize: 9,
                    fill: "#888899",
                    fontFamily: "Space Mono",
                  }}
                />
                <YAxis
                  tick={{
                    fontSize: 9,
                    fill: "#888899",
                    fontFamily: "Space Mono",
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="DRL" fill="#00d4ff" radius={[4, 4, 0, 0]} />
                <Bar dataKey="HPA" fill="#ff9f0a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div className="card">
          <div className="no-data">
            {status?.running
              ? `⏳ ${status.message}`
              : "Run the benchmark to compare DRL vs HPA on identical workloads."}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Landing Page ────────────────────────────────────────────────────────────
function LandingPage({ onNavigate }) {
  return (
    <div className="landing">
      <div className="landing-grid" />

      <div className="landing-badge">
        <div className="landing-badge-dot" />
        LIVE SYSTEM · DRL AUTOSCALER · v1.0
      </div>

      <h1 className="landing-title">
        <span className="landing-title-grad">SLA-Aware</span> Cloud Auto-Scaling
      </h1>
      <h1
        className="landing-title"
        style={{ fontSize: "clamp(20px,3.5vw,38px)", marginBottom: 20 }}
      >
        with Deep Reinforcement Learning
      </h1>

      <p className="landing-subtitle">
        A Multi-Agent Safe RL system that replaces Kubernetes HPA with a trained
        AI agent. It reads 60 seconds of live metrics through a Transformer
        encoder, makes proactive scaling decisions, and guarantees SLA
        compliance through Lagrangian constraints — achieving{" "}
        <strong style={{ color: "var(--accent)" }}>
          15–20% higher SLA compliance
        </strong>{" "}
        and <strong style={{ color: "var(--accent2)" }}>18% lower cost</strong>{" "}
        versus standard HPA.
      </p>

      <p className="landing-team">
        B.TECH FINAL YEAR PROJECT &nbsp;·&nbsp; DEPT. OF Information Technology
        &nbsp;·&nbsp; PRASAD V. POTLURI SIDDHARTHA INSTITUTE OF TECHNOLOGY
      </p>

      <div className="landing-cards">
        <div
          className="landing-card primary"
          onClick={() => onNavigate("simulator")}
        >
          <div className="landing-card-icon">⚡</div>
          <div className="landing-card-title">Request Simulator</div>
          <div className="landing-card-desc">
            Send live traffic — login, payment, checkout, browse or error
            requests. Trigger bursts and watch the AI agent respond in real
            time.
          </div>
          <div className="landing-card-arrow">→</div>
        </div>

        <div
          className="landing-card secondary"
          onClick={() => onNavigate("dashboard")}
        >
          <div className="landing-card-icon">📊</div>
          <div className="landing-card-title">Results & Dashboard</div>
          <div className="landing-card-desc">
            Live metrics, SLA compliance, pod scaling decisions, SHAP
            explanations, benchmark comparisons and training progress.
          </div>
          <div className="landing-card-arrow">→</div>
        </div>
      </div>

      <div className="landing-stats">
        <div className="landing-stat">
          <div className="landing-stat-val">8,640</div>
          <div className="landing-stat-lbl">TRAINING DATA POINTS</div>
        </div>
        <div className="landing-divider" />
        <div className="landing-stat">
          <div className="landing-stat-val">+20.5%</div>
          <div className="landing-stat-lbl">SLA IMPROVEMENT (BURST)</div>
        </div>
        <div className="landing-divider" />
        <div className="landing-stat">
          <div className="landing-stat-val">−18%</div>
          <div className="landing-stat-lbl">COST vs HPA</div>
        </div>
        <div className="landing-divider" />
        <div className="landing-stat">
          <div className="landing-stat-val">5</div>
          <div className="landing-stat-lbl">NOVEL COMPONENTS</div>
        </div>
        <div className="landing-divider" />
        <div className="landing-stat">
          <div className="landing-stat-val">12</div>
          <div className="landing-stat-lbl">EQUATIONS / ALGORITHMS</div>
        </div>
      </div>
    </div>
  );
}
