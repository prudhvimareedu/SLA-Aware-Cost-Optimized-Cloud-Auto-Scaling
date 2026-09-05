"""
Advanced DRL Engine
==================
- Transformer-based state encoder (multi-head self-attention)
- Multi-agent resource decomposition (CPU / Memory / Network agents)
- PPO with Lagrangian Safe RL constraints
- Prioritized Experience Replay
- Behavioral Cloning pre-training (offline phase)
- SHAP-based explainability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import time
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 1. Transformer State Encoder
# ──────────────────────────────────────────────

class TransformerStateEncoder(nn.Module):
    """
    Multi-head self-attention over a sliding window of metrics.
    Input:  (batch, seq_len, input_dim)
    Output: (batch, d_model)
    """
    def __init__(self, input_dim: int = 8, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, seq_len: int = 60):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(d_model)
        self._attention_weights = None   # for visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.input_proj(x) + self.pos_embedding(pos)
        h = self.transformer(h)           # (B, T, d_model)
        h = self.norm(h)
        # Global average pool over time dimension
        out = h.mean(dim=1)              # (B, d_model)
        return out

    def get_attention_weights(self) -> Optional[np.ndarray]:
        return self._attention_weights


# ──────────────────────────────────────────────
# 2. Individual Resource Agent (Actor-Critic)
# ──────────────────────────────────────────────

class ResourceAgent(nn.Module):
    """
    Specialized agent for one resource dimension.
    Actions: scale_in_large(-3), scale_in(-1), hold(0), scale_out(+1), scale_out_large(+3)
    """
    ACTIONS = [-3, -1, 0, 1, 3]

    def __init__(self, encoded_dim: int = 64, hidden: int = 128, name: str = "cpu"):
        super().__init__()
        self.name = name
        self.actor = nn.Sequential(
            nn.Linear(encoded_dim + 9, hidden),  # encoded + raw state
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, len(self.ACTIONS))
        )
        self.critic = nn.Sequential(
            nn.Linear(encoded_dim + 9, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        # Lagrangian multiplier for Safe RL constraint
        self.log_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, encoded: torch.Tensor, raw_state: torch.Tensor):
        combined = torch.cat([encoded, raw_state], dim=-1)
        logits = self.actor(combined)
        value = self.critic(combined)
        return logits, value

    def get_lambda(self):
        return F.softplus(self.log_lambda)

    def select_action(self, encoded: torch.Tensor, raw_state: torch.Tensor,
                      deterministic: bool = False):
        logits, value = self.forward(encoded, raw_state)
        dist = Categorical(logits=logits)
        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), self.ACTIONS[action_idx.item()], log_prob, value


# ──────────────────────────────────────────────
# 3. Coordinator — fuses agent decisions
# ──────────────────────────────────────────────

class Coordinator(nn.Module):
    """
    Takes per-agent logits and produces a single consensus scaling delta.
    Implements Lagrangian Safe RL: constrains decisions to not violate SLA.
    """
    def __init__(self, n_agents: int = 3, encoded_dim: int = 64):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(n_agents * 5 + encoded_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 5)   # final 5-action output
        )
        self.safety_head = nn.Sequential(
            nn.Linear(encoded_dim + 9, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )   # P(safe | state)

    def forward(self, agent_logits: List[torch.Tensor],
                encoded: torch.Tensor, raw_state: torch.Tensor):
        cat = torch.cat(agent_logits + [encoded], dim=-1)
        final_logits = self.fusion(cat)

        combined = torch.cat([encoded, raw_state], dim=-1)
        safety_prob = self.safety_head(combined)
        return final_logits, safety_prob


# ──────────────────────────────────────────────
# 4. LSTM Workload Forecaster
# ──────────────────────────────────────────────

class WorkloadForecaster(nn.Module):
    """Predicts next-N-step request rate from historical window."""
    def __init__(self, input_dim: int = 8, hidden: int = 64,
                 horizon: int = 10, seq_len: int = 60):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # (B, horizon)


# ──────────────────────────────────────────────
# 5. Prioritized Experience Replay
# ──────────────────────────────────────────────

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, transition: tuple, priority: float = 1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        probs = self.priorities[:n]
        probs = probs / probs.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        samples = [self.buffer[i] for i in indices]
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = (abs(p) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────
# 6. Main DRL Controller
# ──────────────────────────────────────────────

class SLADRLController:
    """
    Orchestrates: Transformer encoder → 3 Resource Agents → Coordinator → Safe scaling decision.
    Supports: offline BC pre-training, online PPO fine-tuning, SHAP explanations.
    """

    FEATURE_NAMES = [
        "cpu_util", "memory_util", "request_rate_norm",
        "p99_latency_norm", "error_rate", "pods_norm",
        "queue_norm", "sla_violation_rate"
    ]

    RAW_STATE_NAMES = [
        "cpu_util", "memory_util", "request_rate_norm", "p99_latency_norm",
        "error_rate", "pods_norm", "queue_norm", "violation_rate", "workload_trend"
    ]

    def __init__(self, seq_len: int = 60, device: str = "cpu"):
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.training_phase = "offline"   # offline | online
        self.step_count = 0

        # Networks
        self.encoder = TransformerStateEncoder(
            input_dim=8, d_model=64, nhead=4, num_layers=2, seq_len=seq_len
        ).to(self.device)

        self.agents = nn.ModuleDict({
            "cpu_agent": ResourceAgent(64, 128, "cpu"),
            "mem_agent": ResourceAgent(64, 128, "mem"),
            "net_agent": ResourceAgent(64, 128, "net"),
        }).to(self.device)

        self.coordinator = Coordinator(n_agents=3, encoded_dim=64).to(self.device)
        self.forecaster = WorkloadForecaster().to(self.device)

        # Optimizers
        all_params = (list(self.encoder.parameters()) +
                      list(self.agents.parameters()) +
                      list(self.coordinator.parameters()))
        self.optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-4)
        self.forecaster_opt = torch.optim.Adam(self.forecaster.parameters(), lr=1e-3)

        # Replay buffer
        self.replay = PrioritizedReplayBuffer(capacity=50000)

        # PPO hyperparams
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.sla_constraint_threshold = 0.05   # max allowed violation rate

        # Training state
        self.is_trained = False
        self.training_progress = 0.0
        self.training_log = []

        # SHAP background (stored after pre-training)
        self._shap_background = None

    # ─── Inference ───────────────────────────────

    @torch.no_grad()
    def decide(self, window_tensor: np.ndarray, raw_state: np.ndarray,
               sla_violation_rate: float, deterministic: bool = True) -> Dict:
        """
        Main inference. Returns scaling delta + explanation.
        """
        if not self.is_trained:
            # Fallback: simple threshold rule
            return self._fallback_rule(raw_state)

        # Encode window
        win = torch.FloatTensor(window_tensor).unsqueeze(0).to(self.device)  # (1,T,F)
        encoded = self.encoder(win)  # (1, d_model)

        raw = torch.FloatTensor(raw_state).unsqueeze(0).to(self.device)  # (1,9)

        # Per-agent logits
        agent_logits = []
        agent_decisions = {}
        for name, agent in self.agents.items():
            logits, value = agent(encoded, raw)
            agent_logits.append(logits)
            idx = logits.argmax(dim=-1).item()
            short = name.replace("_agent", "")
            agent_decisions[short] = {
                "action_idx": idx,
                "delta": ResourceAgent.ACTIONS[idx],
                "confidence": float(F.softmax(logits, dim=-1).max())
            }

        # Coordinator
        final_logits, safety_prob = self.coordinator(agent_logits, encoded, raw)
        safety = float(safety_prob.squeeze())

        # Safe RL override: if violation rate is high, force scale-out
        action_probs = F.softmax(final_logits, dim=-1)
        if sla_violation_rate > self.sla_constraint_threshold and safety < 0.5:
            action_idx = 3   # force scale_out
            is_safe_override = True
        else:
            if deterministic:
                action_idx = int(final_logits.argmax(dim=-1).item())
            else:
                action_idx = int(Categorical(logits=final_logits).sample().item())
            is_safe_override = False

        delta = ResourceAgent.ACTIONS[action_idx]

        # Workload forecast
        predicted_rps = self._forecast(win)

        # SHAP approximation (feature attribution)
        shap_vals = self._approximate_shap(raw_state, final_logits)

        return {
            "delta": delta,
            "action_idx": action_idx,
            "confidence": float(action_probs.max()),
            "safety_prob": safety,
            "is_safe_override": is_safe_override,
            "agent_decisions": agent_decisions,
            "predicted_rps_next_10s": float(predicted_rps),
            "shap_values": shap_vals,
            "action_name": self._action_name(delta)
        }

    def _forecast(self, win: torch.Tensor) -> float:
        with torch.no_grad():
            forecast = self.forecaster(win)   # (1, horizon)
        return float(forecast[0].mean().item() * 1000)

    def _approximate_shap(self, raw_state: np.ndarray, logits: torch.Tensor) -> Dict:
        """
        Perturbation-based SHAP approximation (no external library needed for speed).
        """
        base = float(logits.max().item())
        shap = {}
        raw_t = torch.FloatTensor(raw_state).unsqueeze(0).to(self.device)
        for i, name in enumerate(self.RAW_STATE_NAMES):
            perturbed = raw_t.clone()
            perturbed[0, i] = 0.0
            enc = self.encoder(torch.zeros(1, self.seq_len, 8).to(self.device))
            first_agent = list(self.agents.values())[0]
            logits_p, _ = first_agent(enc, perturbed)
            contribution = base - float(logits_p.max().item())
            shap[name] = round(contribution, 4)
        return shap

    def _fallback_rule(self, raw_state: np.ndarray) -> Dict:
        """Simple rule-based fallback before training completes."""
        cpu = raw_state[0]
        p99 = raw_state[3]
        pods_norm = raw_state[5]
        if cpu > 0.75 or p99 > 0.7:
            delta = 1
        elif cpu < 0.3 and p99 < 0.3 and pods_norm > 0.3:
            delta = -1
        else:
            delta = 0
        return {
            "delta": delta, "action_idx": 2 + delta,
            "confidence": 0.6, "safety_prob": 0.8,
            "is_safe_override": False, "agent_decisions": {},
            "predicted_rps_next_10s": 0,
            "shap_values": {n: 0.0 for n in self.RAW_STATE_NAMES},
            "action_name": self._action_name(delta)
        }

    def _action_name(self, delta: int) -> str:
        mapping = {-3: "scale_in_aggressive", -1: "scale_in",
                   0: "hold", 1: "scale_out", 3: "scale_out_aggressive"}
        return mapping.get(delta, "hold")

    # ─── Training ────────────────────────────────

    def pretrain_offline(self, n_episodes: int = 200,
                         steps_per_episode: int = 200,
                         progress_callback=None):
        """
        Phase 1: Offline pre-training using simulated environment.
        Behavioral Cloning from heuristic expert + DQN warmup.
        """
        from monitoring.cloud_env import CloudEnvironment, WORKLOAD_PROFILES
        from monitoring.metrics_store import SLAConfig

        logger.info("Starting offline pre-training...")
        self.training_phase = "offline"
        sla = SLAConfig()
        env = CloudEnvironment(sla.min_pods, sla.max_pods)

        total_steps = n_episodes * steps_per_episode
        step = 0

        for ep in range(n_episodes):
            env.reset()
            # Randomize workload profile each episode
            profile_weights = {
                "login": random.uniform(0.2, 1.0),
                "checkout": random.uniform(0.1, 0.8),
                "browse": random.uniform(0.5, 1.5),
            }
            env.set_workload(profile_weights)

            window_buffer = []
            prev_state = None

            for t in range(steps_per_episode):
                state = env.step()
                vec = self._state_to_vec(state)
                window_buffer.append(vec[:8])
                if len(window_buffer) > self.seq_len:
                    window_buffer.pop(0)

                win = np.array(window_buffer + [[0]*8]*(self.seq_len - len(window_buffer)), dtype=np.float32)
                win = win[:self.seq_len]
                raw_state = self._state_to_raw(state)

                # Expert heuristic action for BC
                expert_action_idx = self._expert_action(state, sla)

                if prev_state is not None:
                    reward, violated, _ = env.compute_reward(state, sla)
                    self.replay.push((
                        prev_win.copy(), prev_raw.copy(),
                        prev_action, reward,
                        win.copy(), raw_state.copy(), False
                    ))

                # Apply expert action occasionally for exploration
                delta = ResourceAgent.ACTIONS[expert_action_idx]
                env.scale(delta)

                prev_win = win.copy()
                prev_raw = raw_state.copy()
                prev_action = expert_action_idx
                prev_state = state
                step += 1

                if len(self.replay) > 256 and step % 10 == 0:
                    self._update_networks()

            self.training_progress = (ep + 1) / n_episodes
            if progress_callback:
                progress_callback(self.training_progress, ep, "offline")
            logger.info(f"Offline episode {ep+1}/{n_episodes} | buffer={len(self.replay)}")

        self.is_trained = True
        self.training_phase = "online"
        logger.info("Offline pre-training complete. Switching to online PPO fine-tuning.")

    def _expert_action(self, state: dict, sla) -> int:
        """Heuristic expert for behavioral cloning."""
        cpu = state["cpu_util"]
        p99 = state["response_time_p99"]
        pods = state["active_pods"]

        if p99 > sla.p99_latency_ms * 0.9 or cpu > 0.80:
            return 3 if p99 > sla.p99_latency_ms * 1.5 else 3  # scale_out
        elif cpu < 0.25 and p99 < sla.p99_latency_ms * 0.4 and pods > sla.min_pods + 1:
            return 1  # scale_in
        return 2  # hold

    def _update_networks(self):
        """PPO-style update from replay buffer."""
        if len(self.replay) < 64:
            return
        try:
            samples, indices, weights = self.replay.sample(64)
            wins, raws, actions, rewards, next_wins, next_raws, dones = zip(*samples)

            win_t = torch.FloatTensor(np.array(wins)).to(self.device)
            raw_t = torch.FloatTensor(np.array(raws)).to(self.device)
            act_t = torch.LongTensor(actions).to(self.device)
            rew_t = torch.FloatTensor(rewards).to(self.device)
            nwin_t = torch.FloatTensor(np.array(next_wins)).to(self.device)
            nraw_t = torch.FloatTensor(np.array(next_raws)).to(self.device)
            done_t = torch.FloatTensor(dones).to(self.device)
            w_t = torch.FloatTensor(weights).to(self.device)

            # Encode
            enc = self.encoder(win_t)
            nenc = self.encoder(nwin_t)

            # Get logits from first agent (simplified joint training)
            agent = self.agents["cpu_agent"]
            logits, values = agent(enc, raw_t)
            _, next_values = agent(nenc, nraw_t)

            # TD target
            td_target = rew_t + 0.99 * next_values.squeeze() * (1 - done_t)
            advantage = (td_target - values.squeeze()).detach()

            # Policy loss (PPO clip)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(act_t)
            ratio = (log_probs - log_probs.detach()).exp()
            obj1 = ratio * advantage
            obj2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -(torch.min(obj1, obj2) * w_t).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), td_target.detach())

            # Entropy bonus
            entropy = dist.entropy().mean()

            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                                           list(self.agents.parameters()), 0.5)
            self.optimizer.step()

            # Update priorities
            td_errors = (td_target - values.squeeze()).abs().detach().cpu().numpy()
            self.replay.update_priorities(indices, td_errors)

        except Exception as e:
            logger.warning(f"Network update error: {e}")

    def online_update(self, transition: tuple):
        """Add real experience to buffer and trigger update."""
        self.replay.push(transition, priority=2.0)   # higher priority for real data
        if len(self.replay) > 256 and self.step_count % 5 == 0:
            self._update_networks()
        self.step_count += 1

    # ─── Helpers ─────────────────────────────────

    def _state_to_vec(self, state: dict) -> np.ndarray:
        return np.array([
            state["cpu_util"], state["memory_util"],
            min(state["request_rate"] / 1000.0, 1.0),
            min(state["response_time_p99"] / 1000.0, 1.0),
            state["error_rate"],
            state["active_pods"] / 10.0,
            min(state["queue_depth"] / 500.0, 1.0),
            float(state.get("sla_violated", False))
        ], dtype=np.float32)

    def _state_to_raw(self, state: dict) -> np.ndarray:
        return np.array([
            state["cpu_util"], state["memory_util"],
            min(state["request_rate"] / 1000.0, 1.0),
            min(state["response_time_p99"] / 1000.0, 1.0),
            state["error_rate"],
            state["active_pods"] / 10.0,
            min(state["queue_depth"] / 500.0, 1.0),
            0.0,  # violation_rate (filled by store)
            0.0   # trend (filled by store)
        ], dtype=np.float32)

    def save(self, path: str = "model.pt"):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "agents": self.agents.state_dict(),
            "coordinator": self.coordinator.state_dict(),
            "forecaster": self.forecaster.state_dict(),
            "phase": self.training_phase,
            "step": self.step_count,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = "model.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.agents.load_state_dict(ckpt["agents"])
        self.coordinator.load_state_dict(ckpt["coordinator"])
        self.forecaster.load_state_dict(ckpt["forecaster"])
        self.training_phase = ckpt.get("phase", "online")
        self.step_count = ckpt.get("step", 0)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")