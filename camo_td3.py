# =============================================================================
# camo_td3.py — Constrained Adaptive Multi-Objective TD3 (CAMO-TD3)
#
# Extends TD3 with four novel components:
#   1. Decomposed Multi-Objective Critics — twin critics for each of 3 reward
#      components (throughput, interference penalty, energy cost)
#   2. Adaptive Lagrangian Weights — dual gradient descent to auto-tune the
#      trade-off between objectives during training
#   3. GRU Belief Encoder — recurrent encoder over the last SEQ_LEN
#      observations, producing a compact belief vector concatenated to state
#   4. Directional Exploration Noise — biased noise that nudges actions toward
#      constraint satisfaction (lower power when PU is at risk)
#
# Public API matches TD3Agent / DDPGAgent:
#   select_action(state, exploration_noise) -> float
#   train_step(replay_buffer, batch_size)   -> dict | None
#   save(directory)
#   load(directory)
# =============================================================================

from __future__ import annotations
import os
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    STATE_DIM, ACTION_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    LR_ACTOR, LR_CRITIC,
    GAMMA_DISCOUNT, TAU,
    POLICY_NOISE, NOISE_CLIP, POLICY_DELAY,
    BATCH_SIZE,
    ALPHA, BETA, GAMMA_REWARD, SINR_THRESHOLD,
    # CAMO-specific
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, BELIEF_DIM, SEQ_LEN,
    LAMBDA1_INIT, LAMBDA2_INIT, LAMBDA3_INIT, LR_LAMBDA,
    LAMBDA_MIN, LAMBDA_MAX,
    MU_BIAS_INIT, NOISE_DECAY_STEPS, VIOLATION_WINDOW,
    CAMO_HIDDEN_DIM,
)


# =============================================================================
# Sequence Replay Buffer
# =============================================================================

class SequenceReplayBuffer:
    """
    Replay buffer that stores full transitions AND maintains per-episode
    observation histories so we can retrieve the last SEQ_LEN observations
    for the GRU encoder at sample time.

    Storage layout:
        _states[i]      : state at timestep i
        _actions[i]     : action at timestep i
        _rewards_*[i]   : decomposed reward components
        _next_states[i] : next state
        _dones[i]       : done flag
        _obs_histories[i]: (SEQ_LEN, STATE_DIM) tensor of preceding observations
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        seq_len:    int = SEQ_LEN,
        max_size:   int = REPLAY_BUFFER_SIZE,
        device:     str = "cpu",
    ):
        self._max_size = max_size
        self._device   = torch.device(device)
        self._ptr      = 0
        self._size     = 0
        self._seq_len  = seq_len

        # Standard transition storage (on device)
        self._states      = torch.zeros((max_size, state_dim),  dtype=torch.float32, device=self._device)
        self._actions     = torch.zeros((max_size, action_dim), dtype=torch.float32, device=self._device)
        self._next_states = torch.zeros((max_size, state_dim),  dtype=torch.float32, device=self._device)
        self._dones       = torch.zeros((max_size, 1),          dtype=torch.float32, device=self._device)

        # Decomposed reward components
        self._rewards_throughput   = torch.zeros((max_size, 1), dtype=torch.float32, device=self._device)
        self._rewards_interference = torch.zeros((max_size, 1), dtype=torch.float32, device=self._device)
        self._rewards_energy       = torch.zeros((max_size, 1), dtype=torch.float32, device=self._device)

        # Observation histories: (max_size, seq_len, state_dim) — stored on device
        self._obs_histories      = torch.zeros((max_size, seq_len, state_dim), dtype=torch.float32, device=self._device)
        self._next_obs_histories = torch.zeros((max_size, seq_len, state_dim), dtype=torch.float32, device=self._device)

        # Episode-local rolling window (built on CPU, copied per add())
        self._current_history = deque(maxlen=seq_len)

    def reset_episode(self, initial_state: np.ndarray) -> None:
        """Call at the start of each episode to reset the observation history."""
        self._current_history.clear()
        # Pad with copies of the initial state
        for _ in range(self._seq_len):
            self._current_history.append(np.array(initial_state, dtype=np.float32))

    def add(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        r_throughput:   float,
        r_interference: float,
        r_energy:       float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        p = self._ptr

        # Build observation history tensors
        hist = np.array(list(self._current_history), dtype=np.float32)  # (seq_len, state_dim)
        self._obs_histories[p] = torch.as_tensor(hist, dtype=torch.float32)

        # Advance history with next_state for next_obs_history
        self._current_history.append(np.array(next_state, dtype=np.float32))
        next_hist = np.array(list(self._current_history), dtype=np.float32)
        self._next_obs_histories[p] = torch.as_tensor(next_hist, dtype=torch.float32)

        # Standard fields
        self._states[p]      = torch.as_tensor(state, dtype=torch.float32)
        self._actions[p]     = torch.as_tensor(action, dtype=torch.float32)
        self._next_states[p] = torch.as_tensor(next_state, dtype=torch.float32)
        self._dones[p, 0]    = float(done)

        self._rewards_throughput[p, 0]   = float(r_throughput)
        self._rewards_interference[p, 0] = float(r_interference)
        self._rewards_energy[p, 0]       = float(r_energy)

        self._ptr  = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self, batch_size: int = BATCH_SIZE):
        idx = torch.randint(0, self._size, (batch_size,), device=self._device)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards_throughput[idx],
            self._rewards_interference[idx],
            self._rewards_energy[idx],
            self._next_states[idx],
            self._dones[idx],
            self._obs_histories[idx],       # (batch, seq_len, state_dim)
            self._next_obs_histories[idx],  # (batch, seq_len, state_dim)
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= MIN_SAMPLES


# =============================================================================
# GRU Belief Encoder
# =============================================================================

class BeliefEncoder(nn.Module):
    """
    GRU-based encoder that maps a sequence of observations to a compact
    belief vector, capturing temporal channel dynamics.

    Input:  (batch, seq_len, state_dim)
    Output: (batch, belief_dim)
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        hidden_dim: int = GRU_HIDDEN_SIZE,
        num_layers: int = GRU_NUM_LAYERS,
        belief_dim: int = BELIEF_DIM,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, belief_dim)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: (batch, seq_len, state_dim)
        _, h_n = self.gru(obs_seq)              # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]                   # (batch, hidden_dim)
        return self.projection(last_hidden)     # (batch, belief_dim)


# =============================================================================
# Actor (augmented state = state + belief)
# =============================================================================

class CAMOActor(nn.Module):
    """
    Policy network: π(s, belief) -> P_s in [0, P_max].
    Takes concatenation of [state, belief_vector].
    """

    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        belief_dim: int   = BELIEF_DIM,
        hidden_dim: int   = CAMO_HIDDEN_DIM,
        action_dim: int   = ACTION_DIM,
        p_max:      float = P_MAX,
    ):
        super().__init__()
        self.p_max = p_max
        input_dim = state_dim + belief_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, belief], dim=-1)
        return self.net(x) * self.p_max


# =============================================================================
# Objective-Specific Critic
# =============================================================================

class ObjectiveCritic(nn.Module):
    """
    Q-value network for a single objective: Q_k(s, belief, a) -> R.
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        belief_dim: int = BELIEF_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = CAMO_HIDDEN_DIM,
    ):
        super().__init__()
        input_dim = state_dim + belief_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state:  torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state, belief, action], dim=-1)
        return self.net(x)


# =============================================================================
# CAMO-TD3 Agent
# =============================================================================

class CAMO_TD3Agent:
    """
    Constrained Adaptive Multi-Objective TD3.

    Decomposes the CRN reward into three objectives and learns separate
    twin critics for each. Lagrangian multipliers are adapted online via
    dual gradient descent. A GRU belief encoder captures channel dynamics.
    Directional exploration noise biases actions toward constraint safety.
    """

    def __init__(
        self,
        state_dim:    int   = STATE_DIM,
        action_dim:   int   = ACTION_DIM,
        p_max:        float = P_MAX,
        lr_actor:     float = LR_ACTOR,
        lr_critic:    float = LR_CRITIC,
        gamma:        float = GAMMA_DISCOUNT,
        tau:          float = TAU,
        policy_noise: float = POLICY_NOISE,
        noise_clip:   float = NOISE_CLIP,
        policy_delay: int   = POLICY_DELAY,
        device:       str   = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.p_max  = p_max

        # ── GRU Belief Encoder + target ──────────────────────────────────────
        self.belief_encoder        = BeliefEncoder(state_dim).to(device)
        self.belief_encoder_target = BeliefEncoder(state_dim).to(device)
        self.belief_encoder_target.load_state_dict(self.belief_encoder.state_dict())

        # ── Actor + target ───────────────────────────────────────────────────
        self.actor        = CAMOActor(state_dim, BELIEF_DIM, CAMO_HIDDEN_DIM, action_dim, p_max).to(device)
        self.actor_target = CAMOActor(state_dim, BELIEF_DIM, CAMO_HIDDEN_DIM, action_dim, p_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # ── 3 pairs of twin critics (6 total) ───────────────────────────────
        # Objective 0: throughput (R_s)
        # Objective 1: interference penalty (-beta * max(0, thr - SINR_p))
        # Objective 2: energy penalty (-gamma * P_s/P_max)
        self.critics     = nn.ModuleList()
        self.critic_tgts = nn.ModuleList()
        for _ in range(6):
            c = ObjectiveCritic(state_dim, BELIEF_DIM, action_dim, CAMO_HIDDEN_DIM).to(device)
            self.critics.append(c)
            ct = ObjectiveCritic(state_dim, BELIEF_DIM, action_dim, CAMO_HIDDEN_DIM).to(device)
            ct.load_state_dict(c.state_dict())
            self.critic_tgts.append(ct)

        # ── Optimizers ───────────────────────────────────────────────────────
        # Combine belief encoder params with actor for joint optimization
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.belief_encoder.parameters()),
            lr=lr_actor,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(),
            lr=lr_critic,
        )

        # ── Adaptive Lagrangian multipliers (softplus-parameterized) ─────────
        # Initialize so that softplus(log_lambda_k) = LAMBDA_K_INIT exactly.
        # Inverse softplus: x = log(exp(target) - 1)
        def _inv_softplus(t: float) -> float:
            # For small t, exp(t)-1 ≈ t, so inv_softplus(t) ≈ log(t)
            if t > 20.0:
                return t  # softplus(x) ≈ x for large x
            return math.log(math.expm1(t)) if t > 1e-4 else math.log(t)

        self._log_lambda1 = torch.tensor(_inv_softplus(LAMBDA1_INIT),
                                         device=device, requires_grad=True, dtype=torch.float32)
        self._log_lambda2 = torch.tensor(_inv_softplus(LAMBDA2_INIT),
                                         device=device, requires_grad=True, dtype=torch.float32)
        self._log_lambda3 = torch.tensor(_inv_softplus(LAMBDA3_INIT),
                                         device=device, requires_grad=True, dtype=torch.float32)
        self.lambda_optimizer = torch.optim.Adam(
            [self._log_lambda1, self._log_lambda2, self._log_lambda3],
            lr=LR_LAMBDA,
        )

        # ── TD3 hyperparameters ──────────────────────────────────────────────
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise * p_max
        self.noise_clip   = noise_clip   * p_max
        self.policy_delay = policy_delay

        self._train_iterations = 0

        # ── Directional exploration noise state ──────────────────────────────
        self._mu_bias         = MU_BIAS_INIT
        self._noise_step      = 0
        self._violation_window = deque(maxlen=VIOLATION_WINDOW)

        # ── Inference-time observation history ───────────────────────────────
        self._obs_history = deque(maxlen=SEQ_LEN)

        # Latest losses for logging
        self.last_critic_losses: list[float] = [0.0] * 6
        self.last_actor_loss:   float | None = None

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def lambda1(self) -> float:
        return float(F.softplus(self._log_lambda1).item())

    @property
    def lambda2(self) -> float:
        return float(F.softplus(self._log_lambda2).item())

    @property
    def lambda3(self) -> float:
        return float(F.softplus(self._log_lambda3).item())

    @property
    def total_steps(self) -> int:
        return self._train_iterations

    # ─── Episode management ──────────────────────────────────────────────────

    def reset_episode(self, initial_state: np.ndarray) -> None:
        """Reset observation history at the start of each episode."""
        self._obs_history.clear()
        for _ in range(SEQ_LEN):
            self._obs_history.append(np.array(initial_state, dtype=np.float32))

    def record_violation(self, sinr_p: float) -> None:
        """Track whether PU SINR constraint was violated this step."""
        self._violation_window.append(1 if sinr_p < SINR_THRESHOLD else 0)

    # ─── Action selection ────────────────────────────────────────────────────

    def select_action(
        self,
        state:             np.ndarray,
        exploration_noise: float = 0.0,
    ) -> float:
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Build observation history tensor
            hist = np.array(list(self._obs_history), dtype=np.float32)
            hist_t = torch.FloatTensor(hist).unsqueeze(0).to(self.device)  # (1, seq_len, state_dim)

            belief = self.belief_encoder(hist_t)  # (1, belief_dim)
            action = self.actor(s, belief).cpu().numpy().flatten()[0]

        # Update history with current observation
        self._obs_history.append(np.array(state, dtype=np.float32))

        if exploration_noise > 0.0:
            # Directional noise: Gaussian with adaptive bias
            self._noise_step += 1
            decay = max(0.0, 1.0 - self._noise_step / NOISE_DECAY_STEPS)

            # Adapt bias: increase negative bias when violations are frequent
            violation_rate = (
                np.mean(list(self._violation_window))
                if len(self._violation_window) > 0
                else 0.0
            )
            adaptive_bias = self._mu_bias * decay * (1.0 + violation_rate)

            noise = np.random.normal(adaptive_bias, exploration_noise)
            action += noise

        return float(np.clip(action, 0.0, self.p_max))

    # ─── Reward decomposition ────────────────────────────────────────────────

    @staticmethod
    def decompose_reward(info: dict) -> tuple[float, float, float]:
        """
        Decompose the scalar CRN reward into its three components using
        the same formula as environment.py.

        Returns (r_throughput, r_interference, r_energy).
        """
        sinr_s = info["sinr_s"]
        sinr_p = info["sinr_p"]
        p_s    = info["p_s"]

        r_throughput   = ALPHA * float(np.log2(1.0 + sinr_s))
        r_interference = -BETA * max(0.0, SINR_THRESHOLD - sinr_p)
        r_energy       = -GAMMA_REWARD * (p_s / P_MAX)

        return r_throughput, r_interference, r_energy

    # ─── Training step ───────────────────────────────────────────────────────

    def train_step(
        self,
        replay_buffer: SequenceReplayBuffer,
        batch_size:    int = BATCH_SIZE,
    ) -> dict | None:
        if not replay_buffer.is_ready:
            return None

        self._train_iterations += 1

        (states, actions,
         r_tput, r_intf, r_energy,
         next_states, dones,
         obs_hist, next_obs_hist) = replay_buffer.sample(batch_size)

        rewards_list = [r_tput, r_intf, r_energy]

        # ── Compute belief vectors ───────────────────────────────────────────
        belief      = self.belief_encoder(obs_hist)
        with torch.no_grad():
            belief_tgt  = self.belief_encoder_target(next_obs_hist)

        # ── Target actions with policy smoothing ─────────────────────────────
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (
                self.actor_target(next_states, belief_tgt) + noise
            ).clamp(0.0, self.p_max)

        # ── Update all 6 critics (twin for each of 3 objectives) ────────────
        total_critic_loss = torch.tensor(0.0, device=self.device)
        for obj_idx in range(3):
            c1_idx = obj_idx * 2
            c2_idx = obj_idx * 2 + 1
            reward_k = rewards_list[obj_idx]

            with torch.no_grad():
                q1_tgt = self.critic_tgts[c1_idx](next_states, belief_tgt, target_actions)
                q2_tgt = self.critic_tgts[c2_idx](next_states, belief_tgt, target_actions)
                q_target = reward_k + self.gamma * (1.0 - dones) * torch.min(q1_tgt, q2_tgt)

            q1 = self.critics[c1_idx](states, belief.detach(), actions)
            q2 = self.critics[c2_idx](states, belief.detach(), actions)
            loss_c1 = F.mse_loss(q1, q_target)
            loss_c2 = F.mse_loss(q2, q_target)

            self.last_critic_losses[c1_idx] = loss_c1.item()
            self.last_critic_losses[c2_idx] = loss_c2.item()

            total_critic_loss = total_critic_loss + loss_c1 + loss_c2

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = None

        # ── Delayed policy + lambda update ───────────────────────────────────
        if self._train_iterations % self.policy_delay == 0:
            # Recompute belief with gradient flow for actor update
            belief_actor = self.belief_encoder(obs_hist)
            actor_actions = self.actor(states, belief_actor)

            # Get Q-values for each objective from critic 1 of each pair
            lam1 = F.softplus(self._log_lambda1)
            lam2 = F.softplus(self._log_lambda2)
            lam3 = F.softplus(self._log_lambda3)

            q_tput   = self.critics[0](states, belief_actor, actor_actions)
            q_intf   = self.critics[2](states, belief_actor, actor_actions)
            q_energy = self.critics[4](states, belief_actor, actor_actions)

            # Scalarized objective: maximize lambda-weighted sum of Q-values
            # Throughput Q is positive (maximize), interference and energy Qs are negative
            actor_loss = -(lam1 * q_tput + lam2 * q_intf + lam3 * q_energy).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_val       = actor_loss.item()
            self.last_actor_loss = actor_loss_val

            # ── Lagrangian update (dual gradient descent) ────────────────────
            # Goal: increase lambda2 when interference constraint is violated,
            #       decrease when satisfied, to maintain balance
            with torch.no_grad():
                mean_intf_reward = r_intf.mean().item()
                # Constraint: interference penalty should be > some threshold
                # If mean interference reward is very negative, violations are high
                constraint_violation = -mean_intf_reward  # positive = bad

            # Update lambdas: maximize lambda * (constraint_slack)
            # Use negative because optimizer minimizes
            lambda_loss = (
                -self._log_lambda1 * r_tput.mean().detach()
                + self._log_lambda2 * constraint_violation
                - self._log_lambda3 * r_energy.mean().detach()
            )

            self.lambda_optimizer.zero_grad()
            lambda_loss.backward()
            self.lambda_optimizer.step()

            # Clamp lambdas to [LAMBDA_MIN, LAMBDA_MAX] to prevent death spiral
            # We clamp in log-space: log_lambda = log(inv_softplus(clamp(softplus(log_lambda))))
            with torch.no_grad():
                for log_lam in [self._log_lambda1, self._log_lambda2, self._log_lambda3]:
                    lam_val = F.softplus(log_lam)
                    clamped  = lam_val.clamp(LAMBDA_MIN, LAMBDA_MAX)
                    # Invert softplus: log_lam = log(exp(clamped) - 1)
                    new_log = torch.log(torch.expm1(clamped).clamp(min=1e-6))
                    log_lam.copy_(new_log)

            # Soft-update all target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.belief_encoder, self.belief_encoder_target)
            for i in range(6):
                self._soft_update(self.critics[i], self.critic_tgts[i])

        return {
            "critic_losses": self.last_critic_losses.copy(),
            "actor_loss":    actor_loss_val,
            "lambda1":       self.lambda1,
            "lambda2":       self.lambda2,
            "lambda3":       self.lambda3,
        }

    # ─── Utilities ───────────────────────────────────────────────────────────

    def _soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, directory: str = "./saved_models/camo_td3") -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   os.path.join(directory, "camo_actor.pth"))
        torch.save(self.belief_encoder.state_dict(),
                   os.path.join(directory, "camo_belief_encoder.pth"))
        torch.save(self.critics.state_dict(),
                   os.path.join(directory, "camo_critics.pth"))
        torch.save({
            "log_lambda1": self._log_lambda1,
            "log_lambda2": self._log_lambda2,
            "log_lambda3": self._log_lambda3,
        }, os.path.join(directory, "camo_lambdas.pth"))
        print(f"[CAMO-TD3] Models saved to '{directory}/'")

    def load(self, directory: str = "./saved_models/camo_td3") -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, "camo_actor.pth"),
                       map_location=self.device)
        )
        self.belief_encoder.load_state_dict(
            torch.load(os.path.join(directory, "camo_belief_encoder.pth"),
                       map_location=self.device)
        )
        self.critics.load_state_dict(
            torch.load(os.path.join(directory, "camo_critics.pth"),
                       map_location=self.device)
        )
        lam = torch.load(os.path.join(directory, "camo_lambdas.pth"),
                         map_location=self.device)
        self._log_lambda1 = lam["log_lambda1"].to(self.device).requires_grad_(True)
        self._log_lambda2 = lam["log_lambda2"].to(self.device).requires_grad_(True)
        self._log_lambda3 = lam["log_lambda3"].to(self.device).requires_grad_(True)

        # Sync targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.belief_encoder_target.load_state_dict(self.belief_encoder.state_dict())
        for i in range(6):
            self.critic_tgts[i].load_state_dict(self.critics[i].state_dict())

        print(f"[CAMO-TD3] Models loaded from '{directory}/'")
