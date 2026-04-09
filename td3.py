# =============================================================================
# td3.py — Twin Delayed Deep Deterministic Policy Gradient (TD3)
#
# Implements the TD3 algorithm from Fujimoto et al., 2018:
#   "Addressing Function Approximation Error in Actor-Critic Methods"
#
# Key TD3 features implemented:
#   1. Twin critics  — two independent Q-networks; use min for target
#   2. Delayed policy updates — update actor every POLICY_DELAY critic steps
#   3. Target policy smoothing — add clipped Gaussian noise to target actions
#   4. Soft target updates  — Polyak averaging with tau
# =============================================================================

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    STATE_DIM, ACTION_DIM, HIDDEN_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    LR_ACTOR, LR_CRITIC,
    GAMMA_DISCOUNT, TAU,
    POLICY_NOISE, NOISE_CLIP, POLICY_DELAY,
    BATCH_SIZE,
)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Circular replay buffer storing (s, a, r, s', done) transitions.
    Pre-allocates numpy arrays for efficiency — no list.append overhead.
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        max_size:   int = REPLAY_BUFFER_SIZE,
        device:     str = "cpu",
    ):
        self._max_size  = max_size
        self._device    = device
        self._ptr       = 0    # write pointer (circular)
        self._size      = 0    # number of valid entries

        # Pre-allocate storage (float32 throughout)
        self._states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self._actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self._rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self._next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self._dones       = np.zeros((max_size, 1),          dtype=np.float32)

    def add(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._states[self._ptr]      = state.astype(np.float32)
        self._actions[self._ptr]     = action.astype(np.float32)
        self._rewards[self._ptr]     = float(reward)
        self._next_states[self._ptr] = next_state.astype(np.float32)
        self._dones[self._ptr]       = float(done)

        self._ptr  = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self, batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch.
        Returns (states, actions, rewards, next_states, dones) as float32 tensors.
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        to_t = lambda arr: torch.FloatTensor(arr[idx]).to(self._device)

        return (
            to_t(self._states),
            to_t(self._actions),
            to_t(self._rewards),
            to_t(self._next_states),
            to_t(self._dones),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= MIN_SAMPLES


# =============================================================================
# Neural Network Modules
# =============================================================================

class Actor(nn.Module):
    """
    Policy network π(s) → P_s ∈ [0, P_max].

    Architecture:
        Linear(state_dim → hidden_dim) → ReLU
        Linear(hidden_dim → hidden_dim) → ReLU
        Linear(hidden_dim → action_dim) → Sigmoid × P_max

    Sigmoid ensures output is always in (0, P_max) without explicit clipping.
    """

    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        hidden_dim: int   = HIDDEN_DIM,
        action_dim: int   = ACTION_DIM,
        p_max:      float = P_MAX,
    ):
        super().__init__()
        self.p_max = p_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.p_max


class Critic(nn.Module):
    """
    Q-value network Q(s, a) → ℝ.

    Input: concatenation of [state, action]  (shape: batch × (state_dim + action_dim))

    Architecture:
        Linear(state_dim + action_dim → hidden_dim) → ReLU
        Linear(hidden_dim → hidden_dim) → ReLU
        Linear(hidden_dim → 1)
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# =============================================================================
# TD3 Agent
# =============================================================================

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient agent.

    Maintains:
        actor, actor_target
        critic1, critic1_target
        critic2, critic2_target
        ... and their Adam optimizers.
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

        # ── Actor + target ──────────────────────────────────────────────────
        self.actor        = Actor(state_dim, HIDDEN_DIM, action_dim, p_max).to(device)
        self.actor_target = Actor(state_dim, HIDDEN_DIM, action_dim, p_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # ── Critic 1 + target ───────────────────────────────────────────────
        self.critic1        = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.critic1_target = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)

        # ── Critic 2 + target ───────────────────────────────────────────────
        self.critic2        = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.critic2_target = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # ── TD3 hyperparameters ─────────────────────────────────────────────
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise * p_max   # scale noise to action range
        self.noise_clip   = noise_clip   * p_max   # scale clip  to action range
        self.policy_delay = policy_delay

        # Internal step counter (used for delayed policy updates)
        self._train_iterations = 0

        # Latest losses for GUI display
        self.last_critic1_loss: float = 0.0
        self.last_critic2_loss: float = 0.0
        self.last_actor_loss:   float | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(
        self,
        state:             np.ndarray,
        exploration_noise: float = 0.0,
    ) -> float:
        """
        Deterministic action from actor + optional Gaussian exploration noise.

        Args:
            state: (STATE_DIM,) float32 numpy array
            exploration_noise: std of Gaussian noise added for exploration

        Returns:
            Scalar float in [0, P_max].
        """
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(s).cpu().numpy().flatten()[0]

        if exploration_noise > 0.0:
            action += np.random.normal(0.0, exploration_noise)

        return float(np.clip(action, 0.0, self.p_max))

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        replay_buffer: ReplayBuffer,
        batch_size:    int = BATCH_SIZE,
    ) -> dict | None:
        """
        Perform one TD3 update.

        Returns a dict with loss values, or None if buffer isn't ready yet.

        TD3 update algorithm:
            1. Sample mini-batch from replay buffer
            2. Compute target actions with clipped noise (target policy smoothing)
            3. Compute target Q = r + γ(1-done) * min(Q1_tgt, Q2_tgt)
            4. Update Critic 1 and Critic 2 via MSE loss
            5. Every POLICY_DELAY steps:
               a. Update actor: minimize -Q1(s, π(s))
               b. Soft-update all four target networks
        """
        if not replay_buffer.is_ready:
            return None

        self._train_iterations += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ── Step 1: Compute target actions with target policy smoothing ──────
        with torch.no_grad():
            # Add clipped Gaussian noise to smooth the target policy
            noise = torch.FloatTensor(actions.shape).normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)   # clip noise first

            target_actions = (self.actor_target(next_states) + noise).clamp(0.0, self.p_max)

            # ── Step 2: Compute target Q-values ─────────────────────────────
            q1_target = self.critic1_target(next_states, target_actions)
            q2_target = self.critic2_target(next_states, target_actions)
            q_target  = rewards + self.gamma * (1.0 - dones) * torch.min(q1_target, q2_target)

        # ── Step 3: Update Critic 1 ──────────────────────────────────────────
        q1_current = self.critic1(states, actions)
        loss_c1    = F.mse_loss(q1_current, q_target)
        self.critic1_optimizer.zero_grad()
        loss_c1.backward()
        self.critic1_optimizer.step()

        # ── Step 4: Update Critic 2 ──────────────────────────────────────────
        q2_current = self.critic2(states, actions)
        loss_c2    = F.mse_loss(q2_current, q_target)
        self.critic2_optimizer.zero_grad()
        loss_c2.backward()
        self.critic2_optimizer.step()

        self.last_critic1_loss = loss_c1.item()
        self.last_critic2_loss = loss_c2.item()
        actor_loss_val         = None

        # ── Step 5: Delayed policy update ────────────────────────────────────
        if self._train_iterations % self.policy_delay == 0:
            # Actor loss: maximize Q1(s, π(s))  →  minimize -Q1(s, π(s))
            actor_actions = self.actor(states)
            actor_loss    = -self.critic1(states, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_val        = actor_loss.item()
            self.last_actor_loss  = actor_loss_val

            # Soft-update all target networks (Polyak averaging)
            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": self.last_critic1_loss,
            "critic2_loss": self.last_critic2_loss,
            "actor_loss":   actor_loss_val,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _soft_update(self, net: nn.Module, target_net: nn.Module) -> None:
        """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, directory: str = "./saved_models") -> None:
        """Save actor and critic network weights to disk."""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),   os.path.join(directory, "actor.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(directory, "critic1.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(directory, "critic2.pth"))
        print(f"[TD3] Models saved to '{directory}/'")

    def load(self, directory: str = "./saved_models") -> None:
        """Load actor and critic weights from disk."""
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, "actor.pth"),   map_location=self.device)
        )
        self.critic1.load_state_dict(
            torch.load(os.path.join(directory, "critic1.pth"), map_location=self.device)
        )
        self.critic2.load_state_dict(
            torch.load(os.path.join(directory, "critic2.pth"), map_location=self.device)
        )
        # Sync target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        print(f"[TD3] Models loaded from '{directory}/'")

    @property
    def total_steps(self) -> int:
        return self._train_iterations
