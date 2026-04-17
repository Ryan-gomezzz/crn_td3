# =============================================================================
# ddpg.py — Deep Deterministic Policy Gradient (DDPG)
#
# Implements the DDPG algorithm from Lillicrap et al., 2015:
#   "Continuous control with deep reinforcement learning"
#
# Key differences from TD3:
#   - Single critic  (no twin critics, no min-trick)
#   - No target policy smoothing noise
#   - Actor updated every step (no policy delay)
#   - Ornstein-Uhlenbeck exploration noise
# =============================================================================

from __future__ import annotations
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    STATE_DIM, ACTION_DIM, HIDDEN_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    LR_ACTOR, LR_CRITIC,
    GAMMA_DISCOUNT, TAU,
    BATCH_SIZE,
)


# =============================================================================
# Ornstein-Uhlenbeck Noise
# =============================================================================

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    Commonly used with DDPG for continuous action spaces.

    dx_t = theta*(mu - x_t)*dt + sigma*dW_t
    """

    def __init__(
        self,
        action_dim: int,
        mu:         float = 0.0,
        theta:      float = 0.15,
        sigma:      float = 0.2,
        dt:         float = 1e-2,
    ):
        self.action_dim = action_dim
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self.reset()

    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        )
        self.state = self.state + dx
        return self.state.copy()

    def set_sigma(self, sigma: float) -> None:
        self.sigma = sigma


# =============================================================================
# Neural Network Modules (identical architecture to TD3)
# =============================================================================

class DDPGActor(nn.Module):
    """
    Policy network π(s) → P_s ∈ [0, P_max].
    Same architecture as TD3 Actor for a fair comparison.
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


class DDPGCritic(nn.Module):
    """
    Single Q-value network Q(s, a) → ℝ.
    (DDPG uses one critic; TD3 uses two to reduce overestimation bias.)
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
# DDPG Agent
# =============================================================================

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent.

    Maintains:
        actor, actor_target
        critic, critic_target
        OUNoise for exploration
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
        device:       str   = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.p_max  = p_max

        # ── Actor + target ──────────────────────────────────────────────────
        self.actor        = DDPGActor(state_dim, HIDDEN_DIM, action_dim, p_max).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # ── Critic + target ─────────────────────────────────────────────────
        self.critic        = DDPGCritic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # ── Hyperparameters ──────────────────────────────────────────────────
        self.gamma = gamma
        self.tau   = tau

        # ── Exploration noise ────────────────────────────────────────────────
        self.ou_noise = OUNoise(
            action_dim=action_dim,
            sigma=0.2,
        )

        # Latest losses for logging
        self.last_critic_loss: float       = 0.0
        self.last_actor_loss:  float       = 0.0

        self._train_iterations: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(
        self,
        state:             np.ndarray,
        exploration_noise: float = 0.0,
    ) -> float:
        """
        Deterministic action from actor + optional OU exploration noise.

        Args:
            state: (STATE_DIM,) float32 numpy array
            exploration_noise: std scale multiplier for OU noise (0 = greedy)

        Returns:
            Scalar float in [0, P_max].
        """
        with torch.no_grad():
            s      = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(s).cpu().numpy().flatten()[0]

        if exploration_noise > 0.0:
            # Scale OU noise std proportional to the exploration_noise argument
            self.ou_noise.sigma = exploration_noise
            noise  = self.ou_noise.sample()[0]
            action = action + noise

        return float(np.clip(action, 0.0, self.p_max))

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        replay_buffer,
        batch_size: int = BATCH_SIZE,
    ) -> dict | None:
        """
        Perform one DDPG update.

        DDPG update algorithm:
            1. Sample mini-batch from replay buffer
            2. Compute target Q = r + γ(1-done) * Q_tgt(s', μ_tgt(s'))
            3. Update Critic via MSE loss
            4. Update Actor: minimize −Q(s, μ(s))
            5. Soft-update both target networks
        """
        if not replay_buffer.is_ready:
            return None

        self._train_iterations += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ── Step 1: Compute target Q-values ─────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_states).clamp(0.0, self.p_max)
            q_target     = rewards + self.gamma * (1.0 - dones) * self.critic_target(next_states, next_actions)

        # ── Step 2: Update Critic ────────────────────────────────────────────
        q_current = self.critic(states, actions)
        loss_c    = F.mse_loss(q_current, q_target)
        self.critic_optimizer.zero_grad()
        loss_c.backward()
        self.critic_optimizer.step()
        self.last_critic_loss = loss_c.item()

        # ── Step 3: Update Actor ─────────────────────────────────────────────
        actor_actions = self.actor(states)
        loss_a        = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        loss_a.backward()
        self.actor_optimizer.step()
        self.last_actor_loss = loss_a.item()

        # ── Step 4: Soft-update target networks ──────────────────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": self.last_critic_loss,
            "actor_loss":  self.last_actor_loss,
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

    def save(self, directory: str = "./saved_models/ddpg") -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, "ddpg_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "ddpg_critic.pth"))
        print(f"[DDPG] Models saved to '{directory}/'")

    def load(self, directory: str = "./saved_models/ddpg") -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, "ddpg_actor.pth"),  map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, "ddpg_critic.pth"), map_location=self.device)
        )
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        print(f"[DDPG] Models loaded from '{directory}/'")

    @property
    def total_steps(self) -> int:
        return self._train_iterations
