# =============================================================================
# environment.py — CRN Environment (Gym-like, no gymnasium dependency)
#
# System model:
#   - 4 nodes: Primary Transmitter (PT), Primary Receiver (PR),
#              Secondary Transmitter (ST), Secondary Receiver (SR)
#   - Nakagami-m fading: |h|^2 ~ Gamma(m, Omega/m), drawn fresh every time step
#     (m=1 recovers Rayleigh/Exponential exactly)
#   - PT transmits at fixed power P_p; ST power P_s is the TD3 action
#   - SINR_p = (P_p * h_pp^2) / (P_s * h_sp^2 + sigma^2)
#   - SINR_s = (P_s * h_ss^2) / (P_p * h_ps^2 + sigma^2)
#   - Reward  = alpha*R_s - beta*max(0, thr - SINR_p) - gamma*(P_s/P_max)
# =============================================================================

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import (
    SIGMA2, P_P, P_MAX, SINR_THRESHOLD,
    ALPHA, BETA, GAMMA_REWARD,
    STATE_DIM, STEPS_PER_EPISODE,
    NAKAGAMI_M, NAKAGAMI_OMEGA,
)


@dataclass
class StepResult:
    state:  np.ndarray   # shape (7,) — next observation
    reward: float
    done:   bool
    info:   dict         # sinr_p, sinr_s, r_s, p_s, h_pp, h_sp, h_ss, h_ps


class CRNEnvironment:
    """
    Cognitive Radio Network environment.

    State vector (7-dimensional):
        [h_pp^2, h_sp^2, h_ss^2, h_ps^2, SINR_p, SINR_s, P_s_prev]

    Action:
        P_s — scalar float in [0, P_max], chosen by the TD3 agent.

    Reward:
        r = alpha * R_s
            - beta  * max(0, SINR_threshold - SINR_p)   # constraint violation penalty
            - gamma * (P_s / P_max)                      # energy efficiency penalty
    """

    def __init__(
        self,
        p_max:             float = P_MAX,
        p_p:               float = P_P,
        sigma2:            float = SIGMA2,
        sinr_threshold:    float = SINR_THRESHOLD,
        steps_per_episode: int   = STEPS_PER_EPISODE,
        alpha:             float = ALPHA,
        beta:              float = BETA,
        gamma_r:           float = GAMMA_REWARD,
        nakagami_m:        float = NAKAGAMI_M,
        nakagami_omega:    float = NAKAGAMI_OMEGA,
        seed:              int | None = None,
    ):
        self.p_max             = p_max
        self.p_p               = p_p
        self.sigma2            = sigma2
        self.sinr_threshold    = sinr_threshold
        self.steps_per_episode = steps_per_episode
        self.alpha             = alpha
        self.beta              = beta
        self.gamma_r           = gamma_r
        self.nakagami_m        = nakagami_m
        self.nakagami_omega    = nakagami_omega

        # Reproducible RNG (independent of global numpy state)
        self.rng = np.random.default_rng(seed)

        # Episode tracking
        self._step_count: int   = 0
        self._p_s_prev:   float = 0.0

        # Last channel gains (stored so GUI can read them)
        self._h_pp_sq: float = 0.0
        self._h_sp_sq: float = 0.0
        self._h_ss_sq: float = 0.0
        self._h_ps_sq: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Start a new episode.
        Draws fresh channel gains and returns the initial 7D state.
        """
        self._step_count = 0
        self._p_s_prev   = 0.0

        h_pp, h_sp, h_ss, h_ps = self._draw_channels()
        sinr_p, sinr_s = self._compute_sinr(h_pp, h_sp, h_ss, h_ps, self._p_s_prev)

        return self._build_state(h_pp, h_sp, h_ss, h_ps, sinr_p, sinr_s, self._p_s_prev)

    def step(self, action: float) -> StepResult:
        """
        Execute one time step.

        Args:
            action: P_s value — the agent's chosen transmit power.
                    Should already be in [0, P_max]; we clip defensively.

        Returns:
            StepResult with next state, reward, done flag, and info dict.
        """
        # Clip action to valid range
        p_s = float(np.clip(action, 0.0, self.p_max))

        # Draw fresh Rayleigh fading channel gains (block-fading model)
        h_pp, h_sp, h_ss, h_ps = self._draw_channels()

        # Compute SINRs and throughput
        sinr_p, sinr_s = self._compute_sinr(h_pp, h_sp, h_ss, h_ps, p_s)
        r_s = float(np.log2(1.0 + sinr_s))          # SU throughput (bits/s/Hz)

        # Compute reward
        reward = self._compute_reward(sinr_p, sinr_s, p_s)

        # Advance counters
        self._step_count += 1
        done = (self._step_count >= self.steps_per_episode)

        # Build next state (stores current p_s as p_s_prev for next step)
        next_state = self._build_state(h_pp, h_sp, h_ss, h_ps, sinr_p, sinr_s, p_s)
        self._p_s_prev = p_s

        info = {
            "sinr_p": sinr_p,
            "sinr_s": sinr_s,
            "r_s":    r_s,
            "p_s":    p_s,
            "h_pp":   h_pp,
            "h_sp":   h_sp,
            "h_ss":   h_ss,
            "h_ps":   h_ps,
        }

        return StepResult(state=next_state, reward=reward, done=done, info=info)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_channels(self) -> tuple[float, float, float, float]:
        """
        Draw Nakagami-m fading channel power gains.
        |h|^2 ~ Gamma(shape=m, scale=Omega/m) for each link independently.
        m=1 exactly recovers Rayleigh (Exponential(Omega)).

        Returns: (h_pp_sq, h_sp_sq, h_ss_sq, h_ps_sq)
        """
        scale = self.nakagami_omega / self.nakagami_m
        h_pp_sq = float(self.rng.gamma(self.nakagami_m, scale))   # PT → PR
        h_sp_sq = float(self.rng.gamma(self.nakagami_m, scale))   # ST → PR
        h_ss_sq = float(self.rng.gamma(self.nakagami_m, scale))   # ST → SR
        h_ps_sq = float(self.rng.gamma(self.nakagami_m, scale))   # PT → SR

        self._h_pp_sq = h_pp_sq
        self._h_sp_sq = h_sp_sq
        self._h_ss_sq = h_ss_sq
        self._h_ps_sq = h_ps_sq

        return h_pp_sq, h_sp_sq, h_ss_sq, h_ps_sq

    def _compute_sinr(
        self,
        h_pp_sq: float, h_sp_sq: float,
        h_ss_sq: float, h_ps_sq: float,
        p_s:     float,
    ) -> tuple[float, float]:
        """
        Compute SINR at Primary Receiver and Secondary Receiver.

        SINR_p = (P_p * h_pp^2) / (P_s * h_sp^2 + sigma^2)
        SINR_s = (P_s * h_ss^2) / (P_p * h_ps^2 + sigma^2)

        Denominator is always > 0 because sigma^2 > 0.
        """
        sinr_p = (self.p_p * h_pp_sq) / (p_s * h_sp_sq + self.sigma2)
        sinr_s = (p_s * h_ss_sq) / (self.p_p * h_ps_sq + self.sigma2)
        return float(sinr_p), float(sinr_s)

    def _compute_reward(self, sinr_p: float, sinr_s: float, p_s: float) -> float:
        """
        Reward = alpha * R_s
                 - beta  * max(0, SINR_threshold - SINR_p)
                 - gamma * (P_s / P_max)

        Positive component: SU throughput (log2(1 + SINR_s))
        Negative component 1: heavy penalty when PU SINR drops below threshold
        Negative component 2: small energy-use penalty
        """
        r_s     = float(np.log2(1.0 + sinr_s))
        penalty = max(0.0, self.sinr_threshold - sinr_p)
        energy  = p_s / self.p_max

        return self.alpha * r_s - self.beta * penalty - self.gamma_r * energy

    def _build_state(
        self,
        h_pp_sq: float, h_sp_sq: float,
        h_ss_sq: float, h_ps_sq: float,
        sinr_p:  float, sinr_s: float,
        p_s_prev: float,
    ) -> np.ndarray:
        """
        Assemble the 7-dimensional state vector as float32.

        State: [h_pp^2, h_sp^2, h_ss^2, h_ps^2, SINR_p, SINR_s, P_s_prev]
        """
        state = np.array(
            [h_pp_sq, h_sp_sq, h_ss_sq, h_ps_sq, sinr_p, sinr_s, p_s_prev],
            dtype=np.float32,
        )
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def observation_space_dim(self) -> int:
        return STATE_DIM

    @property
    def action_space_dim(self) -> int:
        return 1
