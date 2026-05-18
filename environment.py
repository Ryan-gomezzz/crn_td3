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
from collections import deque
from dataclasses import dataclass, field
from config import (
    SIGMA2, P_P, P_MAX, SINR_THRESHOLD,
    ALPHA, BETA, GAMMA_REWARD,
    STATE_DIM, STEPS_PER_EPISODE,
    NAKAGAMI_M, NAKAGAMI_OMEGA,
    IMPERFECT_CSI, CSI_NOISE_STD, CSI_DELAY_STEPS, CSI_DELAY_RHO, CSI_QUANT_BITS,
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
        imperfect_csi:     bool  = IMPERFECT_CSI,
        csi_noise_std:     float = CSI_NOISE_STD,
        csi_delay_steps:   int   = CSI_DELAY_STEPS,
        csi_delay_rho:     float = CSI_DELAY_RHO,
        csi_quant_bits:    int   = CSI_QUANT_BITS,
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

        # Imperfect CSI configuration
        self.imperfect_csi   = imperfect_csi
        self.csi_noise_std   = csi_noise_std
        self.csi_delay_steps = max(0, int(csi_delay_steps))
        self.csi_delay_rho   = float(np.clip(csi_delay_rho, 0.0, 1.0))
        self.csi_quant_bits  = max(0, int(csi_quant_bits))

        # Reproducible RNG (independent of global numpy state)
        self.rng = np.random.default_rng(seed)

        # Episode tracking
        self._step_count: int   = 0
        self._p_s_prev:   float = 0.0

        # Last TRUE channel gains (used by physics; GUI reads these)
        self._h_pp_sq: float = 0.0
        self._h_sp_sq: float = 0.0
        self._h_ss_sq: float = 0.0
        self._h_ps_sq: float = 0.0

        # Last OBSERVED (noisy/delayed) channel gains — what the agent sees
        self._h_pp_obs: float = 0.0
        self._h_sp_obs: float = 0.0
        self._h_ss_obs: float = 0.0
        self._h_ps_obs: float = 0.0

        # Delay buffer for past true gains (for stale-feedback model)
        # Holds the last (csi_delay_steps + 1) tuples; index -1 is current truth.
        self._gain_history: deque = deque(
            maxlen=max(1, self.csi_delay_steps + 1)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Start a new episode.
        Draws fresh channel gains and returns the initial 7D state.
        Under imperfect CSI, the state contains noisy/delayed gain estimates
        and the SINR is computed using those same estimates (the agent's
        view of the world); reward and physics still use the true gains.
        """
        self._step_count = 0
        self._p_s_prev   = 0.0
        self._gain_history.clear()

        h_pp, h_sp, h_ss, h_ps = self._draw_channels()

        # Build observed (possibly noisy/delayed/quantized) gain estimates
        h_pp_o, h_sp_o, h_ss_o, h_ps_o = self._observe_channels(
            h_pp, h_sp, h_ss, h_ps
        )

        # SINRs in the observation are what the agent perceives — use estimates.
        sinr_p_obs, sinr_s_obs = self._compute_sinr(
            h_pp_o, h_sp_o, h_ss_o, h_ps_o, self._p_s_prev
        )

        return self._build_state(
            h_pp_o, h_sp_o, h_ss_o, h_ps_o,
            sinr_p_obs, sinr_s_obs, self._p_s_prev,
        )

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

        # Draw fresh Nakagami-m fading channel gains (block-fading model) — TRUE
        h_pp, h_sp, h_ss, h_ps = self._draw_channels()

        # ── PHYSICS uses TRUE gains ──────────────────────────────────────────
        sinr_p, sinr_s = self._compute_sinr(h_pp, h_sp, h_ss, h_ps, p_s)
        r_s = float(np.log2(1.0 + sinr_s))          # SU throughput (bits/s/Hz)
        reward = self._compute_reward(sinr_p, sinr_s, p_s)

        # ── OBSERVATION uses noisy/delayed/quantized estimates ───────────────
        h_pp_o, h_sp_o, h_ss_o, h_ps_o = self._observe_channels(
            h_pp, h_sp, h_ss, h_ps
        )
        sinr_p_obs, sinr_s_obs = self._compute_sinr(
            h_pp_o, h_sp_o, h_ss_o, h_ps_o, p_s
        )

        # Advance counters
        self._step_count += 1
        done = (self._step_count >= self.steps_per_episode)

        next_state = self._build_state(
            h_pp_o, h_sp_o, h_ss_o, h_ps_o,
            sinr_p_obs, sinr_s_obs, p_s,
        )
        self._p_s_prev = p_s

        info = {
            "sinr_p":     sinr_p,        # TRUE SINR at PR (used for outage stats)
            "sinr_s":     sinr_s,        # TRUE SINR at SR
            "sinr_p_obs": sinr_p_obs,    # Estimate seen by agent
            "sinr_s_obs": sinr_s_obs,
            "r_s":        r_s,
            "p_s":        p_s,
            "h_pp":       h_pp,
            "h_sp":       h_sp,
            "h_ss":       h_ss,
            "h_ps":       h_ps,
            "h_pp_obs":   h_pp_o,
            "h_sp_obs":   h_sp_o,
            "h_ss_obs":   h_ss_o,
            "h_ps_obs":   h_ps_o,
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

    def _observe_channels(
        self,
        h_pp_sq: float, h_sp_sq: float,
        h_ss_sq: float, h_ps_sq: float,
    ) -> tuple[float, float, float, float]:
        """
        Produce the SU's noisy view of the channel gains.

        Imperfect-CSI model (composition of three standard impairments):
          1. Feedback delay  — partial mixing of stale and current gain via rho
          2. Estimation noise — additive Gaussian, std proportional to true gain
          3. Quantization     — uniform quantizer over [0, 4*Omega] (optional)

        When `imperfect_csi=False` returns the true gains unchanged.
        """
        # Always push current truth into the delay buffer
        self._gain_history.append((h_pp_sq, h_sp_sq, h_ss_sq, h_ps_sq))

        if not self.imperfect_csi:
            self._h_pp_obs = h_pp_sq
            self._h_sp_obs = h_sp_sq
            self._h_ss_obs = h_ss_sq
            self._h_ps_obs = h_ps_sq
            return h_pp_sq, h_sp_sq, h_ss_sq, h_ps_sq

        # ── 1. Stale gain (feedback delay) ───────────────────────────────────
        if self.csi_delay_steps > 0 and len(self._gain_history) >= self.csi_delay_steps + 1:
            stale = self._gain_history[0]   # oldest in window
        else:
            # Buffer not yet full at episode start — fall back to current truth
            stale = self._gain_history[0]

        rho = self.csi_delay_rho
        cur = (h_pp_sq, h_sp_sq, h_ss_sq, h_ps_sq)
        mixed = tuple(rho * c + (1.0 - rho) * s for c, s in zip(cur, stale))

        # ── 2. Estimation noise (multiplicative-style: std ~ sigma * |h|^2) ──
        sigma = self.csi_noise_std
        noise = self.rng.normal(0.0, sigma, size=4) * np.array(mixed)
        noisy = np.maximum(np.array(mixed) + noise, 0.0)   # clip negatives

        # ── 3. Quantization (optional) ───────────────────────────────────────
        if self.csi_quant_bits > 0:
            levels = 2 ** self.csi_quant_bits
            top = 4.0 * self.nakagami_omega   # cover ~99% of Gamma mass
            step = top / levels
            noisy = np.minimum(noisy, top - 1e-9)
            noisy = np.round(noisy / step) * step

        h_pp_o, h_sp_o, h_ss_o, h_ps_o = (float(x) for x in noisy)

        self._h_pp_obs = h_pp_o
        self._h_sp_obs = h_sp_o
        self._h_ss_obs = h_ss_o
        self._h_ps_obs = h_ps_o

        return h_pp_o, h_sp_o, h_ss_o, h_ps_o

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
