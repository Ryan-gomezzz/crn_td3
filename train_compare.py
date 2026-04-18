# =============================================================================
# train_compare.py — TD3 vs DDPG Headless Comparison Training
#
# Trains both TD3 and DDPG on the Nakagami-m CRN environment,
# collects performance metrics, and generates a PDF report with:
#   - SINR vs Bit Error Rate (BER) scatter + theoretical BPSK curve
#   - Secondary User Throughput (TD3 vs DDPG)
#   - Primary User Throughput (TD3 vs DDPG)
#   - Outage Probability (TD3 vs DDPG)
#   - Episode Reward Curves (TD3 vs DDPG)
#   - Results Summary Table
#
# Usage:
#   python train_compare.py [--episodes N] [--steps-per-ep M] [--output PATH]
#
# Kaggle defaults: 500 episodes × 200 steps, ~10 min on GPU.
# =============================================================================

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.special import erfc

# ── Matplotlib non-interactive backend (must be set before importing pyplot) ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator, LogFormatter

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    STATE_DIM, ACTION_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    EXPLORATION_NOISE_STD, EXPLORATION_NOISE_END,
    BATCH_SIZE, SINR_THRESHOLD, NAKAGAMI_M, NAKAGAMI_OMEGA,
    GRAD_UPDATES_PER_STEP,
)
from environment import CRNEnvironment
from td3  import TD3Agent,  ReplayBuffer
from ddpg import DDPGAgent
from utils import RollingStats, ExplorationNoise

# ─── CLI defaults ─────────────────────────────────────────────────────────────
DEFAULT_EPISODES      = 500      # Episodes per algorithm
DEFAULT_STEPS_PER_EP  = 200      # Steps per episode (overrides env default)
DEFAULT_OUTPUT        = "results/crn_comparison_report.pdf"
PRINT_EVERY           = 50       # Console progress frequency
MAX_SCATTER_PTS       = 3000     # Cap scatter-plot points per algorithm
OUTAGE_WINDOW_SIZE    = 500      # Rolling window for outage probability
DEFAULT_CHECKPOINT_EVERY = 0     # 0 = disabled; set to e.g. 3000 to save mid-run PNGs

# ─── Plot style ───────────────────────────────────────────────────────────────
TD3_COLOR  = "#1f77b4"   # matplotlib default blue
DDPG_COLOR = "#d62728"   # matplotlib default red
ALPHA_FILL = 0.15


# =============================================================================
# Metrics container
# =============================================================================

@dataclass
class RunMetrics:
    """All collected metrics for a single training run."""
    name: str

    # Per-episode scalars
    rewards:        List[float] = field(default_factory=list)
    su_throughputs: List[float] = field(default_factory=list)  # bits/s/Hz (SU)
    pu_throughputs: List[float] = field(default_factory=list)  # bits/s/Hz (PU)
    outage_probs:   List[float] = field(default_factory=list)  # fraction of steps
    avg_bers:       List[float] = field(default_factory=list)  # mean BER per episode

    # Step-level data for SINR vs BER scatter (sampled for readability)
    sinr_db_pts: List[float] = field(default_factory=list)
    ber_pts:     List[float] = field(default_factory=list)

    # Final summary stats (filled after training)
    final_avg_reward:      float = 0.0
    final_avg_su_tput:     float = 0.0
    final_avg_pu_tput:     float = 0.0
    final_outage_prob:     float = 0.0
    final_avg_ber:         float = 0.0
    training_time_sec:     float = 0.0


# =============================================================================
# Core training loop (algorithm-agnostic)
# =============================================================================

def save_checkpoint_plots(
    metrics: "RunMetrics",
    current_ep: int,
    checkpoint_dir: str,
    color: str = "#555555",
) -> None:
    """Save BER / SU Throughput / Outage Probability PNGs at a mid-run checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    episodes = np.arange(1, current_ep + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # BER (log-scale)
    ax = axes[0]
    raw = np.array(metrics.avg_bers)
    sm  = smooth(raw, window=20)
    ax.semilogy(episodes, raw, color=color, alpha=0.2, lw=0.7)
    ax.semilogy(episodes, sm,  color=color, lw=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average BER")
    ax.set_title(f"{metrics.name} — BER (ep {current_ep})")
    ax.grid(True, which="both", alpha=0.3)

    # SU Throughput
    ax = axes[1]
    raw = np.array(metrics.su_throughputs)
    sm  = smooth(raw, window=20)
    ax.plot(episodes, raw, color=color, alpha=0.2, lw=0.7)
    ax.plot(episodes, sm,  color=color, lw=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (bits/s/Hz)")
    ax.set_title(f"{metrics.name} — SU Throughput (ep {current_ep})")
    ax.grid(True, alpha=0.3)

    # Outage Probability
    ax = axes[2]
    raw = np.array(metrics.outage_probs)
    sm  = smooth(raw, window=20)
    ax.plot(episodes, raw, color=color, alpha=0.2, lw=0.7)
    ax.plot(episodes, sm,  color=color, lw=2.0)
    ax.axhline(0.05, color="gray", ls="--", lw=1.2, label="5% target")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Outage Probability")
    ax.set_title(f"{metrics.name} — Outage (ep {current_ep})")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{metrics.name} Checkpoint — Episode {current_ep}  "
        f"(Nakagami-m={NAKAGAMI_M})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    fname = os.path.join(
        checkpoint_dir,
        f"checkpoint_{metrics.name.lower()}_ep{current_ep:05d}.png",
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [checkpoint] {fname}")


def run_algorithm(
    algo_name:       str,
    agent,
    n_episodes:      int,
    steps_per_ep:    int,
    verbose:         bool = True,
    checkpoint_every: int = 0,
    checkpoint_dir:  str  = "results",
) -> RunMetrics:
    """
    Train `agent` for `n_episodes` episodes and collect metrics.

    Args:
        algo_name:    Label for printing / PDF ("TD3" or "DDPG")
        agent:        TD3Agent or DDPGAgent instance
        n_episodes:   Number of training episodes
        steps_per_ep: Steps per episode (used when creating the environment)
        verbose:      Whether to print progress

    Returns:
        RunMetrics populated with all collected data.
    """
    metrics = RunMetrics(name=algo_name)

    env = CRNEnvironment(steps_per_episode=steps_per_ep)
    buffer = ReplayBuffer(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_size=REPLAY_BUFFER_SIZE,
        device=agent.device,
    )
    noise_sched = ExplorationNoise(
        std_start   = EXPLORATION_NOISE_STD * P_MAX,
        std_end     = EXPLORATION_NOISE_END  * P_MAX,
        decay_steps = n_episodes,
    )
    rolling = RollingStats(window=100)

    # Rolling outage window across all steps
    sinr_window = deque(maxlen=OUTAGE_WINDOW_SIZE)

    # Track scatter-point budget per episode (uniform sampling)
    scatter_budget = max(1, MAX_SCATTER_PTS // max(n_episodes, 1))

    t_start = time.perf_counter()

    if verbose:
        print()
        print("=" * 68)
        print(f"  [{algo_name}] Starting training   "
              f"episodes={n_episodes}  steps/ep={steps_per_ep}  "
              f"Nakagami-m={NAKAGAMI_M}")
        print(f"  Device: {agent.device}")
        print("=" * 68)

    for ep in range(n_episodes):
        state          = env.reset()
        ep_reward      = 0.0
        ep_sinr_s_list = []
        ep_ber_list    = []
        ep_rs_list     = []
        ep_rp_list     = []
        ep_outage_list = []

        # Scatter sampling: choose which steps to record this episode
        sample_steps = set(
            np.random.choice(steps_per_ep, size=min(scatter_budget, steps_per_ep), replace=False)
        )

        for t in range(steps_per_ep):
            action = agent.select_action(state, noise_sched.current_std)
            result = env.step(action)

            # Store transition
            buffer.add(
                state,
                np.array([action], dtype=np.float32),
                result.reward,
                result.state,
                result.done,
            )

            # Metrics
            sinr_s   = result.info["sinr_s"]
            sinr_p   = result.info["sinr_p"]
            r_s      = result.info["r_s"]                        # SU throughput
            r_p      = float(np.log2(1.0 + sinr_p))             # PU throughput
            ber      = float(0.5 * erfc(math.sqrt(max(0.0, sinr_s))))
            sinr_db  = 10.0 * math.log10(max(1e-9, sinr_s))

            sinr_window.append(sinr_s)
            ep_sinr_s_list.append(sinr_s)
            ep_ber_list.append(ber)
            ep_rs_list.append(r_s)
            ep_rp_list.append(r_p)
            ep_outage_list.append(1 if sinr_s < SINR_THRESHOLD else 0)

            # Scatter sample
            if t in sample_steps:
                metrics.sinr_db_pts.append(sinr_db)
                metrics.ber_pts.append(ber)

            ep_reward += result.reward
            state      = result.state

            # Train — GRAD_UPDATES_PER_STEP gradient steps per env step
            # keeps GPU utilization high despite small network size
            if buffer.is_ready:
                for _ in range(GRAD_UPDATES_PER_STEP):
                    agent.train_step(buffer, BATCH_SIZE)

        # End of episode aggregation
        noise_sched.step()
        rolling.push(ep_reward)

        metrics.rewards.append(ep_reward)
        metrics.su_throughputs.append(float(np.mean(ep_rs_list)))
        metrics.pu_throughputs.append(float(np.mean(ep_rp_list)))
        metrics.outage_probs.append(float(np.mean(ep_outage_list)))
        metrics.avg_bers.append(float(np.mean(ep_ber_list)))

        if verbose and (ep + 1) % PRINT_EVERY == 0:
            avg100 = rolling.mean()
            elapsed = time.perf_counter() - t_start
            print(
                f"  ep {ep+1:>5}/{n_episodes}  "
                f"reward={ep_reward:>8.3f}  avg100={avg100:>8.3f}  "
                f"outage={metrics.outage_probs[-1]:.3f}  "
                f"SU_tput={metrics.su_throughputs[-1]:.3f}  "
                f"PU_tput={metrics.pu_throughputs[-1]:.3f}  "
                f"elapsed={elapsed:.0f}s"
            )

        # ── Mid-run checkpoint ────────────────────────────────────────────────
        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0 and (ep + 1) < n_episodes:
            _color = TD3_COLOR if algo_name == "TD3" else DDPG_COLOR
            save_checkpoint_plots(metrics, ep + 1, checkpoint_dir, color=_color)

    metrics.training_time_sec = time.perf_counter() - t_start

    # Final stats over last 100 episodes (or all if fewer)
    tail = min(100, n_episodes)
    metrics.final_avg_reward  = float(np.mean(metrics.rewards[-tail:]))
    metrics.final_avg_su_tput = float(np.mean(metrics.su_throughputs[-tail:]))
    metrics.final_avg_pu_tput = float(np.mean(metrics.pu_throughputs[-tail:]))
    metrics.final_outage_prob = float(np.mean(metrics.outage_probs[-tail:]))
    metrics.final_avg_ber     = float(np.mean(metrics.avg_bers[-tail:]))

    if verbose:
        print(f"\n  [{algo_name}] Done in {metrics.training_time_sec:.1f}s")
        print(f"  Final avg reward  : {metrics.final_avg_reward:.4f}")
        print(f"  Final SU throughput: {metrics.final_avg_su_tput:.4f} bits/s/Hz")
        print(f"  Final PU throughput: {metrics.final_avg_pu_tput:.4f} bits/s/Hz")
        print(f"  Final outage prob  : {metrics.final_outage_prob:.4f}")
        print(f"  Final avg BER      : {metrics.final_avg_ber:.6f}")

    return metrics


# =============================================================================
# Smoothing helper
# =============================================================================

def smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Moving-average smoothing for cleaner plots."""
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=float)
    if window >= len(arr):
        return arr
    kernel = np.ones(window) / window
    pad    = np.pad(arr, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


# =============================================================================
# Theoretical BER curves for reference
# =============================================================================

def theoretical_bpsk_ber(snr_db: np.ndarray) -> np.ndarray:
    """Instantaneous BPSK BER: 0.5 * erfc(sqrt(SNR_linear))."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_lin))


def nakagami_avg_ber_bpsk(snr_db: np.ndarray, m: float = NAKAGAMI_M) -> np.ndarray:
    """
    Average BPSK BER over Nakagami-m fading (closed-form).

    BER_avg = I_m(gamma_bar)  where gamma_bar = 10^(snr_db/10)

    For BPSK under Nakagami-m: exact closed-form via the regularized
    incomplete Beta function (Simon & Alouini, 2005, eq. 8.98):

        P_b(E) = ((1-mu)/2)^m * sum_{k=0}^{m-1} C(m-1+k, k) * ((1+mu)/2)^k

    where mu = sqrt(gamma_bar / (m + gamma_bar)).

    Only valid for integer m.
    """
    snr_lin  = 10.0 ** (snr_db / 10.0)
    m_int    = int(round(m))
    mu       = np.sqrt(snr_lin / (m_int + snr_lin))

    ber = np.zeros_like(snr_lin)
    coeff_base = ((1.0 - mu) / 2.0) ** m_int
    for k in range(m_int):
        binom = math.comb(m_int - 1 + k, k)
        ber  += coeff_base * binom * ((1.0 + mu) / 2.0) ** k
    return np.clip(ber, 1e-12, 0.5)


# =============================================================================
# PDF Report Generator
# =============================================================================

def generate_pdf(
    td3_metrics:  RunMetrics,
    ddpg_metrics: RunMetrics,
    output_path:  str,
    n_episodes:   int,
    steps_per_ep: int,
) -> None:
    """
    Render all plots to a multi-page PDF report.

    Pages:
      1. Title / Summary Table
      2. SINR vs BER (scatter + theoretical curves)
      3. Secondary User Throughput vs Episodes
      4. Primary User Throughput vs Episodes
      5. Outage Probability vs Episodes
      6. Episode Reward Curve
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "lines.linewidth":  1.8,
    })

    episodes = np.arange(1, n_episodes + 1)

    with PdfPages(output_path) as pdf:

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Title + Summary Table
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        title_txt = (
            "CRN Power Control: TD3 vs DDPG\n"
            "Nakagami-m Fading Channel Performance Report"
        )
        fig.text(0.5, 0.88, title_txt, ha="center", va="top",
                 fontsize=18, fontweight="bold")
        fig.text(0.5, 0.80,
                 f"Nakagami-m = {NAKAGAMI_M}  |  Episodes = {n_episodes}  |  "
                 f"Steps/Episode = {steps_per_ep}  |  SINR Threshold = {SINR_THRESHOLD} dB",
                 ha="center", va="top", fontsize=11, color="#444444")

        # Summary table
        col_labels = ["Metric", "TD3", "DDPG", "Winner"]
        metrics_rows = [
            ("Avg Reward (last 100 ep)",
             f"{td3_metrics.final_avg_reward:.4f}",
             f"{ddpg_metrics.final_avg_reward:.4f}",
             "TD3" if td3_metrics.final_avg_reward >= ddpg_metrics.final_avg_reward else "DDPG"),
            ("SU Throughput (bits/s/Hz)",
             f"{td3_metrics.final_avg_su_tput:.4f}",
             f"{ddpg_metrics.final_avg_su_tput:.4f}",
             "TD3" if td3_metrics.final_avg_su_tput >= ddpg_metrics.final_avg_su_tput else "DDPG"),
            ("PU Throughput (bits/s/Hz)",
             f"{td3_metrics.final_avg_pu_tput:.4f}",
             f"{ddpg_metrics.final_avg_pu_tput:.4f}",
             "TD3" if td3_metrics.final_avg_pu_tput >= ddpg_metrics.final_avg_pu_tput else "DDPG"),
            ("Outage Probability",
             f"{td3_metrics.final_outage_prob:.4f}",
             f"{ddpg_metrics.final_outage_prob:.4f}",
             "TD3" if td3_metrics.final_outage_prob <= ddpg_metrics.final_outage_prob else "DDPG"),
            ("Average BER",
             f"{td3_metrics.final_avg_ber:.6f}",
             f"{ddpg_metrics.final_avg_ber:.6f}",
             "TD3" if td3_metrics.final_avg_ber <= ddpg_metrics.final_avg_ber else "DDPG"),
            ("Training Time (s)",
             f"{td3_metrics.training_time_sec:.1f}",
             f"{ddpg_metrics.training_time_sec:.1f}",
             "—"),
        ]

        table = ax.table(
            cellText   = metrics_rows,
            colLabels  = col_labels,
            cellLoc    = "center",
            loc        = "center",
            bbox       = [0.05, 0.08, 0.90, 0.62],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)

        # Style header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

        # Highlight winner column
        for i in range(1, len(metrics_rows) + 1):
            winner = metrics_rows[i - 1][3]
            cell   = table[i, 3]
            if winner == "TD3":
                cell.set_facecolor("#d4edfa")
                cell.set_text_props(color=TD3_COLOR, fontweight="bold")
            elif winner == "DDPG":
                cell.set_facecolor("#fde8e8")
                cell.set_text_props(color=DDPG_COLOR, fontweight="bold")

        # Alternate row shading
        for i in range(1, len(metrics_rows) + 1):
            bg = "#f8f8f8" if i % 2 == 0 else "white"
            for j in range(3):
                table[i, j].set_facecolor(bg)

        fig.text(
            0.5, 0.02,
            "Ramaiah Institute of Technology, Bangalore  |  "
            "Cognitive Radio Network — RL Power Allocation",
            ha="center", fontsize=9, color="#888888"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — SINR vs BER
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 6))

        # Theoretical curves
        snr_range = np.linspace(-5, 25, 300)
        ax.semilogy(snr_range, theoretical_bpsk_ber(snr_range),
                    "k--", linewidth=1.4, label="BPSK Theoretical (AWGN)", alpha=0.7)
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, NAKAGAMI_M),
                    "k-.", linewidth=1.4,
                    label=f"BPSK Avg BER (Nakagami-m={int(NAKAGAMI_M)})", alpha=0.7)

        # Simulated scatter
        if td3_metrics.sinr_db_pts:
            ax.scatter(td3_metrics.sinr_db_pts,  td3_metrics.ber_pts,
                       color=TD3_COLOR,  alpha=0.35, s=10, label="TD3 (simulated)")
        if ddpg_metrics.sinr_db_pts:
            ax.scatter(ddpg_metrics.sinr_db_pts, ddpg_metrics.ber_pts,
                       color=DDPG_COLOR, alpha=0.35, s=10, label="DDPG (simulated)")

        # Binned mean BER
        for m_obj, color, name in [
            (td3_metrics,  TD3_COLOR,  "TD3"),
            (ddpg_metrics, DDPG_COLOR, "DDPG"),
        ]:
            if m_obj.sinr_db_pts:
                pts = np.array(list(zip(m_obj.sinr_db_pts, m_obj.ber_pts)))
                bins = np.linspace(-5, 25, 25)
                bin_centers, bin_means = [], []
                for i in range(len(bins) - 1):
                    mask = (pts[:, 0] >= bins[i]) & (pts[:, 0] < bins[i + 1])
                    if mask.sum() > 0:
                        bin_centers.append((bins[i] + bins[i + 1]) / 2)
                        bin_means.append(np.mean(pts[mask, 1]))
                if bin_centers:
                    ax.semilogy(bin_centers, bin_means,
                                color=color, linewidth=2.2,
                                marker="o", markersize=4,
                                label=f"{name} Mean BER")

        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(
            f"SINR vs BER — Nakagami-m={int(NAKAGAMI_M)} Fading Channel\n"
            f"(BPSK Modulation, Secondary User)"
        )
        ax.set_xlim(-5, 25)
        ax.set_ylim(1e-6, 0.6)
        ax.legend(loc="lower left")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3 — Secondary User Throughput
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for m_obj, color in [(td3_metrics, TD3_COLOR), (ddpg_metrics, DDPG_COLOR)]:
            raw  = np.array(m_obj.su_throughputs)
            sm   = smooth(raw, window=20)
            ax.plot(episodes, raw,  color=color, alpha=0.25, linewidth=0.8)
            ax.plot(episodes, sm,   color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
            ax.fill_between(episodes,
                            np.maximum(0, sm - raw.std()),
                            sm + raw.std(),
                            color=color, alpha=ALPHA_FILL)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Throughput (bits/s/Hz)")
        ax.set_title("Secondary User (SU) Throughput vs Episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 4 — Primary User Throughput
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for m_obj, color in [(td3_metrics, TD3_COLOR), (ddpg_metrics, DDPG_COLOR)]:
            raw = np.array(m_obj.pu_throughputs)
            sm  = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(episodes, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
            ax.fill_between(episodes,
                            np.maximum(0, sm - raw.std()),
                            sm + raw.std(),
                            color=color, alpha=ALPHA_FILL)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Throughput (bits/s/Hz)")
        ax.set_title("Primary User (PU) Throughput vs Episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 5 — Outage Probability
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for m_obj, color in [(td3_metrics, TD3_COLOR), (ddpg_metrics, DDPG_COLOR)]:
            raw = np.array(m_obj.outage_probs)
            sm  = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(episodes, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")

        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1.2, label="5% target")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Outage Probability")
        ax.set_title(
            f"Outage Probability vs Episodes\n"
            f"(SINR_s < {SINR_THRESHOLD} threshold, rolling per episode)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 6 — Episode Reward Curve
        # ══════════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)

        for idx, (m_obj, color) in enumerate(
            [(td3_metrics, TD3_COLOR), (ddpg_metrics, DDPG_COLOR)]
        ):
            ax   = axes[idx]
            raw  = np.array(m_obj.rewards)
            sm   = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.2, linewidth=0.7)
            ax.plot(episodes, sm,  color=color, linewidth=2.2, label="Smoothed reward")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Reward")
            ax.set_title(f"{m_obj.name} Reward Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Episode Reward Curves — TD3 vs DDPG", fontsize=14, y=1.01)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 7 — Combined overlay reward comparison
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))

        for m_obj, color in [(td3_metrics, TD3_COLOR), (ddpg_metrics, DDPG_COLOR)]:
            raw = np.array(m_obj.rewards)
            sm  = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.18, linewidth=0.7)
            ax.plot(episodes, sm,  color=color, linewidth=2.2, label=f"{m_obj.name}")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title("TD3 vs DDPG — Reward Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "CRN TD3 vs DDPG Comparison Report"
        d["Author"]  = "Ramaiah Institute of Technology"
        d["Subject"] = f"Nakagami-m={NAKAGAMI_M} CRN Power Allocation via RL"

    print(f"\n  PDF report saved: {os.path.abspath(output_path)}")


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TD3 and DDPG on the CRN and generate a PDF comparison report."
    )
    p.add_argument("--episodes",     type=int,   default=DEFAULT_EPISODES,
                   help=f"Number of training episodes per algorithm (default {DEFAULT_EPISODES})")
    p.add_argument("--steps-per-ep", type=int,   default=DEFAULT_STEPS_PER_EP,
                   help=f"Steps per episode (default {DEFAULT_STEPS_PER_EP})")
    p.add_argument("--output",       type=str,   default=DEFAULT_OUTPUT,
                   help=f"Output PDF path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Global random seed (default 42)")
    p.add_argument("--no-ddpg",          action="store_true",
                   help="Skip DDPG and only run TD3 (useful for quick smoke-test)")
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                   help="Save intermediate PNG graphs every N episodes (0 = disabled)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)

    # GPU speed: enable TF32 on Ampere/Ada GPUs (RTX 30xx/40xx) — uses tensor
    # cores for matmul with negligible precision loss, ~1.5-2x faster on MLPs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    checkpoint_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "results"

    print("\n" + "=" * 68)
    print("  CRN TD3 vs DDPG — Comparison Training & Report Generator")
    print(f"  Nakagami-m = {NAKAGAMI_M}  |  Episodes = {args.episodes}"
          f"  |  Steps/ep = {args.steps_per_ep}")
    if args.checkpoint_every > 0:
        print(f"  Checkpoint graphs every {args.checkpoint_every} episodes -> {checkpoint_dir}/")
    print("=" * 68)

    # ── TD3 ────────────────────────────────────────────────────────────────────
    td3_agent = TD3Agent()
    td3_metrics = run_algorithm(
        algo_name        = "TD3",
        agent            = td3_agent,
        n_episodes       = args.episodes,
        steps_per_ep     = args.steps_per_ep,
        checkpoint_every = args.checkpoint_every,
        checkpoint_dir   = checkpoint_dir,
    )

    # ── DDPG ───────────────────────────────────────────────────────────────────
    if not args.no_ddpg:
        ddpg_agent = DDPGAgent()
        ddpg_metrics = run_algorithm(
            algo_name        = "DDPG",
            agent            = ddpg_agent,
            n_episodes       = args.episodes,
            steps_per_ep     = args.steps_per_ep,
            checkpoint_every = args.checkpoint_every,
            checkpoint_dir   = checkpoint_dir,
        )
    else:
        # Dummy metrics for single-algo run
        ddpg_metrics = RunMetrics(name="DDPG (skipped)")

    # ── Generate PDF ───────────────────────────────────────────────────────────
    print("\n  Generating PDF report…")
    generate_pdf(
        td3_metrics  = td3_metrics,
        ddpg_metrics = ddpg_metrics,
        output_path  = args.output,
        n_episodes   = args.episodes,
        steps_per_ep = args.steps_per_ep,
    )

    print("\n  All done!")
    print(f"  Report: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
