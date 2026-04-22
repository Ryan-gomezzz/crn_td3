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
from camo_td3 import CAMO_TD3Agent, SequenceReplayBuffer
from utils import RollingStats, ExplorationNoise

# ─── CLI defaults ─────────────────────────────────────────────────────────────
DEFAULT_EPISODES      = 500      # Episodes per algorithm
DEFAULT_STEPS_PER_EP  = 200      # Steps per episode (overrides env default)
DEFAULT_OUTPUT        = "results/crn_comparison_report.pdf"
PRINT_EVERY           = 50       # Console progress frequency
MAX_SCATTER_PTS       = 12000    # Cap scatter-plot points per algorithm (≥ n_episodes for good coverage)
OUTAGE_WINDOW_SIZE    = 500      # Rolling window for outage probability
DEFAULT_CHECKPOINT_EVERY = 0     # 0 = disabled; set to e.g. 3000 to save mid-run PNGs

# ─── Plot style ───────────────────────────────────────────────────────────────
TD3_COLOR      = "#1f77b4"   # matplotlib default blue
DDPG_COLOR     = "#d62728"   # matplotlib default red
CAMO_TD3_COLOR = "#ff7f0e"   # matplotlib default orange
ALPHA_FILL     = 0.15

ALGO_COLORS = {"TD3": TD3_COLOR, "DDPG": DDPG_COLOR, "CAMO-TD3": CAMO_TD3_COLOR}


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

    # Step-level data for SU SINR vs BER scatter (sampled for readability)
    sinr_db_pts: List[float] = field(default_factory=list)
    ber_pts:     List[float] = field(default_factory=list)

    # Step-level data for PU SINR vs BER scatter
    pu_sinr_db_pts: List[float] = field(default_factory=list)
    pu_ber_pts:     List[float] = field(default_factory=list)

    # Per-episode PU BER (for BER vs episode curve)
    avg_pu_bers: List[float] = field(default_factory=list)

    # Final summary stats (filled after training)
    final_avg_reward:      float = 0.0
    final_avg_su_tput:     float = 0.0
    final_avg_pu_tput:     float = 0.0
    final_outage_prob:     float = 0.0
    final_avg_ber:         float = 0.0
    final_avg_pu_ber:      float = 0.0
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
    """Save BER / SU Throughput / Outage / SINR-vs-BER PNGs at a mid-run checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    episodes = np.arange(1, current_ep + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Top-left: SU BER vs Episode ──────────────────────────────────────────
    ax = axes[0, 0]
    raw = np.array(metrics.avg_bers)
    sm  = smooth(raw, window=20)
    ax.semilogy(episodes, raw, color=color, alpha=0.2, lw=0.7)
    ax.semilogy(episodes, sm,  color=color, lw=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average BER")
    ax.set_title(f"{metrics.name} — SU BER vs Episode (ep {current_ep})")
    ax.grid(True, which="both", alpha=0.3)

    # ── Top-right: SU Throughput vs Episode ──────────────────────────────────
    ax = axes[0, 1]
    raw = np.array(metrics.su_throughputs)
    sm  = smooth(raw, window=20)
    ax.plot(episodes, raw, color=color, alpha=0.2, lw=0.7)
    ax.plot(episodes, sm,  color=color, lw=2.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (bits/s/Hz)")
    ax.set_title(f"{metrics.name} — SU Throughput vs Episode (ep {current_ep})")
    ax.grid(True, alpha=0.3)

    # ── Bottom-left: Outage Probability vs Episode ────────────────────────────
    ax = axes[1, 0]
    raw = np.array(metrics.outage_probs)
    sm  = smooth(raw, window=20)
    ax.plot(episodes, raw, color=color, alpha=0.2, lw=0.7)
    ax.plot(episodes, sm,  color=color, lw=2.0)
    ax.axhline(0.05, color="gray", ls="--", lw=1.2, label="5% target")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Outage Probability")
    ax.set_title(f"{metrics.name} — Outage Probability (ep {current_ep})")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom-right: SINR vs BER scatter (SU + PU) ───────────────────────────
    ax = axes[1, 1]
    if metrics.sinr_db_pts:
        ax.scatter(metrics.sinr_db_pts, metrics.ber_pts,
                   color=color, alpha=0.25, s=8, label="SU")
    if metrics.pu_sinr_db_pts:
        ax.scatter(metrics.pu_sinr_db_pts, metrics.pu_ber_pts,
                   color="orange", alpha=0.25, s=8, label="PU")
    # Theoretical Nakagami curve
    _snr = np.linspace(-5, 25, 200)
    _ber = nakagami_avg_ber_bpsk(_snr, NAKAGAMI_M)
    ax.semilogy(_snr, _ber, "k-", lw=1.4, alpha=0.7,
                label=f"Nakagami-m={int(NAKAGAMI_M)} Theory")
    ax.axvline(x=10*np.log10(SINR_THRESHOLD), color="red", ls="--",
               lw=1.2, label="SINR Threshold")
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("BER")
    ax.set_xlim(-5, 20)
    ax.set_ylim(1e-6, 0.6)
    ax.set_title(f"{metrics.name} — SINR vs BER (ep {current_ep})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"{metrics.name} Checkpoint — Episode {current_ep}  "
        f"(Nakagami-m={NAKAGAMI_M}  |  P_MAX={P_MAX}W)",
        fontsize=13,
    )
    plt.tight_layout()

    fname = os.path.join(
        checkpoint_dir,
        f"checkpoint_{metrics.name.lower()}_ep{current_ep:05d}.png",
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [checkpoint] {fname}")

    # Also save a self-contained checkpoint PDF with all metrics pages
    _save_checkpoint_pdf(metrics, current_ep, checkpoint_dir)


def _save_checkpoint_pdf(
    metrics: "RunMetrics",
    current_ep: int,
    checkpoint_dir: str,
) -> None:
    """Save a full single-algorithm PDF report at a checkpoint episode."""
    import json

    episodes = np.arange(1, current_ep + 1)
    color    = TD3_COLOR if metrics.name == "TD3" else DDPG_COLOR
    snr_rng  = np.linspace(-5, 25, 300)

    pdf_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{metrics.name.lower()}_ep{current_ep:05d}_report.pdf",
    )

    with PdfPages(pdf_path) as pdf:

        # ── Summary stats ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.axis("off")
        fig.text(0.5, 0.92,
                 f"{metrics.name} — Checkpoint Report  (Episode {current_ep})",
                 ha="center", fontsize=16, fontweight="bold")
        fig.text(0.5, 0.85,
                 f"Nakagami-m={NAKAGAMI_M}  |  P_MAX={P_MAX}W  |  Steps/ep={current_ep}",
                 ha="center", fontsize=10, color="#555555")

        tail = min(100, current_ep)
        rows = [
            ("Episodes trained",         str(current_ep),                         ""),
            ("Avg Reward (last 100 ep)",  f"{float(np.mean(metrics.rewards[-tail:])):.4f}",  ""),
            ("SU Throughput (bits/s/Hz)", f"{float(np.mean(metrics.su_throughputs[-tail:])):.4f}", ""),
            ("PU Throughput (bits/s/Hz)", f"{float(np.mean(metrics.pu_throughputs[-tail:])):.4f}", ""),
            ("Outage Probability",        f"{float(np.mean(metrics.outage_probs[-tail:])):.4f}", "5% target"),
            ("Avg SU BER",                f"{float(np.mean(metrics.avg_bers[-tail:])):.6f}",    ""),
            ("Avg PU BER",                f"{float(np.mean(metrics.avg_pu_bers[-tail:])):.6f}" if metrics.avg_pu_bers else "N/A", ""),
            ("Training Time (s)",         f"{metrics.training_time_sec:.1f}",                   ""),
        ]
        table = ax.table(cellText=rows, colLabels=["Metric", "Value", "Note"],
                         cellLoc="center", loc="center",
                         bbox=[0.05, 0.10, 0.90, 0.68])
        table.auto_set_font_size(False); table.set_fontsize(11)
        for j in range(3):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── SU SINR vs BER ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.semilogy(snr_rng, theoretical_bpsk_ber(snr_rng), "k--", lw=1.4,
                    label="BPSK (AWGN)", alpha=0.7)
        ax.semilogy(snr_rng, nakagami_avg_ber_bpsk(snr_rng, NAKAGAMI_M), "k-", lw=1.4,
                    label=f"Nakagami-m={int(NAKAGAMI_M)} Theory", alpha=0.7)
        if metrics.sinr_db_pts:
            ax.scatter(metrics.sinr_db_pts, metrics.ber_pts,
                       color=color, alpha=0.3, s=8, label=f"{metrics.name} SU (simulated)")
            pts  = np.array(list(zip(metrics.sinr_db_pts, metrics.ber_pts)))
            bins = np.linspace(-5, 25, 25)
            bx   = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)
                    if ((pts[:,0]>=bins[i])&(pts[:,0]<bins[i+1])).sum()>0]
            by   = [np.mean(pts[(pts[:,0]>=bins[i])&(pts[:,0]<bins[i+1]),1])
                    for i in range(len(bins)-1)
                    if ((pts[:,0]>=bins[i])&(pts[:,0]<bins[i+1])).sum()>0]
            if bx:
                ax.semilogy(bx, by, color=color, lw=2.2, marker="o", ms=4,
                            label=f"{metrics.name} Mean BER")
        ax.set_xlabel("SINR (dB)"); ax.set_ylabel("BER")
        ax.set_title(f"{metrics.name} — SU SINR vs BER (ep {current_ep})")
        ax.set_xlim(-5, 25); ax.set_ylim(1e-6, 0.6)
        ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── PU SINR vs BER ───────────────────────────────────────────────────
        if metrics.pu_sinr_db_pts:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.semilogy(snr_rng, nakagami_avg_ber_bpsk(snr_rng, NAKAGAMI_M), "k-", lw=1.4,
                        label=f"Nakagami-m={int(NAKAGAMI_M)} Theory", alpha=0.7)
            ax.scatter(metrics.pu_sinr_db_pts, metrics.pu_ber_pts,
                       color=color, alpha=0.3, s=8, label=f"{metrics.name} PU (simulated)")
            pts  = np.array(list(zip(metrics.pu_sinr_db_pts, metrics.pu_ber_pts)))
            bins = np.linspace(-5, 25, 25)
            bx   = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)
                    if ((pts[:,0]>=bins[i])&(pts[:,0]<bins[i+1])).sum()>0]
            by   = [np.mean(pts[(pts[:,0]>=bins[i])&(pts[:,0]<bins[i+1]),1])
                    for i in range(len(bins)-1)
                    if ((pts[:,0]>=bins[i])&(pts[:,0]<bins[i+1])).sum()>0]
            if bx:
                ax.semilogy(bx, by, color=color, lw=2.2, marker="s", ms=4,
                            label=f"{metrics.name} PU Mean BER")
            ax.axvline(x=10*np.log10(SINR_THRESHOLD), color="green", ls="--",
                       lw=1.4, label="PU SINR Threshold")
            ax.set_xlabel("PU SINR (dB)"); ax.set_ylabel("BER")
            ax.set_title(f"{metrics.name} — PU SINR vs BER (ep {current_ep})")
            ax.set_xlim(-5, 25); ax.set_ylim(1e-6, 0.6)
            ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── SU Throughput / Outage / Reward ──────────────────────────────────
        for data, ylabel, title, extra in [
            (metrics.su_throughputs, "Throughput (bits/s/Hz)",
             f"{metrics.name} — SU Throughput (ep {current_ep})", None),
            (metrics.outage_probs,   "Outage Probability",
             f"{metrics.name} — Outage Probability (ep {current_ep})", 0.05),
            (metrics.rewards,        "Episode Reward",
             f"{metrics.name} — Reward Curve (ep {current_ep})", None),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            raw = np.array(data); sm = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.2, lw=0.7)
            ax.plot(episodes, sm,  color=color, lw=2.2)
            if extra is not None:
                ax.axhline(extra, color="gray", ls="--", lw=1.2, label=f"{extra} target")
                ax.set_ylim(0, 1.0); ax.legend()
            ax.set_xlabel("Episode"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.grid(True, alpha=0.3); fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = f"CRN {metrics.name} Checkpoint Report — ep {current_ep}"

    print(f"  [checkpoint PDF] {pdf_path}")


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
        ep_sinr_s_list  = []
        ep_ber_list     = []
        ep_rs_list      = []
        ep_rp_list      = []
        ep_outage_list  = []
        ep_pu_ber_list  = []

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
            ber         = float(0.5 * erfc(math.sqrt(max(0.0, sinr_s))))
            sinr_db     = 10.0 * math.log10(max(1e-9, sinr_s))
            pu_ber      = float(0.5 * erfc(math.sqrt(max(0.0, sinr_p))))
            pu_sinr_db  = 10.0 * math.log10(max(1e-9, sinr_p))

            sinr_window.append(sinr_s)
            ep_sinr_s_list.append(sinr_s)
            ep_ber_list.append(ber)
            ep_rs_list.append(r_s)
            ep_rp_list.append(r_p)
            ep_outage_list.append(1 if sinr_s < SINR_THRESHOLD else 0)
            ep_pu_ber_list.append(pu_ber)

            # Scatter sample (SU + PU)
            if t in sample_steps:
                metrics.sinr_db_pts.append(sinr_db)
                metrics.ber_pts.append(ber)
                metrics.pu_sinr_db_pts.append(pu_sinr_db)
                metrics.pu_ber_pts.append(pu_ber)

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
        metrics.avg_pu_bers.append(float(np.mean(ep_pu_ber_list)))

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
    metrics.final_avg_pu_ber  = float(np.mean(metrics.avg_pu_bers[-tail:]))

    if verbose:
        print(f"\n  [{algo_name}] Done in {metrics.training_time_sec:.1f}s")
        print(f"  Final avg reward  : {metrics.final_avg_reward:.4f}")
        print(f"  Final SU throughput: {metrics.final_avg_su_tput:.4f} bits/s/Hz")
        print(f"  Final PU throughput: {metrics.final_avg_pu_tput:.4f} bits/s/Hz")
        print(f"  Final outage prob  : {metrics.final_outage_prob:.4f}")
        print(f"  Final avg BER      : {metrics.final_avg_ber:.6f}")

    return metrics


# =============================================================================
# CAMO-TD3 training loop (uses SequenceReplayBuffer + decomposed rewards)
# =============================================================================

def run_camo_algorithm(
    agent:           "CAMO_TD3Agent",
    n_episodes:      int,
    steps_per_ep:    int,
    verbose:         bool = True,
    checkpoint_every: int = 0,
    checkpoint_dir:  str  = "results",
) -> RunMetrics:
    """
    Train CAMO-TD3 agent. Same metric collection as run_algorithm but uses
    the SequenceReplayBuffer and decomposed reward storage.
    """
    algo_name = "CAMO-TD3"
    metrics = RunMetrics(name=algo_name)

    env = CRNEnvironment(steps_per_episode=steps_per_ep)
    buffer = SequenceReplayBuffer(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        seq_len=8,  # from config.SEQ_LEN
        max_size=REPLAY_BUFFER_SIZE,
        device=agent.device,
    )
    noise_sched = ExplorationNoise(
        std_start   = EXPLORATION_NOISE_STD * P_MAX,
        std_end     = EXPLORATION_NOISE_END * P_MAX,
        decay_steps = n_episodes,
    )
    rolling = RollingStats(window=100)
    sinr_window = deque(maxlen=OUTAGE_WINDOW_SIZE)
    scatter_budget = max(1, MAX_SCATTER_PTS // max(n_episodes, 1))

    t_start = time.perf_counter()

    if verbose:
        print()
        print("=" * 68)
        print(f"  [{algo_name}] Starting training   "
              f"episodes={n_episodes}  steps/ep={steps_per_ep}  "
              f"Nakagami-m={NAKAGAMI_M}")
        print(f"  Device: {agent.device}")
        print(f"  Components: GRU Belief Encoder + 6 Critics + Adaptive Lagrangian")
        print("=" * 68)

    for ep in range(n_episodes):
        state = env.reset()
        agent.reset_episode(state)
        buffer.reset_episode(state)

        ep_reward       = 0.0
        ep_sinr_s_list  = []
        ep_ber_list     = []
        ep_rs_list      = []
        ep_rp_list      = []
        ep_outage_list  = []
        ep_pu_ber_list  = []

        sample_steps = set(
            np.random.choice(steps_per_ep, size=min(scatter_budget, steps_per_ep), replace=False)
        )

        for t in range(steps_per_ep):
            action = agent.select_action(state, noise_sched.current_std)
            result = env.step(action)

            # Decompose reward
            r_tput, r_intf, r_energy = CAMO_TD3Agent.decompose_reward(result.info)

            buffer.add(
                state,
                np.array([action], dtype=np.float32),
                r_tput, r_intf, r_energy,
                result.state,
                result.done,
            )

            # Track PU violations for directional noise
            agent.record_violation(result.info["sinr_p"])

            # Metrics (same as run_algorithm)
            sinr_s   = result.info["sinr_s"]
            sinr_p   = result.info["sinr_p"]
            r_s      = result.info["r_s"]
            r_p      = float(np.log2(1.0 + sinr_p))
            ber      = float(0.5 * erfc(math.sqrt(max(0.0, sinr_s))))
            sinr_db  = 10.0 * math.log10(max(1e-9, sinr_s))
            pu_ber     = float(0.5 * erfc(math.sqrt(max(0.0, sinr_p))))
            pu_sinr_db = 10.0 * math.log10(max(1e-9, sinr_p))

            sinr_window.append(sinr_s)
            ep_sinr_s_list.append(sinr_s)
            ep_ber_list.append(ber)
            ep_rs_list.append(r_s)
            ep_rp_list.append(r_p)
            ep_outage_list.append(1 if sinr_s < SINR_THRESHOLD else 0)
            ep_pu_ber_list.append(pu_ber)

            if t in sample_steps:
                metrics.sinr_db_pts.append(sinr_db)
                metrics.ber_pts.append(ber)
                metrics.pu_sinr_db_pts.append(pu_sinr_db)
                metrics.pu_ber_pts.append(pu_ber)

            ep_reward += result.reward
            state      = result.state

            if buffer.is_ready:
                for _ in range(GRAD_UPDATES_PER_STEP):
                    agent.train_step(buffer, BATCH_SIZE)

        noise_sched.step()
        rolling.push(ep_reward)

        metrics.rewards.append(ep_reward)
        metrics.su_throughputs.append(float(np.mean(ep_rs_list)))
        metrics.pu_throughputs.append(float(np.mean(ep_rp_list)))
        metrics.outage_probs.append(float(np.mean(ep_outage_list)))
        metrics.avg_bers.append(float(np.mean(ep_ber_list)))
        metrics.avg_pu_bers.append(float(np.mean(ep_pu_ber_list)))

        if verbose and (ep + 1) % PRINT_EVERY == 0:
            avg100 = rolling.mean()
            elapsed = time.perf_counter() - t_start
            print(
                f"  ep {ep+1:>5}/{n_episodes}  "
                f"reward={ep_reward:>8.3f}  avg100={avg100:>8.3f}  "
                f"outage={metrics.outage_probs[-1]:.3f}  "
                f"SU_tput={metrics.su_throughputs[-1]:.3f}  "
                f"PU_tput={metrics.pu_throughputs[-1]:.3f}  "
                f"lam=[{agent.lambda1:.2f},{agent.lambda2:.2f},{agent.lambda3:.3f}]  "
                f"elapsed={elapsed:.0f}s"
            )

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0 and (ep + 1) < n_episodes:
            save_checkpoint_plots(metrics, ep + 1, checkpoint_dir, color=CAMO_TD3_COLOR)

    metrics.training_time_sec = time.perf_counter() - t_start

    tail = min(100, n_episodes)
    metrics.final_avg_reward  = float(np.mean(metrics.rewards[-tail:]))
    metrics.final_avg_su_tput = float(np.mean(metrics.su_throughputs[-tail:]))
    metrics.final_avg_pu_tput = float(np.mean(metrics.pu_throughputs[-tail:]))
    metrics.final_outage_prob = float(np.mean(metrics.outage_probs[-tail:]))
    metrics.final_avg_ber     = float(np.mean(metrics.avg_bers[-tail:]))
    metrics.final_avg_pu_ber  = float(np.mean(metrics.avg_pu_bers[-tail:]))

    if verbose:
        print(f"\n  [{algo_name}] Done in {metrics.training_time_sec:.1f}s")
        print(f"  Final avg reward  : {metrics.final_avg_reward:.4f}")
        print(f"  Final SU throughput: {metrics.final_avg_su_tput:.4f} bits/s/Hz")
        print(f"  Final PU throughput: {metrics.final_avg_pu_tput:.4f} bits/s/Hz")
        print(f"  Final outage prob  : {metrics.final_outage_prob:.4f}")
        print(f"  Final avg BER      : {metrics.final_avg_ber:.6f}")
        print(f"  Final lambdas      : [{agent.lambda1:.3f}, {agent.lambda2:.3f}, {agent.lambda3:.4f}]")

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
    all_metrics:  list["RunMetrics"] | tuple,
    output_path:  str,
    n_episodes:   int,
    steps_per_ep: int,
    # Legacy 2-arg signature support: generate_pdf(td3_m, ddpg_m, path, ...)
    _legacy_ddpg: "RunMetrics | None" = None,
) -> None:
    """
    Render all plots to a multi-page PDF report.
    Accepts a list of RunMetrics (any number of algorithms).
    """
    # ── Handle legacy call signature: generate_pdf(td3_m, ddpg_m, path, n, s) ──
    if isinstance(all_metrics, RunMetrics):
        # Called as generate_pdf(td3_metrics, ddpg_metrics, output_path, n, s)
        td3_m = all_metrics
        ddpg_m = output_path  # second positional arg
        output_path = n_episodes  # third positional arg
        n_episodes = steps_per_ep  # fourth
        steps_per_ep = _legacy_ddpg  # fifth
        all_metrics = [td3_m, ddpg_m]

    # Filter out empty/skipped metrics
    all_metrics = [m for m in all_metrics if m.rewards]

    if not all_metrics:
        print("  [generate_pdf] No metrics to report — skipping PDF.")
        return

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
    algo_names = " vs ".join(m.name for m in all_metrics)

    def _color_for(m):
        return ALGO_COLORS.get(m.name, "#555555")

    def _binned_mean(x_pts, y_pts, lo=-5, hi=25, n_bins=25):
        """Compute binned means for scatter overlay."""
        pts = np.column_stack([x_pts, y_pts])
        bins = np.linspace(lo, hi, n_bins)
        bx, by = [], []
        for i in range(len(bins) - 1):
            mask = (pts[:, 0] >= bins[i]) & (pts[:, 0] < bins[i + 1])
            if mask.sum() > 0:
                bx.append((bins[i] + bins[i + 1]) / 2)
                by.append(np.mean(pts[mask, 1]))
        return bx, by

    with PdfPages(output_path) as pdf:

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 1 — Title + Summary Table
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        title_txt = (
            f"CRN Power Control: {algo_names}\n"
            "Nakagami-m Fading Channel Performance Report"
        )
        fig.text(0.5, 0.88, title_txt, ha="center", va="top",
                 fontsize=18, fontweight="bold")
        fig.text(0.5, 0.80,
                 f"Nakagami-m = {NAKAGAMI_M}  |  Episodes = {n_episodes}  |  "
                 f"Steps/Episode = {steps_per_ep}  |  SINR Threshold = {SINR_THRESHOLD} dB",
                 ha="center", va="top", fontsize=11, color="#444444")

        # Build dynamic summary table
        col_labels = ["Metric"] + [m.name for m in all_metrics] + ["Winner"]

        metric_fields = [
            ("Avg Reward (last 100 ep)", "final_avg_reward",  "max", ".4f"),
            ("SU Throughput (bits/s/Hz)","final_avg_su_tput", "max", ".4f"),
            ("PU Throughput (bits/s/Hz)","final_avg_pu_tput", "max", ".4f"),
            ("Outage Probability",       "final_outage_prob", "min", ".4f"),
            ("Average BER",              "final_avg_ber",     "min", ".6f"),
            ("Training Time (s)",        "training_time_sec", None,  ".1f"),
        ]

        metrics_rows = []
        for label, attr, best_fn, fmt in metric_fields:
            vals = [getattr(m, attr) for m in all_metrics]
            row = [label] + [f"{v:{fmt}}" for v in vals]
            if best_fn == "max":
                winner = all_metrics[int(np.argmax(vals))].name
            elif best_fn == "min":
                winner = all_metrics[int(np.argmin(vals))].name
            else:
                winner = "-"
            row.append(winner)
            metrics_rows.append(row)

        table = ax.table(
            cellText   = metrics_rows,
            colLabels  = col_labels,
            cellLoc    = "center",
            loc        = "center",
            bbox       = [0.02, 0.08, 0.96, 0.62],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10 if len(all_metrics) > 2 else 11)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#2c3e50")
            table[0, j].set_text_props(color="white", fontweight="bold")

        for i in range(1, len(metrics_rows) + 1):
            winner = metrics_rows[i - 1][-1]
            cell = table[i, len(col_labels) - 1]
            wcolor = ALGO_COLORS.get(winner)
            if wcolor:
                cell.set_facecolor("#e8f4e8")
                cell.set_text_props(color=wcolor, fontweight="bold")
            bg = "#f8f8f8" if i % 2 == 0 else "white"
            for j in range(len(col_labels) - 1):
                table[i, j].set_facecolor(bg)

        fig.text(
            0.5, 0.02,
            "Ramaiah Institute of Technology, Bangalore  |  "
            "Cognitive Radio Network - RL Power Allocation",
            ha="center", fontsize=9, color="#888888"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 2 — SU SINR vs BER
        # ══════════════════════════════════════════════════════════════════════
        snr_range = np.linspace(-5, 25, 300)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(snr_range, theoretical_bpsk_ber(snr_range),
                    "k--", linewidth=1.4, label="BPSK Theoretical (AWGN)", alpha=0.7)
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, NAKAGAMI_M),
                    "k-.", linewidth=1.4,
                    label=f"BPSK Avg BER (Nakagami-m={int(NAKAGAMI_M)})", alpha=0.7)

        for m_obj in all_metrics:
            color = _color_for(m_obj)
            if m_obj.sinr_db_pts:
                ax.scatter(m_obj.sinr_db_pts, m_obj.ber_pts,
                           color=color, alpha=0.35, s=10, label=f"{m_obj.name} (simulated)")
                bx, by = _binned_mean(m_obj.sinr_db_pts, m_obj.ber_pts)
                if bx:
                    ax.semilogy(bx, by, color=color, linewidth=2.2,
                                marker="o", markersize=4, label=f"{m_obj.name} Mean BER")

        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(
            f"SINR vs BER - Nakagami-m={int(NAKAGAMI_M)} Fading Channel\n"
            f"(BPSK Modulation, Secondary User)"
        )
        ax.set_xlim(-5, 25); ax.set_ylim(1e-6, 0.6)
        ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 3 — PU SINR vs BER
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(snr_range, theoretical_bpsk_ber(snr_range),
                    "k--", linewidth=1.4, label="BPSK Theoretical (AWGN)", alpha=0.7)
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, NAKAGAMI_M),
                    "k-.", linewidth=1.4,
                    label=f"BPSK Avg BER (Nakagami-m={int(NAKAGAMI_M)})", alpha=0.7)

        for m_obj in all_metrics:
            color = _color_for(m_obj)
            if m_obj.pu_sinr_db_pts:
                ax.scatter(m_obj.pu_sinr_db_pts, m_obj.pu_ber_pts,
                           color=color, alpha=0.25, s=10, label=f"{m_obj.name} PU (simulated)")
                bx, by = _binned_mean(m_obj.pu_sinr_db_pts, m_obj.pu_ber_pts)
                if bx:
                    ax.semilogy(bx, by, color=color, linewidth=2.2,
                                marker="s", markersize=4, label=f"{m_obj.name} PU Mean BER")

        ax.axvline(x=10*np.log10(SINR_THRESHOLD), color="green", ls="--",
                   lw=1.4, label=f"PU SINR Threshold ({SINR_THRESHOLD:.0f} linear)")
        ax.set_xlabel("PU SINR (dB)")
        ax.set_ylabel("Bit Error Rate (BER)")
        ax.set_title(
            f"Primary User SINR vs BER - Nakagami-m={int(NAKAGAMI_M)} Fading Channel\n"
            f"(BPSK Modulation | Effect of SU Interference)"
        )
        ax.set_xlim(-5, 25); ax.set_ylim(1e-6, 0.6)
        ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 4 — Secondary User Throughput
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.su_throughputs); sm = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(episodes, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
            ax.fill_between(episodes, np.maximum(0, sm - raw.std()), sm + raw.std(),
                            color=color, alpha=ALPHA_FILL)
        ax.set_xlabel("Episode"); ax.set_ylabel("Throughput (bits/s/Hz)")
        ax.set_title("Secondary User (SU) Throughput vs Episodes")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 5 — Primary User Throughput
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.pu_throughputs); sm = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(episodes, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
            ax.fill_between(episodes, np.maximum(0, sm - raw.std()), sm + raw.std(),
                            color=color, alpha=ALPHA_FILL)
        ax.set_xlabel("Episode"); ax.set_ylabel("Throughput (bits/s/Hz)")
        ax.set_title("Primary User (PU) Throughput vs Episodes")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 6 — Outage Probability
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.outage_probs); sm = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(episodes, sm,  color=color, linewidth=2.0, label=f"{m_obj.name} (smoothed)")
        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1.2, label="5% target")
        ax.set_xlabel("Episode"); ax.set_ylabel("Outage Probability")
        ax.set_title(f"Outage Probability vs Episodes\n"
                     f"(SINR_s < {SINR_THRESHOLD} threshold, rolling per episode)")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.0)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 7 — Individual Reward Curves
        # ══════════════════════════════════════════════════════════════════════
        n_algos = len(all_metrics)
        cols = min(n_algos, 3)
        rows_grid = math.ceil(n_algos / cols)
        fig, axes = plt.subplots(rows_grid, cols, figsize=(5 * cols, 5 * rows_grid),
                                 sharey=False, squeeze=False)
        for idx, m_obj in enumerate(all_metrics):
            ax = axes[idx // cols][idx % cols]
            color = _color_for(m_obj)
            raw = np.array(m_obj.rewards); sm = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.2, linewidth=0.7)
            ax.plot(episodes, sm,  color=color, linewidth=2.2, label="Smoothed reward")
            ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
            ax.set_title(f"{m_obj.name} Reward Curve"); ax.legend(); ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for idx in range(n_algos, rows_grid * cols):
            axes[idx // cols][idx % cols].set_visible(False)
        fig.suptitle(f"Episode Reward Curves - {algo_names}", fontsize=14, y=1.01)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ══════════════════════════════════════════════════════════════════════
        # PAGE 8 — Combined overlay reward comparison
        # ══════════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m_obj in all_metrics:
            color = _color_for(m_obj)
            raw = np.array(m_obj.rewards); sm = smooth(raw, window=20)
            ax.plot(episodes, raw, color=color, alpha=0.18, linewidth=0.7)
            ax.plot(episodes, sm,  color=color, linewidth=2.2, label=f"{m_obj.name}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
        ax.set_title(f"{algo_names} - Reward Comparison")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"CRN {algo_names} Comparison Report"
        d["Author"]  = "Ramaiah Institute of Technology"
        d["Subject"] = f"Nakagami-m={NAKAGAMI_M} CRN Power Allocation via RL"

    print(f"\n  PDF report saved: {os.path.abspath(output_path)}")


# =============================================================================
# Entry point
# =============================================================================

VALID_AGENTS = {"td3", "ddpg", "camo-td3"}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TD3, DDPG, and/or CAMO-TD3 on the CRN and generate a PDF comparison report."
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
    p.add_argument("--parallel",         action="store_true",
                   help="Train algorithms in parallel on the same GPU")
    p.add_argument("--agents", type=str, default=None,
                   help="Comma-separated list of agents to train: td3,ddpg,camo-td3  "
                        "(default: td3,ddpg — overrides --no-ddpg)")
    return p.parse_args()


def _train_worker(algo_name, n_episodes, steps_per_ep, checkpoint_every, checkpoint_dir, seed):
    """Worker function for parallel training (runs in a separate process)."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if algo_name == "TD3":
        agent = TD3Agent()
        return run_algorithm("TD3", agent, n_episodes, steps_per_ep,
                             checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir)
    elif algo_name == "DDPG":
        agent = DDPGAgent()
        return run_algorithm("DDPG", agent, n_episodes, steps_per_ep,
                             checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir)
    elif algo_name == "CAMO-TD3":
        agent = CAMO_TD3Agent()
        return run_camo_algorithm(agent, n_episodes, steps_per_ep,
                                  checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def main() -> None:
    args = parse_args()

    # Determine which agents to train
    if args.agents:
        agent_list = [a.strip().lower() for a in args.agents.split(",")]
        for a in agent_list:
            if a not in VALID_AGENTS:
                print(f"  ERROR: Unknown agent '{a}'. Valid: {', '.join(sorted(VALID_AGENTS))}")
                sys.exit(1)
    elif args.no_ddpg:
        agent_list = ["td3"]
    else:
        agent_list = ["td3", "ddpg"]

    # Canonical names for display
    DISPLAY_NAMES = {"td3": "TD3", "ddpg": "DDPG", "camo-td3": "CAMO-TD3"}
    agent_display = [DISPLAY_NAMES[a] for a in agent_list]

    # Reproducibility
    np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)

    # GPU speed: enable TF32 on Ampere/Ada GPUs (RTX 30xx/40xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    checkpoint_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "results"

    print("\n" + "=" * 68)
    print(f"  CRN {' vs '.join(agent_display)} -- Comparison Training & Report Generator")
    print(f"  Nakagami-m = {NAKAGAMI_M}  |  Episodes = {args.episodes}"
          f"  |  Steps/ep = {args.steps_per_ep}")
    if args.checkpoint_every > 0:
        print(f"  Checkpoint graphs every {args.checkpoint_every} episodes -> {checkpoint_dir}/")
    if args.parallel and len(agent_list) > 1:
        print(f"  Mode: PARALLEL ({' + '.join(agent_display)} training simultaneously)")
    print("=" * 68)

    all_metrics: list[RunMetrics] = []

    if args.parallel and len(agent_list) > 1:
        # ── Parallel training ─────────────────────────────────────────────────
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=len(agent_list)) as pool:
            futures = {}
            for i, name in enumerate(agent_display):
                f = pool.submit(_train_worker, name, args.episodes,
                                args.steps_per_ep, args.checkpoint_every,
                                checkpoint_dir, args.seed + i)
                futures[name] = f

            for name in agent_display:
                all_metrics.append(futures[name].result())
    else:
        # ── Sequential training ───────────────────────────────────────────────
        for i, algo_key in enumerate(agent_list):
            name = DISPLAY_NAMES[algo_key]
            seed_i = args.seed + i

            np.random.seed(seed_i)
            import torch; torch.manual_seed(seed_i)

            if algo_key == "td3":
                agent = TD3Agent()
                m = run_algorithm("TD3", agent, args.episodes, args.steps_per_ep,
                                  checkpoint_every=args.checkpoint_every,
                                  checkpoint_dir=checkpoint_dir)
            elif algo_key == "ddpg":
                agent = DDPGAgent()
                m = run_algorithm("DDPG", agent, args.episodes, args.steps_per_ep,
                                  checkpoint_every=args.checkpoint_every,
                                  checkpoint_dir=checkpoint_dir)
            elif algo_key == "camo-td3":
                agent = CAMO_TD3Agent()
                m = run_camo_algorithm(agent, args.episodes, args.steps_per_ep,
                                       checkpoint_every=args.checkpoint_every,
                                       checkpoint_dir=checkpoint_dir)
            all_metrics.append(m)

    # ── Generate PDF ───────────────────────────────────────────────────────────
    print("\n  Generating PDF report...")
    generate_pdf(
        all_metrics  = all_metrics,
        output_path  = args.output,
        n_episodes   = args.episodes,
        steps_per_ep = args.steps_per_ep,
    )

    print("\n  All done!")
    print(f"  Report: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
