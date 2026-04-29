# =============================================================================
# train_compare_colab.py — TD3 vs DDPG vs CAMO-TD3 with Checkpoint/Resume
#
# New features over train_compare.py:
#   - save_checkpoint()  : saves agent weights + metrics JSON every N episodes
#   - load_checkpoint()  : finds latest checkpoint and resumes from it
#   - run_algorithm() / run_camo_algorithm() accept start_episode + pre-metrics
# =============================================================================

from __future__ import annotations
import argparse, glob, json, math, os, sys, time
from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import erfc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from config import (
    STATE_DIM, ACTION_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    EXPLORATION_NOISE_STD, EXPLORATION_NOISE_END,
    BATCH_SIZE, SINR_THRESHOLD, NAKAGAMI_M, NAKAGAMI_OMEGA,
    GRAD_UPDATES_PER_STEP,
)
from environment import CRNEnvironment
from td3       import TD3Agent,  ReplayBuffer
from ddpg      import DDPGAgent
from camo_td3  import CAMO_TD3Agent, SequenceReplayBuffer
from utils     import RollingStats, ExplorationNoise

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_EPISODES         = 3000
DEFAULT_STEPS_PER_EP     = 200
DEFAULT_OUTPUT           = "results/crn_comparison_report.pdf"
DEFAULT_CHECKPOINT_EVERY = 300
PRINT_EVERY              = 50
MAX_SCATTER_PTS          = 8000
OUTAGE_WINDOW_SIZE       = 500

TD3_COLOR      = "#1f77b4"
DDPG_COLOR     = "#d62728"
CAMO_TD3_COLOR = "#ff7f0e"
ALPHA_FILL     = 0.15
ALGO_COLORS    = {"TD3": TD3_COLOR, "DDPG": DDPG_COLOR, "CAMO-TD3": CAMO_TD3_COLOR}

# =============================================================================
# Metrics container
# =============================================================================

@dataclass
class RunMetrics:
    name: str
    rewards:        List[float] = field(default_factory=list)
    su_throughputs: List[float] = field(default_factory=list)
    pu_throughputs: List[float] = field(default_factory=list)
    outage_probs:   List[float] = field(default_factory=list)
    avg_bers:       List[float] = field(default_factory=list)
    avg_pu_bers:    List[float] = field(default_factory=list)
    sinr_db_pts:    List[float] = field(default_factory=list)
    ber_pts:        List[float] = field(default_factory=list)
    pu_sinr_db_pts: List[float] = field(default_factory=list)
    pu_ber_pts:     List[float] = field(default_factory=list)
    final_avg_reward:   float = 0.0
    final_avg_su_tput:  float = 0.0
    final_avg_pu_tput:  float = 0.0
    final_outage_prob:  float = 0.0
    final_avg_ber:      float = 0.0
    final_avg_pu_ber:   float = 0.0
    training_time_sec:  float = 0.0


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _metrics_to_dict(m: RunMetrics, episode: int) -> dict:
    return {
        "name": m.name, "episode": episode,
        "rewards":        m.rewards,
        "su_throughputs": m.su_throughputs,
        "pu_throughputs": m.pu_throughputs,
        "outage_probs":   m.outage_probs,
        "avg_bers":       m.avg_bers,
        "avg_pu_bers":    m.avg_pu_bers,
        # cap scatter pts to keep JSON small
        "sinr_db_pts":    m.sinr_db_pts[-2000:],
        "ber_pts":        m.ber_pts[-2000:],
        "pu_sinr_db_pts": m.pu_sinr_db_pts[-2000:],
        "pu_ber_pts":     m.pu_ber_pts[-2000:],
        "training_time_sec": m.training_time_sec,
    }

def _metrics_from_dict(d: dict) -> tuple[RunMetrics, int]:
    m = RunMetrics(name=d["name"])
    m.rewards        = d["rewards"]
    m.su_throughputs = d["su_throughputs"]
    m.pu_throughputs = d["pu_throughputs"]
    m.outage_probs   = d["outage_probs"]
    m.avg_bers       = d["avg_bers"]
    m.avg_pu_bers    = d.get("avg_pu_bers", [])
    m.sinr_db_pts    = d.get("sinr_db_pts", [])
    m.ber_pts        = d.get("ber_pts", [])
    m.pu_sinr_db_pts = d.get("pu_sinr_db_pts", [])
    m.pu_ber_pts     = d.get("pu_ber_pts", [])
    m.training_time_sec = d.get("training_time_sec", 0.0)
    return m, int(d["episode"])


def save_checkpoint(
    algo_name: str,
    episode: int,
    metrics: RunMetrics,
    agent,
    checkpoint_dir: str,
) -> str:
    """Save agent weights + metrics JSON. Returns checkpoint directory path."""
    ep_dir = os.path.join(checkpoint_dir, f"{algo_name.replace('-','_')}_ep{episode:05d}")
    os.makedirs(ep_dir, exist_ok=True)
    agent.save(ep_dir)
    with open(os.path.join(ep_dir, "metrics.json"), "w") as f:
        json.dump(_metrics_to_dict(metrics, episode), f)
    print(f"  ✔ Checkpoint saved → {ep_dir}")
    return ep_dir


def find_latest_checkpoint(algo_name: str, checkpoint_dir: str) -> str | None:
    """Return the path of the latest checkpoint for algo_name, or None."""
    pattern = os.path.join(
        checkpoint_dir,
        f"{algo_name.replace('-','_')}_ep*",
        "metrics.json",
    )
    found = sorted(glob.glob(pattern))
    return os.path.dirname(found[-1]) if found else None


def load_checkpoint(algo_name: str, checkpoint_dir: str):
    """
    Load the latest checkpoint for algo_name.
    Returns (metrics, start_episode, ckpt_path) or (None, 0, None) if none found.
    """
    ckpt_path = find_latest_checkpoint(algo_name, checkpoint_dir)
    if not ckpt_path:
        return None, 0, None
    with open(os.path.join(ckpt_path, "metrics.json")) as f:
        d = json.load(f)
    metrics, start_ep = _metrics_from_dict(d)
    print(f"  ↩ Checkpoint found → {ckpt_path}  (resuming from episode {start_ep})")
    return metrics, start_ep, ckpt_path


# =============================================================================
# Smoothing helpers
# =============================================================================

def smooth(values, window: int = 20) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=float)
    if window >= len(arr):
        return arr
    kernel = np.ones(window) / window
    pad = np.pad(arr, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def theoretical_bpsk_ber(snr_db: np.ndarray) -> np.ndarray:
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_lin))


def nakagami_avg_ber_bpsk(snr_db: np.ndarray, m: float = NAKAGAMI_M) -> np.ndarray:
    snr_lin = 10.0 ** (snr_db / 10.0)
    m_int = int(round(m))
    mu = np.sqrt(snr_lin / (m_int + snr_lin))
    ber = np.zeros_like(snr_lin)
    coeff_base = ((1.0 - mu) / 2.0) ** m_int
    for k in range(m_int):
        binom = math.comb(m_int - 1 + k, k)
        ber += coeff_base * binom * ((1.0 + mu) / 2.0) ** k
    return np.clip(ber, 1e-12, 0.5)


# =============================================================================
# Training loops
# =============================================================================

def run_algorithm(
    algo_name: str,
    agent,
    n_episodes: int,
    steps_per_ep: int,
    verbose: bool = True,
    checkpoint_every: int = 0,
    checkpoint_dir: str = "results",
    resume: bool = True,
) -> RunMetrics:

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    start_ep = 0
    metrics  = RunMetrics(name=algo_name)

    if resume:
        prev_metrics, start_ep, ckpt_path = load_checkpoint(algo_name, checkpoint_dir)
        if prev_metrics is not None:
            metrics = prev_metrics
            agent.load(ckpt_path)
            if start_ep >= n_episodes:
                print(f"  [{algo_name}] Already complete ({start_ep}/{n_episodes} ep). Skipping.")
                _finalize_metrics(metrics, n_episodes)
                return metrics

    env    = CRNEnvironment(steps_per_episode=steps_per_ep)
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, REPLAY_BUFFER_SIZE, device=agent.device)
    noise_sched = ExplorationNoise(
        std_start   = EXPLORATION_NOISE_STD * P_MAX,
        std_end     = EXPLORATION_NOISE_END  * P_MAX,
        decay_steps = n_episodes,
    )
    # Fast-forward noise scheduler to current episode
    for _ in range(start_ep):
        noise_sched.step()

    rolling = RollingStats(window=100)
    for r in metrics.rewards:
        rolling.push(r)

    sinr_window    = deque(maxlen=OUTAGE_WINDOW_SIZE)
    scatter_budget = max(1, MAX_SCATTER_PTS // max(n_episodes, 1))
    t_start        = time.perf_counter()

    if verbose:
        remaining = n_episodes - start_ep
        print(f"\n{'='*68}")
        print(f"  [{algo_name}] episodes={n_episodes}  steps/ep={steps_per_ep}  "
              f"start_ep={start_ep}  remaining={remaining}")
        print(f"  Device: {agent.device}  |  Nakagami-m={NAKAGAMI_M}")
        print(f"{'='*68}")

    for ep in range(start_ep, n_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_sinr_s_list, ep_ber_list = [], []
        ep_rs_list, ep_rp_list      = [], []
        ep_outage_list, ep_pu_ber_list = [], []

        sample_steps = set(np.random.choice(
            steps_per_ep, size=min(scatter_budget, steps_per_ep), replace=False))

        for t in range(steps_per_ep):
            action = agent.select_action(state, noise_sched.current_std)
            result = env.step(action)
            buffer.add(state, np.array([action], dtype=np.float32),
                       result.reward, result.state, result.done)

            sinr_s  = result.info["sinr_s"]; sinr_p = result.info["sinr_p"]
            r_s     = result.info["r_s"]
            r_p     = float(np.log2(1.0 + sinr_p))
            ber     = float(0.5 * erfc(math.sqrt(max(0.0, sinr_s))))
            sinr_db = 10.0 * math.log10(max(1e-9, sinr_s))
            pu_ber      = float(0.5 * erfc(math.sqrt(max(0.0, sinr_p))))
            pu_sinr_db  = 10.0 * math.log10(max(1e-9, sinr_p))

            sinr_window.append(sinr_s)
            ep_sinr_s_list.append(sinr_s); ep_ber_list.append(ber)
            ep_rs_list.append(r_s);        ep_rp_list.append(r_p)
            ep_outage_list.append(1 if sinr_s < SINR_THRESHOLD else 0)
            ep_pu_ber_list.append(pu_ber)

            if t in sample_steps:
                metrics.sinr_db_pts.append(sinr_db); metrics.ber_pts.append(ber)
                metrics.pu_sinr_db_pts.append(pu_sinr_db); metrics.pu_ber_pts.append(pu_ber)

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
        metrics.training_time_sec += time.perf_counter() - t_start
        t_start = time.perf_counter()

        if verbose and (ep + 1) % PRINT_EVERY == 0:
            print(f"  ep {ep+1:>5}/{n_episodes}  "
                  f"reward={ep_reward:>8.3f}  avg100={rolling.mean():>8.3f}  "
                  f"outage={metrics.outage_probs[-1]:.3f}  "
                  f"SU_tput={metrics.su_throughputs[-1]:.3f}")

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0:
            save_checkpoint(algo_name, ep + 1, metrics, agent, checkpoint_dir)

    _finalize_metrics(metrics, n_episodes)
    return metrics


def run_camo_algorithm(
    agent: CAMO_TD3Agent,
    n_episodes: int,
    steps_per_ep: int,
    verbose: bool = True,
    checkpoint_every: int = 0,
    checkpoint_dir: str = "results",
    resume: bool = True,
) -> RunMetrics:
    algo_name = "CAMO-TD3"

    start_ep = 0
    metrics  = RunMetrics(name=algo_name)

    if resume:
        prev_metrics, start_ep, ckpt_path = load_checkpoint(algo_name, checkpoint_dir)
        if prev_metrics is not None:
            metrics = prev_metrics
            agent.load(ckpt_path)
            if start_ep >= n_episodes:
                print(f"  [{algo_name}] Already complete ({start_ep}/{n_episodes} ep). Skipping.")
                _finalize_metrics(metrics, n_episodes)
                return metrics

    env    = CRNEnvironment(steps_per_episode=steps_per_ep)
    buffer = SequenceReplayBuffer(STATE_DIM, ACTION_DIM, seq_len=8,
                                  max_size=REPLAY_BUFFER_SIZE, device=agent.device)
    noise_sched = ExplorationNoise(
        std_start   = EXPLORATION_NOISE_STD * P_MAX,
        std_end     = EXPLORATION_NOISE_END  * P_MAX,
        decay_steps = n_episodes,
    )
    for _ in range(start_ep):
        noise_sched.step()

    rolling = RollingStats(window=100)
    for r in metrics.rewards:
        rolling.push(r)

    sinr_window    = deque(maxlen=OUTAGE_WINDOW_SIZE)
    scatter_budget = max(1, MAX_SCATTER_PTS // max(n_episodes, 1))
    t_start        = time.perf_counter()

    if verbose:
        print(f"\n{'='*68}")
        print(f"  [{algo_name}] episodes={n_episodes}  start_ep={start_ep}  "
              f"remaining={n_episodes-start_ep}")
        print(f"  GRU Belief Encoder + 6 Critics + Adaptive Lagrangian")
        print(f"{'='*68}")

    for ep in range(start_ep, n_episodes):
        state = env.reset()
        agent.reset_episode(state)
        buffer.reset_episode(state)

        ep_reward = 0.0
        ep_sinr_s_list, ep_ber_list = [], []
        ep_rs_list, ep_rp_list      = [], []
        ep_outage_list, ep_pu_ber_list = [], []

        sample_steps = set(np.random.choice(
            steps_per_ep, size=min(scatter_budget, steps_per_ep), replace=False))

        for t in range(steps_per_ep):
            action = agent.select_action(state, noise_sched.current_std)
            result = env.step(action)
            r_tput, r_intf, r_energy = CAMO_TD3Agent.decompose_reward(result.info)
            buffer.add(state, np.array([action], dtype=np.float32),
                       r_tput, r_intf, r_energy, result.state, result.done)
            agent.record_violation(result.info["sinr_p"])

            sinr_s  = result.info["sinr_s"]; sinr_p = result.info["sinr_p"]
            r_s     = result.info["r_s"]
            r_p     = float(np.log2(1.0 + sinr_p))
            ber     = float(0.5 * erfc(math.sqrt(max(0.0, sinr_s))))
            sinr_db = 10.0 * math.log10(max(1e-9, sinr_s))
            pu_ber      = float(0.5 * erfc(math.sqrt(max(0.0, sinr_p))))
            pu_sinr_db  = 10.0 * math.log10(max(1e-9, sinr_p))

            sinr_window.append(sinr_s)
            ep_sinr_s_list.append(sinr_s); ep_ber_list.append(ber)
            ep_rs_list.append(r_s);        ep_rp_list.append(r_p)
            ep_outage_list.append(1 if sinr_s < SINR_THRESHOLD else 0)
            ep_pu_ber_list.append(pu_ber)
            if t in sample_steps:
                metrics.sinr_db_pts.append(sinr_db); metrics.ber_pts.append(ber)
                metrics.pu_sinr_db_pts.append(pu_sinr_db); metrics.pu_ber_pts.append(pu_ber)

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
        metrics.training_time_sec += time.perf_counter() - t_start
        t_start = time.perf_counter()

        if verbose and (ep + 1) % PRINT_EVERY == 0:
            print(f"  ep {ep+1:>5}/{n_episodes}  "
                  f"reward={ep_reward:>8.3f}  avg100={rolling.mean():>8.3f}  "
                  f"lam=[{agent.lambda1:.2f},{agent.lambda2:.2f},{agent.lambda3:.3f}]")

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0:
            save_checkpoint(algo_name, ep + 1, metrics, agent, checkpoint_dir)

    _finalize_metrics(metrics, n_episodes)
    return metrics


def _finalize_metrics(metrics: RunMetrics, n_episodes: int) -> None:
    tail = min(100, len(metrics.rewards))
    if tail == 0:
        return
    metrics.final_avg_reward  = float(np.mean(metrics.rewards[-tail:]))
    metrics.final_avg_su_tput = float(np.mean(metrics.su_throughputs[-tail:]))
    metrics.final_avg_pu_tput = float(np.mean(metrics.pu_throughputs[-tail:]))
    metrics.final_outage_prob = float(np.mean(metrics.outage_probs[-tail:]))
    metrics.final_avg_ber     = float(np.mean(metrics.avg_bers[-tail:]))
    metrics.final_avg_pu_ber  = float(np.mean(metrics.avg_pu_bers[-tail:]))


# =============================================================================
# PDF report (same as original)
# =============================================================================

def generate_pdf(all_metrics, output_path, n_episodes, steps_per_ep):
    all_metrics = [m for m in all_metrics if m.rewards]
    if not all_metrics:
        print("  No metrics to report — skipping PDF.")
        return
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                          "axes.titlesize": 13, "axes.labelsize": 12})
    n_ep_actual = max(len(m.rewards) for m in all_metrics)
    episodes = np.arange(1, n_ep_actual + 1)

    def _color(m): return ALGO_COLORS.get(m.name, "#555555")
    def _binned_mean(x, y, lo=-5, hi=25, n_bins=25):
        pts = np.column_stack([x, y])
        bins = np.linspace(lo, hi, n_bins)
        bx, by = [], []
        for i in range(len(bins)-1):
            mask = (pts[:,0]>=bins[i]) & (pts[:,0]<bins[i+1])
            if mask.sum()>0:
                bx.append((bins[i]+bins[i+1])/2); by.append(np.mean(pts[mask,1]))
        return bx, by

    snr_range = np.linspace(-5, 25, 300)
    algo_names = " vs ".join(m.name for m in all_metrics)

    with PdfPages(output_path) as pdf:

        # PAGE 1 — Summary table
        fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
        fig.text(0.5, 0.90, f"CRN Power Control: {algo_names}\nNakagami-m Fading Performance Report",
                 ha="center", fontsize=17, fontweight="bold")
        fig.text(0.5, 0.82,
                 f"m={NAKAGAMI_M}  |  Episodes={n_ep_actual}  |  Steps/ep={steps_per_ep}  "
                 f"|  SINR_th={SINR_THRESHOLD}",
                 ha="center", fontsize=10, color="#444")
        col_labels = ["Metric"] + [m.name for m in all_metrics] + ["Winner"]
        rows, fields = [], [
            ("Avg Reward (last 100)",    "final_avg_reward",  "max", ".4f"),
            ("SU Throughput (bits/s/Hz)","final_avg_su_tput", "max", ".4f"),
            ("PU Throughput (bits/s/Hz)","final_avg_pu_tput", "max", ".4f"),
            ("Outage Probability",       "final_outage_prob", "min", ".4f"),
            ("Average BER",              "final_avg_ber",     "min", ".6f"),
            ("Training Time (s)",        "training_time_sec", None,  ".1f"),
        ]
        for label, attr, best_fn, fmt in fields:
            vals = [getattr(m, attr) for m in all_metrics]
            row  = [label] + [f"{v:{fmt}}" for v in vals]
            if best_fn == "max": winner = all_metrics[int(np.argmax(vals))].name
            elif best_fn == "min": winner = all_metrics[int(np.argmin(vals))].name
            else: winner = "-"
            row.append(winner); rows.append(row)
        tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                       loc="center", bbox=[0.02, 0.08, 0.96, 0.62])
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for j in range(len(col_labels)):
            tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 2 — SU SINR vs BER
        fig, ax = plt.subplots(figsize=(10,6))
        ax.semilogy(snr_range, theoretical_bpsk_ber(snr_range), "k--", lw=1.4, label="BPSK AWGN", alpha=0.7)
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, NAKAGAMI_M), "k-.",
                    lw=1.4, label=f"Nakagami-m={int(NAKAGAMI_M)}", alpha=0.7)
        for m in all_metrics:
            c = _color(m)
            if m.sinr_db_pts:
                ax.scatter(m.sinr_db_pts, m.ber_pts, color=c, alpha=0.3, s=10, label=f"{m.name} (sim)")
                bx, by = _binned_mean(m.sinr_db_pts, m.ber_pts)
                if bx: ax.semilogy(bx, by, color=c, lw=2.2, marker="o", ms=4, label=f"{m.name} mean")
        ax.set(xlabel="SINR (dB)", ylabel="BER", xlim=(-5,25), ylim=(1e-6,0.6),
               title="SU SINR vs BER — Nakagami-m Fading")
        ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 3 — PU SINR vs BER
        fig, ax = plt.subplots(figsize=(10,6))
        ax.semilogy(snr_range, nakagami_avg_ber_bpsk(snr_range, NAKAGAMI_M), "k-.",
                    lw=1.4, label=f"Nakagami-m={int(NAKAGAMI_M)}", alpha=0.7)
        for m in all_metrics:
            c = _color(m)
            if m.pu_sinr_db_pts:
                ax.scatter(m.pu_sinr_db_pts, m.pu_ber_pts, color=c, alpha=0.25, s=10, label=f"{m.name} PU")
                bx, by = _binned_mean(m.pu_sinr_db_pts, m.pu_ber_pts)
                if bx: ax.semilogy(bx, by, color=c, lw=2.2, marker="s", ms=4)
        ax.axvline(x=10*np.log10(SINR_THRESHOLD), color="green", ls="--", lw=1.4, label="PU threshold")
        ax.set(xlabel="PU SINR (dB)", ylabel="BER", xlim=(-5,25), ylim=(1e-6,0.6),
               title="PU SINR vs BER — Effect of SU Interference")
        ax.legend(loc="lower left"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGES 4-6 — SU Tput, PU Tput, Outage
        for data_key, ylabel, title, hline in [
            ("su_throughputs", "Throughput (bits/s/Hz)", "SU Throughput vs Episodes", None),
            ("pu_throughputs", "Throughput (bits/s/Hz)", "PU Throughput vs Episodes", None),
            ("outage_probs",   "Outage Probability",      "Outage Probability vs Episodes", 0.05),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for m in all_metrics:
                c = _color(m); raw = np.array(getattr(m, data_key))
                ep_x = np.arange(1, len(raw)+1)
                sm = smooth(raw, window=20)
                ax.plot(ep_x, raw, color=c, alpha=0.2, lw=0.7)
                ax.plot(ep_x, sm,  color=c, lw=2.0, label=m.name)
                ax.fill_between(ep_x, np.maximum(0, sm-raw.std()), sm+raw.std(), color=c, alpha=ALPHA_FILL)
            if hline: ax.axhline(hline, color="gray", ls="--", lw=1.2, label="5% target"); ax.set_ylim(0,1)
            ax.set(xlabel="Episode", ylabel=ylabel, title=title)
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 7 — Individual reward curves
        n_algos = len(all_metrics)
        cols = min(n_algos, 3); rows_g = math.ceil(n_algos/cols)
        fig, axes = plt.subplots(rows_g, cols, figsize=(5*cols, 5*rows_g), squeeze=False)
        for idx, m in enumerate(all_metrics):
            ax = axes[idx//cols][idx%cols]; c = _color(m)
            raw = np.array(m.rewards); ep_x = np.arange(1, len(raw)+1)
            ax.plot(ep_x, raw, color=c, alpha=0.2, lw=0.7)
            ax.plot(ep_x, smooth(raw,20), color=c, lw=2.2, label="Smoothed")
            ax.set(xlabel="Episode", ylabel="Episode Reward", title=f"{m.name} Reward Curve")
            ax.legend(); ax.grid(True, alpha=0.3)
        for idx in range(n_algos, rows_g*cols):
            axes[idx//cols][idx%cols].set_visible(False)
        fig.suptitle("Episode Reward Curves", fontsize=14); fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # PAGE 8 — Overlay reward comparison
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m in all_metrics:
            c = _color(m); raw = np.array(m.rewards); ep_x = np.arange(1, len(raw)+1)
            ax.plot(ep_x, raw, color=c, alpha=0.18, lw=0.7)
            ax.plot(ep_x, smooth(raw,20), color=c, lw=2.2, label=m.name)
        ax.set(xlabel="Episode", ylabel="Episode Reward", title=f"{algo_names} — Reward Comparison")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        d = pdf.infodict()
        d["Title"] = f"CRN {algo_names} Report"

    print(f"\n  PDF report saved: {os.path.abspath(output_path)}")


# =============================================================================
# Entry point
# =============================================================================

VALID_AGENTS = {"td3", "ddpg", "camo-td3"}
DISPLAY      = {"td3": "TD3", "ddpg": "DDPG", "camo-td3": "CAMO-TD3"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",          type=int,  default=DEFAULT_EPISODES)
    p.add_argument("--steps-per-ep",      type=int,  default=DEFAULT_STEPS_PER_EP)
    p.add_argument("--output",            type=str,  default=DEFAULT_OUTPUT)
    p.add_argument("--seed",              type=int,  default=42)
    p.add_argument("--agents",            type=str,  default="td3,ddpg,camo-td3")
    p.add_argument("--checkpoint-every",  type=int,  default=DEFAULT_CHECKPOINT_EVERY)
    p.add_argument("--checkpoint-dir",    type=str,  default="results/checkpoints")
    p.add_argument("--no-resume",         action="store_true",
                   help="Ignore existing checkpoints and train from scratch")
    return p.parse_args()


def main():
    args = parse_args()
    agent_list = [a.strip().lower() for a in args.agents.split(",")]
    resume = not args.no_resume

    np.random.seed(args.seed)
    import torch; torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  CRN Comparison: {' vs '.join(DISPLAY[a] for a in agent_list)}")
    print(f"  Episodes={args.episodes} | Steps/ep={args.steps_per_ep} | "
          f"Checkpoint every {args.checkpoint_every} ep")
    print(f"  Resume={resume} | Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*68}")

    all_metrics = []
    for i, key in enumerate(agent_list):
        np.random.seed(args.seed + i)
        import torch; torch.manual_seed(args.seed + i)
        name = DISPLAY[key]
        if key == "td3":
            agent = TD3Agent()
            m = run_algorithm(name, agent, args.episodes, args.steps_per_ep,
                              checkpoint_every=args.checkpoint_every,
                              checkpoint_dir=args.checkpoint_dir,
                              resume=resume)
        elif key == "ddpg":
            agent = DDPGAgent()
            m = run_algorithm(name, agent, args.episodes, args.steps_per_ep,
                              checkpoint_every=args.checkpoint_every,
                              checkpoint_dir=args.checkpoint_dir,
                              resume=resume)
        elif key == "camo-td3":
            agent = CAMO_TD3Agent()
            m = run_camo_algorithm(agent, args.episodes, args.steps_per_ep,
                                   checkpoint_every=args.checkpoint_every,
                                   checkpoint_dir=args.checkpoint_dir,
                                   resume=resume)
        all_metrics.append(m)

    print("\n  Generating PDF report...")
    generate_pdf(all_metrics, args.output, args.episodes, args.steps_per_ep)
    print("\n  All done!")

if __name__ == "__main__":
    main()
