# =============================================================================
# main.py — CRN TD3 training loop
#
# Exposes run_training(broadcast_fn) for use by server.py (WebSocket mode).
# Run standalone with: python main.py  (console-only, no GUI)
# =============================================================================

SAVE_INTERVAL = 100   # Save model checkpoint every N completed episodes

import math
import numpy as np
from collections import deque
from scipy.special import erfc

from config import (
    STATE_DIM, ACTION_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    EXPLORATION_NOISE_STD, EXPLORATION_NOISE_END,
    BATCH_SIZE, TRAINING_EPISODES, STEPS_PER_EPISODE,
    SINR_THRESHOLD,
    BROADCAST_INTERVAL, SCATTER_WINDOW, OUTAGE_WINDOW,
)
from environment import CRNEnvironment
from td3         import TD3Agent, ReplayBuffer
from utils       import RollingStats, ExplorationNoise, Logger


def make_components():
    """Construct a fresh set of training components."""
    env    = CRNEnvironment()
    agent  = TD3Agent()
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, REPLAY_BUFFER_SIZE,
                          device=agent.device)
    rolling = RollingStats(window=100)
    noise   = ExplorationNoise(
        std_start   = EXPLORATION_NOISE_STD * P_MAX,
        std_end     = EXPLORATION_NOISE_END  * P_MAX,
        decay_steps = TRAINING_EPISODES,
    )
    logger  = Logger(log_to_file=False)
    return env, agent, buffer, rolling, noise, logger


def run_training(broadcast_fn=None):
    """
    Core TD3 training loop.

    broadcast_fn(packet: dict) is called every BROADCAST_INTERVAL steps with
    the current metrics snapshot. Pass None for silent console-only mode.
    """
    if broadcast_fn is None:
        broadcast_fn = lambda _: None

    env, agent, buffer, rolling, noise_scheduler, logger = make_components()

    # ── Runtime state ──────────────────────────────────────────────────────────
    episode        = 0
    episode_reward = 0.0
    global_step    = 0
    state          = env.reset()

    # ── Rolling metric windows ─────────────────────────────────────────────────
    constraint_window = deque(maxlen=100)
    power_window      = deque(maxlen=200)
    rs_window         = deque(maxlen=200)
    reward_history    = deque(maxlen=20)
    sinr_window       = deque(maxlen=OUTAGE_WINDOW)   # for outage probability
    scatter_buffer    = deque(maxlen=SCATTER_WINDOW)  # (sinr_db, ber) pairs

    best_reward = [-float("inf")]
    peak_r_s    = [0.0]
    avg100      = 0.0

    # ── Insights helper ────────────────────────────────────────────────────────
    def _constraint_rate():
        return (sum(constraint_window) / len(constraint_window)
                if constraint_window else 0.0)

    def _reward_trend():
        if len(reward_history) >= 6:
            mid   = len(reward_history) // 2
            hist  = list(reward_history)
            delta = sum(hist[mid:]) / mid - sum(hist[:mid]) / mid
            if delta > 5:   return "improving"
            if delta < -5:  return "declining"
        return "stable"

    def _training_stage():
        from utils import training_status
        return training_status(episode, len(buffer), avg100)

    print("=" * 70)
    print("  CRN TD3 — Cognitive Radio Network Power Allocation Training")
    print("  Nakagami-m fading channel  |  WebSocket metrics stream active")
    print("=" * 70)

    while episode < TRAINING_EPISODES:
        # ── Select action & step environment ──────────────────────────────────
        action = agent.select_action(state, noise_scheduler.current_std)
        result = env.step(action)

        # ── Store transition ──────────────────────────────────────────────────
        buffer.add(
            state,
            np.array([action], dtype=np.float32),
            result.reward,
            result.state,
            result.done,
        )

        episode_reward += result.reward
        state           = result.state
        global_step    += 1

        # ── Per-step metric computation ────────────────────────────────────────
        sinr_s   = result.info["sinr_s"]
        sinr_p   = result.info["sinr_p"]
        r_s      = result.info["r_s"]

        # Instantaneous BPSK BER from current SINR
        ber = float(0.5 * erfc(math.sqrt(max(0.0, sinr_s))))

        # dB conversion (guard against zero/negative)
        sinr_s_db = 10.0 * math.log10(max(1e-9, sinr_s))
        sinr_p_db = 10.0 * math.log10(max(1e-9, sinr_p))

        # Rolling outage probability
        sinr_window.append(sinr_s)
        outage_prob = (sum(1 for v in sinr_window if v < SINR_THRESHOLD)
                       / len(sinr_window))

        # Scatter buffer for BER vs SINR chart
        scatter_buffer.append({"x": round(sinr_s_db, 3), "y": round(ber, 6)})

        # Constraint / power / throughput tracking
        constraint_window.append(1 if sinr_p >= SINR_THRESHOLD else 0)
        power_window.append(action)
        rs_window.append(r_s)
        if r_s > peak_r_s[0]:
            peak_r_s[0] = r_s

        # TD3 training step
        if buffer.is_ready:
            agent.train_step(buffer, BATCH_SIZE)

        # ── Broadcast metrics every BROADCAST_INTERVAL steps ──────────────────
        if global_step % BROADCAST_INTERVAL == 0:
            packet = {
                "step":            global_step,
                "episode":         episode,
                "sinr_s_db":       round(sinr_s_db, 3),
                "sinr_p_db":       round(sinr_p_db, 3),
                "ber":             round(ber, 6),
                "throughput":      round(r_s, 4),
                "outage_prob":     round(outage_prob, 4),
                "p_s":             round(float(action), 4),
                "reward":          round(float(result.reward), 4),
                "avg100_reward":   round(float(avg100), 4),
                "episode_count":   episode,
                "constraint_rate": round(_constraint_rate(), 4),
                "training_stage":  _training_stage(),
                "reward_trend":    _reward_trend(),
                "scatter":         list(scatter_buffer),
            }
            broadcast_fn(packet)

        # ── Episode boundary ───────────────────────────────────────────────────
        if result.done:
            rolling.push(episode_reward)
            avg100 = rolling.mean()

            logger.log(
                episode     = episode,
                steps       = STEPS_PER_EPISODE,
                reward      = episode_reward,
                avg100      = avg100,
                su_rate     = r_s,
                pu_sinr     = sinr_p,
                power       = action,
                buffer_size = len(buffer),
            )

            if episode > 0 and episode % SAVE_INTERVAL == 0:
                ckpt_dir = f"./checkpoints/ep_{episode:04d}"
                agent.save(ckpt_dir)
                print(f"  [Checkpoint] Saved at episode {episode} -> {ckpt_dir}/")

            if episode_reward > best_reward[0]:
                best_reward[0] = episode_reward
            reward_history.append(episode_reward)

            noise_scheduler.step()

            episode        += 1
            episode_reward  = 0.0
            state           = env.reset()

    # ── Training complete ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  Training complete!  Episodes: {episode}  Final Avg100: {avg100:.4f}")
    print("  Saving final model to ./saved_models/ ...")
    print("=" * 70)
    agent.save("./saved_models")


if __name__ == "__main__":
    run_training()
