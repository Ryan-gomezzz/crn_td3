# =============================================================================
# main.py — Entry point for the CRN TD3 simulation
#
# Architecture: single-threaded training + render loop.
#   - The speed slider controls how many env.step() + agent.train_step()
#     calls happen between successive pygame renders.
#   - The GUI remains responsive at all speeds (event pump in inner loop).
#   - Model is checkpointed every SAVE_INTERVAL episodes to ./checkpoints/
#
# Run: python main.py
# =============================================================================

SAVE_INTERVAL = 100   # Save model checkpoint every N completed episodes

import sys
import numpy as np
import pygame

from config import (
    STATE_DIM, ACTION_DIM, P_MAX,
    REPLAY_BUFFER_SIZE, MIN_SAMPLES,
    EXPLORATION_NOISE_STD, EXPLORATION_NOISE_END,
    BATCH_SIZE, TRAINING_EPISODES, STEPS_PER_EPISODE,
    SINR_THRESHOLD, FPS_CAP,
)
from environment   import CRNEnvironment
from td3           import TD3Agent, ReplayBuffer
from visualization import PygameRenderer
from utils         import RollingStats, ExplorationNoise, Logger


def make_components():
    """Construct a fresh set of training components (called at start and on Reset)."""
    env    = CRNEnvironment()
    agent  = TD3Agent()
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, REPLAY_BUFFER_SIZE,
                          device=agent.device)
    rolling = RollingStats(window=100)
    noise   = ExplorationNoise(
        std_start  = EXPLORATION_NOISE_STD * P_MAX,
        std_end    = EXPLORATION_NOISE_END  * P_MAX,
        decay_steps= TRAINING_EPISODES,
    )
    logger  = Logger(log_to_file=False)
    return env, agent, buffer, rolling, noise, logger


def main():
    # ── Initialise components ──────────────────────────────────────────────────
    env, agent, buffer, rolling, noise_scheduler, logger = make_components()

    renderer = PygameRenderer()
    renderer.init()
    clock    = pygame.time.Clock()

    # ── Training state dict (mutable, passed to renderer every frame) ──────────
    ts = {
        "episode":         0,
        "step":            0,
        "reward":          0.0,
        "avg100":          0.0,
        "su_rate":         0.0,
        "pu_sinr":         0.0,
        "p_s":             0.0,
        "sinr_p":          0.0,
        "sinr_s":          0.0,
        "r_s":             0.0,
        "exploration_noise": EXPLORATION_NOISE_STD * P_MAX,
        "is_training":     False,
        "env_state":       None,
        "buffer_size":     0,
        "paused":          False,
        "total_episodes":  TRAINING_EPISODES,
        "steps_ep":        STEPS_PER_EPISODE,
        "_episode_done":   False,
        "insights":        {},
    }

    # ── Runtime variables ──────────────────────────────────────────────────────
    episode         = 0
    episode_reward  = 0.0
    state           = env.reset()
    ts["env_state"] = state

    # ── Insights tracking ──────────────────────────────────────────────────────
    from collections import deque
    constraint_window = deque(maxlen=100)   # 1=satisfied, 0=violated (per step)
    power_window      = deque(maxlen=200)   # recent P_s values
    rs_window         = deque(maxlen=200)   # recent R_s values
    reward_history    = deque(maxlen=20)    # recent episode rewards for trend
    best_reward       = [-float("inf")]  # list so closure can mutate
    peak_r_s          = [0.0]

    def _compute_insights(episode, buffer_size, is_training, avg100):
        """Build the insights dict passed to InsightsPanel.render()."""
        from utils import training_status
        stage = training_status(episode, buffer_size, avg100)

        cr     = (sum(constraint_window) / len(constraint_window)) if constraint_window else 0.0
        avg_pw = (sum(power_window) / len(power_window)) if power_window else 0.0
        avg_rs = (sum(rs_window) / len(rs_window)) if rs_window else 0.0
        viols  = sum(1 for v in constraint_window if v == 0)

        # Reward trend based on recent episode history
        if len(reward_history) >= 6:
            first_half  = list(reward_history)[:len(reward_history)//2]
            second_half = list(reward_history)[len(reward_history)//2:]
            delta = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
            if delta > 5:
                trend = "↑  Improving"
            elif delta < -5:
                trend = "↓  Declining"
            else:
                trend = "→  Stable"
        else:
            trend = "→  Collecting data"

        # Power trend
        if len(power_window) >= 40:
            half = len(power_window) // 2
            pw_vals = list(power_window)
            pw_delta = sum(pw_vals[half:]) / half - sum(pw_vals[:half]) / half
            pw_trend = "↓  Reducing" if pw_delta < -0.02 else ("↑  Increasing" if pw_delta > 0.02 else "→  Stable")
        else:
            pw_trend = "→  Stable"

        # Policy insight text
        if not is_training:
            insight = "Filling replay buffer with random exploration actions."
        elif stage == "Exploring":
            insight = "Agent exploring the action space. Expect high PU violations and negative rewards."
        elif cr < 0.4:
            insight = "Frequent PU SINR violations. Agent has not yet discovered the interference-reward trade-off."
        elif cr < 0.7:
            insight = "Learning phase: agent is reducing interference. Power levels trending down when h_sp is high."
        elif avg_pw < 0.15:
            insight = "Agent found safe low-power policy. May be stuck in zero-throughput local minimum."
        elif cr >= 0.85 and avg_rs > 0.3:
            insight = "Agent balancing SU throughput against PU protection. Policy near-optimal!"
        else:
            insight = "Training in progress. Agent learning channel-adaptive power control."

        return {
            "stage":             stage,
            "episode":           episode,
            "total_episodes":    TRAINING_EPISODES,
            "constraint_rate":   cr,
            "avg_reward":        avg100,
            "best_reward":       best_reward[0],
            "avg_power":         avg_pw,
            "avg_r_s":           avg_rs,
            "peak_r_s":          peak_r_s[0],
            "reward_trend":      trend,
            "power_trend":       pw_trend,
            "buffer_size":       buffer_size,
            "is_training":       is_training,
            "violations_last100": viols,
            "policy_insight":    insight,
        }

    running = True

    print("=" * 80)
    print("  Cognitive Radio Network — TD3 Power Allocation Training")
    print("  Ramaiah Institute of Technology, Bangalore")
    print("  Controls: Space=Pause  R=Reset  I=Toggle Interference")
    print("=" * 80)

    while running and episode < TRAINING_EPISODES:
        # ── Handle GUI events ─────────────────────────────────────────────────
        events = renderer.handle_events()

        if events["quit"]:
            running = False
            break

        if events["pause_toggle"]:
            ts["paused"] = renderer.paused

        if events["reset"]:
            # Re-initialise everything
            env, agent, buffer, rolling, noise_scheduler, logger = make_components()
            episode        = 0
            episode_reward = 0.0
            state          = env.reset()
            ts["env_state"] = state
            ts["episode"]   = 0
            ts["avg100"]    = 0.0
            renderer.clear_plots()
            print("\n[Reset] Training restarted from scratch.\n")

        if events["toggle_interference"]:
            pass  # renderer tracks its own show_interf flag

        # Speed is read directly from renderer.current_speed
        steps_per_frame = renderer.current_speed

        # ── Training inner loop ───────────────────────────────────────────────
        if not renderer.paused:
            for inner in range(steps_per_frame):
                # Select action (with exploration noise)
                action = agent.select_action(state, noise_scheduler.current_std)

                # Environment step
                result = env.step(action)

                # Store transition in replay buffer
                buffer.add(
                    state,
                    np.array([action], dtype=np.float32),
                    result.reward,
                    result.state,
                    result.done,
                )

                episode_reward += result.reward
                state           = result.state

                # Update training state dict with latest step info
                ts.update({
                    "step":            env.step_count,
                    "reward":          result.reward,
                    "su_rate":         result.info["r_s"],
                    "pu_sinr":         result.info["sinr_p"],
                    "p_s":             action,
                    "sinr_p":          result.info["sinr_p"],
                    "sinr_s":          result.info["sinr_s"],
                    "r_s":             result.info["r_s"],
                    "env_state":       state,
                    "buffer_size":     len(buffer),
                    "is_training":     buffer.is_ready,
                    "exploration_noise": noise_scheduler.current_std,
                    "_episode_done":   False,
                })

                # Push step-level data to live plots
                renderer.push_step_data(result.info["r_s"], result.info["sinr_p"])

                # Track insights stats
                satisfied = 1 if result.info["sinr_p"] >= SINR_THRESHOLD else 0
                constraint_window.append(satisfied)
                power_window.append(action)
                rs_window.append(result.info["r_s"])
                if result.info["r_s"] > peak_r_s[0]:
                    peak_r_s[0] = result.info["r_s"]

                # Update insights panel in training state
                ts["insights"] = _compute_insights(
                    episode, len(buffer), buffer.is_ready, ts["avg100"]
                )

                # TD3 training step
                if buffer.is_ready:
                    agent.train_step(buffer, BATCH_SIZE)

                # Keep pygame event queue alive during fast inner loops
                if inner % 20 == 0:
                    pygame.event.pump()

                # ── Episode boundary ──────────────────────────────────────────
                if result.done:
                    avg = rolling.mean() if len(rolling) > 0 else episode_reward
                    rolling.push(episode_reward)
                    avg = rolling.mean()

                    ts["avg100"]      = avg
                    ts["episode"]     = episode
                    ts["_episode_done"] = True

                    # Push episode-level reward to reward plot
                    renderer.push_episode_reward(episode_reward, avg)

                    # Console log
                    logger.log(
                        episode        = episode,
                        steps          = STEPS_PER_EPISODE,
                        reward         = episode_reward,
                        avg100         = avg,
                        su_rate        = result.info["r_s"],
                        pu_sinr        = result.info["sinr_p"],
                        power          = action,
                        buffer_size    = len(buffer),
                    )

                    # Periodic checkpoint save
                    if episode > 0 and episode % SAVE_INTERVAL == 0:
                        ckpt_dir = f"./checkpoints/ep_{episode:04d}"
                        agent.save(ckpt_dir)
                        print(f"  [Checkpoint] Saved at episode {episode} → {ckpt_dir}/")

                    # Update best reward and trend tracking
                    if episode_reward > best_reward[0]:
                        best_reward[0] = episode_reward
                    reward_history.append(episode_reward)

                    # Decay exploration noise
                    noise_scheduler.step()

                    # Start next episode
                    episode        += 1
                    episode_reward  = 0.0
                    state           = env.reset()
                    ts["env_state"] = state
                    ts["episode"]   = episode

                    if episode >= TRAINING_EPISODES:
                        break

        # ── Render frame ──────────────────────────────────────────────────────
        renderer.update(ts)
        ts["_episode_done"] = False   # reset per-frame flag after render
        clock.tick(FPS_CAP)

    # ── Training complete ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  Training complete!  Episodes: {episode}  Final Avg100: {rolling.mean():.4f}")
    print("  Saving final model weights to ./saved_models/ ...")
    print("=" * 80)
    agent.save("./saved_models")

    # Keep window open so user can see the final state
    print("  [Close the window to exit]")
    while True:
        events = renderer.handle_events()
        if events["quit"]:
            break
        # Update display with final state (no training)
        renderer.update(ts)
        clock.tick(30)

    renderer.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
