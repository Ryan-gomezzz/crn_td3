# 06 — Training Process

## Phase Overview

Training unfolds in three distinct phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Exploration   │ Phase 2: Learning      │ Phase 3: Converge │
│ (ep 0 → ~100)         │ (ep 100 → ~1500)       │ (ep 1500 → 3000)  │
│                        │                        │                   │
│ Buffer filling         │ Reward improving       │ Reward stable     │
│ Random-ish actions     │ Power decreasing       │ Constraints met   │
│ High violations        │ Fewer violations       │ PU protected      │
│ Reward: ~ -1000        │ Reward: -500 → 0       │ Reward: 0 → +5    │
└────────────────────────┴────────────────────────┴───────────────────┘
```

---

## Phase 1: Exploration (Buffer Filling)

**Trigger:** Buffer size < MIN_SAMPLES (1000 transitions)

During this phase, the agent takes **pure exploration actions** — the actor network is used, but with significant Gaussian noise added (std = 0.1 × P_max). Since the network is randomly initialised, actions are essentially random.

**What you see in the GUI:**
- ST node transmitting at random power levels
- Frequent red PT→PR links (PU SINR violated)
- Reward plot showing highly negative values (~ −1000)
- Throughput plot fluctuating randomly
- Buffer size counter climbing rapidly

**Why this matters:**
The replay buffer needs diverse experiences before training can begin. If training started with only a few highly correlated transitions, the critic would overfit immediately.

---

## Phase 2: Learning

**Trigger:** Buffer size ≥ MIN_SAMPLES (1000 transitions)

Now `agent.train_step()` is called at every environment step. The critic networks start fitting the Q-function, and the actor gradually shifts toward lower-power policies.

### Early Learning (ep 100–500)

The critics initially output **large, inaccurate Q-values** because they're fitting noisy, sparse data. The actor loss is high. Rewards improve slowly.

**Key observation:** The agent often discovers that P_s = 0 (silent) is a safe fallback — no interference, no energy cost, but also no throughput. Rewards temporarily cluster around −0.01 × (P_max/P_max) = −0.01 (pure energy penalty for doing nothing).

### Mid Learning (ep 500–1500)

The critics have fit a reasonable Q-function. The actor learns to:
1. Use low-to-medium power when h_sp is large (protect PU)
2. Use higher power when h_ss is large AND h_sp is small

Constraint violations drop from ~80% to ~20–30%. Rewards move from −100 toward 0.

**Key observation:** The policy noise (σ = 0.2 × P_max, decaying) acts as a curriculum — early on, the agent explores widely; later, it refines its policy.

### Late Learning (ep 1500–3000)

The policy approaches its converged form. Power control becomes adaptive: high P_s in favourable channels, low P_s when interference risk is high. The reward curve flattens as the policy stabilises.

---

## Exploration Noise Schedule

Exploration noise decays **linearly** from σ_start to σ_end over all training episodes:

```
σ(ep) = σ_start + (ep / T) × (σ_end - σ_start)

σ_start = 0.10 × P_max = 0.10 W
σ_end   = 0.01 × P_max = 0.01 W
T       = 3000 episodes
```

This means:
- Episode 0: noise std = 0.10 W (10% of P_max)
- Episode 1500: noise std = 0.055 W (5.5% of P_max)
- Episode 3000: noise std = 0.01 W (1% of P_max)

The agent exploits more and more as training progresses.

---

## Replay Buffer Sampling

At each training step, a random mini-batch of 128 transitions is sampled **uniformly** from the replay buffer. This has two key benefits:

1. **Breaks temporal correlations** — consecutive transitions are highly correlated (same episode, similar channels). Mixing them with older transitions reduces gradient variance.
2. **Experience reuse** — each transition can be sampled multiple times, improving data efficiency.

The buffer uses a circular (FIFO) write policy: when full (100,000 transitions), the oldest transitions are overwritten. This ensures the buffer always reflects recent policy behaviour.

---

## Checkpointing

Model weights are saved at two points:

| Save Type | Directory | Trigger |
|-----------|-----------|---------|
| Periodic checkpoint | `./checkpoints/ep_XXXX/` | Every 100 episodes |
| Final model | `./saved_models/` | End of training or window close |

Each save directory contains three files:
- `actor.pth` — actor network weights (~275 KB)
- `critic1.pth` — first critic weights (~276 KB)
- `critic2.pth` — second critic weights (~276 KB)

To resume from a checkpoint, call `agent.load("./checkpoints/ep_0500/")`.

---

## Computational Cost

On a modern CPU (no GPU required for this problem size):

| Metric | Approx. Value |
|--------|--------------|
| Time per episode (1× speed) | ~0.5–2 seconds |
| Time per episode (Max speed) | <0.1 seconds |
| Full 3000-episode training | 20–60 minutes |
| GPU speedup | ~2–5× (small network; GPU overhead dominates) |

The bottleneck is Python loop overhead and pygame rendering, not the neural network forward/backward passes.
