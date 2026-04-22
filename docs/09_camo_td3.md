# CAMO-TD3: Constrained Adaptive Multi-Objective TD3

> **Module:** `camo_td3.py`
> **Config block:** `config.py` (CAMO-TD3 section)

CAMO-TD3 extends the standard TD3 algorithm with four novel components designed to improve constraint-aware power allocation in Cognitive Radio Networks.

---

## Architecture Overview

```
                     Observation History (last 8 states)
                              |
                     +--------v--------+
                     | GRU Belief      |
                     | Encoder         |
                     | (2-layer GRU)   |
                     +--------+--------+
                              |
                        belief (16-dim)
                              |
         +--------------------+--------------------+
         |                                         |
  [state || belief]                         [state || belief || action]
         |                                         |
  +------v------+      +------v------+  +------v------+  +------v------+
  |   Actor     |      | Twin Critics |  | Twin Critics |  | Twin Critics |
  | pi(s,b)     |      | Q_throughput |  | Q_interfer.  |  | Q_energy     |
  +------+------+      +-------------+  +--------------+  +--------------+
         |                    |                |                  |
      action P_s        lambda_1 *        lambda_2 *         lambda_3 *
                         Q_tput       +    Q_intf       +     Q_energy
                              |                |                  |
                              +--------> Scalarized Loss <--------+
                                              |
                                    Actor Gradient Update
```

---

## Component 1: Decomposed Multi-Objective Critics

Instead of a single scalar Q-value, CAMO-TD3 maintains **6 critic networks** (3 twin pairs), one pair per reward component:

| Pair | Objective | Reward Component |
|------|-----------|-----------------|
| Q1a, Q1b | Throughput | `alpha * log2(1 + SINR_s)` |
| Q2a, Q2b | Interference | `-beta * max(0, threshold - SINR_p)` |
| Q3a, Q3b | Energy | `-gamma * (P_s / P_max)` |

Each pair uses the TD3 twin-critic trick (take the minimum for target computation) to reduce overestimation bias per objective.

The actor loss is a **weighted sum** of the three Q-values:

```
L_actor = -(lambda_1 * Q_tput + lambda_2 * Q_intf + lambda_3 * Q_energy)
```

---

## Component 2: Adaptive Lagrangian Weights

The weights `lambda_1`, `lambda_2`, `lambda_3` are **not fixed** -- they are learned via dual gradient descent:

- When PU SINR constraint violations are frequent, `lambda_2` (interference weight) **increases**, pushing the actor toward safer power levels
- When throughput is low relative to what's achievable, `lambda_1` **adjusts** to rebalance
- `lambda_3` stays small, providing gentle energy efficiency pressure

The lambdas are parameterized as `softplus(log_lambda_k)` to ensure positivity, and updated with their own Adam optimizer (learning rate: `LR_LAMBDA = 0.001`).

**Initial values:**
| Lambda | Init | Role |
|--------|------|------|
| lambda_1 | 1.0 | Throughput weight |
| lambda_2 | 10.0 | Interference constraint weight |
| lambda_3 | 0.01 | Energy penalty weight |

---

## Component 3: GRU Belief Encoder

A 2-layer GRU processes the **last 8 observations** (the `SEQ_LEN` parameter) and outputs a compact **16-dimensional belief vector**. This captures temporal channel dynamics that a single observation cannot:

- Channel gain trends (fading patterns across time)
- SINR trajectory (is PU SINR improving or degrading?)
- Previous power allocation history

The belief vector is concatenated with the current state before being fed to both the actor and all critics:

```
augmented_input = [state (7-dim) || belief (16-dim)] = 23-dim
```

The GRU encoder has its own target network (Polyak-averaged) for stable target Q computation.

---

## Component 4: Directional Exploration Noise

Standard TD3 uses zero-mean Gaussian noise. CAMO-TD3 adds a **bias term** that nudges exploration toward constraint satisfaction:

```
noise = N(mu_bias * decay * (1 + violation_rate), sigma)
```

- `mu_bias = -0.15` (negative = lower power = protect PU)
- `decay`: linear decay from 1.0 to 0.0 over `NOISE_DECAY_STEPS` (200k steps)
- `violation_rate`: rolling fraction of PU SINR violations over the last 100 steps

This means early in training, when the agent hasn't learned the constraint boundary, exploration is biased toward safer (lower power) actions. As training progresses, the bias decays and the agent relies on its learned policy.

---

## Configuration Parameters

All CAMO-TD3 parameters are in `config.py`:

```python
# GRU Belief Encoder
GRU_HIDDEN_SIZE    = 64       # GRU hidden state width
GRU_NUM_LAYERS     = 2        # Number of GRU layers
BELIEF_DIM         = 16       # Compressed belief dimension
SEQ_LEN            = 8        # Observation history length

# Adaptive Lagrangian
LAMBDA1_INIT       = 1.0      # Throughput weight
LAMBDA2_INIT       = 10.0     # Interference constraint weight
LAMBDA3_INIT       = 0.01     # Energy weight
LR_LAMBDA          = 0.001    # Lambda learning rate

# Directional Noise
MU_BIAS_INIT       = -0.15    # Noise bias (negative = lower power)
NOISE_DECAY_STEPS  = 200000   # Steps over which bias decays
VIOLATION_WINDOW   = 100      # Rolling window for violation tracking

# Network
CAMO_HIDDEN_DIM    = 512      # Hidden layer width
```

---

## Replay Buffer: SequenceReplayBuffer

CAMO-TD3 uses `SequenceReplayBuffer` instead of the standard `ReplayBuffer`. Key differences:

1. **Decomposed rewards**: stores `r_throughput`, `r_interference`, `r_energy` separately (not a scalar sum)
2. **Observation histories**: stores `(SEQ_LEN, STATE_DIM)` tensors for both current and next states, enabling GRU inference at sample time
3. **Episode management**: call `buffer.reset_episode(initial_state)` at the start of each episode to initialize the rolling observation window

---

## Usage

### Train all three algorithms
```bash
python train_compare.py --agents td3,ddpg,camo-td3 --episodes 1500 --output results/report_3way.pdf
```

### Train only CAMO-TD3 (smoke test)
```bash
python train_compare.py --agents camo-td3 --episodes 100 --output results/camo_only.pdf
```

### Compare CAMO-TD3 vs TD3
```bash
python train_compare.py --agents td3,camo-td3 --episodes 2000 --output results/td3_vs_camo.pdf
```

### All three in parallel
```bash
python train_compare.py --agents td3,ddpg,camo-td3 --episodes 1500 --parallel --output results/report_parallel.pdf
```

---

## Expected Behavior

- **Early training**: CAMO-TD3 may converge slightly slower than TD3 due to the additional complexity (GRU warmup, lambda adaptation)
- **Mid training**: Lagrangian weights should stabilize; lambda_2 typically settles between 5-15 depending on how frequently the constraint is violated
- **Late training**: CAMO-TD3 should achieve **comparable or better SU throughput** with **lower outage probability** than vanilla TD3, as the decomposed critics and adaptive weights provide finer control over the throughput-interference trade-off

---

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| Actor (23-dim input) | ~275k |
| Belief Encoder (2-layer GRU + projection) | ~40k |
| 6 Critics (24-dim input each) | ~1.66M |
| Lagrangian multipliers | 3 |
| **Total** | **~1.97M** |

Compare to vanilla TD3 (~790k params) -- CAMO-TD3 is ~2.5x larger, primarily due to the 6 critic networks.

---

*Ramaiah Institute of Technology, Bangalore*
*Cognitive Radio Network -- RL Power Allocation*
