# CAMO-TD3 Methodology Document

> **Algorithm:** Constrained Adaptive Multi-Objective TD3
> **Application:** Power Allocation in Cognitive Radio Networks under Nakagami-m Fading
> **Project:** Ramaiah Institute of Technology, Bangalore

---

## 1. Problem Statement and Motivation

### 1.1 The Cognitive Radio Network Problem

In a Cognitive Radio Network (CRN), a **Secondary User (SU)** shares the spectrum of a **Primary User (PU)**. The SU must maximize its own throughput while ensuring the PU's communication quality remains above a minimum threshold.

Two RL algorithms (TD3 and DDPG) have been used for this problem, but they share a fundamental limitation: they treat the multi-objective trade-off (SU throughput vs. PU protection vs. energy efficiency) as a **fixed-weight scalar reward**. This has three consequences:

1. **Manual tuning burden**: The weights `alpha`, `beta`, `gamma` must be hand-tuned for every channel condition, threshold, and fading scenario.
2. **No constraint guarantee**: A scalar reward cannot guarantee constraint satisfaction — only soft-penalize violations. An aggressive SU might still violate the PU constraint if the throughput gain outweighs the penalty.
3. **Memoryless policy**: Standard TD3/DDPG treat each state independently, ignoring temporal channel dynamics (slow fading, correlated interference patterns).

### 1.2 What CAMO-TD3 Changes

CAMO-TD3 addresses these three limitations with four architectural innovations:

| Issue | CAMO-TD3 Solution |
|-------|------------------|
| Fixed weights | **Adaptive Lagrangian multipliers** learned via dual gradient descent |
| No constraint guarantee | **Decomposed multi-objective critics** (Q-value per objective) |
| Memoryless policy | **GRU belief encoder** over last 8 observations |
| Blind exploration | **Directional exploration noise** biased toward constraint safety |

---

## 2. System Model

### 2.1 Network Topology

Four nodes:
- **PT** — Primary Transmitter (fixed power `P_p = 1.0 W`)
- **PR** — Primary Receiver
- **ST** — Secondary Transmitter (variable power `P_s`, chosen by RL agent)
- **SR** — Secondary Receiver

Four links (channel power gains):
- `|h_pp|^2` : PT -> PR
- `|h_sp|^2` : ST -> PR (interference on PU)
- `|h_ss|^2` : ST -> SR
- `|h_ps|^2` : PT -> SR (interference on SU)

### 2.2 Nakagami-m Fading Model

Each link's power gain is drawn independently each time step:

```
|h|^2 ~ Gamma(shape = m, scale = Omega / m)
```

- `m = 3` (moderate fading severity; `m=1` recovers Rayleigh, `m -> infinity` approaches AWGN)
- `Omega = 1.0` (mean link power)

### 2.3 SINR Formulas

Signal-to-Interference-plus-Noise Ratio at each receiver:

```
SINR_p = (P_p * |h_pp|^2) / (P_s * |h_sp|^2 + sigma^2)     [Primary User]

SINR_s = (P_s * |h_ss|^2) / (P_p * |h_ps|^2 + sigma^2)     [Secondary User]
```

where `sigma^2 = 1e-3 W` is the AWGN noise power.

### 2.4 Throughput (Shannon Capacity)

```
R_s = log2(1 + SINR_s)     [bits/s/Hz, SU]
R_p = log2(1 + SINR_p)     [bits/s/Hz, PU]
```

### 2.5 BER (BPSK Modulation)

```
BER = 0.5 * erfc(sqrt(SINR))
```

### 2.6 Constraint

The core constraint: the PU must maintain `SINR_p >= gamma_th` where `gamma_th = 1.0` (linear, equivalent to ~0 dB). When `SINR_p < gamma_th`, the PU is in **outage**.

---

## 3. MDP Formulation

### 3.1 State Space (7-dimensional)

```
s_t = [|h_pp|^2, |h_sp|^2, |h_ss|^2, |h_ps|^2, SINR_p, SINR_s, P_s_prev]
```

**Reasoning:** The first four channel gains describe the current propagation environment. The two SINRs give immediate feedback from the last action. `P_s_prev` provides action continuity information.

### 3.2 Action Space

```
a_t = P_s in [0, P_max]     where P_max = 3.0 W
```

**Reasoning for P_max = 3.0:** During tuning, we observed that `P_max = 2.0` left the SU unable to punch through Nakagami deep fades (outage > 20%). Increasing to 3.0 gave the agent enough headroom to achieve `<15%` outage while still respecting the PU constraint most of the time.

### 3.3 Original Scalar Reward (used by TD3/DDPG)

```
r = alpha * R_s - beta * max(0, gamma_th - SINR_p) - gamma * (P_s / P_max)
   |___________|   |______________________________|   |_______________|
   SU throughput    PU constraint violation            Energy penalty
```

With `alpha = 1.0`, `beta = 1.5`, `gamma = 0.005`.

### 3.4 CAMO-TD3 Decomposed Reward (new)

```
r_throughput   = alpha * R_s                        >= 0  (maximize)
r_interference = -beta * max(0, gamma_th - SINR_p)  <= 0  (keep near zero)
r_energy       = -gamma * (P_s / P_max)             <= 0  (minimize magnitude)
```

These are **not summed** — each is stored separately and learned by its own critic.

---

## 4. CAMO-TD3 Architecture

### 4.1 Component 1: GRU Belief Encoder

**Mathematical Formulation:**

Given the last `L = SEQ_LEN = 8` observations:
```
O_t = [s_{t-7}, s_{t-6}, ..., s_{t-1}, s_t]     (shape: 8 x 7)
```

A 2-layer GRU processes this sequence:
```
h_0 = 0
For i = 1..8:
    z_i, r_i, n_i = sigmoid/tanh gates (GRU cell 1 + 2)
    h_i = update(h_{i-1}, o_i)
b_t = W_proj * h_L + c_proj     (projection to belief_dim = 16)
```

**Why GRU over LSTM or Transformer?**
- LSTM has more gates -> more parameters, slower training on small sequences
- Transformers need longer sequences (typically 32+) to beat GRU; at `SEQ_LEN = 8`, GRU is strictly better
- GRU has empirically been shown to match LSTM on RL benchmarks with fewer parameters

**Why `SEQ_LEN = 8`?**
- Nakagami block fading refreshes each timestep, so perfect correlation is not expected
- But the agent's previous **actions** and their consequences carry information across 2-3 steps (action inertia, learning cause-effect)
- 8 steps = 1 frame in a 200-step episode = enough to capture short-term dynamics without training cost explosion
- Tested: SEQ_LEN=4 (under-capacity), SEQ_LEN=16 (marginal gain, 2x memory)

**Why `belief_dim = 16`?**
- Compresses 56-dim observation history (8 x 7) down to 16 -- a 3.5x compression ratio
- Smaller than state_dim x 2 to force the encoder to extract features, not memorize

### 4.2 Component 2: Decomposed Multi-Objective Critics

**Mathematical Formulation:**

Instead of a single scalar Q-value, CAMO-TD3 maintains **six critic networks** — one twin pair per objective:

```
Q_1^{throughput}(s, b, a), Q_2^{throughput}(s, b, a)       — throughput critics
Q_1^{interference}(s, b, a), Q_2^{interference}(s, b, a)   — interference critics
Q_1^{energy}(s, b, a), Q_2^{energy}(s, b, a)               — energy critics
```

where `b` is the belief vector from the GRU and `s` is the current state.

Each pair uses the TD3 twin-critic target smoothing:
```
y_k = r_k + gamma_discount * (1 - d) * min(Q_1^k_target, Q_2^k_target)(s', b', a')
```

and minimizes MSE loss:
```
L_critic_k = (Q(s, b, a) - y_k)^2     for each k in {throughput, interference, energy}
```

**Why decomposed critics?**

A single critic learning `Q(s, a) = E[sum r]` mixes all objectives into one signal. If throughput is on the order of `+2` and interference penalties are on the order of `-15`, the critic's output is dominated by the interference term — making it hard to distinguish "low reward because we lost throughput" from "low reward because we hit the constraint."

With decomposed critics:
- `Q^{throughput}` learns **only** the expected SU throughput
- `Q^{interference}` learns **only** the expected PU violation penalty
- These signals are cleanly separable, enabling the Lagrangian to trade them off explicitly

### 4.3 Component 3: Adaptive Lagrangian Weights

**Mathematical Formulation:**

The actor maximizes a **weighted combination** of the three Q-values:
```
L_actor = -E[lambda_1 * Q^{tput}(s, b, a) + lambda_2 * Q^{intf}(s, b, a) + lambda_3 * Q^{energy}(s, b, a)]
```

where the weights `lambda_k` are **learned parameters**, not hand-tuned.

Positivity is enforced via softplus parameterization:
```
lambda_k = softplus(log_lambda_k) = log(1 + exp(log_lambda_k))
```

The Lagrangian update uses dual gradient descent:
```
Violation signal:   v = -E[r_{interference}]                 (positive when violations occur)
Throughput signal:  t = E[r_{throughput}]                    (we want this to grow)
Energy signal:      e = E[r_{energy}]                        (small negative)

L_lambda = -log_lambda_1 * t + log_lambda_2 * v - log_lambda_3 * e
```

Intuition of the update (after gradient step):
- If PU violations are frequent (`v` large positive) -> `lambda_2` increases -> actor penalizes interference more
- If throughput is high (`t` large positive) -> `lambda_1` decreases slightly (constraint now active)
- Lambdas are **clamped** to `[LAMBDA_MIN, LAMBDA_MAX] = [0.1, 20.0]` to prevent runaway values

**Why Lagrangian over fixed weights?**

This is the **primal-dual formulation** of the constrained optimization problem:
```
maximize   E[R_s]
subject to P(SINR_p < gamma_th) <= epsilon    (outage probability constraint)
           E[P_s] <= P_max
```

Classical Lagrangian theory (Boyd & Vandenberghe, *Convex Optimization*, 2004, Ch. 5) tells us the optimal solution is a saddle point:
```
L(policy, lambda) = primal_objective - lambda * constraint_slack
```

Dual gradient ascent on lambda converges to the optimal multipliers **without requiring manual weight tuning** — the agent discovers them as training progresses. This is the same mathematical foundation used in:
- **TRPO/PPO-Lagrangian** (Achiam et al., 2017, "Constrained Policy Optimization")
- **Safety-Gym benchmarks** (OpenAI, 2019)
- **CMDP theory** (Altman, 1999, *Constrained Markov Decision Processes*)

### 4.4 Component 4: Directional Exploration Noise

**Mathematical Formulation:**

Standard TD3 uses symmetric Gaussian noise:
```
epsilon ~ N(0, sigma^2)
a_explore = pi(s) + epsilon
```

CAMO-TD3 uses a **biased** Gaussian with adaptive mean:
```
mu(t, violation_rate) = mu_bias * decay(t) * (1 + violation_rate)

decay(t) = max(0, 1 - t / NOISE_DECAY_STEPS)

epsilon ~ N(mu(t, violation_rate), sigma^2)
```

where:
- `mu_bias = -0.05` (small negative: biases toward lower power)
- `violation_rate`: rolling fraction of PU SINR violations over last 100 steps

**Why directional bias?**

Consider what happens with symmetric zero-mean noise early in training:
- Actor outputs some power level (say 1.5 W)
- Gaussian noise adds N(0, 0.3^2) -> actions span [0.6, 2.4] with equal probability
- Half the explored actions **increase** interference, half **decrease** it
- The agent learns from both, but needs many violations to learn "don't increase"

With directional noise biased toward lower power:
- Exploration naturally probes **safer** actions more often
- When violations are frequent, the bias strengthens (`violation_rate` multiplier)
- As training progresses and the agent becomes competent, `decay(t)` -> 0 and the bias vanishes
- Result: **faster convergence with fewer catastrophic violations during exploration**

This is analogous to "safe exploration" techniques in the robotics literature (García & Fernández, 2015, *A Comprehensive Survey on Safe RL*).

---

## 5. Why CAMO-TD3 Outperforms TD3 and DDPG

### 5.1 Theoretical Advantages

| Dimension | TD3 | DDPG | CAMO-TD3 |
|-----------|-----|------|----------|
| Critics | Twin (2) | Single (1) | **6 (twin per objective)** |
| Reward signal | Scalar sum | Scalar sum | **3-way decomposed** |
| Constraint handling | Fixed penalty | Fixed penalty | **Adaptive Lagrangian** |
| Temporal memory | None (7-D state only) | None | **8-step GRU history** |
| Exploration | Symmetric Gaussian | OU noise | **Directional adaptive** |
| Overestimation bias | Reduced (min of twins) | Not reduced | **Reduced per objective** |

### 5.2 Expected Performance Gains

1. **Lower outage probability**: Adaptive lambda_2 automatically increases when violations occur, providing a self-correcting constraint mechanism. TD3/DDPG need manual tuning of `beta` for every scenario.

2. **Higher SU throughput at the same outage**: The decomposed critics let the actor extract more throughput because it can distinguish throughput-limited states from constraint-limited states. In TD3/DDPG, a state where the agent lost throughput for an innocuous reason looks identical to one where it hit the constraint.

3. **Better adaptation to channel dynamics**: The GRU belief captures temporal patterns (e.g., "the last 3 steps had deep fades -> expect recovery soon"), letting the agent be more aggressive after fade recovery.

4. **Safer exploration**: Fewer catastrophic PU violations during training = faster convergence + fewer real-world safety concerns if deployed.

### 5.3 Trade-offs

CAMO-TD3 is not a free lunch:

- **2.5x parameter count** (6 critics + GRU + belief projection)
- **~2-3x slower per training step** (more forward/backward passes)
- **Slower initial convergence** (lambdas need to stabilize before policy can converge)
- **More hyperparameters** (lambda inits, LR_LAMBDA, MU_BIAS, SEQ_LEN, BELIEF_DIM)

For small-scale problems where 1000 episodes suffice, TD3 may converge faster in wall-clock time. CAMO-TD3 shines in constraint-tight scenarios where TD3 cannot satisfy the constraint at all, or where the constraint threshold changes dynamically.

---

## 6. Hyperparameter Configuration & Reasoning

### 6.1 Inherited TD3 Parameters (unchanged)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `HIDDEN_DIM` | 512 | Wider network improves convergence on Nakagami fades (tested 256 and 512) |
| `BATCH_SIZE` | 256 | GPU-efficient batch for RTX 4060 Laptop |
| `LR_ACTOR`, `LR_CRITIC` | 3e-4 | Standard TD3 learning rate (Fujimoto et al., 2018) |
| `GAMMA_DISCOUNT` | 0.99 | Standard for infinite-horizon continuous control |
| `TAU` | 0.005 | Slow target network updates — stable for 6 critics |
| `POLICY_NOISE` | 0.2 | Target smoothing noise (fraction of `P_max`) |
| `NOISE_CLIP` | 0.5 | Target smoothing clip |
| `POLICY_DELAY` | 2 | Update actor every 2 critic updates |
| `REPLAY_BUFFER_SIZE` | 200,000 | 1000 episodes x 200 steps of history |
| `GRAD_UPDATES_PER_STEP` | 2 | UTD ratio for GPU utilization |

### 6.2 CAMO-TD3 Specific Parameters

#### GRU Belief Encoder

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `GRU_HIDDEN_SIZE` | 64 | Hidden state width — balances capacity vs. overfitting |
| `GRU_NUM_LAYERS` | 2 | Single layer is under-capacity; 3+ overfits |
| `SEQ_LEN` | 8 | ~4% of an episode; enough for short-term channel trends |
| `BELIEF_DIM` | 16 | 3.5x compression from 56 (8 x 7) raw history |

#### Adaptive Lagrangian

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `LAMBDA1_INIT` | 3.0 | **Was 1.0** — increased to prioritize throughput from day 1 (prevents zero-power collapse) |
| `LAMBDA2_INIT` | 1.0 | **Was 10.0** — reduced because the reward already has `beta = 1.5`; `10 * 1.5 = 15x` was way too aggressive |
| `LAMBDA3_INIT` | 0.01 | Energy is a minor consideration; small weight keeps it as a tiebreaker |
| `LR_LAMBDA` | 0.0005 | **Was 0.001** — slower lambda adaptation prevents early oscillation |
| `LAMBDA_MIN` | 0.1 | Floor — prevents any objective from being fully ignored |
| `LAMBDA_MAX` | 20.0 | Ceiling — prevents runaway constraint weighting (death spiral) |

**Why these values changed from the literature defaults:**

In the initial implementation following standard CMDP literature (LAMBDA_INIT around the expected rewards), the agent collapsed into a "don't transmit" policy — SU throughput was 0.03 bits/s/Hz and outage was 99.97%. Root cause analysis showed:

1. `LAMBDA2 = 10.0` combined with `BETA = 1.5` gave the interference objective 15x the weight of throughput.
2. Early violations pushed `lambda_2` even higher, worsening the imbalance.
3. Once the agent learned zero power -> no violations, there was no gradient to pull it back.

**Fix applied:** Rebalance inits so throughput dominates at start, slow the adaptation, and clamp the lambdas to prevent runaway.

#### Directional Exploration

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `MU_BIAS_INIT` | -0.05 | **Was -0.15** — reduced because -0.15 was too aggressive (compounded the zero-power collapse) |
| `NOISE_DECAY_STEPS` | 200,000 | ~1000 episodes; bias fades as agent learns |
| `VIOLATION_WINDOW` | 100 | Rolling window for computing violation rate for adaptive bias |

---

## 7. Summary of Expected Results

After proper tuning, CAMO-TD3 should demonstrate:

| Metric | TD3 Baseline | DDPG Baseline | CAMO-TD3 Target |
|--------|-------------|--------------|------------------|
| SU Throughput (bits/s/Hz) | ~2.0 | ~1.6 | **>= 2.0** |
| Outage Probability | ~11% | ~28% | **<= 8%** |
| PU Throughput | ~0.5 | ~0.9 | **>= 0.9** |
| Average BER (SU) | ~0.028 | ~0.062 | **<= 0.028** |

The headline claim: **CAMO-TD3 achieves comparable or better SU throughput than TD3 with significantly lower outage probability**, because the adaptive Lagrangian mechanism automatically finds the optimal trade-off for the given channel conditions without manual tuning.

---

## 8. Mathematical Summary (Optimization Problem)

### 8.1 Original (TD3/DDPG) Objective — Scalar

```
maximize_{pi}  E_{tau ~ pi}[sum_t (alpha * R_s - beta * max(0, gamma_th - SINR_p) - gamma * P_s/P_max)]
```

**Weakness:** Weights `alpha, beta, gamma` are fixed hyperparameters; no constraint guarantee.

### 8.2 CAMO-TD3 Objective — Constrained Multi-Objective

**Primal:**
```
maximize_{pi}    E[sum_t R_s(s_t, a_t)]                   (SU throughput)

subject to       E[sum_t max(0, gamma_th - SINR_p)]  <=  delta    (constraint)
                 E[sum_t P_s]                        <=  P_budget
```

**Lagrangian (dual):**
```
L(pi, lambda_1, lambda_2, lambda_3) =
      lambda_1 * E[sum_t R_s]
    - lambda_2 * (E[sum_t max(0, gamma_th - SINR_p)] - delta)
    - lambda_3 * (E[sum_t P_s] - P_budget)
```

**Actor update (maximize L over pi):**
```
pi <- pi + eta_pi * grad_pi L
```

**Dual update (minimize L over lambda — gradient ascent on negative):**
```
lambda_2 <- lambda_2 + eta_lambda * (E[sum_t max(0, gamma_th - SINR_p)] - delta)
```

**Convergence:** Saddle-point theorem (Sion's minimax, 1958) guarantees convergence to the optimal `(pi*, lambda*)` under standard convexity assumptions. Even when the policy is parameterized by a neural network (non-convex), empirical studies show Lagrangian updates converge in practice (Ray et al., 2019, *Benchmarking Safe Exploration*).

---

## 9. References

1. **Fujimoto et al., 2018.** "Addressing Function Approximation Error in Actor-Critic Methods" (TD3 paper). ICML.
2. **Lillicrap et al., 2015.** "Continuous control with deep reinforcement learning" (DDPG paper). ICLR.
3. **Achiam et al., 2017.** "Constrained Policy Optimization." ICML.
4. **Ray, Achiam, Amodei, 2019.** "Benchmarking Safe Exploration in Deep Reinforcement Learning." arXiv:1910.01708.
5. **Altman, 1999.** *Constrained Markov Decision Processes*. Chapman & Hall/CRC.
6. **Simon & Alouini, 2005.** *Digital Communication over Fading Channels*. Wiley. (Nakagami-m BER formulas.)
7. **Boyd & Vandenberghe, 2004.** *Convex Optimization*. Cambridge University Press. (Ch. 5: Duality.)
8. **Cho et al., 2014.** "Learning Phrase Representations using RNN Encoder-Decoder" (GRU paper). EMNLP.
9. **García & Fernández, 2015.** "A Comprehensive Survey on Safe Reinforcement Learning." JMLR.

---

*Ramaiah Institute of Technology, Bangalore*
*Department of Electronics and Communication Engineering*
*Cognitive Radio Network — RL-Based Power Allocation*
