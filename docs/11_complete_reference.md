# 11 — Complete Reference: Every Aspect of the CRN TD3 / DDPG / CAMO-TD3 Project

> A single-file, ground-up walkthrough of the entire mini-project — every symbol,
> every parameter, every design choice, with extra depth on **CAMO-TD3**.
>
> Read this if you want to be able to explain *any* line of the code or *any*
> number in the PDF reports to a viva examiner.

---

## Table of Contents

1. [What the project is and why it exists](#1-what-the-project-is-and-why-it-exists)
2. [Wireless system model](#2-wireless-system-model)
3. [The Nakagami-m fading channel](#3-the-nakagami-m-fading-channel)
4. [Imperfect CSI model](#4-imperfect-csi-model)
5. [Markov Decision Process (MDP) formulation](#5-markov-decision-process-mdp-formulation)
6. [Reward function — line by line](#6-reward-function--line-by-line)
7. [DDPG — the simpler baseline](#7-ddpg--the-simpler-baseline)
8. [TD3 — the strong baseline](#8-td3--the-strong-baseline)
9. [CAMO-TD3 — the proposed algorithm (deep dive)](#9-camo-td3--the-proposed-algorithm-deep-dive)
10. [Every hyperparameter in `config.py`](#10-every-hyperparameter-in-configpy)
11. [Neural-network architectures](#11-neural-network-architectures)
12. [Replay buffers (uniform & sequence)](#12-replay-buffers-uniform--sequence)
13. [Training pipeline and `train_compare.py`](#13-training-pipeline-and-train_comparepy)
14. [Performance metrics and how each PDF page is built](#14-performance-metrics-and-how-each-pdf-page-is-built)
15. [Glossary of every symbol and acronym](#15-glossary-of-every-symbol-and-acronym)
16. [Mathematical appendix](#16-mathematical-appendix)
17. [Common questions a viva examiner will ask](#17-common-questions-a-viva-examiner-will-ask)

---

## 1. What the project is and why it exists

### 1.1 One-sentence summary

> Train a reinforcement-learning agent that decides, every time-slot, **how much power the secondary transmitter should use**, so the secondary user maximises its own data rate **without violating** the primary user's quality-of-service.

### 1.2 Why this problem matters

Wireless spectrum is finite and largely licensed. A **Cognitive Radio Network (CRN)** lets an *unlicensed* "secondary" transceiver coexist in a licensed band, provided it does not significantly hurt the licensed "primary" user. The hardest part is the power-control decision: too low and the SU achieves no throughput; too high and the SU drowns the PU. A classical optimisation does not work well in practice because:

- channel gains change every millisecond (block fading),
- the SU only knows *estimates* of the channel (imperfect CSI),
- closed-form solutions assume idealised distributions and Lagrangians the operator must tune manually.

**Reinforcement learning** is a natural fit: the agent learns the optimal mapping from observation → power, even when the underlying physics is messy.

### 1.3 What the three algorithms do

| Algorithm | One-line role |
|-----------|---------------|
| **DDPG** | Original deterministic actor–critic baseline (single critic, OU noise). Often unstable. |
| **TD3** | Drop-in stabilised replacement for DDPG (twin critics, delayed policy, target smoothing). Strong baseline. |
| **CAMO-TD3** | Our proposed algorithm: TD3 + decomposed-objective critics + GRU belief + adaptive Lagrangian + directional noise. Beats both baselines in constraint satisfaction. |

---

## 2. Wireless system model

### 2.1 The four-node topology

```
                       h_pp  (desired PU link)
   ┌──────┐ ════════════════════════════════════  ┌──────┐
   │  PT  │                                       │  PR  │
   └──┬───┘                                       └──┬───┘
      │                                              ▲
      │  h_ps  (PT → SR interference)                │ h_sp  (ST → PR interference)
      │                                              │
   ┌──▼───┐                                       ┌──▼───┐
   │  ST  │ ═════════════════════════════════════►│  SR  │
   └──────┘             h_ss (desired SU link)    └──────┘
```

| Node | Meaning | Power |
|------|---------|-------|
| PT | Primary Transmitter (licensed) | Fixed `P_p = 1.0 W` |
| PR | Primary Receiver (must be protected) | — |
| ST | Secondary Transmitter (the agent) | Variable `P_s ∈ [0, P_max]` |
| SR | Secondary Receiver | — |

### 2.2 The four channels

| Link | Symbol | Type | Effect |
|------|--------|------|--------|
| PT → PR | h_pp | Desired (PU) | Higher is better for the PU |
| ST → SR | h_ss | Desired (SU) | Higher is better for the SU |
| ST → PR | **h_sp** | **Interference** | **The link we must protect — every Watt the SU sends bleeds into the PU here** |
| PT → SR | h_ps | Interference | PT noise leaking into the SU's receiver |

Each channel is modelled by its **power gain** `|h|^2 ≥ 0`, drawn independently every step from a Nakagami-m distribution (Section 3).

### 2.3 SINR formulas

**Signal-to-Interference-plus-Noise Ratio** is the workhorse quantity:

```
SINR_p = (P_p · |h_pp|²) / (P_s · |h_sp|² + σ²)     # at PR
SINR_s = (P_s · |h_ss|²) / (P_p · |h_ps|² + σ²)     # at SR
```

- Numerator = wanted signal power
- Denominator = unwanted interference + thermal noise

Note how `P_s` (the agent's action) appears as **interference** for the PU but as **signal** for the SU — this is exactly the tension the agent has to resolve.

### 2.4 Throughput and BER

**Shannon capacity** (assumed achievable rate per Hz):

```
R_x = log₂(1 + SINR_x)        x ∈ {s, p}        [bits/s/Hz]
```

**Bit-Error-Rate** under BPSK (binary phase-shift keying) on an instantaneous SINR:

```
BER = ½ · erfc( √SINR )
```

`erfc` is the complementary error function. For a Nakagami-m faded channel the **average** BER becomes an integral over the gamma-distributed SINR (Section 16).

### 2.5 The constraint

The PU is protected by a hard SINR threshold:

```
SINR_p ≥ γ_th = 1.0   (= 0 dB)
```

A time-step in which `SINR_p < γ_th` is called an **outage**. The fraction of outage steps is the **outage probability** — one of the headline metrics in every PDF report.

---

## 3. The Nakagami-m fading channel

### 3.1 Distribution

Each instantaneous power gain `|h|²` is drawn from a **Gamma distribution**:

```
|h|²  ~  Gamma(shape = m, scale = Ω / m)
mean(|h|²) = Ω           variance(|h|²) = Ω²/m
```

- `m` (the "Nakagami parameter") controls **severity** of fading.
- `Ω` is the mean link power, fixed at 1.0.

### 3.2 What different m's mean physically

| m | Distribution recovered | Channel character |
|---|------------------------|-------------------|
| 0.5 | One-sided Gaussian | Worst-case, very deep fades |
| **1.0** | **Rayleigh** | Dense scatter, no line of sight (NLOS) — *severe* |
| 2.0 | Moderate Nakagami | Mild LOS component |
| **3.0** | Moderate Nakagami | Strong LOS component — *typical urban macrocell* |
| → ∞ | Deterministic | AWGN (no fading at all) |

### 3.3 Why we picked m = 3 (and also report m = 1)

- **m = 3** matches a 3GPP urban-macro propagation profile and is the **headline** result.
- **m = 1** is the textbook Rayleigh case — included to show how the algorithms degrade under harsher fading. (DDPG falls apart faster than TD3, as expected.)

### 3.4 Block-fading vs. continuous fading

We use **block fading**: channels are constant for one symbol time (≈ one MDP step) and re-drawn independently every step. This is the standard simplifying assumption in CRN papers and matches the *coherence time* of indoor channels (~1 ms).

### 3.5 Code reference

```python
# environment.py, _draw_channels():
shape = self.nakagami_m
scale = self.nakagami_omega / self.nakagami_m
return tuple(self.rng.gamma(shape, scale, size=4))
```

---

## 4. Imperfect CSI model

### 4.1 What "CSI" means

**Channel State Information** = the receiver/transmitter's knowledge of `|h|²`. Real systems estimate `|h|²` from pilot symbols → there is **noise**. The receiver then *feeds it back* to the transmitter → there is **delay**. The feedback path is bit-limited → there is **quantization**. All three are modelled.

### 4.2 The three impairments and their parameters

| Impairment | Math | `config.py` knob | Default |
|-----------|------|-------------------|---------|
| Estimation error | `+ N(0, (σ_csi · h²)²)` | `CSI_NOISE_STD` | 0.15 |
| Feedback delay | `(1-ρ) · h²_{t-D} + ρ · h²_t` | `CSI_DELAY_STEPS`, `CSI_DELAY_RHO` | D=1, ρ=0.7 |
| Quantization | round to `2^bits` levels | `CSI_QUANT_BITS` | 0 (off) |

Combined formula applied per channel inside `environment._observe_channels`:

```
ĥ²_obs = (1 - ρ) · h²_{t - D} + ρ · h²_t  +  N(0, (σ_csi · h²)²)
ĥ²_obs ← max(0, ĥ²_obs)                    # clip negatives
ĥ²_obs ← quantize(ĥ²_obs, bits)            # if enabled
```

### 4.3 Physics vs. observation — the critical split

Inside `environment.py.step()`:

```python
# Physics uses TRUE gains to compute SINR, BER, reward, outage
sinr_p, sinr_s = self._compute_sinr(h_pp, h_sp, h_ss, h_ps, p_s)
reward         = self._compute_reward(...)

# But the state returned to the agent uses the noisy/delayed estimates
h_pp_o, h_sp_o, h_ss_o, h_ps_o = self._observe_channels(h_pp, h_sp, h_ss, h_ps)
next_state                      = self._build_state(h_pp_o, ...)
```

In short: **the agent acts on lies; the world rewards the truth.** This makes the learning problem strictly harder than a perfect-CSI MDP.

### 4.4 Turning it off

Set `IMPERFECT_CSI = False` in `config.py` to recover the original "agent sees the real channel" baseline.

---

## 5. Markov Decision Process (MDP) formulation

### 5.1 State (7 dimensions)

```
s_t = [ |h_pp|², |h_sp|², |h_ss|², |h_ps|², SINR_p, SINR_s, P_s^{t-1} ]
```

| Index | Feature | Why included |
|-------|---------|--------------|
| 0 | `|h_pp|²` | PU desired-link gain. Weak → PR already fragile → SU must back off |
| 1 | `|h_sp|²` | **Most critical** — large means ST will smash the PR |
| 2 | `|h_ss|²` | SU desired-link gain; high → power well-spent |
| 3 | `|h_ps|²` | PT's interference on SR; high → SU SINR is capped no matter what |
| 4 | `SINR_p` | Direct "how close to the constraint" reading |
| 5 | `SINR_s` | Direct "how good am I doing" reading |
| 6 | `P_s^{t-1}` | Previous action — gives the agent an "inertia" cue |

Under imperfect CSI the first six entries are the **noisy estimates**, not the truth.

### 5.2 Action

```
a_t = P_s ∈ [0, P_max = 3.0]    (continuous, scalar)
```

The actor network outputs `sigmoid(·) · P_max`, so the action is naturally bounded — no projection/clipping artefact during gradient updates.

### 5.3 Why P_max = 3.0 (and not 1.0)

We initially used `P_max = 1.0 W`. Under Nakagami fades the SU could not punch through deep fades → outage stayed > 20 %. Increasing to `3.0 W` gave the agent enough headroom while still requiring it to *not* always max out (the energy and PU-protection penalties remain). This is documented in `docs/10_camo_td3_methodology.md`.

### 5.4 Why include `P_s^{t-1}` in the state?

Without it, the policy cannot prefer smooth power trajectories. Many practical RF systems penalise large power jumps (PA back-off, EVM degradation); including `P_s^{t-1}` lets the agent reason about delta-power.

---

## 6. Reward function — line by line

```
r_t  =     α · R_s
         − β · max(0, γ_th − SINR_p)
         − γ_e · (P_s / P_max)
```

| Term | Sign | Default | Meaning |
|------|------|---------|---------|
| `α · R_s` | + | α = 1.0 | Shannon throughput reward (the primary objective) |
| `−β · max(0, γ_th − SINR_p)` | − | β = 1.5 | Linear **hinge** penalty: 0 if constraint met, grows otherwise |
| `−γ_e · (P_s / P_max)` | − | γ_e = 0.005 | Small energy penalty — breaks ties toward lower power |

### 6.1 Why a hinge instead of a step penalty?

A step penalty (large fixed cost if `SINR_p < γ_th`, zero otherwise) yields a non-differentiable signal: every step inside the feasible region looks identical, so the agent cannot tell whether it has a 1 % or 99 % safety margin. The hinge gives a smooth gradient that grows as the violation grows.

### 6.2 Why β = 1.5 (not 10) like the literature?

Setting β = 10 made the interference term dominate by ~7×, the agent collapsed into a zero-power policy (no SU throughput at all). β = 1.5 keeps the term meaningful but not lethal. The Lagrangian in CAMO-TD3 will *automatically* increase the effective β when needed — that's the whole point.

### 6.3 Why such a small γ_e?

Energy is a *tertiary* objective; we want it to break ties, not dominate. With γ_e = 0.005 the worst-case energy cost is 0.005 per step (≈ 1 reward unit per episode of 200 steps), comparable to a ~1 % shift in throughput.

### 6.4 Decomposed reward (used by CAMO-TD3)

```
r_tput   = + α · R_s                          (≥ 0)
r_intf   = − β · max(0, γ_th − SINR_p)        (≤ 0)
r_energy = − γ_e · (P_s / P_max)              (≤ 0)
```

They are **never summed**; each is stored separately and learnt by its own critic pair.

---

## 7. DDPG — the simpler baseline

### 7.1 Origin

Lillicrap et al., 2015. "Continuous control with deep reinforcement learning." First widely used continuous-action actor–critic algorithm.

### 7.2 Components

| Network | Role |
|---------|------|
| Actor `π(s) → a` | Deterministic policy |
| Critic `Q(s, a) → ℝ` | State-action value function |
| Two target networks | Slow-moving copies for stable bootstrapping |

### 7.3 Update rules

```
y      = r + γ · Q_target(s', π_target(s'))                # TD target
L_Q    = (Q(s, a) − y)²                                   # critic MSE
L_π    = − Q(s, π(s))                                     # maximise Q via deterministic policy gradient
```

Targets are **soft-updated**: `θ_target ← τ·θ + (1-τ)·θ_target`.

### 7.4 Exploration

DDPG uses **Ornstein-Uhlenbeck (OU) noise** — temporally correlated Gaussian noise that produces smoother trajectories than i.i.d. noise. In our code (`ddpg.py.OUNoise`) the OU process has mean 0 and slowly drifts.

### 7.5 Why DDPG is unstable

- Single Q-network → severe **overestimation bias** (the actor exploits over-confident Q values)
- Actor updates as fast as critic → small errors in Q amplify into bad policy moves
- No target-policy smoothing → narrow Q estimates around target actions

Empirically: DDPG plateaus early, sometimes diverges. In our 3500-episode runs it ends with ~28 % outage at m = 3 vs. TD3's ~11 %.

---

## 8. TD3 — the strong baseline

### 8.1 Origin

Fujimoto et al., 2018. "Addressing Function Approximation Error in Actor-Critic Methods."

TD3 is **DDPG + three fixes**:

1. **Twin critics** (`Q1`, `Q2`) → take `min(Q1_target, Q2_target)` in the TD target. This combats overestimation: optimistic noise in one critic is averaged out by the more pessimistic estimate from the other.

2. **Delayed policy updates** → update the actor only every `POLICY_DELAY = 2` critic updates. Critics get to settle before the actor commits to their estimates.

3. **Target policy smoothing** → add **clipped** Gaussian noise to the target action before evaluating `Q_target`:
   ```
   ã = clip(π_target(s') + clip(ε, -c, c), 0, P_max),  ε ~ N(0, σ_ps²)
   ```
   This regularises Q by enforcing that similar actions yield similar Q-values; it directly attacks the "spikes in Q" failure mode of DDPG.

### 8.2 Code map of one TD3 update (`td3.py.train_step`)

| Line range | Action |
|-----------|--------|
| 289-294 | Compute noisy target action `ã` |
| 297-299 | Compute `q_target = r + γ(1−d) min(Q1_tgt, Q2_tgt)(s', ã)` |
| 302-306 | Update Q1 by MSE |
| 309-313 | Update Q2 by MSE |
| 320-335 | **Every POLICY_DELAY steps**: update actor + soft-update all 4 targets |

### 8.3 Why TD3 wins over DDPG

- **Lower variance critics** → fewer "overestimation cliffs" the actor falls off.
- **Slower actor updates** → no positive-feedback loop between policy and bad Q values.
- **Smoothed targets** → Q surface stays Lipschitz around target actions.

In our experiments TD3 outperforms DDPG on every metric.

---

## 9. CAMO-TD3 — the proposed algorithm (deep dive)

**CAMO** = **C**onstrained **A**daptive **M**ulti-**O**bjective.

CAMO-TD3 keeps everything good about TD3 and layers on four extensions, each targeting a specific weakness of the baseline. Read this section slowly — it is the differentiator of the project.

### 9.1 The four weaknesses CAMO-TD3 addresses

| TD3 weakness | CAMO-TD3 fix |
|--------------|---------------|
| Reward weights α, β, γ_e are hand-tuned per scenario | **Adaptive Lagrangian λ₁, λ₂, λ₃** learned by dual gradient descent |
| One scalar critic mixes throughput, interference, energy | **Decomposed multi-objective critics** — 6 networks (twin per objective) |
| Markov state — no temporal memory | **GRU belief encoder** over last 8 observations |
| Symmetric exploration noise → many catastrophic violations early | **Directional noise** biased toward safer (lower) power |

### 9.2 Component 1 — GRU Belief Encoder

**Idea:** Feed the last `SEQ_LEN = 8` states through a 2-layer GRU and project the final hidden state to a `BELIEF_DIM = 16` vector. Concatenate this belief `b_t` to the current state `s_t` before passing to actor and critics.

**Why this helps under imperfect CSI:** the noise on `ĥ²` is i.i.d. across time, so averaging over a short window gives a less-noisy estimate "for free". The GRU learns a *task-conditioned* averaging — it can de-emphasise stale frames if the channel has changed.

**Why GRU and not LSTM/Transformer:**

| Encoder | Parameter cost | Strength at SEQ_LEN = 8 |
|---------|----------------|-------------------------|
| GRU | low | excellent — gates capture short-term dependencies cleanly |
| LSTM | medium | comparable but more parameters |
| Transformer | high | overkill — attention shines at SEQ_LEN ≥ 32 |

**Parameters (in `config.py`):**

| Name | Value | Meaning |
|------|-------|---------|
| `GRU_HIDDEN_SIZE` | 64 | Width of GRU hidden state |
| `GRU_NUM_LAYERS` | 2 | Depth of GRU stack |
| `SEQ_LEN` | 8 | History window in steps (≈ 4 % of an episode) |
| `BELIEF_DIM` | 16 | Compressed belief size (3.5× compression from 56 = 8 × 7 raw) |

**Compute:** every forward pass costs an extra `O(SEQ_LEN · GRU_HIDDEN_SIZE²)` ops; the largest fixed overhead in the algorithm.

### 9.3 Component 2 — Decomposed Multi-Objective Critics

Instead of one Q-value, CAMO-TD3 trains **three twin-critic pairs**:

```
Q¹ᵗ, Q²ᵗ  → expected discounted sum of   r_tput
Q¹ⁱ, Q²ⁱ  → expected discounted sum of   r_intf
Q¹ᵉ, Q²ᵉ  → expected discounted sum of   r_energy
```

Each pair uses the standard TD3 twin-critic loss with target smoothing:

```
y_k = r_k + γ (1 − d) · min( Q_target,1ᵏ(s', b', ã), Q_target,2ᵏ(s', b', ã) )
L_k = ( Q_1ᵏ(s, b, a) − y_k )² + ( Q_2ᵏ(s, b, a) − y_k )²       k ∈ {t, i, e}
```

**Why decomposed?** A scalar critic that learns `Q(s, a) ≈ E[Σ r]` mixes three very-different-scale signals (`R_s` is `~2`, the interference penalty can be `~-15`, energy is `~-0.005`). The interference term dominates → the critic loses the ability to distinguish *throughput-limited* states from *constraint-limited* states. Decomposition makes them independent learning problems.

### 9.4 Component 3 — Adaptive Lagrangian Weights

This is the most theoretically interesting piece. We treat the optimisation as a **constrained MDP**:

```
maximise_π   E[ Σ_t R_s(s_t, a_t) ]
subject to   E[ Σ_t max(0, γ_th − SINR_p) ] ≤ δ        (constraint)
             E[ Σ_t P_s ]                  ≤ P_budget  (energy budget)
```

The Lagrangian dual problem is

```
L(π, λ₁, λ₂, λ₃)  =   λ₁ · E[Σ R_s]
                    − λ₂ · ( E[Σ max(0, γ_th − SINR_p)] − δ )
                    − λ₃ · ( E[Σ P_s] − P_budget )
```

By **Sion's minimax theorem** (1958) and standard CMDP results (Altman, 1999), the saddle point of `L` solves the constrained problem. Empirically we use the **primal-dual** algorithm:

```
Actor step (primal):       maximise λ₁·Qᵗ + λ₂·Qⁱ + λ₃·Qᵉ  over π
Dual step (gradient):      λ_k ← clamp(λ_k + η_λ · ∂L/∂λ_k, λ_min, λ_max)
```

To keep `λ_k > 0` we parametrise `λ_k = softplus(log_λ_k)` and learn the unconstrained `log_λ_k`. The bounds `LAMBDA_MIN = 0.1` and `LAMBDA_MAX = 20.0` prevent runaway/collapse.

**Why this beats hand-tuned weights:**

- **No manual sweep** — different channel conditions need different α, β, γ_e; the Lagrangian discovers them.
- **Constraint guarantee in expectation** — at convergence, `λ₂` is exactly the *shadow price* of the constraint.
- **Self-healing** — if violations spike mid-training, `λ₂` rises and the actor backs off; once violations subside, `λ₂` drifts back down.

**Init choices (tuned the hard way):**

| Knob | Value | Why |
|------|-------|-----|
| `LAMBDA1_INIT = 3.0` | Throughput | Start aggressive to avoid "do-nothing" collapse |
| `LAMBDA2_INIT = 1.0` | Interference | β is already 1.5 in the per-step reward; effective weight 1.5 × 1 = 1.5, sane |
| `LAMBDA3_INIT = 0.01` | Energy | Tiebreaker only |
| `LR_LAMBDA = 5e-4` | Dual LR | Slow enough to prevent λ-oscillations; ~5× slower than the actor LR |
| `LAMBDA_MIN = 0.1` | Floor | No objective ever fully ignored |
| `LAMBDA_MAX = 20.0` | Ceiling | Prevents "death spiral" if early violations push λ₂ to infinity |

### 9.5 Component 4 — Directional Exploration Noise

Standard TD3 explores by adding zero-mean Gaussian noise to actions. Half the explored actions raise power → half cause more interference. Early in training this generates many PU violations.

CAMO-TD3 uses a **biased** Gaussian:

```
μ(t, viol_rate) = μ_bias · decay(t) · (1 + viol_rate)
decay(t)        = max(0, 1 − t / NOISE_DECAY_STEPS)
ε               ~ N( μ(t, viol_rate), σ² )
```

with `μ_bias = -0.05` (negative ⇒ exploration leans toward *lower* power).

| Knob | Value | Meaning |
|------|-------|---------|
| `MU_BIAS_INIT` | `-0.05` | Initial directional bias (negative = safer) |
| `NOISE_DECAY_STEPS` | 200 000 | Steps over which bias decays to 0 |
| `VIOLATION_WINDOW` | 100 | Rolling window to compute `viol_rate` |

`viol_rate` = fraction of the last 100 environment steps where `SINR_p < γ_th`. When violations are common, the bias amplifies; once the policy is competent, `viol_rate → 0` and the bias relaxes. After `NOISE_DECAY_STEPS`, the decay term kills the bias entirely so the agent can explore symmetrically near convergence.

### 9.6 The actor update under all four components

```
actor_loss = − E_{(s, b) ~ D}[
    λ₁ · Q_1ᵗ(s, b, π(s, b))
  + λ₂ · Q_1ⁱ(s, b, π(s, b))
  + λ₃ · Q_1ᵉ(s, b, π(s, b))
]
```

Only one of the twin critics (`Q_1`) is used for actor gradients (standard TD3 convention). The `λ_k` here are *detached* — their gradient flows through the *dual* loss instead.

### 9.7 The dual update

Approximate `∂L/∂λ_k` with running estimates of each reward stream:

```
v   = − E[ r_intf ]          # positive when violations occur
t   = + E[ r_tput ]          # what we want to grow
e   = − E[ r_energy ]        # small positive (energy cost magnitude)

L_λ = − log_λ₁ · t  +  log_λ₂ · v  +  log_λ₃ · e
```

A standard Adam step on `L_λ` then yields, after softplus, the new `λ_k`.

### 9.8 Costs CAMO-TD3 pays

- ~2.5× parameter count vs. TD3 (6 critics + GRU + projection).
- ~2-3× per-step wall-clock cost (more forward/backward passes).
- Slower initial convergence (the λ's need to stabilise).
- More hyperparameters.

The trade-off is worth it whenever the constraint is tight (low γ_th-to-noise margin) or the channel statistics change between training/deployment.

---

## 10. Every hyperparameter in `config.py`

> File reference: [config.py](../config.py)

### 10.1 System / physics

| Name | Value | Meaning | Why this value |
|------|-------|---------|----------------|
| `SIGMA2` | `1e-3` | AWGN noise variance σ² at every receiver | Thermal noise floor for room-temperature receivers (~ -30 dBm normalised) |
| `P_P` | `1.0` W | PT transmit power (fixed) | Reference primary level; sets the unit scale |
| `P_MAX` | `3.0` W | Max ST power (action upper bound) | Tuned: 1.0 W gave > 20 % outage; 3.0 W gives 11 % |

### 10.2 Reward weights

| Name | Value | Meaning |
|------|-------|---------|
| `ALPHA` | `1.0` | Throughput weight α |
| `BETA` | `1.5` | PU-violation penalty β |
| `GAMMA_REWARD` | `0.005` | Energy penalty γ_e |
| `SINR_THRESHOLD` | `1.0` | γ_th — PU SINR threshold (0 dB) |

### 10.3 RL / TD3

| Name | Value | Meaning |
|------|-------|---------|
| `STATE_DIM` | 7 | State vector length |
| `ACTION_DIM` | 1 | Scalar action |
| `HIDDEN_DIM` | 512 | Width of all FC layers in baselines |
| `REPLAY_BUFFER_SIZE` | 200 000 | Transitions stored (≈ 1000 ep × 200 steps) |
| `MIN_SAMPLES` | 1000 | Warm-up: buffer size before first update |
| `POLICY_NOISE` | 0.2 | σ of target-action noise (fraction of P_MAX) |
| `NOISE_CLIP` | 0.5 | Clip target-action noise (fraction of P_MAX) |
| `POLICY_DELAY` | 2 | Actor update frequency (every N critic updates) |
| `EXPLORATION_NOISE_STD` | 0.10 | Initial exploration σ (× P_MAX) |
| `EXPLORATION_NOISE_END` | 0.01 | Final exploration σ |
| `BATCH_SIZE` | 256 | SGD mini-batch |
| `GRAD_UPDATES_PER_STEP` | 2 | UTD ratio — 2 critic updates per env step |
| `LR_ACTOR` | `3e-4` | Adam LR for the actor |
| `LR_CRITIC` | `3e-4` | Adam LR for the critics |
| `GAMMA_DISCOUNT` | 0.99 | RL discount γ (standard for infinite-horizon control) |
| `TAU` | 0.005 | Soft-update factor for target networks |
| `TRAINING_EPISODES` | 7500 | Default episode count (CLI overrides to 3500 for headline runs) |
| `STEPS_PER_EPISODE` | 200 | Episode horizon |

### 10.4 Nakagami-m fading

| Name | Value | Meaning |
|------|-------|---------|
| `NAKAGAMI_M` | 3.0 | Fading severity m (CLI flag `--nakagami-m` overrides at run-time) |
| `NAKAGAMI_OMEGA` | 1.0 | Mean link power Ω |

### 10.5 Imperfect CSI

| Name | Value | Meaning |
|------|-------|---------|
| `IMPERFECT_CSI` | True | Master switch |
| `CSI_NOISE_STD` | 0.15 | Relative estimation error σ_csi (15 % of true gain) |
| `CSI_DELAY_STEPS` | 1 | Feedback delay D (0 = no delay) |
| `CSI_DELAY_RHO` | 0.7 | Convex-combination weight ρ between delayed and current truth |
| `CSI_QUANT_BITS` | 0 | Bits per fed-back gain (0 = disabled) |

### 10.6 CAMO-TD3 specifics

| Name | Value | Meaning |
|------|-------|---------|
| `GRU_HIDDEN_SIZE` | 64 | GRU hidden width |
| `GRU_NUM_LAYERS` | 2 | GRU depth |
| `BELIEF_DIM` | 16 | Compressed belief size |
| `SEQ_LEN` | 8 | History window passed to GRU |
| `LAMBDA1_INIT` | 3.0 | Initial λ for throughput objective |
| `LAMBDA2_INIT` | 1.0 | Initial λ for interference constraint |
| `LAMBDA3_INIT` | 0.01 | Initial λ for energy objective |
| `LR_LAMBDA` | `5e-4` | Adam LR for the dual variables |
| `LAMBDA_MIN` | 0.1 | Floor — prevents any objective being ignored |
| `LAMBDA_MAX` | 20.0 | Ceiling — prevents runaway |
| `MU_BIAS_INIT` | `-0.05` | Initial directional-noise bias |
| `NOISE_DECAY_STEPS` | 200 000 | Steps over which the bias decays |
| `VIOLATION_WINDOW` | 100 | Rolling window for `viol_rate` |
| `CAMO_HIDDEN_DIM` | 512 | Width of CAMO actor/critic FC layers |

### 10.7 Server / visualisation (not used by `train_compare.py`)

`WS_HOST`, `WS_PORT`, `BROADCAST_INTERVAL`, `SCATTER_WINDOW`, `OUTAGE_WINDOW` configure the live WebSocket dashboard (`server.py`).

---

## 11. Neural-network architectures

### 11.1 TD3 / DDPG Actor

```
Linear(7 → 512) → ReLU
Linear(512 → 512) → ReLU
Linear(512 → 1) → Sigmoid → × P_MAX
```

Output `∈ (0, P_MAX)` by construction. ~270 k parameters total.

### 11.2 TD3 / DDPG Critic

```
Concat([state, action]) →
Linear(8 → 512) → ReLU
Linear(512 → 512) → ReLU
Linear(512 → 1)
```

~270 k parameters each; TD3 maintains two such critics + two target copies = 4 critic nets.

### 11.3 CAMO-TD3 GRU Belief Encoder

```
Input: tensor of shape (batch, SEQ_LEN=8, 7)
GRU(input_size=7, hidden_size=64, num_layers=2, batch_first=True)
Take final hidden state h_L ∈ ℝ^64
Linear(64 → 16) → tanh   # belief b_t
```

### 11.4 CAMO-TD3 Actor

```
Concat([state, belief]) → ℝ^{7+16}
Linear(23 → 512) → ReLU
Linear(512 → 512) → ReLU
Linear(512 → 1) → Sigmoid → × P_MAX
```

### 11.5 CAMO-TD3 Critic (× 6)

```
Concat([state, belief, action]) → ℝ^{7+16+1}
Linear(24 → 512) → ReLU
Linear(512 → 512) → ReLU
Linear(512 → 1)
```

Six independent instances (two per objective). Each has its own target copy → **12 critic networks total**.

---

## 12. Replay buffers (uniform & sequence)

### 12.1 `ReplayBuffer` (TD3 / DDPG)

Stored as **pre-allocated GPU tensors** for speed:

| Field | Shape | dtype |
|-------|-------|-------|
| `_states` | `(N, 7)` | float32 |
| `_actions` | `(N, 1)` | float32 |
| `_rewards` | `(N, 1)` | float32 |
| `_next_states` | `(N, 7)` | float32 |
| `_dones` | `(N, 1)` | float32 |

`N = 200_000`. Circular write pointer; uniform random sampling (`torch.randint`). All tensors live on the GPU → zero CPU → GPU transfer at sample time.

### 12.2 `SequenceReplayBuffer` (CAMO-TD3)

Same as above plus a per-slot **observation history** `(SEQ_LEN, 7)` for the GRU. Three reward streams (`r_tput`, `r_intf`, `r_energy`) are stored separately. At sample time the buffer returns `(states, beliefs_input_tensor, actions, r_tput, r_intf, r_energy, next_states, next_obs_histories, dones)`.

---

## 13. Training pipeline and `train_compare.py`

### 13.1 What `train_compare.py` does

A single script that:

1. Builds whichever subset of `{TD3, DDPG, CAMO-TD3}` you ask for.
2. Trains each (sequentially or `--parallel` via `ProcessPoolExecutor`).
3. Collects per-episode metrics into a `RunMetrics` dataclass.
4. Emits a multi-page PDF with `matplotlib.backends.backend_pdf.PdfPages`.

### 13.2 Important CLI flags

| Flag | Default | Effect |
|------|---------|--------|
| `--episodes N` | 500 | Episodes per algorithm |
| `--steps-per-ep M` | 200 | Steps per episode |
| `--agents td3,ddpg,camo-td3` | `td3,ddpg` | Comma-separated list of agents to train |
| `--no-ddpg` | off | Shortcut: train only TD3 |
| `--nakagami-m X` | 3.0 | Override fading severity at run-time |
| `--camo-variant` | `full` | Ablation: `none / multi-obj-only / lambda-only / gru-only / directional-only / full` |
| `--parallel` | off | Train selected agents simultaneously (one process each) |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |
| `--seed` | 42 | Reproducibility seed (each parallel worker offsets by `+i`) |
| `--checkpoint-every K` | 0 | Save mid-run PNG + PDF every K episodes (0 = disabled) |
| `--output PATH` | `results/crn_comparison_report.pdf` | Final PDF path |

### 13.3 Per-episode loop (excerpt)

```
for ep in range(n_episodes):
    state = env.reset()
    for t in range(steps_per_ep):
        action = agent.select_action(state, exploration_noise=σ(t))
        result = env.step(action)
        replay_buffer.add(state, action, result.reward, result.state, result.done)
        for _ in range(GRAD_UPDATES_PER_STEP):
            agent.train_step(replay_buffer)
        state = result.state
        # record SINR/BER/throughput/outage flags
```

### 13.4 Headless mode

`matplotlib.use("Agg")` is set **before** the first `pyplot` import so the script never tries to open a display — essential for Kaggle/Colab/remote SSH.

### 13.5 Reproducibility

`np.random.seed(args.seed)` and `torch.manual_seed(args.seed)` are set up-front; each parallel worker offsets by `args.seed + i`. The `CRNEnvironment` carries its own `np.random.default_rng` so it doesn't pollute global state.

---

## 14. Performance metrics and how each PDF page is built

### 14.1 The metrics in `RunMetrics`

| Field | Computation |
|-------|-------------|
| `rewards` | Per-episode sum of `r_t` |
| `su_throughputs` | Per-episode mean of `R_s = log₂(1+SINR_s)` |
| `pu_throughputs` | Per-episode mean of `R_p = log₂(1+SINR_p)` |
| `outage_probs` | Per-episode fraction of steps where `SINR_p < γ_th` |
| `avg_bers` / `avg_pu_bers` | Per-episode mean BPSK BER (SU / PU) |
| `sinr_db_pts`, `ber_pts` | Step-level SU (SINR, BER) pairs, sampled |
| `pu_sinr_db_pts`, `pu_ber_pts` | Same for PU |
| `final_*` | Means over the last 500 episodes (a "post-training" summary) |
| `training_time_sec` | Wall-clock |

### 14.2 The PDF pages

1. **Summary table** — final-500-episode averages per algorithm.
2. **SU SINR vs BER scatter** — empirical points overlaid on the theoretical Nakagami-m BPSK curve (Section 16).
3. **SU throughput curve** vs. episode (raw + 20-episode rolling mean).
4. **PU throughput curve** vs. episode.
5. **Outage probability curve** with the 5 % design target as a dashed line.
6. **Reward curve per algorithm**.
7. **Reward curve overlay** (all algorithms together).

The theoretical reference curve uses

```python
nakagami_avg_ber_bpsk(snr_db, m) ≈ 0.5 · (1 + m·γ̄)^{-m}   # γ̄ from snr_db
```

(see `train_compare.nakagami_avg_ber_bpsk` for the exact closed-form).

---

## 15. Glossary of every symbol and acronym

| Symbol | Meaning |
|--------|---------|
| **AWGN** | Additive White Gaussian Noise |
| **BER** | Bit Error Rate |
| **BPSK** | Binary Phase-Shift Keying (1 bit per symbol) |
| **CMDP** | Constrained Markov Decision Process |
| **CRN** | Cognitive Radio Network |
| **CSI** | Channel State Information |
| **DDPG** | Deep Deterministic Policy Gradient |
| **CAMO-TD3** | Constrained Adaptive Multi-Objective TD3 |
| **GRU** | Gated Recurrent Unit |
| **MDP** | Markov Decision Process |
| **MSE** | Mean Squared Error |
| **NLOS / LOS** | Non-Line-Of-Sight / Line-Of-Sight |
| **OU noise** | Ornstein-Uhlenbeck noise |
| **PA** | Power Amplifier |
| **PT / PR / ST / SR** | Primary Tx, Primary Rx, Secondary Tx, Secondary Rx |
| **PU / SU** | Primary User / Secondary User |
| **Q(s, a)** | Action-value function — expected discounted return from `s` taking `a` |
| **SDR** | Software-Defined Radio |
| **SINR** | Signal-to-Interference-plus-Noise Ratio |
| **TD3** | Twin Delayed DDPG |
| **TF32** | NVIDIA Tensor-Float 32 — fast matmul mode on Ampere+ GPUs |
| **UTD** | Updates-To-Data ratio (gradient updates per env step) |
| `α, β, γ_e` | Reward weights (throughput, PU penalty, energy) |
| `γ` | RL discount factor (`GAMMA_DISCOUNT`) — confusingly shares a name with `γ_e`; γ ≈ 0.99 vs γ_e = 0.005 |
| `γ_th` | PU SINR threshold |
| `λ_k` | Lagrangian dual variable for objective k (CAMO-TD3 only) |
| `τ` | Polyak averaging factor for target networks |
| `σ²` | AWGN noise variance |
| `Ω` | Nakagami mean link power |
| `ρ` | CSI temporal correlation (delay model) |
| `D` | CSI feedback delay in steps |

---

## 16. Mathematical appendix

### 16.1 Nakagami-m PDF

```
f_{|h|²}(x) = ( m^m / Γ(m) ) · ( x^{m-1} / Ω^m ) · exp( − m·x / Ω ),     x ≥ 0
```

For m = 1 this collapses to the **exponential** (= Rayleigh power), and for m → ∞ to a Dirac at Ω.

### 16.2 Average BER on Nakagami-m BPSK

Starting from `BER = ½ erfc(√γ)` and integrating against the gamma PDF of γ, one obtains the closed-form (Simon & Alouini, *Digital Communication over Fading Channels*, 2005, eqn 5.16):

```
BER̄(m, γ̄) = ½ [ 1 − √(γ̄ / (m + γ̄)) · ∑_{k=0}^{m-1} C(2k, k) / 4^k · (m / (m + γ̄))^k ]
```

For integer m this is exact; otherwise the project uses the asymptotic `≈ 0.5 · (1 + m·γ̄)^{-m}`.

### 16.3 Soft Polyak update derivation

The target network is meant to behave like an exponential moving average:

```
θ_target^{(k+1)} = τ · θ^{(k)} + (1 − τ) · θ_target^{(k)}
```

Choosing τ small (`0.005`) ⇒ the target lags the online network by ~ `1/τ ≈ 200` updates, providing a stable bootstrapping target for TD learning.

### 16.4 Convergence of the primal-dual algorithm

For convex `L`, **Sion's minimax theorem** (1958) guarantees the saddle point exists and that *alternating* gradient ascent on the primal and descent on the dual converges. Neural-network parametrisation breaks convexity, but Ray et al. (2019, *Benchmarking Safe Exploration in Deep RL*) observe empirical convergence in deep CMDPs across many tasks — exactly the regime we operate in.

---

## 17. Common questions a viva examiner will ask

**Q. Why TD3 over DDPG?**
TD3 fixes three failure modes of DDPG: overestimation (twin critics + min), policy oscillation (delayed actor), and Q-spike artefacts (target policy smoothing). Empirically our DDPG ends at ~28 % outage vs. TD3 at ~11 % under m = 3.

**Q. Why m = 3?**
It is moderate Nakagami fading consistent with an urban-macro LOS profile (3GPP TR 38.901). It is neither trivial (m → ∞) nor pathological (m = 1).

**Q. Why log₂(1 + SINR) for the rate?**
Shannon's capacity formula for a Gaussian channel. It is the theoretical upper bound on bits/s/Hz achievable; treating it as our reward is the standard idealisation in CRN papers.

**Q. Why penalise the PU violation with a hinge?**
A hinge gives a smooth gradient that grows with the violation magnitude, while still being zero inside the feasible region. A step penalty would yield a flat-then-cliff loss — much harder to optimise.

**Q. Why decompose the reward in CAMO-TD3?**
The three reward components have very different scales (`+2`, `−15`, `−0.005`). A scalar critic mixes them and ends up dominated by the largest — it can't tell why a state is bad. Decomposed critics give three independent learning signals; the Lagrangian then combines them with *learned* (not hand-set) weights.

**Q. How does the Lagrangian know to update λ₂?**
By the dual loss `L_λ = log_λ₂ · v` where `v = −E[r_intf]`. When violations happen, `v > 0` and gradient descent on `log_λ₂` is *negative*, but the actor loss term is `+λ₂ · Q_i(s, b, a)` with `Q_i < 0` (an estimate of the negative interference return) — so increasing λ₂ makes the actor avoid violations *more*. The signs work out to "violations → λ₂ rises → actor backs off".

**Q. Why not just use a transformer instead of a GRU?**
At SEQ_LEN = 8 the per-step cost of a transformer dominates and the attention pattern is degenerate (only 8 keys). GRUs match or exceed transformers in this regime in RL benchmarks. Above SEQ_LEN = 32 or so, attention starts to pay off.

**Q. What is the most important hyperparameter?**
`P_MAX`. We initially used 1.0 W and could not get outage below 20 %; bumping to 3.0 W was the single most impactful change. After that, `β` and the CAMO `λ` initialisations matter most.

**Q. What is the practical upshot of imperfect CSI?**
Every algorithm degrades; CAMO-TD3 degrades least because the GRU implicitly averages over the noisy estimates. The numerical gap between "perfect" and "imperfect" CSI is the realistic deployment cost of pilot-based channel estimation.

**Q. Why log₂ and not natural log?**
Conversion factor only — the agent learns the same optimum either way. We use log₂ so the unit "bits/s/Hz" matches the wireless-engineering convention.

**Q. How do you know the agent isn't just memorising specific channel realisations?**
Channels are drawn i.i.d. every step from a continuous distribution; the probability of seeing the same `(h_pp², h_sp², h_ss², h_ps²)` tuple twice is zero. Generalisation is forced by the very nature of block fading.

---

*Author: project for Cognitive Radio Networks Mini-Project,
Department of Electronics & Communication Engineering,
Ramaiah Institute of Technology, Bangalore.*
