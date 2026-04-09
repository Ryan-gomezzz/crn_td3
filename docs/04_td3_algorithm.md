# 04 — TD3 Algorithm: Twin Delayed DDPG

## Background

TD3 (Twin Delayed Deep Deterministic Policy Gradient) was introduced by Fujimoto, van Hoof, and Meger in 2018. It addresses three well-known failure modes of the original DDPG algorithm:

| DDPG Failure Mode | TD3 Solution |
|-------------------|-------------|
| **Overestimation bias** — Q-function overestimates value, leading to poor policies | **Twin critics** — take the minimum of two Q-estimates |
| **High variance in policy updates** — frequent updates amplify noise | **Delayed policy update** — update actor only every N critic steps |
| **Brittle target policy** — small changes in target actor cause large Q-target variance | **Target policy smoothing** — add clipped noise to target actions |

---

## Algorithm Walkthrough

### Initialisation

```
Actor π_φ with parameters φ              (maps state → action)
Critic Q_{θ₁}, Q_{θ₂} with parameters θ₁, θ₂   (maps (state,action) → Q-value)
Target copies: π_{φ'}, Q_{θ₁'}, Q_{θ₂'}  (initialised identical to originals)
Replay buffer B of capacity 100,000
```

### Interaction Loop (per time step)

```
1. Observe state s_t
2. Select action: a_t = π_φ(s_t) + ε,  ε ~ N(0, σ_explore)
3. Clip a_t to [0, P_max]
4. Execute a_t in environment → receive r_t, s_{t+1}
5. Store (s_t, a_t, r_t, s_{t+1}, done) in replay buffer B
```

### Training Loop (per training step)

**Step 1 — Sample mini-batch**

Sample N = 128 transitions from B:
{(s_i, a_i, r_i, s'_i, done_i)} for i = 1..N

**Step 2 — Compute target actions with smoothing noise**

$$\tilde{a} = \pi_{\phi'}(s') + \text{clip}(\epsilon,\ -c,\ c), \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}})$$
$$\tilde{a} = \text{clip}(\tilde{a},\ 0,\ P_{max})$$

where c = 0.5 (noise clip) and σ_noise = 0.2 (target policy noise).

> **Why smooth?** Without noise, the critic can exploit narrow peaks in the Q-function where the target actor places actions. Adding clipped noise forces the critic to learn a smoother, more robust Q-landscape.

**Step 3 — Compute target Q-value**

$$y = r + \gamma (1 - \text{done}) \cdot \min\!\left(Q_{\theta_1'}(s', \tilde{a}),\ Q_{\theta_2'}(s', \tilde{a})\right)$$

> **Why take the minimum?** Both critics tend to overestimate — taking the minimum provides a conservative (pessimistic) estimate of the true value, which has been shown empirically to yield better policies.

**Step 4 — Update both critics**

$$\mathcal{L}(\theta_k) = \frac{1}{N} \sum_i \left(Q_{\theta_k}(s_i, a_i) - y_i\right)^2, \quad k \in \{1, 2\}$$

Both critics are updated independently via gradient descent.

**Step 5 — Delayed actor update (every d = 2 critic steps)**

$$\nabla_\phi J = \frac{1}{N} \sum_i \nabla_a Q_{\theta_1}(s_i, a)\big|_{a=\pi_\phi(s_i)} \cdot \nabla_\phi \pi_\phi(s_i)$$

The actor is updated to maximise Q₁ (only one critic used here — using both would cause double-counting since they are correlated).

**Step 6 — Soft-update target networks (every d steps)**

$$\phi' \leftarrow \tau \phi + (1 - \tau) \phi'$$
$$\theta_k' \leftarrow \tau \theta_k + (1 - \tau) \theta_k', \quad k \in \{1, 2\}$$

with τ = 0.005. This slow, gradual update keeps target networks stable while still tracking the learned policy.

---

## Hyperparameter Justification

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Discount γ | 0.99 | High discount — agent cares about long-term PU protection, not just immediate reward |
| Replay buffer | 100,000 | Large enough to break temporal correlations; small enough to fit in memory |
| Batch size | 128 | Balance between gradient quality and update frequency |
| Learning rate | 3×10⁻⁴ | Adam LR; proven stable for continuous control tasks |
| τ (soft update) | 0.005 | Very slow target update — prevents catastrophic overwriting |
| Policy noise σ | 0.2 × P_max | Smooths Q in action space proportional to action range |
| Noise clip c | 0.5 × P_max | Prevents extreme target actions from destabilising Q |
| Policy delay d | 2 | Actor sees more stable Q estimates before each update |
| Exploration noise | 0.1 → 0.01 × P_max | Decays linearly from 10% to 1% of P_max over training |

---

## Why TD3 for CRN?

| Requirement | Why TD3 Satisfies It |
|-------------|---------------------|
| Continuous action space (P_s ∈ [0,1]) | TD3 is designed for continuous actions; DQN cannot handle this |
| Deterministic optimal policy | Actor outputs a single deterministic P_s, not a distribution |
| Stable learning in noisy environment | Twin critics + delayed updates prevent divergence from channel noise |
| Safety-aware (PU constraint) | Heavy reward penalty enforces constraint without explicit safety layers |
| Sample efficiency | Replay buffer allows reuse of past experience |

---

## Convergence Behaviour

A well-trained TD3 agent on this CRN task should show:

1. **Episodes 0–100:** Pure exploration. Rewards ≈ −1000 (random power, maximum violations)
2. **Episodes 100–500:** Buffer fills; critic losses are high but decreasing. Agent starts avoiding maximum power.
3. **Episodes 500–1500:** Actor updates produce noticeably lower power levels. Constraint violations decrease.
4. **Episodes 1500–3000:** Policy converges to near-optimal power control. Rewards stabilise near positive values.

The reward curve follows an approximately **logarithmic growth** from negative to positive, which is characteristic of TD3 on constraint-satisfaction problems.
