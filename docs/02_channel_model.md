# 02 — Channel Model: Rayleigh Fading

## What is Rayleigh Fading?

In real wireless environments, a signal doesn't travel on a single straight path from transmitter to receiver. Instead, it reflects off buildings, trees, vehicles, and other objects, arriving at the receiver via **multiple paths** simultaneously — each with a different delay, phase, and amplitude. This is called **multipath propagation**.

**Rayleigh fading** is the statistical model used when:
- There is **no dominant line-of-sight (LoS) path** between transmitter and receiver
- The signal arrives via many independent scattered paths
- The receiver sees constructive and destructive interference between these paths

This is the standard model for urban wireless environments, which is why it's used here.

---

## Mathematical Model

The complex baseband channel coefficient h is modeled as:

$$h = h_I + j \cdot h_Q$$

where h_I and h_Q are independent zero-mean Gaussian random variables:

$$h_I, h_Q \sim \mathcal{N}\left(0, \frac{1}{2}\right)$$

The **magnitude** |h| then follows a **Rayleigh distribution**, and crucially, the **channel power gain** |h|² follows an **Exponential distribution**:

$$|h|^2 \sim \text{Exponential}(\lambda = 1) \quad \Rightarrow \quad \mathbb{E}[|h|^2] = 1$$

In code, this is simply:

```python
h_sq = np.random.exponential(1.0)   # |h|² ~ Exp(1)
```

---

## Block Fading Model

This simulation uses **block fading** — the channel remains constant for one time step (one resource block), then changes independently for the next step. This means:

- At each time step t, four new independent channel gains are drawn:
  - h_pp(t), h_sp(t), h_ss(t), h_ps(t)
- Each draw is independent of all previous draws (i.i.d. fading)
- The agent must adapt its power P_s to each new channel realisation

This models a fast-fading environment where the coherence time equals one transmission slot — a common assumption in OFDM-based cognitive radio systems.

---

## Why This Makes the Problem Hard

Because channels change **every single time step**, the agent cannot simply memorise a fixed power level. It must learn a **policy** — a mapping from observed channel states to transmit power — that works well on average across all possible channel realisations.

### Channel Statistics

For a single Exponential(1) random variable:

| Metric | Value |
|--------|-------|
| Mean E[|h|²] | 1.0 |
| Variance Var[|h|²] | 1.0 |
| Probability P(|h|² < 0.1) | ≈ 9.5% (deep fade) |
| Probability P(|h|² > 3.0) | ≈ 5.0% (strong channel) |

This means ~10% of the time any given link is in a **deep fade** — the channel is nearly blocked. The agent must handle these edge cases gracefully.

---

## Channel Instantiations and Their Effect

| Scenario | h_pp | h_sp | h_ss | h_ps | Optimal P_s | Reason |
|----------|------|------|------|------|------------|--------|
| Strong primary link, weak interference | High | Low | High | Low | High | PT→PR strong; ST can transmit powerfully without harming PU |
| Weak h_pp, strong h_sp | Low | High | Any | Any | Very Low | Even small P_s damages PR (weak PT signal + strong ST interference) |
| h_ss in deep fade | Any | Any | Low | Any | Low | No point transmitting; SU link is blocked anyway |
| Ideal conditions | High | Low | High | Low | ~P_max | Maximum SU throughput while keeping PU safe |

The TD3 agent learns to map these channel observations directly to the appropriate power level.

---

## AWGN Noise

Additive White Gaussian Noise (AWGN) is present at every receiver:

$$n \sim \mathcal{CN}(0, \sigma^2), \quad \sigma^2 = 10^{-3} \text{ W}$$

This ensures that even when all interference is zero (P_s = 0), there is still a noise floor. It also prevents division-by-zero in the SINR formula — the denominator is always at least σ² > 0.
