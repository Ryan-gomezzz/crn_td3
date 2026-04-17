# 02 — Channel Model: Nakagami-m Fading (m = 3)

## What is Nakagami-m Fading?

**Nakagami-m** is a general-purpose fading model that covers a range of fading severities with a single parameter m ≥ 0.5:

| m value | Distribution | Physical meaning |
|---------|-------------|-----------------|
| m = 0.5 | One-sided Gaussian | Worst-case fading |
| **m = 1** | **Rayleigh / Exponential** | **No line-of-sight, urban scatter** |
| m = 2 | Moderate fading | Partial LoS |
| **m = 3** | **Mild fading** | **Strong LoS, suburban/indoor** |
| m → ∞ | AWGN (no fading) | Ideal channel |

This project uses **m = 3**, which models a scenario with a relatively strong line-of-sight component — less volatile than pure Rayleigh (m=1) but still with significant channel variation.

---

## Mathematical Model

The **channel power gain** |h|² under Nakagami-m follows a **Gamma distribution**:

$$|h|^2 \sim \text{Gamma}\!\left(m,\; \frac{\Omega}{m}\right), \quad \mathbb{E}[|h|^2] = \Omega$$

where:
- **m = 3** — fading severity parameter (integer for closed-form BER)
- **Ω = 1.0** — mean channel power (normalised)

The PDF is:

$$f(x) = \frac{m^m x^{m-1}}{\Gamma(m)\,\Omega^m} \exp\!\left(-\frac{mx}{\Omega}\right), \quad x \geq 0$$

In code (`environment.py`):

```python
scale   = nakagami_omega / nakagami_m   # = 1.0/3.0
h_sq    = rng.gamma(nakagami_m, scale)  # Gamma(3, 1/3) → E[h²]=1, Var=1/3
```

With m=3 the variance of |h|² drops to Ω²/m = 1/3 (vs 1 for Rayleigh), meaning **less extreme deep fades**.

---

## Average BER for BPSK under Nakagami-m (m = 3)

For BPSK modulation the **instantaneous** BER given SINR γ is:

$$\text{BER}(\gamma) = \frac{1}{2}\,\text{erfc}\!\left(\sqrt{\gamma}\right)$$

The **average** BER over Nakagami-m fading (integer m, Simon & Alouini, 2005, eq. 8.98):

$$\bar{P}_b = \left(\frac{1-\mu}{2}\right)^m \sum_{k=0}^{m-1} \binom{m-1+k}{k} \left(\frac{1+\mu}{2}\right)^k, \quad \mu = \sqrt{\frac{\bar{\gamma}}{m + \bar{\gamma}}}$$

where $\bar{\gamma}$ is the average SINR. This closed-form is plotted on the SINR vs BER page of the PDF report.

---

## Block Fading Model

This simulation uses **block fading** — the channel is drawn fresh every time step, independently across all four links:

- **h_pp** — Primary Transmitter → Primary Receiver
- **h_sp** — Secondary Transmitter → Primary Receiver  *(interference link)*
- **h_ss** — Secondary Transmitter → Secondary Receiver *(desired SU link)*
- **h_ps** — Primary Transmitter → Secondary Receiver  *(interference link)*

At each step, four independent `Gamma(3, 1/3)` samples are drawn. The agent observes the resulting channel gains and SINRs, then chooses transmit power P_s.

---

## Nakagami-m = 3 vs Rayleigh (m = 1): Key Differences

| Property | Rayleigh (m=1) | Nakagami-m=3 |
|----------|---------------|--------------|
| |h|² distribution | Exponential(1) | Gamma(3, 1/3) |
| Mean |h|² | 1.0 | 1.0 |
| Variance |h|² | 1.0 | **0.33** |
| P(deep fade: |h|² < 0.1) | ≈ 9.5% | **≈ 0.2%** |
| BER at 10 dB SNR | ≈ 5×10⁻³ | **≈ 1×10⁻⁴** |
| Reward learning | Harder (volatile) | Smoother convergence |

The reduced variance makes the reward signal **less noisy**, so both TD3 and DDPG converge more cleanly — making algorithmic differences more visible in the comparison plots.

---

## AWGN Noise

Additive White Gaussian Noise is present at every receiver:

$$n \sim \mathcal{CN}(0,\,\sigma^2), \quad \sigma^2 = 10^{-3}\text{ W}$$

This provides a noise floor and prevents division-by-zero in SINR denominators.
