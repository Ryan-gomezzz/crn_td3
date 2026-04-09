# 03 — Reinforcement Learning Formulation

## The Core Problem

We frame the CRN power control problem as a **Markov Decision Process (MDP)**:

> An agent (the TD3 algorithm) interacts with the CRN environment at discrete time steps. At each step it observes the current state, takes an action, receives a reward, and transitions to the next state. Its goal is to find a policy that maximises the long-term cumulative discounted reward.

---

## State Space

**Dimension:** 7 (continuous)

The state vector s_t captures all information the agent needs to make an intelligent power control decision:

$$s_t = \left[ |h_{pp}|^2,\ |h_{sp}|^2,\ |h_{ss}|^2,\ |h_{ps}|^2,\ \text{SINR}_p,\ \text{SINR}_s,\ P_s^{(t-1)} \right]$$

| Index | Feature | Why It's Included |
|-------|---------|------------------|
| 0 | \|h_pp\|² | Tells agent how strong the PU's own link is — a weak h_pp means SINR_p is already fragile |
| 1 | \|h_sp\|² | Most critical: high h_sp means even small P_s will strongly interfere with PR |
| 2 | \|h_ss\|² | Tells agent how efficiently SU can use power — high h_ss = more throughput per Watt |
| 3 | \|h_ps\|² | PT's interference to SR — limits max achievable SINR_s regardless of P_s |
| 4 | SINR_p | Current PU quality — how close to the protection threshold |
| 5 | SINR_s | Current SU quality — direct measure of what the agent is optimising |
| 6 | P_s^(t-1) | Previous action — gives the agent a sense of momentum/inertia |

### Why Include SINR Directly?

The channel gains alone don't tell the full story — SINR depends on the product of channel gain and power. Including SINR_p and SINR_s gives the agent an immediate, actionable signal of the current system state without requiring it to derive these from the raw channel gains.

---

## Action Space

**Dimension:** 1 (continuous)

$$a_t = P_s \in [0, P_{max}] = [0, 1.0] \text{ W}$$

The actor network outputs a single scalar representing the secondary transmitter's power level for the current time step. The sigmoid activation on the output ensures the value is automatically bounded in (0, P_max) without any external clipping.

---

## Reward Function

The reward function encodes the three competing objectives of the SU:

$$r_t = \underbrace{\alpha \cdot R_s}_{\text{(1) maximise SU throughput}} - \underbrace{\beta \cdot \max\!\left(0,\ \gamma_{th} - \text{SINR}_p\right)}_{\text{(2) penalise PU violation}} - \underbrace{\gamma \cdot \frac{P_s}{P_{max}}}_{\text{(3) energy efficiency}}$$

### Term 1: SU Throughput Reward (α = 1.0)

$$\alpha \cdot R_s = \alpha \cdot \log_2(1 + \text{SINR}_s)$$

This is the primary objective. Shannon capacity gives a logarithmic reward — doubling SINR does not double the reward. This naturally prevents the agent from over-investing in very high power levels.

### Term 2: PU Protection Penalty (β = 10.0)

$$-\beta \cdot \max(0,\ \gamma_{th} - \text{SINR}_p)$$

This is a **linear hinge penalty** — zero when SINR_p ≥ γ_th (constraint satisfied), and growing linearly with the shortfall otherwise. The large weight β = 10.0 makes constraint violations extremely costly, forcing the agent to prioritise PU protection.

> **Design choice:** β = 10.0 means a constraint violation of 1.0 SINR unit costs the agent 10 units of reward — roughly equivalent to losing ~7 bits/s/Hz of SU throughput. This ensures the agent never willingly violates the PU constraint when it can be avoided.

### Term 3: Energy Penalty (γ = 0.01)

$$-\gamma \cdot \frac{P_s}{P_{max}}$$

A small regularisation term that encourages energy efficiency. Without it, the agent might find policies that use full power P_max in situations where lower power would also satisfy the constraint, wasting energy unnecessarily.

---

## Episode Structure

| Parameter | Value |
|-----------|-------|
| Episode length | 200 time steps |
| Total training episodes | 3,000 |
| Total training steps | 600,000 |

At the **start of each episode**:
- Channel gains are reset (new random draw)
- P_s_prev is reset to 0
- Episode reward accumulator is cleared

At **each time step within an episode**:
1. New channel gains are drawn (Rayleigh block fading)
2. Agent observes state s_t
3. Agent selects action a_t = P_s
4. Environment computes SINR, R_s, reward r_t
5. Transition (s_t, a_t, r_t, s_{t+1}) stored in replay buffer
6. TD3 training step executed

---

## Markov Property

The state is **Markov** (the future depends only on the current state, not the history) because:
- Channel gains are drawn i.i.d. at each step — no memory of past channels
- SINR values are fully determined by current channels and P_s
- P_s_prev is included in the state so the agent has all relevant past information

---

## Optimal Policy (Intuition)

After convergence, the optimal policy should:

1. **Observe h_sp carefully** — if h_sp is large, keep P_s small to protect PR
2. **Observe h_ss** — if h_ss is large, transmit more to exploit the good SU channel
3. **Monitor SINR_p margin** — if SINR_p is well above γ_th, the agent has room to increase P_s
4. **Balance** the trade-off dynamically at every time step

This is essentially a **water-filling** type solution, but learned adaptively via deep RL rather than derived analytically.
