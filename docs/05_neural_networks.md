# 05 — Neural Network Architectures

## Actor Network (Policy)

The actor network implements the policy π_φ: S → A, mapping observed states to transmit power decisions.

```
Input Layer        Hidden Layer 1     Hidden Layer 2     Output Layer
  (7 neurons)       (256 neurons)      (256 neurons)      (1 neuron)

 [h_pp²   ]                                              
 [h_sp²   ]         ┌──────────┐       ┌──────────┐     ┌──────────┐
 [h_ss²   ] ──────► │  Linear  │─ReLU─►│  Linear  │─ReLU►│  Linear  │─Sigmoid─► × P_max ─► P_s
 [h_ps²   ]         │  7→256   │       │  256→256 │     │  256→1   │
 [SINR_p  ]         └──────────┘       └──────────┘     └──────────┘
 [SINR_s  ]
 [P_s_prev]
```

### Architecture Details

| Layer | Type | Input → Output | Activation | Purpose |
|-------|------|----------------|------------|---------|
| 1 | Linear | 7 → 256 | ReLU | Feature extraction from state |
| 2 | Linear | 256 → 256 | ReLU | Abstract representation learning |
| 3 | Linear | 256 → 1 | Sigmoid × P_max | Bounded action output |

### Why Sigmoid × P_max for Output?

The sigmoid function σ(x) ∈ (0, 1) guarantees the output is strictly within the valid action range (0, P_max). This is preferable to tanh × (P_max/2) + (P_max/2) because:
- Output is always strictly positive (P_s = 0 exactly is unreachable, which is fine physically)
- No need for external clipping during inference
- Gradient flows through sigmoid without vanishing at the boundaries for typical inputs

### Why ReLU Hidden Activations?

ReLU (Rectified Linear Unit) is used for the hidden layers because:
- **Computationally efficient** — simple thresholding operation
- **No vanishing gradient problem** — gradient is 1 for positive inputs
- **Sparse activation** — roughly half the neurons are inactive at any input, acting as implicit regularisation
- **State-of-the-art** for continuous control tasks (superior to tanh/sigmoid in hidden layers)

### Parameter Count

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Layer 1 | 7 × 256 = 1,792 | 256 | 2,048 |
| Layer 2 | 256 × 256 = 65,536 | 256 | 65,792 |
| Layer 3 | 256 × 1 = 256 | 1 | 257 |
| **Total** | | | **68,097** |

---

## Critic Network (Q-Function)

Two independent critic networks Q_{θ₁} and Q_{θ₂} each estimate the Q-value for a given (state, action) pair.

```
State Input     Action Input
  (7 neurons)    (1 neuron)
      │               │
      └───────┬────────┘
              │ Concatenate
              │ (8 neurons)
              ▼
        ┌──────────┐       ┌──────────┐     ┌──────────┐
        │  Linear  │─ReLU─►│  Linear  │─ReLU►│  Linear  │─►  Q-value
        │  8→256   │       │  256→256 │     │  256→1   │   (scalar)
        └──────────┘       └──────────┘     └──────────┘
```

### Architecture Details

| Layer | Type | Input → Output | Activation |
|-------|------|----------------|------------|
| Input concat | cat([s, a]) | 7+1 = 8 | — |
| 1 | Linear | 8 → 256 | ReLU |
| 2 | Linear | 256 → 256 | ReLU |
| 3 | Linear | 256 → 1 | None (linear) |

### Why No Output Activation on Critic?

The Q-value is a real number with no predefined range — it can be positive or negative, and its magnitude grows with the discount factor and episode length. Any bounded activation (sigmoid, tanh) would limit the expressiveness of the Q-function and cause learning to stall.

### Parameter Count (per critic)

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Layer 1 | 8 × 256 = 2,048 | 256 | 2,304 |
| Layer 2 | 256 × 256 = 65,536 | 256 | 65,792 |
| Layer 3 | 256 × 1 = 256 | 1 | 257 |
| **Total** | | | **68,353** |

**Total critic parameters (×2):** 136,706

---

## Total Network Parameters

| Network | Parameters |
|---------|-----------|
| Actor | 68,097 |
| Actor Target | 68,097 |
| Critic 1 | 68,353 |
| Critic 1 Target | 68,353 |
| Critic 2 | 68,353 |
| Critic 2 Target | 68,353 |
| **Grand Total** | **409,606** |

This is a relatively small network — intentionally so. The CRN problem has a 7-dimensional state and 1-dimensional action; a larger network would overfit and train slower without meaningful improvement.

---

## Target Networks

Each of the three networks (actor, critic 1, critic 2) has a **target copy** that is updated via **Polyak averaging (soft update)**:

$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

With τ = 0.005, the target network moves very slowly — it takes approximately 1/τ = 200 actor updates to substantially change. This prevents the "moving target" problem where the Q-function tries to hit a target that changes too rapidly, causing divergence.

---

## Initialisation

All linear layers use **PyTorch's default initialisation**: Kaiming uniform initialisation, which is designed for ReLU networks and ensures gradients are well-scaled at the start of training.
