# Documentation Index — CRN TD3 Power Allocation

> **Project:** Cognitive Radio Network Power Allocation via Twin Delayed DDPG  
> **Institution:** Ramaiah Institute of Technology, Bangalore  
> **Framework:** PyTorch + Pygame

---

## Contents

| # | Document | Description |
|---|----------|-------------|
| [01](01_system_model.md) | **System Model** | Four-node CRN topology, nodes, links, SINR formulas |
| [02](02_channel_model.md) | **Channel Model** | Rayleigh fading, block-fading assumption, AWGN |
| [03](03_rl_formulation.md) | **RL Formulation** | MDP definition, state/action/reward spaces |
| [04](04_td3_algorithm.md) | **TD3 Algorithm** | Full algorithm walkthrough with equations |
| [05](05_neural_networks.md) | **Neural Networks** | Actor/Critic architectures, parameter counts |
| [06](06_training_process.md) | **Training Process** | Phases, exploration schedule, checkpointing |
| [07](07_gui_guide.md) | **GUI Guide** | Panel layout, link colours, controls reference |
| [08](08_results_interpretation.md) | **Results** | How to read plots, expected training curve, diagnostics |

---

## Quick Reference

### Channel Model
```
|h|² ~ Exponential(1)     (Rayleigh power gain)
σ² = 1e-3 W               (noise power)
```

### SINR Formulas
```
SINR_p = (P_p × h_pp²) / (P_s × h_sp² + σ²)
SINR_s = (P_s × h_ss²) / (P_p × h_ps² + σ²)
R_s    = log2(1 + SINR_s)  [bits/s/Hz]
```

### Reward
```
r = 1.0 × R_s  -  10.0 × max(0, 2.0 - SINR_p)  -  0.01 × (P_s / P_max)
      ↑ throughput      ↑ PU penalty                  ↑ energy penalty
```

### TD3 Key Numbers
```
State dim: 7     Action dim: 1     Hidden: 256×256
Buffer: 100,000  Batch: 128        Episodes: 3,000
τ = 0.005        γ = 0.99          Policy delay: 2
```
