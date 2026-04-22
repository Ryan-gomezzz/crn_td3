# Documentation Index — CRN TD3 / DDPG Power Allocation

> **Project:** Cognitive Radio Network Power Allocation via TD3 and DDPG  
> **Institution:** Ramaiah Institute of Technology, Bangalore  
> **Framework:** PyTorch · Matplotlib · FastAPI

---

## Contents

| # | Document | Description |
|---|----------|-------------|
| [01](01_system_model.md) | **System Model** | Four-node CRN topology, nodes, links, SINR formulas |
| [02](02_channel_model.md) | **Channel Model** | Nakagami-m fading (m=3), block-fading assumption, AWGN |
| [03](03_rl_formulation.md) | **RL Formulation** | MDP definition, state/action/reward spaces |
| [04](04_td3_algorithm.md) | **TD3 Algorithm** | Full TD3 walkthrough with equations |
| [05](05_neural_networks.md) | **Neural Networks** | Actor/Critic architectures, parameter counts |
| [06](06_training_process.md) | **Training Process** | Phases, exploration schedule, checkpointing |
| [07](07_gui_guide.md) | **GUI Guide** | Panel layout, link colours, controls reference |
| [08](08_results_interpretation.md) | **Results** | How to read plots, expected training curve, diagnostics |
| [09](09_camo_td3.md) | **CAMO-TD3 Algorithm** | Constrained Adaptive Multi-Objective TD3 — GRU belief encoder, decomposed critics, adaptive Lagrangian |
| [10](10_camo_td3_methodology.md) | **CAMO-TD3 Methodology** | Full methodology: problem, formulation, parameter reasoning, why better than TD3/DDPG (for reports/reviews) |
| [KAGGLE_GUIDE](KAGGLE_GUIDE.md) | **Kaggle Guide** | Step-by-step notebook setup, GPU tips, troubleshooting |

---

## Quick Reference

### Channel Model (Nakagami-m, m = 3)
```
|h|² ~ Gamma(m=3, Ω/m)    (Nakagami-m power gain, m=1 recovers Rayleigh)
σ² = 1e-3 W               (noise power)
```

### SINR Formulas
```
SINR_p = (P_p × h_pp²) / (P_s × h_sp² + σ²)
SINR_s = (P_s × h_ss²) / (P_p × h_ps² + σ²)
R_s    = log2(1 + SINR_s)  [SU throughput, bits/s/Hz]
R_p    = log2(1 + SINR_p)  [PU throughput, bits/s/Hz]
BER    = 0.5 × erfc(√SINR_s)  [BPSK instantaneous BER]
```

### Reward
```
r = 1.0 × R_s  -  10.0 × max(0, 2.0 - SINR_p)  -  0.01 × (P_s / P_max)
      ↑ SU throughput    ↑ PU protection penalty      ↑ energy penalty
```

### TD3 Key Numbers
```
State dim: 7     Action dim: 1     Hidden: 256×256
Buffer: 100,000  Batch: 128        Episodes: 3,000,000
τ = 0.005        γ = 0.99          Policy delay: 2
```

### DDPG Key Differences vs TD3
```
Single critic (no twin min-trick)
No target policy smoothing noise
Actor updated every step (no policy delay)
Ornstein-Uhlenbeck exploration noise
```

### CAMO-TD3 Key Additions vs TD3
```
GRU Belief Encoder: 2-layer GRU, seq_len=8, belief_dim=16
6 Critics: twin critics for throughput, interference, energy
Adaptive Lagrangian: lambda_1=1.0, lambda_2=10.0, lambda_3=0.01
Directional Noise: mu_bias=-0.15, decays over 200k steps
```

### Comparison Script
```bash
# Train TD3 + DDPG and generate PDF report
python train_compare.py --episodes 500 --output results/report.pdf

# Train all three (TD3 + DDPG + CAMO-TD3)
python train_compare.py --agents td3,ddpg,camo-td3 --episodes 1500 --output results/report_3way.pdf

# TD3 only (quick smoke-test)
python train_compare.py --episodes 100 --no-ddpg --output results/td3_only.pdf
```
