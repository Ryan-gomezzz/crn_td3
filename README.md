# Cognitive Radio Network — TD3 Power Allocation

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&pause=1000&color=FFFFFF&background=00000000&center=true&width=800&lines=Cognitive+Radio+Network;TD3+Power+Allocation" alt="Animated Heading" />
</p>

**Deep Reinforcement Learning for Intelligent Spectrum Management**

*Academic Mini-Project guided by Dr. Chitra M  | by Aditya Gangwani, Ryan Gomez, Shreya Revankar and Sneha Tapadar*

---

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.3%2B-00B140?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</div>

---

## What This Project Does

A **Cognitive Radio (SU)** learns — entirely through trial and error — to share spectrum with a licensed Primary User while:

- **Maximising** its own data throughput (Shannon capacity)
- **Protecting** the Primary User from harmful interference
- **Minimising** energy consumption

The learning agent is a **TD3 (Twin Delayed DDPG)** neural network controller. Watch it evolve in real time through a Pygame GUI that shows animated radio links, live training plots, and an AI insights panel.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   COGNITIVE RADIO NETWORK                       │
│                                                                 │
│    PT ══════[ h_pp ]══════════════════════════════> PR          │
│  (fixed)        Primary Link (must be protected!)     (victim)  │
│    │                                                    ▲       │
│    │ h_ps                                        h_sp  │        │
│    │ (interference                          (interference│      │
│    │  to SU)                                 to PU!)   │        │
│    ▼                                                    │       │
│    SR <══════[ h_ss ]══════════════════════════════  ST         │
│  (goal)         SU Link (maximise throughput!)     (agent)      │
│                                                     P_s ∈       │
│                                                   [0, P_max]    │
└─────────────────────────────────────────────────────────────────┘
```

### Channel Model: Rayleigh Block-Fading

Each of the four links experiences independent **Rayleigh fading** — the channel power gain is drawn fresh every time step from an Exponential distribution:

```
|h|² ~ Exp(1)    i.e.   E[|h|²] = 1,   Var[|h|²] = 1
```

This forces the agent to generalise across all possible channel realisations, not just memorise a single operating point.

---

## SINR & Reward Formulation

**Signal-to-Interference-plus-Noise Ratios:**

```
              P_p × |h_pp|²                      P_s × |h_ss|²
SINR_p = ─────────────────────      SINR_s = ─────────────────────
          P_s × |h_sp|² + σ²                  P_p × |h_ps|² + σ²
```

**SU Throughput (Shannon Capacity):**
```
R_s = log₂(1 + SINR_s)   [bits/s/Hz]
```

**Reward Function (three competing objectives):**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   r = α · R_s  −  β · max(0, γ_th − SINR_p)  −  γ · (P_s / P_max)       │
│         ▲                      ▲                        ▲               │
│    Throughput              PU violation              Energy             | 
│    reward                  penalty                   penalty            │
│    (α = 1.0)               (β = 10.0)                (γ = 0.01)         │
│                            γ_th = 2.0                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## TD3 Algorithm

**Twin Delayed Deep Deterministic Policy Gradient** (Fujimoto et al., 2018)

```
┌────────────────────────────────────────────────────────────────────┐
│                      TD3 AGENT                                     │
│                                                                    │
│  ┌─────────────┐    select_action(s)    ┌──────────────────────┐   │
│  │    ACTOR    │ ─────────────────────► │   ENVIRONMENT (CRN)  │   │
│  │  π_φ(s)     │                        │                      │   │
│  │  7→256→256→1│ ◄── soft update ──     │  Rayleigh channels   │   │
│  └─────────────┘        (τ=0.005)       │  SINR computation    │   │
│         ▲                               │  Reward function     │   │
│    actor loss                           └──────────┬───────────┘   │
│   -Q₁(s,π(s))           s, a, r, s'    │                      |    │
│         │               ─────────────► │ REPLAY BUFFER        |    │
│  ┌──────┴──────┐                       │ (100,000 transitions)|    │
│  │  CRITIC 1   │ ◄──── sample batch ── └──────────────────────┘    │
│  │  Q_θ₁(s,a) │                                                    │
│  │ 8→256→256→1 │   Target Q = r + γ·min(Q₁', Q₂')                  │
│  └─────────────┘                                                   │
│  ┌─────────────┐   ← MSE loss on both critics every step           │
│  │  CRITIC 2   │   ← Actor update every 2nd critic step            │
│  │  Q_θ₂(s,a) │   ← Target policy smoothing (clipped noise)        │
│  └─────────────┘                                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Why TD3 for This Problem?

| Feature | Benefit |
|---------|---------|
| Continuous action space | P_s is real-valued; DQN cannot handle this |
| Twin critics | Prevents Q-value overestimation → stable learning |
| Delayed policy updates | More stable actor; reduces variance |
| Target policy smoothing | Robust Q-function across action space |
| Replay buffer | Breaks temporal correlation in correlated channel samples |

---

## Real-Time Pygame GUI

The **1620×900 pixel window** is divided into four panels:

```
┌──────────────────── STATUS BAR ─────────────────────────────────────────────┐
│  Episode | Step | Reward | Avg100 | Noise | Buffer | [ Training Status ]    │
├─────────────────────────┬──────────────────────┬──────────────────────────  ┤
│                         │  Episode Reward       │                           │
│   NETWORK               │  ████ raw + avg100    │    AI INSIGHTS            │
│   VISUALIZATION         ├──────────────────────┤                            │
│                         │  SU Throughput R_s    │  • Training Stage badge   │
│   PT ──────────── PR    │  ████                 │  • PU Protection Rate     │
│   │ (green=OK/red=bad)  ├──────────────────────┤  • Reward Trend arrow      │
│   │                     │  PU SINR              │  • Policy Insights text   │
│   ST ──────────── SR    │  ████ — — threshold   │  • Violation Counter      │
│      (blue = SU link)   │                       │  • Live stats             │
├─────────────────────────┴──────────────────────┴──────────────────────────  ┤
│  [ 1× ] [ 5× ] [ 10× ] [ Max ]    [ Pause ]  [ Interf ]  [ Reset ]          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Link Colour Coding

| Link | Colour | Condition |
|------|--------|-----------|
| PT → PR | **Solid Green** | PU SINR ≥ 2.0 (threshold met — PU happy) |
| PT → PR | **Solid Red** | PU SINR < 2.0 (constraint violated!) |
| ST → SR | **Solid Blue** | SU desired link (width ∝ throughput) |
| ST → PR | **Dashed Red** | SU→PU interference (toggle with I key) |
| PT → SR | **Dashed Orange** | PU→SU interference (toggle with I key) |

---

## Training Progression

```
Reward
   │
+5 ┤                                                 ╭──────────────── Converged
   │                                          ╭──────╯
 0 ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╭──╯  ← zero crossing
   │                                   ╭───╯
-100┤                           ╭───────╯
   │                    ╭───────╯         ← avg100 reward (yellow)
-500┤          ╭─────────╯
   │  ─────────╯   raw reward (gray, noisy)
-1000┤─────────────────────────────────────────────────────────► Episode
   0       500       1000      1500      2000      2500      3000
   │         │          │
   Exploring  Learning   Converging
```

---

## Installation & Running

```bash
# Clone or navigate to project
cd crn_td3

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies (PyTorch, Pygame, NumPy)
pip install -r requirements.txt

# Launch the simulation
python main.py
```

**That's it.** The Pygame window opens immediately and training begins.

---

## Controls

| Input | Action |
|-------|--------|
| `Space` | Pause / Resume training |
| `R` | Reset — restart training from scratch |
| `I` | Toggle interference link visibility |
| `1×` button | 1 step per frame (slow, fully animated) |
| `5×` button | 5 steps per frame |
| `10×` button | 10 steps per frame |
| `Max` button | Full episode per frame (maximum speed) |

---

## File Structure

```
crn_td3/
│
├── main.py             ← Entry point: training loop + GUI integration
├── environment.py      ← CRN physics: Rayleigh fading, SINR, reward
├── td3.py              ← TD3: Actor, Critic, ReplayBuffer, TD3Agent
├── visualization.py    ← Pygame GUI: NetworkPanel, PlotSurface, InsightsPanel
├── config.py           ← All hyperparameters and constants
├── utils.py            ← RollingStats, ExplorationNoise, Logger
│
├── requirements.txt    ← pip dependencies
├── README.md           ← This file
│
├── docs/               ← Detailed documentation
│   ├── index.md
│   ├── 01_system_model.md
│   ├── 02_channel_model.md
│   ├── 03_rl_formulation.md
│   ├── 04_td3_algorithm.md
│   ├── 05_neural_networks.md
│   ├── 06_training_process.md
│   ├── 07_gui_guide.md
│   └── 08_results_interpretation.md
│
├── checkpoints/        ← Auto-saved every 100 episodes
│   ├── ep_0100/
│   ├── ep_0200/
│   └── ...
│
└── saved_models/       ← Final model after training
    ├── actor.pth
    ├── critic1.pth
    └── critic2.pth
```

---

## Hyperparameter Reference

| Category | Parameter | Value | Role |
|----------|-----------|-------|------|
| **System** | P_p | 1.0 W | PT fixed transmit power |
| | P_max | 1.0 W | ST maximum power cap |
| | σ² | 10⁻³ W | AWGN noise floor |
| | SINR_th | 2.0 | PU protection threshold (~3 dB) |
| **Reward** | α | 1.0 | SU throughput weight |
| | β | 10.0 | PU violation penalty weight |
| | γ | 0.01 | Energy efficiency penalty |
| **TD3** | discount | 0.99 | Future reward discount |
| | τ | 0.005 | Soft target update rate |
| | π noise | 0.2·P_max | Target policy smoothing σ |
| | noise clip | 0.5·P_max | Target noise clipping range |
| | policy delay | 2 | Actor update period |
| **Training** | episodes | 3,000 | Total training episodes |
| | steps/ep | 200 | Steps per episode |
| | batch size | 128 | Mini-batch size |
| | LR | 3×10⁻⁴ | Adam learning rate (actor + critics) |
| | buffer | 100,000 | Replay buffer capacity |

---

## Academic References

1. **Fujimoto, S., van Hoof, H., & Meger, D.** (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

2. **Haykin, S.** (2005). *Cognitive radio: brain-empowered wireless communications.* IEEE Journal on Selected Areas in Communications, 23(2), 201–220.

3. **Goldsmith, A.** (2005). *Wireless Communications.* Cambridge University Press.

4. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction.* MIT Press.

---

<div align="center">

*Built with PyTorch · Pygame · NumPy*

**guided by Dr. Chitra M and developed by Aditya Gangwani, Ryan Gomez, Shreya Revankar and Sneha Tapadar**

</div>
