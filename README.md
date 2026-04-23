<p align="center">
  <h1 align="center">🧠 CRN-TD3 — Cognitive Radio Network Power Allocation via Deep Reinforcement Learning</h1>
  <p align="center">
    <strong>TD3 · DDPG · CAMO-TD3 &nbsp;|&nbsp; Nakagami-<em>m</em> Fading Channel &nbsp;|&nbsp; Real-Time Dashboard</strong>
  </p>
  <p align="center">
    <em>Ramaiah Institute of Technology, Bangalore</em><br/>
    Aditya Gangwani · Ryan Gomez · Shreya Revankar · Sneha Tapadar
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" alt="React">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Model](#-system-model)
- [Algorithms](#-algorithms)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output & Results](#-output--results)
- [Examples](#-examples)
- [Technologies Used](#-technologies-used)
- [Performance Notes](#-performance-notes)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This project implements **deep reinforcement learning agents** for **dynamic power allocation** in a **Cognitive Radio Network (CRN)** under **Nakagami-*m* fading channels**.

In a CRN, a Secondary User (SU) must dynamically adjust its transmit power to maximize its own throughput while protecting the Primary User (PU) from harmful interference. This is formulated as a continuous-action RL problem and solved using three algorithms:

| Algorithm | Description |
|-----------|-------------|
| **TD3** | Twin Delayed DDPG — addresses overestimation bias with twin critics, delayed policy updates, and target policy smoothing |
| **DDPG** | Deep Deterministic Policy Gradient — baseline actor-critic with Ornstein–Uhlenbeck exploration |
| **CAMO-TD3** | *Constrained Adaptive Multi-Objective TD3* — a novel extension with GRU belief encoding, decomposed multi-objective critics, adaptive Lagrangian weights, and directional exploration noise |

The project includes a **real-time browser dashboard** (React + WebSocket) for live training visualization, a **headless comparison training script** with automated PDF report generation, and a **Kaggle-ready notebook** for cloud GPU training.

---

## ✨ Features

- **Three RL Algorithms** — TD3, DDPG, and the novel CAMO-TD3, all sharing a unified API for fair comparison
- **Nakagami-*m* Fading Channel** — Configurable fading severity (`m=1` recovers Rayleigh, `m=3` default moderate fading)
- **Real-Time Dashboard** — React + Recharts frontend connected via FastAPI WebSocket showing live SINR, BER, throughput, outage probability, and network topology
- **Automated PDF Reports** — Multi-page comparison reports with SINR vs BER scatter plots, throughput curves, outage probability, reward curves, and summary tables
- **Cloud-Ready** — Kaggle/Colab notebook (`crn_10k_training.ipynb`) for GPU-accelerated training up to 10,000 episodes
- **Checkpointing** — Periodic model saves and intermediate checkpoint plots/PDFs during long training runs
- **GPU-Optimized Replay Buffer** — Pre-allocated on-device tensor storage with zero-copy batch sampling
- **Comprehensive Documentation** — 10+ documentation pages covering system model, channel model, RL formulation, algorithm details, and GUI guide

---

## 📡 System Model

The environment simulates a **4-node underlay CRN** topology:

```
  Primary Transmitter (PT) ──── h_pp ────→ Primary Receiver (PR)
         │                                         ↑
         │ h_ps (interference)            h_sp (interference)
         ↓                                         │
  Secondary Receiver (SR) ←── h_ss ──── Secondary Transmitter (ST)
```

### Channel Model

- **Fading**: Nakagami-*m* block fading — `|h|² ~ Gamma(m, Ω/m)`, drawn fresh each time step
- **Noise**: AWGN with power `σ² = 1e-3 W`

### SINR Formulas

```
SINR_p = (P_p × h_pp²) / (P_s × h_sp² + σ²)     — Primary User SINR
SINR_s = (P_s × h_ss²) / (P_p × h_ps² + σ²)     — Secondary User SINR
```

### Reward Function

```
r = α × log₂(1 + SINR_s) − β × max(0, SINR_thr − SINR_p) − γ × (P_s / P_max)
    ├── SU throughput        ├── PU protection penalty        └── Energy penalty
```

### RL Formulation

| Component | Details |
|-----------|---------|
| **State** (7D) | `[h_pp², h_sp², h_ss², h_ps², SINR_p, SINR_s, P_s_prev]` |
| **Action** (1D) | `P_s ∈ [0, P_max]` — continuous transmit power |
| **Reward** | Throughput − interference penalty − energy penalty |
| **Episode** | 200 time steps (configurable channel coherence blocks) |

---

## 🤖 Algorithms

### TD3 (Twin Delayed DDPG)

Based on [Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477):

- **Twin Critics**: Two independent Q-networks; uses `min(Q1, Q2)` for target to reduce overestimation
- **Delayed Policy Updates**: Actor updated every 2 critic steps
- **Target Policy Smoothing**: Clipped Gaussian noise added to target actions
- **Soft Target Updates**: Polyak averaging with `τ = 0.005`

### DDPG (Deep Deterministic Policy Gradient)

Based on [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971):

- Single critic (no twin min-trick)
- No target policy smoothing noise
- Actor updated every step (no policy delay)
- Ornstein–Uhlenbeck (OU) exploration noise for temporally correlated exploration

### CAMO-TD3 (Constrained Adaptive Multi-Objective TD3)

A novel extension of TD3 with four key innovations:

1. **GRU Belief Encoder** — A 2-layer GRU processes the last 8 observations to produce a 16D belief vector capturing temporal channel dynamics
2. **Decomposed Multi-Objective Critics** — 6 critics total (twin critics × 3 objectives: throughput, interference, energy), each learning objective-specific Q-values
3. **Adaptive Lagrangian Weights** — Dual gradient descent auto-tunes the trade-off weights `λ₁, λ₂, λ₃` online, clamped to `[0.1, 20.0]` to prevent collapse
4. **Directional Exploration Noise** — Biased noise nudges actions toward constraint satisfaction (lower power when PU violations are frequent)

---

## 📁 Project Structure

```
crn_td3/
├── config.py                  # All hyperparameters and constants (single source of truth)
├── environment.py             # CRN environment (Gym-like API, no external dependency)
├── td3.py                     # TD3 agent: Actor, Critic, ReplayBuffer, TD3Agent
├── ddpg.py                    # DDPG agent: DDPGActor, DDPGCritic, OUNoise, DDPGAgent
├── camo_td3.py                # CAMO-TD3: BeliefEncoder, CAMOActor, ObjectiveCritic,
│                              #   SequenceReplayBuffer, CAMO_TD3Agent
├── utils.py                   # Helpers: RollingStats, ExplorationNoise, Logger
├── main.py                    # TD3-only training loop (standalone or WebSocket mode)
├── server.py                  # FastAPI WebSocket server (streams live metrics to browser)
├── train_compare.py           # Headless multi-algorithm comparison + PDF report generator
├── crn_10k_training.ipynb     # Kaggle/Colab notebook for cloud GPU training (10k episodes)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── frontend/                  # React + Vite real-time dashboard
│   ├── package.json           # Node dependencies (React 19, Recharts, Vite 8)
│   ├── vite.config.js         # Vite configuration
│   ├── index.html             # HTML entry point
│   └── src/
│       ├── App.jsx            # Main app shell with WebSocket connection status
│       ├── main.jsx           # React entry point
│       ├── App.css            # Global styles
│       ├── index.css          # Base styles
│       ├── hooks/
│       │   └── useWebSocket.js # Custom WebSocket hook for live data streaming
│       ├── components/
│       │   ├── Dashboard.jsx       # Main dashboard layout
│       │   ├── NetworkTopology.jsx  # SVG network topology visualization
│       │   ├── BerVsSinrChart.jsx   # BER vs SINR scatter plot (Recharts)
│       │   ├── ThroughputChart.jsx  # Throughput time series chart
│       │   ├── OutageProbChart.jsx  # Outage probability chart
│       │   └── StatsPanel.jsx       # Live statistics panel
│       └── lib/               # Utility libraries
│
├── docs/                      # Detailed technical documentation
│   ├── index.md               # Documentation index with quick reference
│   ├── 01_system_model.md     # System model: topology, nodes, links, SINR
│   ├── 02_channel_model.md    # Nakagami-m fading, block-fading, AWGN
│   ├── 03_rl_formulation.md   # MDP definition, state/action/reward spaces
│   ├── 04_td3_algorithm.md    # TD3 walkthrough with equations
│   ├── 05_neural_networks.md  # Actor/Critic architectures, parameter counts
│   ├── 06_training_process.md # Training phases, exploration schedule, checkpointing
│   ├── 07_gui_guide.md        # Dashboard panel layout, controls reference
│   ├── 08_results_interpretation.md # How to read plots, diagnostics
│   ├── 09_camo_td3.md         # CAMO-TD3 algorithm overview
│   ├── 10_camo_td3_methodology.md  # Full CAMO-TD3 methodology for reports
│   ├── COLAB_GUIDE.md         # Google Colab setup instructions
│   └── KAGGLE_GUIDE.md        # Kaggle notebook setup and GPU tips
│
└── results/                   # Generated reports and checkpoint plots
    ├── crn_comparison_report_*.pdf  # Multi-algorithm comparison PDF reports
    ├── report_3way*.pdf             # Three-way (TD3 vs DDPG vs CAMO-TD3) reports
    ├── checkpoint_*.png             # Mid-run checkpoint visualizations
    └── smoke_test.pdf               # Quick validation report
```

---

## 🚀 Installation

### Prerequisites

- **Python** ≥ 3.10
- **Node.js** ≥ 18 (for the frontend dashboard only)
- **CUDA-capable GPU** (recommended for faster training; CPU works but is slower)

### 1. Clone the Repository

```bash
git clone https://github.com/Ryan-gomezzz/crn_td3.git
cd crn_td3
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---------|---------|
| `torch >= 2.0.0` | Neural networks, GPU acceleration |
| `numpy >= 1.24.0` | Numerical computation |
| `scipy >= 1.11.0` | BER calculation (`erfc` function) |
| `matplotlib >= 3.7.0` | Plotting and PDF report generation |
| `fastapi >= 0.111.0` | WebSocket server |
| `uvicorn[standard] >= 0.29.0` | ASGI server for FastAPI |
| `websockets >= 12.0` | WebSocket protocol support |

### 4. Install Frontend Dependencies (Optional — for Dashboard)

```bash
cd frontend
npm install
cd ..
```

---

## 💻 Usage

### Mode 1: Real-Time Dashboard (WebSocket + Browser)

Start the backend server and the frontend dev server:

```bash
# Terminal 1 — Start the WebSocket server + TD3 training
python server.py

# Terminal 2 — Start the React dev server
cd frontend
npm run dev
```

Open your browser at **`http://localhost:5173`** to see live training metrics including:
- Network topology with animated channel gains
- SINR vs BER scatter plot with theoretical Nakagami-*m* curve
- SU throughput time series
- Outage probability chart
- Real-time statistics panel (reward, training stage, constraint rate)

> **Production Build**: To serve the dashboard from the backend directly:
> ```bash
> cd frontend && npm run build && cd ..
> python server.py
> # Open http://localhost:8000
> ```

### Mode 2: Standalone TD3 Training (Console Only)

```bash
python main.py
```

Trains the TD3 agent for 7,500 episodes (default) with console logging. Saves checkpoints to `./checkpoints/` and the final model to `./saved_models/`.

### Mode 3: Multi-Algorithm Comparison + PDF Report

The recommended way to run experiments:

```bash
# Default: TD3 vs DDPG (500 episodes each)
python train_compare.py

# TD3 vs DDPG with more episodes
python train_compare.py --episodes 3000 --output results/report_3000ep.pdf

# All three algorithms: TD3 vs DDPG vs CAMO-TD3
python train_compare.py --agents td3,ddpg,camo-td3 --episodes 1500

# TD3 only (quick smoke test)
python train_compare.py --episodes 100 --no-ddpg --output results/smoke_test.pdf

# With intermediate checkpoint graphs every 750 episodes
python train_compare.py --episodes 10000 --agents td3,ddpg,camo-td3 \
  --checkpoint-every 750 --output results/full_report.pdf

# Parallel training (multi-process, same GPU)
python train_compare.py --agents td3,ddpg --parallel --episodes 1000
```

### Mode 4: Kaggle / Colab Notebook

Upload and run `crn_10k_training.ipynb` on [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/) with GPU acceleration enabled. The notebook:

1. Installs dependencies
2. Writes all source files via `%%writefile` magic
3. Trains all three algorithms for 10,000 episodes
4. Generates a comprehensive PDF report in `/kaggle/working/results/`

See `docs/KAGGLE_GUIDE.md` and `docs/COLAB_GUIDE.md` for detailed setup instructions.

---

## ⚙️ Configuration

All hyperparameters are centralized in **`config.py`**. Key parameters:

### System Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIGMA2` | `1e-3` | AWGN noise power (Watts) |
| `P_P` | `1.0` | Primary Transmitter fixed power (Watts) |
| `P_MAX` | `3.0` | Maximum Secondary Transmitter power (Watts) |
| `SINR_THRESHOLD` | `1.0` | Minimum acceptable PU SINR (~0 dB) |

### Reward Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA` | `1.0` | SU throughput weight |
| `BETA` | `1.5` | PU SINR constraint violation penalty |
| `GAMMA_REWARD` | `0.005` | Energy usage penalty |

### RL / TD3 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIDDEN_DIM` | `512` | Hidden layer width (Actor & Critic) |
| `REPLAY_BUFFER_SIZE` | `200,000` | Maximum replay buffer capacity |
| `BATCH_SIZE` | `256` | Mini-batch size |
| `LR_ACTOR` / `LR_CRITIC` | `3e-4` | Adam learning rates |
| `GAMMA_DISCOUNT` | `0.99` | Discount factor |
| `TAU` | `0.005` | Soft target update rate |
| `POLICY_NOISE` | `0.2` | Target policy smoothing noise (fraction of P_MAX) |
| `POLICY_DELAY` | `2` | Actor update frequency |
| `TRAINING_EPISODES` | `7,500` | Total training episodes |
| `STEPS_PER_EPISODE` | `200` | Steps per episode |

### Nakagami-*m* Fading

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NAKAGAMI_M` | `3.0` | Fading severity (`1` = Rayleigh, `3` = moderate Nakagami) |
| `NAKAGAMI_OMEGA` | `1.0` | Mean power per link |

### CAMO-TD3 Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRU_HIDDEN_SIZE` | `64` | GRU encoder hidden width |
| `GRU_NUM_LAYERS` | `2` | GRU depth |
| `BELIEF_DIM` | `16` | Compressed belief vector dimension |
| `SEQ_LEN` | `8` | Observation history length |
| `LAMBDA1_INIT` / `LAMBDA2_INIT` / `LAMBDA3_INIT` | `3.0 / 1.0 / 0.01` | Initial Lagrangian weights |
| `LR_LAMBDA` | `0.0005` | Lagrangian multiplier learning rate |
| `MU_BIAS_INIT` | `-0.05` | Directional noise bias (negative → lower power to protect PU) |

### WebSocket / Server

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WS_HOST` | `0.0.0.0` | Server bind address |
| `WS_PORT` | `8000` | Server port |
| `BROADCAST_INTERVAL` | `50` | Broadcast metrics every N steps |

---

## 📊 Output & Results

### Model Checkpoints

| Location | Contents |
|----------|----------|
| `./saved_models/` | Final trained model weights (`.pth` files) |
| `./checkpoints/ep_XXXX/` | Periodic checkpoint weights during training |
| `./saved_models/ddpg/` | DDPG-specific model weights |
| `./saved_models/camo_td3/` | CAMO-TD3 model weights + Lagrangian multipliers |

### PDF Reports

Generated by `train_compare.py` into the `results/` directory:

| Report Page | Content |
|------------|---------|
| **Page 1** | Title page with summary statistics table (avg reward, throughput, outage, BER, training time) with winner highlighting |
| **Page 2** | SU SINR vs BER scatter plot with theoretical BPSK (AWGN) and Nakagami-*m* curves |
| **Page 3** | PU SINR vs BER scatter plot with SINR threshold marker |
| **Page 4** | Secondary User throughput curves over episodes |
| **Page 5** | Primary User throughput curves over episodes |
| **Page 6** | Outage probability over episodes (with 5% target line) |
| **Page 7** | Individual reward curves per algorithm |
| **Page 8** | Combined reward comparison overlay |

### Checkpoint Plots

When `--checkpoint-every N` is specified, intermediate PNG plots and single-algorithm PDF reports are saved to the `results/` directory at every N episodes.

### Console Logs

Training progress is printed to the console with formatted tables:

```
  Episode |  Steps |    Reward |    Avg100 |  SU_Rate |  PU_SINR |   Power |   Buffer
----------------------------------------------------------------------------------
      50  |    200 |   87.4321 |   65.1234 |   1.8765 |   3.4567 |  1.2340 |    10000
```

---

## 📝 Examples

### Quick Smoke Test

```bash
python train_compare.py --episodes 100 --no-ddpg --output results/smoke_test.pdf
```

Expected: ~30 seconds on GPU, produces a single-algorithm report confirming the pipeline works.

### Standard Comparison (TD3 vs DDPG)

```bash
python train_compare.py --episodes 3000 --output results/comparison_3000ep.pdf
```

Expected: ~30 minutes on GPU. The PDF will show TD3 generally achieving higher throughput and lower outage probability than DDPG.

### Full 3-Way Comparison with Checkpoints

```bash
python train_compare.py \
  --agents td3,ddpg,camo-td3 \
  --episodes 10000 \
  --checkpoint-every 750 \
  --output results/crn_comparison_report_10000ep.pdf \
  --seed 42
```

Expected: ~3–5 hours on GPU (T4/P100). Produces a comprehensive multi-page PDF with intermediate checkpoint graphs saved as PNGs and PDFs.

### Live Dashboard Training

```bash
# Terminal 1
python server.py

# Terminal 2
cd frontend && npm run dev
```

Open `http://localhost:5173` — you'll see real-time charts updating every 50 environment steps, including an animated SVG network topology showing channel gains and power levels.

---

## 🛠 Technologies Used

### Backend

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | ≥ 3.10 | Core language |
| **PyTorch** | ≥ 2.0 | Neural network training and inference |
| **NumPy** | ≥ 1.24 | Array operations, channel simulation |
| **SciPy** | ≥ 1.11 | BER calculation (`erfc`), statistical functions |
| **Matplotlib** | ≥ 3.7 | Plotting and PDF report generation |
| **FastAPI** | ≥ 0.111 | WebSocket server for real-time dashboard |
| **Uvicorn** | ≥ 0.29 | ASGI server |

### Frontend

| Technology | Version | Purpose |
|-----------|---------|---------|
| **React** | 19 | UI framework |
| **Vite** | 8 | Build tool and dev server |
| **Recharts** | 3.8 | Charting library for live metrics |
| **WebSocket** | Native | Real-time data streaming |

### Infrastructure

| Tool | Purpose |
|------|---------|
| **Kaggle / Google Colab** | Cloud GPU training |
| **Git** | Version control |

---

## 📈 Performance Notes

- **GPU Acceleration**: CUDA is auto-detected. Training on a T4 GPU is ~5–10× faster than CPU.
- **TF32 Optimization**: Enabled by default on Ampere/Ada GPUs (RTX 30xx/40xx) for ~2× speedup.
- **GPU-Resident Replay Buffer**: Tensors are pre-allocated on the GPU, eliminating CPU→GPU transfer overhead during sampling.
- **Update-to-Data (UTD) Ratio**: 2 gradient updates per environment step keeps GPU utilization high despite the small network size.
- **Typical Training Times** (T4 GPU):
  - 500 episodes × 200 steps: ~10 minutes
  - 3,000 episodes: ~45 minutes
  - 10,000 episodes (3 algorithms): ~3–5 hours
- **CAMO-TD3** is approximately **2–3× slower** per episode than TD3/DDPG due to the GRU encoder, 6 critics, and Lagrangian updates.
- **Memory**: The replay buffer pre-allocates ~200K transitions. For CAMO-TD3 with sequence history, expect ~1–2 GB GPU memory usage.

### Known Limitations

- The environment uses a **simplified 4-node topology** without path loss or mobility.
- Channel gains are drawn **i.i.d. per time step** (block fading) — no temporal correlation.
- BER is computed using the **instantaneous BPSK formula**, not through actual bit-level simulation.
- The CAMO-TD3 Lagrangian update can occasionally exhibit oscillatory behavior in the early training phases.

---

## 🔮 Future Improvements

- [ ] **Multi-SU Scaling** — Extend the environment to multiple secondary user pairs with inter-SU interference
- [ ] **Path Loss & Mobility** — Add distance-dependent path loss and node mobility models
- [ ] **Spectrum Sensing Integration** — Combine RL power control with energy detection or cyclostationary spectrum sensing
- [ ] **OFDM Extension** — Generalize to multi-carrier systems with sub-channel power allocation
- [ ] **Transfer Learning** — Pre-train on one channel condition and fine-tune on another for faster adaptation
- [ ] **Prioritized Experience Replay** — Replace uniform sampling with priority-weighted sampling for faster convergence
- [ ] **SAC Comparison** — Add Soft Actor-Critic for entropy-regularized exploration comparison
- [ ] **Deployment** — Docker containerization for reproducible cloud deployment
- [ ] **Multi-Agent RL** — Decentralized training for scenarios with non-cooperative secondary users

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes with clear, descriptive messages:
   ```bash
   git commit -m "feat: add prioritized experience replay to TD3"
   ```
4. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** with a detailed description of your changes

### Guidelines

- Follow the existing code style (type hints, docstrings, `# ──` section separators)
- Keep all hyperparameters in `config.py` — avoid magic numbers in other files
- Add documentation to `docs/` for any new algorithmic components
- Ensure `train_compare.py --episodes 100 --no-ddpg` passes as a basic smoke test

---

## 📄 License

This project is available under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

### References

- **TD3**: Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- **DDPG**: Lillicrap, T. P., et al. (2015). *Continuous Control with Deep Reinforcement Learning.* ICLR 2016. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- **Nakagami-*m* BER**: Simon, M. K., & Alouini, M.-S. (2005). *Digital Communication over Fading Channels.* Wiley, 2nd Edition.
- **CRN Power Control**: Haykin, S. (2005). *Cognitive Radio: Brain-Empowered Wireless Communications.* IEEE JSAC.

### Institution

Developed at **Ramaiah Institute of Technology, Bangalore** under the guidance of **Dr. Chitra M**.

### Authors

- Aditya Gangwani
- Ryan Gomez
- Shreya Revankar
- Sneha Tapadar

---

<p align="center">
  <sub>Built with ❤️ using PyTorch, FastAPI, and React</sub>
</p>
