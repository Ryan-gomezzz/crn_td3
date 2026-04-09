# 07 — GUI Guide: Reading the Visualization

## Window Layout

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  TOP STATUS BAR — Episode | Step | Reward | Avg100 | Noise | Buffer | Status    │
├──────────────────────┬───────────────────────────┬──────────────────────────────┤
│                      │   Episode Reward Plot      │                              │
│   CRN NETWORK        │   (raw + 100-ep avg)       │                              │
│   VISUALIZATION      ├───────────────────────────┤   AI INSIGHTS PANEL          │
│                      │   SU Throughput Plot       │   (training analysis)        │
│   [Animated nodes,   │   (R_s per step)           │                              │
│    links, pulses]    ├───────────────────────────┤                              │
│                      │   PU SINR Plot             │                              │
│                      │   (with threshold line)    │                              │
├──────────────────────┴───────────────────────────┴──────────────────────────────┤
│  BOTTOM CONTROL BAR — Speed | Pause | Interference | Reset | Keyboard hints      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Left Panel: CRN Network Visualization

### Nodes

| Node | Color | Position |
|------|-------|----------|
| PT (Primary Transmitter) | Blue | Top-left |
| PR (Primary Receiver) | Green | Top-right |
| ST (Secondary Transmitter) | Red | Bottom-left |
| SR (Secondary Receiver) | Purple | Bottom-right |

**Halo colors:**
- **Green halo** — node performing well (SINR above threshold for PU nodes; non-zero throughput for SU nodes)
- **Red/orange halo** — node under stress (PU SINR violated, or SU in deep fade)

### Links

| Link | Style | Color | Meaning |
|------|-------|-------|---------|
| PT → PR | Solid | **Green** | PU SINR ≥ threshold (constraint satisfied) |
| PT → PR | Solid | **Red** | PU SINR < threshold (constraint violated!) |
| ST → SR | Solid Blue | Thin–Thick | SU link; line width ∝ SU throughput R_s |
| ST → PR | Dashed Red | Fixed | Interference from SU to PU |
| PT → SR | Dashed Orange | Fixed | Interference from PU to SU |

### Signal Pulses

Small filled circles travel along each link, driven by a continuously advancing phase variable. They provide a visual indication that the channel is active. In the early training stages they appear random; by convergence the PT→PR pulse is always green.

### Power Bar

A vertical bar on the left side of the panel shows ST's current transmit power P_s:
- **Green fill** — low power (good, safe)
- **Yellow fill** — medium power (caution zone)
- **Red fill** — high power (likely causing PU interference)

### Numeric Readouts

Values displayed near each node:
- Near PT: `P_p = 1.00 W` (constant)
- Near PR: `SINR_p = X.XX` (coloured green/red based on threshold)
- Near ST: `P_s = X.XXX W`
- Near SR: `SINR_s = X.XX` and `R_s = X.XXX b/s/Hz`

---

## Right Panel: Live Plots

### Plot 1: Episode Reward

- **Gray line:** Raw episode reward (highly noisy)
- **Yellow line:** Rolling 100-episode average (smooth trend)
- **What to look for:** Yellow line should trend upward from large negative values toward zero and eventually positive

### Plot 2: SU Throughput (R_s)

- **Blue line:** SU throughput at each step within the current episode, in bits/s/Hz
- **What to look for:** As training progresses, this should hover at 0–2 b/s/Hz, with the agent achieving higher values more consistently

### Plot 3: PU SINR

- **Cyan line:** SINR_p at each step
- **Yellow dashed line:** Protection threshold (SINR_th = 2.0)
- **What to look for:** In early training, SINR_p frequently dips below the threshold (red zone). By convergence, it should stay above the dashed line most of the time.

---

## Far-Right Panel: AI Insights

This panel provides a **real-time analytical summary** of training progress:

| Widget | Description |
|--------|-------------|
| Training Stage badge | Current phase: Exploring / Learning / Converging |
| Progress bar | Episode count as percentage of total training |
| PU Protection rate | % of recent steps where SINR_p ≥ threshold (target: 100%) |
| SU Throughput stats | Current episode average and all-time peak |
| Reward trend arrow | ↑ Improving / → Stable / ↓ Declining |
| Average power trend | Shows whether agent is learning to use less power |
| Best episode | Highest reward achieved so far |
| Policy insight text | Plain-English description of current agent behaviour |
| Violation count | How many constraint breaks in last 100 steps |

---

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume training |
| `R` | Reset — restart training from scratch |
| `I` | Toggle interference link visibility |

### Bottom Bar Buttons

| Button | Action |
|--------|--------|
| `1×` | 1 environment step per render frame |
| `5×` | 5 steps per frame (5× faster) |
| `10×` | 10 steps per frame (10× faster) |
| `Max` | Full episode per frame (maximum speed, animation disabled) |
| `Pause / Resume` | Same as Space |
| `Interf: ON/OFF` | Toggle dashed interference links |
| `Reset` | Same as R |

### Recommended Workflow

1. Start with **1× speed** to watch the early exploration phase visually
2. Switch to **10× or Max** speed to let the agent train through episode 500+
3. Switch back to **1×** when reward starts improving to watch the converged policy in detail
4. Use **Pause** to freeze and read current numeric values carefully
