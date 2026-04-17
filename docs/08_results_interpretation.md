# 08 — Interpreting Training Results

## The Story of a Successful Training Run

### Stage 1: Exploration (Episodes 0–100)

**Reward:** ~ −1000 per episode  
**PU Constraint Satisfaction:** ~20–40%  
**Average Power:** ~0.5 W (essentially random)

The agent has no learned policy. Its neural network outputs are based on random initialisation, and the exploration noise adds additional randomness. Most actions transmit at moderate-to-high power, causing frequent PU SINR violations.

**Expected console log:**
```
Episode |  Steps |    Reward |    Avg100
      0 |    200 | -1168.130 | -1168.130
      5 |    200 |  -138.630 |  -895.199
     10 |    200 |  -108.137 |  -542.332
```

The high variance (some episodes near -1000, some near -100) is because channel realisations vary — some episodes draw channel conditions where even random power control happens to be safe.

---

### Stage 2: Early Learning (Episodes 100–500)

**Reward:** −500 to −100  
**PU Constraint Satisfaction:** 40–60%  
**Average Power:** 0.2–0.4 W (declining)

The critic networks start fitting the Q-function. The most important early discovery the agent makes: **transmitting with P_s = 0 always satisfies the PU constraint** (no SU interference). This is a local minimum — zero reward from throughput, but no penalty. You'll see episodes where the agent transmits very little and achieves near-zero reward (rather than the large negative values of Stage 1).

**What to look for:** Reward plot average line (yellow) starting to trend upward. Power bar in the GUI showing mostly low/medium fills.

---

### Stage 3: Mid Learning (Episodes 500–1500)

**Reward:** −100 to 0  
**PU Constraint Satisfaction:** 60–85%  
**Average Power:** 0.05–0.2 W

Now the agent starts discovering that **selective high power** is possible when channel conditions allow it. The policy differentiates between:
- High h_sp (bad): keep P_s low → protect PU
- Low h_sp + high h_ss (good): increase P_s → harvest SU throughput

The reward curve shows an upward trend but with high variance. The PU SINR plot shows fewer dips below the threshold line.

**Key metric to watch:** PU Protection Rate in the AI Insights panel. Should be crossing 70% by episode 1000.

---

### Stage 4: Convergence (Episodes 1500–3000)

**Reward:** 0 to +5  
**PU Constraint Satisfaction:** >90%  
**Average Power:** Channel-adaptive (0.01 – 0.5 W depending on h_sp)

The policy is now near-optimal. The agent has learned:
1. If h_sp is large (high interference risk) → P_s ≈ 0
2. If h_ss is large and h_sp is small → P_s ≈ 0.3–0.7 W
3. The exact balance between reward and penalty

**What convergence looks like:**
- PT→PR link is almost always green in the GUI
- SU throughput plot shows consistent non-zero values
- PU SINR plot stays mostly above the yellow threshold line
- Reward average (yellow line) has flattened at a positive value

---

## Reading the Plots

### Episode Reward Plot

```
Reward
   │
 0 ┼─────────────────────────────────────────────────── target
   │                                          ╭─────────
-50│                                    ╭─────╯
   │                              ╭─────╯
-100│                        ╭─────╯
   │                   ╭─────╯
-500│             ╭─────╯  ← avg100 (yellow)
   │       raw (gray, noisy)
-1000┼────────────────────────────────────────────────────► Episode
   0        500        1000       1500       2000       3000
```

**Key features:**
- Raw line (gray): highly noisy, shows single-episode performance
- Average line (yellow): smooth trend, key indicator of learning progress
- The "hockey stick" shape (flat negative → rapidly improving → stable) is the hallmark of successful TD3 convergence

### PU SINR Plot

```
SINR_p
     │     (good — constraint satisfied)
 10  │▓▓░▓░▓▓▓░░▓▓░░▓▓▓▓░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  2  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ threshold
  1  │░░░▓░░░░▓░░░░░░░░░░░░░░░░░░░░░░░░
  0  ├─────────────────────────────────── Step
                (early)        (late)
```
- **▓**: SINR above threshold (good)
- **░**: SINR below threshold (violation)
- Pattern should shift from mostly-░ to mostly-▓ over training

---

## Diagnostic Checklist

| Symptom | Likely Cause | Check |
|---------|-------------|-------|
| Reward stuck at ~ −0.01 | Agent learned to always transmit P_s = 0 | Look at power bar — is it always empty? Check avg power in Insights |
| Reward oscillating wildly | Learning rate too high or batch size too small | Check console for loss values |
| PU violation rate never drops below 50% | β penalty insufficient | Increase β in config.py |
| Reward never goes positive | SINR_s too low even at P_s = P_max | Check h_ps interference level; may need longer training |
| Training very slow | GPU not detected | Check PyTorch device — add `print(agent.device)` to main.py |

---

## Academic Takeaways

After a successful 3000-episode training run, the key results to report are:

1. **Convergence:** Reward increases from ≈ −1000 to ≈ +2–5 b/s/Hz (before penalties)
2. **Constraint satisfaction:** PU protection rate achieves >90% without explicit hard constraints — emergent from reward shaping
3. **Adaptive power control:** Agent dynamically adjusts P_s based on channel state — mimicking the theoretical optimal power control policy
4. **Sample efficiency:** Convergence in ~600,000 environment steps — achievable in <1 hour on CPU
5. **No domain knowledge required:** The agent discovers the interference management strategy purely from reward signals, demonstrating the power of model-free RL for cognitive radio

---

## TD3 vs DDPG Comparison (PDF Report)

Run `python train_compare.py` to generate the multi-page PDF report. Here is what each page tells you:

### Page 2 — SINR vs BER

This is the most academically significant plot. It shows:
- **Simulated scatter points** from all steps during training (colored by algorithm)
- **Theoretical BPSK BER curve** (AWGN, no fading) — the best possible lower bound
- **Average BER curve** over Nakagami-m=3 fading — the expected BER after averaging over channel statistics
- **Binned mean BER** per algorithm — shows which algorithm achieves lower BER at each SINR level

**What to look for:** TD3's binned mean should fall closer to the theoretical curve than DDPG's, indicating it achieves better power control and thus lower BER at a given SINR.

### Pages 3 & 4 — Throughput

- **SU Throughput** (Page 3): R_s = log₂(1 + SINR_s). TD3 should achieve higher SU throughput because its twin-critic design reduces Q-value overestimation and leads to a more efficient power policy.
- **PU Throughput** (Page 4): R_p = log₂(1 + SINR_p). Both algorithms must protect PU; a higher PU throughput indicates the SU is interfering less, which is desirable.

### Page 5 — Outage Probability

Outage = fraction of steps where SINR_s < threshold. As training progresses this should fall. The 5% reference line is a typical QoS target. TD3's lower-variance Q-learning should converge to a lower outage probability.

### Pages 6 & 7 — Reward Curves

Compare the learning speed and final reward. TD3 typically:
- Converges faster due to twin critics reducing overestimation
- Achieves higher final reward
- Has smoother convergence (less oscillation)

DDPG often:
- Converges faster in early episodes (no policy delay)
- Can overestimate Q-values, leading to instability in later training
- May oscillate around a suboptimal policy
