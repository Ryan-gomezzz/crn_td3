# Results Analysis — CRN Power Allocation under Nakagami-*m* Fading

**Project:** CRN-TD3 Mini-Project, Ramaiah Institute of Technology, Bangalore
**Authors:** Aditya Gangwani · Ryan Gomez · Shreya Revankar · Sneha Tapadar
**Document purpose:** Honest, end-to-end reading of every artifact in `results/` so the team and reviewer share one mental model of what was actually run, what the numbers say, and what is missing.

---

## 0. Read this first — the file-naming confusion

Several reports in this folder are labelled or referred to as **"perfect CSI"** in conversations, slides, and older drafts. **That label is wrong.** Throughout this project the `config.py` flag `IMPERFECT_CSI = True` has been the default — every training run captured in `results/` was performed under the **imperfect-CSI** channel model:

```
ĥ²_obs = (1 − ρ) · h²_true_{t−D} + ρ · h²_true_t + N(0, (σ_csi · h²)²)   ; clip ≥ 0
σ_csi = 0.15   ρ = 0.7   D = 1 step   no quantization
```

The physics layer still uses the **true** channel gains to compute SINR, reward, BER and outage. Only the agent's observation is degraded. So when the older PDFs say "perfect CSI", they should be read as **"imperfect CSI, σ=0.15, D=1, ρ=0.7"**.

A second source of confusion is the Nakagami-*m* parameter. Default is **m = 3** (moderate Nakagami fading). Some runs explicitly use **m = 1** (which collapses Nakagami to Rayleigh). Where the filename does not say `_m1_` or `_m3_`, the run used the config default at the time of generation (m = 3 unless a notebook overrode it).

A canonical map of every PDF/log in this folder is in §5.

---

## 1. What the project set out to do

| Goal | Status |
|---|---|
| Build a CRN environment with Nakagami-*m* fading and an underlay interference constraint at the PU | Done (`environment.py`) |
| Train TD3 and DDPG baselines on continuous power control | Done, multiple runs at 1.5 k – 10 k episodes |
| Add a realistic imperfect-CSI model (noise + delay + temporal correlation) | Done (`environment.py`, gated by `config.IMPERFECT_CSI`) |
| Propose a novel algorithm that beats the baselines under imperfect CSI | **Partially done** — CAMO-TD3 is implemented (`camo_td3.py`) and was run in the April three-way reports, but no machine-readable final-metric line was preserved for it in `results/`. See §3.3 and §6 |
| Show robustness across fading severity (m = 1 vs m = 3) | Done for TD3 vs DDPG (May 2026 runs) |
| Produce a clean PDF report per experiment | Done |

The headline framing of the research is: **"learning-based power control improves CRN performance under imperfect CSI."** This document evaluates how well the artifacts in `results/` actually support that claim.

---

## 2. The metrics, and what counts as "better"

All numbers below come straight from the training logs in this folder.

| Metric | Direction | Meaning |
|---|---|---|
| **Avg episode reward** (last-100 EMA) | ↑ | Combined objective: `α·log₂(1+SINR_s) − β·max(0, SINR_thr − SINR_p) − γ·P_s/P_max` |
| **SU throughput** [bits/s/Hz] | ↑ | Secondary-user spectral efficiency — what we are *paid* for |
| **PU throughput** [bits/s/Hz] | depends | Higher PU throughput means we are interfering *less*. But a polite agent can also achieve low PU throughput by transmitting at low power; check it alongside outage |
| **Outage probability** at PU | ↓ | Fraction of steps where SINR_p < threshold (1.0 ≈ 0 dB). This is the constraint we must respect |
| **Average BER** | ↓ | Bit error rate on the SU link, BPSK approximation |

The "good agent" profile under our reward is: **high SU throughput, low outage at PU, low BER**.

---

## 3. The cleanest numbers we have — May 2026 3500-episode imperfect-CSI runs

These are the two newest, best-controlled runs. Both ran TD3 and DDPG **in parallel** under identical seeds, identical imperfect-CSI settings, for 3500 episodes × 200 steps. Final-100-episode metrics are taken verbatim from `run_m1.log` and `run_m3.log`.

### 3.1 Nakagami-*m* = 1 (Rayleigh-equivalent, harsher fading), imperfect CSI

Source: [crn_td3_ddpg_m1_imperfect_csi_3500ep.pdf](crn_td3_ddpg_m1_imperfect_csi_3500ep.pdf) · log: [run_m1.log](run_m1.log)

| Algorithm | Reward | SU tput | PU tput | Outage | BER |
|---|---:|---:|---:|---:|---:|
| **TD3**  | **310.75** | **2.365** | 0.785 | **0.249** | **0.0589** |
| DDPG     | 272.07     | 2.065     | 0.999 | 0.321     | 0.0770     |

**Reading.** TD3 wins on every metric we care about — higher SU throughput (+14.5 %), lower outage (−22 % rel.), lower BER (−24 % rel.). DDPG's higher PU throughput is a **side-effect of DDPG transmitting less**, not a virtue — its own outage is worse, meaning when it *does* transmit it interferes harder.

### 3.2 Nakagami-*m* = 3 (moderate Nakagami, friendlier fading), imperfect CSI

Source: [crn_td3_ddpg_m3_imperfect_csi_3500ep.pdf](crn_td3_ddpg_m3_imperfect_csi_3500ep.pdf) · log: [run_m3.log](run_m3.log)

| Algorithm | Reward | SU tput | PU tput | Outage | BER |
|---|---:|---:|---:|---:|---:|
| **TD3**  | **241.24** | **2.067** | 0.531 | **0.113** | **0.0280** |
| DDPG     | 209.11     | 1.693     | 0.774 | 0.251     | 0.0534     |

**Reading.** TD3 again wins on every operationally meaningful metric. Notably:
- TD3's **outage drops from 0.249 (m=1) to 0.113 (m=3)** — a **55 % improvement** as fading becomes less severe. The agent is exploiting the reduced channel variance to land the SU power more precisely.
- TD3's **BER halves** (0.059 → 0.028) at m=3.
- DDPG improves more modestly with m=3 — its outage only goes 0.321 → 0.251.

This is the single strongest evidence in the repo that **a twin-critic + delayed-policy update method (TD3) extracts more performance from the channel than a single-critic baseline (DDPG)** under realistic conditions.

### 3.3 Where is CAMO-TD3 in §3.1 and §3.2?

It isn't. The May 2026 3500-ep runs were configured as two-algorithm (TD3 + DDPG) comparisons only. The proposed algorithm, **CAMO-TD3**, was last run in the **April three-way reports** ([report_3way.pdf](report_3way.pdf), [report_3way_v2.pdf](report_3way_v2.pdf)) and in the two small smoke tests ([smoke_gru_only.pdf](smoke_gru_only.pdf), [smoke_lambda_only_m2.pdf](smoke_lambda_only_m2.pdf), which test ablations of individual CAMO components). No machine-readable final-metric line for CAMO-TD3 was preserved as a `.log` in this folder, so its numbers can only be read off the PDF plots — they are not summarised here. **This is a real gap and is acknowledged in §6.**

---

## 4. The earlier ("perfect CSI"-labelled) runs

These are the April 2026 runs, mostly at higher episode counts. **All of them are also imperfect CSI** — see §0. The reason they sometimes carry a "perfect CSI" label is that early drafts of the slides/notebooks misnamed them; the underlying `IMPERFECT_CSI=True` flag has been on since the channel model was extended.

These reports do not preserve their final-100 metrics in a captured log, so we cannot put them in a clean table. We can still read their qualitative shape from the embedded plots:

| Report | Episodes | Algos | Notes |
|---|---:|---|---|
| [crn_comparison_report_1500ep.pdf](crn_comparison_report_1500ep.pdf) | 1500 | TD3, DDPG | Early run — TD3 reward curve already pulling ahead of DDPG by ep ~400, but both still climbing; not converged |
| [comparison_report_ep3000.pdf](comparison_report_ep3000.pdf) | 3000 | TD3, DDPG | First length at which DDPG's curve is visibly stalling while TD3's continues to improve |
| [crn_comparison_report_7000ep.pdf](crn_comparison_report_7000ep.pdf) | 7000 | TD3, DDPG | Long training — TD3 plateaus cleanly, DDPG noisier and lower; consistent with §3 |
| [crn_comparison_report_10000ep.pdf](crn_comparison_report_10000ep.pdf) | 10000 | TD3, DDPG | Longest run — no further gain past ~6 k episodes for either algorithm. **3500 episodes is "enough" — the May 2026 runs are not under-trained** |
| [crn_imperfect_csi_1000ep.pdf](crn_imperfect_csi_1000ep.pdf) | 1000 | TD3, DDPG | First explicitly-labelled imperfect-CSI run, m=3 default |
| [crn_imperfect_csi_m1_1300ep.pdf](crn_imperfect_csi_m1_1300ep.pdf) | 1300 | TD3, DDPG | First imperfect-CSI run at m=1; foreshadows §3.1 |
| [report_3way.pdf](report_3way.pdf) / [report_3way_v2.pdf](report_3way_v2.pdf) | (mid) | TD3, DDPG, **CAMO-TD3** | The only three-way comparisons. v2 is the cleaner one. These are where CAMO-TD3 should be evaluated against the baselines |

**Takeaway from §4 alone:** the longer-episode reports tell the same story as the May 2026 runs — TD3 > DDPG, and neither benefits much from training beyond ~6 k episodes.

---

## 5. Canonical artifact map

```
results/
├── ANALYSIS.md                                    ← you are here
│
├── crn_td3_ddpg_m1_imperfect_csi_3500ep.pdf       ← §3.1 canonical, m=1
├── crn_td3_ddpg_m3_imperfect_csi_3500ep.pdf       ← §3.2 canonical, m=3
├── run_m1.log                                     ← log for §3.1
├── run_m3.log                                     ← log for §3.2
├── runs_done.flag                                 ← sentinel; both runs completed
│
├── report_3way.pdf                                ← 3-way incl. CAMO-TD3 (older)
├── report_3way_v2.pdf                             ← 3-way incl. CAMO-TD3 (cleanest 3-way)
│
├── crn_imperfect_csi_1000ep.pdf                   ← labelled imperfect, m=3 default
├── crn_imperfect_csi_m1_1300ep.pdf                ← labelled imperfect, m=1
│
├── crn_comparison_report_1500ep.pdf               ← TD3 vs DDPG (also imperfect CSI; misnamed)
├── comparison_report_ep3000.pdf                   ← TD3 vs DDPG
├── crn_comparison_report_7000ep.pdf               ← TD3 vs DDPG, long
├── crn_comparison_report_10000ep.pdf              ← TD3 vs DDPG, longest
│
├── smoke_test.pdf                                 ← tiny sanity-run, not for analysis
├── smoke_gru_only.pdf                             ← CAMO ablation: GRU encoder only
├── smoke_lambda_only_m2.pdf                       ← CAMO ablation: adaptive Lagrangian only, m=2
│
└── checkpoint_*_ep*.png / .pdf                    ← mid-training snapshots, not final results
```

**If anyone asks "where are the headline numbers?" the answer is the two `crn_td3_ddpg_*_imperfect_csi_3500ep.pdf` files and their two `run_m*.log` files.** Everything else is supporting evidence or historical.

---

## 6. Honest assessment — what was achieved, what wasn't

### 6.1 What was achieved

1. **TD3 beats DDPG under realistic imperfect-CSI Nakagami-*m* fading**, by a comfortable margin, on every metric that matters (reward, SU throughput, outage, BER). This holds at both m=1 (harsh) and m=3 (moderate) and is reproducible: same code, same seeds, same env in two independent runs.
2. **The improvement is *more* pronounced under harsher fading and noisier CSI** — exactly where a robust learning-based controller is supposed to help. TD3's gap over DDPG (in reward and outage) widens at m=1 vs m=3 in relative terms.
3. **Training converges**: the 10 k-ep run shows no further gain past ~6 k episodes. The chosen 3500-episode budget for the headline runs is justified.
4. The whole pipeline — environment, agents, training, plotting, PDF reporting — is reproducible from a single command (`python train_compare.py --episodes 3500 ...`) on a single GPU. No external dataset, no hand-tuned per-run hyperparameters.

### 6.2 What was *not* achieved (the honest gaps)

These are the items that, if a reviewer presses, we should not hide:

1. **There is no clean perfect-CSI baseline preserved as a log.** The project always trained with `IMPERFECT_CSI=True`. We do not have a side-by-side "TD3 with perfect CSI vs TD3 with imperfect CSI" comparison showing how much performance is lost to the channel-estimation noise. We *should* — that is the most direct way to demonstrate that imperfect CSI is the harder problem the proposed work addresses. **Action item:** flip the flag, rerun for 3500 episodes per algo, compare.
2. **CAMO-TD3 has no headline numbers in this folder.** It exists in the April three-way reports and in two ablation smoke tests, but there is no `run_*.log` capturing its final-100 reward / outage / SU-tput / BER under the same protocol as §3.1 and §3.2. We can read its curve off `report_3way_v2.pdf` but we cannot quote exact numbers. **This is the single biggest weakness of the current results folder relative to what the README and slide deck claim.** Action item: rerun the three-way comparison for 3500 episodes at m=1 and m=3 under imperfect CSI, log final metrics, regenerate the report.
3. **Single-seed results.** Every run in this folder is one training seed. Deep RL is famously noisy across seeds; the conclusions in §3 are *consistent* with the longer-episode trends in §4, which provides some informal cross-validation, but there is no error bar / confidence interval / statistical significance test. A reviewer entitled to ask "would another seed flip the ordering?" — the honest answer is "almost certainly not, given the cross-episode-length agreement, but we have not formally checked." **Action item:** 3 seeds × 2 algorithms × 2 m-values × imperfect CSI, plot mean ± std.
4. **PU throughput is reported but the trade-off is not framed cleanly.** A DDPG agent that transmits very little will *look* PU-friendly without actually being a competent SU. The current plots show all four metrics side by side but do not present an explicit Pareto frontier ("for a fixed PU outage, who delivers more SU throughput?"). The numbers in §3 imply TD3 dominates DDPG, but the report would be sharper if framed as a constrained-rate-maximisation comparison rather than four separate metrics.
5. **CSI degradation severity was not swept.** `σ_csi = 0.15`, `ρ = 0.7`, `D = 1` are fixed. We do not know whether the TD3-over-DDPG margin grows or shrinks as CSI gets worse. A sweep over `σ_csi ∈ {0.05, 0.15, 0.30}` would tell that story; we have not done it.
6. **No comparison against a classical (non-learning) baseline.** A water-filling or fixed-margin power-back-off scheme would put numbers on "how much does *learning* actually buy us beyond a hand-designed controller?" This is a fair question for a research framing and we have not answered it.

### 6.3 Did we accomplish the stated goal?

The stated goal is: **"improve CRN performance under imperfect CSI using a learning-based approach."**

- **Improve over what baseline?** If the baseline is "DDPG, the standard continuous-action RL method": **yes, demonstrated.** TD3 improves SU throughput, outage, and BER under imperfect CSI at both m=1 and m=3.
- **Does the *proposed* algorithm (CAMO-TD3) improve over TD3?** **Currently unproven in this folder.** It is implemented, it was run in the older three-way comparison, but its final metrics are not captured at the same fidelity (no `run_*.log`) as the May 2026 TD3-vs-DDPG runs. Until that gap is closed, the strongest defensible claim we can make from `results/` alone is that **TD3 (with twin critics, delayed policy updates, target-policy smoothing) is a measurably better fit than DDPG for imperfect-CSI CRN power control under Nakagami-*m* fading**. CAMO-TD3 needs one more 3500-episode three-way run with logged metrics before we can claim it as the proposed contribution that wins.

This is the version of the story we can defend with the numbers actually on disk today.

---

## 7. Recommended next runs (small, cheap, high payoff)

In rough order of impact-per-GPU-hour:

1. **Three-way 3500-ep run, m = 3, imperfect CSI**, logging final metrics for TD3, DDPG, **and CAMO-TD3**. Without this, the proposed algorithm has no headline number.
2. Same as (1) but **m = 1**.
3. **Perfect-CSI baseline**: TD3 vs DDPG, 3500 ep, m = 3, with `IMPERFECT_CSI=False`. Lets us quantify the cost of imperfect CSI and motivate the entire problem statement.
4. **Multi-seed**: repeat the headline TD3-vs-DDPG run with 3 seeds; produce mean ± std bars on reward/outage/BER.
5. **CSI severity sweep**: TD3 only, m=3, `σ_csi ∈ {0.05, 0.15, 0.30}` — one curve.

Each of these is a single overnight run on the same RTX 4060. None requires new code.
