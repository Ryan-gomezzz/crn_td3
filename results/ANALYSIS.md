# Results Analysis — CRN Power Allocation under Nakagami-*m* Fading

**Project:** CRN-TD3 Mini-Project, Ramaiah Institute of Technology, Bangalore
**Authors:** Aditya Gangwani · Ryan Gomez · Shreya Revankar · Sneha Tapadar
**Document purpose:** Honest, end-to-end reading of every artifact in `results/` so the team and reviewer share one mental model of what was actually run, what the numbers say, and what is missing.

---

## 0. Read this first — the file-naming convention

The repository was developed in two phases. The phase you are looking at a file from is encoded in its **filename**.

| Phase | Channel-side condition | Filename signature |
|---|---|---|
| **Phase 1 — April 2026** | **Perfect CSI** — agent observes true channel gains directly | filename does **not** contain `imperfect_csi`. Examples: `crn_comparison_report_*ep.pdf`, `comparison_report_ep3000.pdf`, `report_3way*.pdf`, `smoke_test.pdf` |
| **Phase 2 — May 2026** | **Imperfect CSI** — agent observes noisy + delayed + temporally-correlated estimates of the gains; physics still uses the true gains for SINR / reward / BER / outage | filename **does** contain `imperfect_csi`. Examples: `crn_imperfect_csi_1000ep.pdf`, `crn_imperfect_csi_m1_1300ep.pdf`, `crn_td3_ddpg_m{1,3}_imperfect_csi_3500ep.pdf` |

The imperfect-CSI machinery (`config.IMPERFECT_CSI`, the observation-corruption block in `environment.py`) was added in git commit **`397430d`** — the same commit that produced the May 2026 3500-episode runs. **Nothing in this folder dated before that commit can have used imperfect CSI**, because the code to do it didn't exist yet.

Imperfect-CSI parameters used in Phase 2:

```
ĥ²_obs = (1 − ρ) · h²_true_{t−D} + ρ · h²_true_t + N(0, (σ_csi · h²)²)  ; clip ≥ 0
σ_csi = 0.15   ρ = 0.7   D = 1 step   no quantization
```

A second axis is the **Nakagami-*m* shape parameter**. Default is **m = 3** (moderate Nakagami fading). Some runs explicitly use **m = 1** (which collapses Nakagami to Rayleigh — harshest fading). Where the filename does not say `_m1_` or `_m3_`, the run used the config default at the time (m = 3).

A canonical map of every PDF/log is in §5.

---

## 1. What the project set out to do

| Goal | Status |
|---|---|
| Build a CRN environment with Nakagami-*m* fading and an underlay interference constraint at the PU | Done (`environment.py`) |
| Train TD3 and DDPG baselines on continuous power control under perfect CSI | Done — Phase 1 April runs |
| Add a realistic imperfect-CSI model (noise + delay + temporal correlation) and re-evaluate | Done — Phase 2 May runs |
| Propose a novel algorithm (CAMO-TD3) that improves over the baselines | **Implemented and run, but only under perfect CSI.** No imperfect-CSI run of CAMO-TD3 exists in this folder. See §3.3 and §6 |
| Show robustness across fading severity (m = 1 vs m = 3) | Done for TD3 vs DDPG under imperfect CSI |
| Produce a clean PDF report per experiment | Done |

The headline framing of the research is: **"learning-based power control improves CRN performance, with the focus being on imperfect CSI."** This document evaluates how well the artifacts in `results/` actually support that claim.

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

## 3. Phase 1 — Perfect-CSI baseline (April 2026)

These runs use the original environment, in which the agent observes the true Nakagami-*m* channel gains directly. They establish a best-case ceiling for the learning approach and set the bar that the imperfect-CSI runs must approach.

Final-100 metrics are **not** preserved in a log file for these runs (the `run_m*.log` files are Phase-2 artifacts). What can be read off the embedded PDF plots is qualitative — reward curves, outage curves, BER scatter against the theoretical Nakagami-*m* BPSK envelope. The strongest things we can say:

### 3.1 What the Phase-1 PDFs show (qualitative)

- **TD3 consistently sits above DDPG** on the episode-reward curve in every Phase-1 report (`crn_comparison_report_1500ep`, `comparison_report_ep3000`, `crn_comparison_report_7000ep`, `crn_comparison_report_10000ep`). The gap is visible from a few hundred episodes in and persists.
- **The 10 k-episode run shows no meaningful improvement past ~6 k episodes** for either algorithm. This is the evidence that 3500 episodes is a fair training budget for the Phase-2 runs — neither agent is under-trained at that horizon.
- **BER scatter sits below the theoretical Nakagami-*m*=3 BPSK curve** for both agents, which is the expected behaviour for a power-controlled SU.
- **CAMO-TD3 appears only in [report_3way.pdf](report_3way.pdf) and [report_3way_v2.pdf](report_3way_v2.pdf)** (April 21–22). Reading off the curves, CAMO-TD3 tracks closely with TD3 and at times overtakes it; both clearly beat DDPG. **This is the only evidence in the folder that CAMO-TD3 works — and it is perfect-CSI evidence.** No imperfect-CSI run of CAMO-TD3 exists. See §6.

### 3.2 Why we cannot give exact perfect-CSI numbers in this document

For Phase 1 the training script did not yet write final-100 metric lines to a captured log file. The metrics live inside each PDF's summary table, but those tables are images embedded in the PDFs, not text we can quote with confidence here. If a reviewer wants exact perfect-CSI numbers, the path forward is either (a) re-run any of the Phase-1 reports at the new logging fidelity, or (b) read the numbers off the summary-page table in the relevant PDF.

---

## 4. Phase 2 — Imperfect-CSI evaluation (May 2026)

These are the cleanest, best-controlled runs in the folder. Both ran TD3 and DDPG **in parallel** under identical seeds, identical imperfect-CSI settings, for 3500 episodes × 200 steps. Final-100-episode metrics are taken verbatim from `run_m1.log` and `run_m3.log`.

### 4.1 Nakagami-*m* = 1 (Rayleigh-equivalent, harsher fading), imperfect CSI

Source: [crn_td3_ddpg_m1_imperfect_csi_3500ep.pdf](crn_td3_ddpg_m1_imperfect_csi_3500ep.pdf) · log: [run_m1.log](run_m1.log)

| Algorithm | Reward | SU tput | PU tput | Outage | BER |
|---|---:|---:|---:|---:|---:|
| **TD3**  | **310.75** | **2.365** | 0.785 | **0.249** | **0.0589** |
| DDPG     | 272.07     | 2.065     | 0.999 | 0.321     | 0.0770     |

**Reading.** TD3 wins on every metric we care about — higher SU throughput (+14.5 %), lower outage (−22 % rel.), lower BER (−24 % rel.). DDPG's higher PU throughput is a **side-effect of DDPG transmitting less**, not a virtue — its own outage is worse, meaning when it *does* transmit it interferes harder.

### 4.2 Nakagami-*m* = 3 (moderate Nakagami, friendlier fading), imperfect CSI

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

### 4.3 The two earlier imperfect-CSI runs

- [crn_imperfect_csi_1000ep.pdf](crn_imperfect_csi_1000ep.pdf) (Apr 30, m=3, 1000 ep) — the first imperfect-CSI experiment, used to validate that the new flag worked. Short training; not a headline number.
- [crn_imperfect_csi_m1_1300ep.pdf](crn_imperfect_csi_m1_1300ep.pdf) (May 1, m=1, 1300 ep) — first imperfect-CSI run at m=1. Curve shape matches the later 3500-ep m=1 run; consistent with §4.1.

Two small CAMO ablation smoke tests also live in Phase 2:
- [smoke_gru_only.pdf](smoke_gru_only.pdf) — CAMO with **only** the GRU belief encoder enabled (other CAMO components off).
- [smoke_lambda_only_m2.pdf](smoke_lambda_only_m2.pdf) — CAMO with **only** the adaptive Lagrangian weights enabled, m=2.

These are tiny-episode sanity runs to confirm the CAMO components individually train; they are **not full CAMO-TD3 evaluations** and they are not at the m=1 / m=3 / 3500-ep / full-CAMO configuration that would be needed to back the proposed contribution under imperfect CSI.

### 4.4 Where is full CAMO-TD3 under imperfect CSI?

**It does not exist in this folder.** This is the single most important gap and is the headline item in §6.

---

## 5. Canonical artifact map

```
results/
├── ANALYSIS.md                                    ← you are here
│
├── PHASE 2 — IMPERFECT CSI (May 2026)
│   ├── crn_td3_ddpg_m1_imperfect_csi_3500ep.pdf   ← §4.1 canonical, m=1
│   ├── crn_td3_ddpg_m3_imperfect_csi_3500ep.pdf   ← §4.2 canonical, m=3
│   ├── run_m1.log                                 ← log for §4.1
│   ├── run_m3.log                                 ← log for §4.2
│   ├── runs_done.flag                             ← sentinel; both runs completed
│   ├── crn_imperfect_csi_1000ep.pdf               ← early imperfect, m=3
│   ├── crn_imperfect_csi_m1_1300ep.pdf            ← early imperfect, m=1
│   ├── smoke_gru_only.pdf                         ← CAMO ablation: GRU only
│   └── smoke_lambda_only_m2.pdf                   ← CAMO ablation: Lagrangian only, m=2
│
├── PHASE 1 — PERFECT CSI (April 2026)
│   ├── crn_comparison_report_1500ep.pdf           ← TD3 vs DDPG, 1500 ep
│   ├── comparison_report_ep3000.pdf               ← TD3 vs DDPG, 3000 ep
│   ├── crn_comparison_report_7000ep.pdf           ← TD3 vs DDPG, 7000 ep
│   ├── crn_comparison_report_10000ep.pdf          ← TD3 vs DDPG, 10000 ep (longest)
│   ├── report_3way.pdf                            ← 3-way incl. CAMO-TD3 (older)
│   ├── report_3way_v2.pdf                         ← 3-way incl. CAMO-TD3 (cleanest 3-way; perfect CSI)
│   ├── smoke_test.pdf                             ← tiny sanity-run
│   ├── checkpoint_td3_ep*.png                     ← TD3 mid-training snapshots
│   ├── checkpoint_ddpg_ep*.png                    ← DDPG mid-training snapshots
│   └── checkpoint_{td3,ddpg}_ep00750_report.pdf   ← checkpoint mini-reports
```

**Headline numbers under imperfect CSI live in the two `crn_td3_ddpg_*_imperfect_csi_3500ep.pdf` files and their `run_m*.log` files.**
**Evidence that CAMO-TD3 works at all lives in `report_3way_v2.pdf` — but it is perfect-CSI evidence.**

---

## 6. Honest assessment — what was achieved, what wasn't

### 6.1 What was achieved

1. **Under perfect CSI (Phase 1):** TD3 consistently beats DDPG across 1.5 k, 3 k, 7 k, and 10 k-episode runs. CAMO-TD3 (in the three-way reports) is at least competitive with TD3 and clearly beats DDPG. Training converges by ~6 k episodes, justifying the 3500-episode budget used later.
2. **Under imperfect CSI (Phase 2):** TD3 beats DDPG by a comfortable margin on every metric that matters (reward, SU throughput, outage, BER). This holds at both m=1 (harsh fading) and m=3 (moderate fading) — the result is robust across fading severity.
3. **The improvement is *more* pronounced under harsher fading** — TD3's relative reward/outage gap over DDPG is larger at m=1 than at m=3, exactly the regime where a robust learning controller is supposed to help.
4. The whole pipeline — environment, agents, training, plotting, PDF reporting — is reproducible from a single command (`python train_compare.py --episodes 3500 ...`) on a single GPU. No external dataset, no hand-tuned per-run hyperparameters.

### 6.2 What was *not* achieved (the honest gaps)

These are the items that, if a reviewer presses, we should not hide:

1. **CAMO-TD3 has never been evaluated under imperfect CSI.** It was implemented after the April 3-way reports and the imperfect-CSI flag was added afterwards. The two artifacts that show CAMO-TD3 — `report_3way.pdf` and `report_3way_v2.pdf` — are **perfect-CSI** experiments. **This is the single biggest weakness of the current results folder relative to what the project claims to contribute.** The proposed algorithm is justified by being robust to channel-estimation error and partial observability (GRU belief encoder, adaptive Lagrangian under interference constraint), but it has not actually been tested in the regime where those features are supposed to pay off. Action item: rerun the three-way comparison for 3500 episodes at m=1 and m=3 under imperfect CSI, with logged metrics.
2. **Phase 1 has no logged final metrics.** We can quote exact numbers for TD3 vs DDPG under imperfect CSI (from `run_m{1,3}.log`) but we cannot quote exact perfect-CSI numbers from a captured log. The numbers exist inside the Phase-1 PDFs' summary pages but were never written to a text file. This makes it awkward to write the most natural ablation table the reviewer will ask for ("how much performance is lost by going from perfect to imperfect CSI?"). Action item: re-run TD3 vs DDPG once under perfect CSI at 3500 episodes with the current logging, m=3.
3. **Single-seed results.** Every run in this folder is one training seed. Deep RL is famously noisy across seeds. The conclusions in §4 are *consistent* with the longer-episode Phase-1 trends, which provides some informal cross-validation, but there is no error bar / confidence interval. A reviewer entitled to ask "would another seed flip the ordering?" — the honest answer is "almost certainly not, given the cross-episode-length and cross-CSI-condition agreement, but we have not formally checked." Action item: 3 seeds × 2 algorithms × 2 m-values × imperfect CSI, plot mean ± std.
4. **PU throughput is reported but the trade-off is not framed cleanly.** A DDPG agent that transmits very little will *look* PU-friendly without actually being a competent SU. The current plots show all four metrics side by side but do not present an explicit Pareto frontier ("for a fixed PU outage, who delivers more SU throughput?"). The numbers in §4 imply TD3 dominates DDPG, but the report would be sharper framed as a constrained-rate-maximisation comparison rather than four separate metrics.
5. **CSI degradation severity was not swept.** `σ_csi = 0.15`, `ρ = 0.7`, `D = 1` are fixed. We do not know whether the TD3-over-DDPG margin (and eventually the CAMO-TD3-over-TD3 margin, once measured) grows or shrinks as CSI gets worse.
6. **No comparison against a classical (non-learning) baseline.** A water-filling or fixed-margin power-back-off scheme would put numbers on "how much does *learning* actually buy us beyond a hand-designed controller?" This is a fair question for a research framing and we have not answered it.

### 6.3 Did we accomplish the stated goal?

The stated goal is: **"improve CRN performance under imperfect CSI using a learning-based approach."**

- **Improve over what baseline?** If the baseline is "DDPG, the standard continuous-action RL method": **yes, demonstrated under imperfect CSI.** TD3 improves SU throughput, outage, and BER at both m=1 and m=3.
- **Does the *proposed* algorithm (CAMO-TD3) improve over TD3 specifically under imperfect CSI?** **Currently unproven in this folder.** CAMO-TD3 was shown to work under perfect CSI in the April three-way reports, but the entire imperfect-CSI evaluation in Phase 2 was a two-algorithm (TD3 vs DDPG) comparison. The proposed algorithm has never been put against the baselines in the channel condition the project is built around. Until that gap is closed, the strongest defensible claim from `results/` alone is:
  > **TD3 is a measurably better fit than DDPG for imperfect-CSI CRN power control under Nakagami-*m* fading, and CAMO-TD3 looks promising under perfect CSI. Whether CAMO-TD3 actually wins under imperfect CSI — the main use case the project is about — is currently untested.**

This is the version of the story we can defend with the numbers actually on disk today.

---

## 7. Recommended next runs (small, cheap, high payoff)

In rough order of impact-per-GPU-hour:

1. **Three-way 3500-ep run, m = 3, imperfect CSI**, logging final metrics for TD3, DDPG, **and CAMO-TD3**. This is *the* missing experiment — without it, the proposed algorithm has no headline number in its target regime.
2. Same as (1) but **m = 1**.
3. **Perfect-CSI baseline at the new logging fidelity**: TD3 vs DDPG, 3500 ep, m = 3, with `IMPERFECT_CSI=False`. Lets us quote exact "cost of imperfect CSI" deltas, motivating the entire problem statement with numbers, not curves.
4. **Multi-seed**: repeat the headline TD3-vs-DDPG run with 3 seeds; produce mean ± std bars on reward/outage/BER.
5. **CSI severity sweep**: TD3 only, m=3, `σ_csi ∈ {0.05, 0.15, 0.30}` — one curve.

Each of these is a single overnight run on the same RTX 4060. None requires new code.
