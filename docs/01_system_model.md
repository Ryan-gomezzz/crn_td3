# 01 — System Model: The Cognitive Radio Network

## Overview

This project simulates a **Cognitive Radio Network (CRN)** — a wireless communication system where an intelligent secondary user (SU) learns to share spectrum with a licensed primary user (PU) without causing harmful interference.

---

## The Four-Node Topology

The network consists of exactly four nodes, forming two transceiver pairs:

```
                    h_pp (desired PU link)
   ┌────────┐ ══════════════════════════════════ ┌────────┐
   │   PT   │                                    │   PR   │
   │Primary │                                    │Primary │
   │  Tx    │                                    │  Rx    │
   └────────┘                                    └────────┘
       │  ╲                                      ▲   ▲
       │   ╲  h_ps (PT→SR interference)         │   │
       │    ╲                                   │   │
  h_sp│     ╲                                  │   │
 (ST→PR      ╲                                │   │ h_sp
 interference) ╲                              │   │(ST→PR)
       │         ╲                            │   │
       ▼           ╲                          │   │
   ┌────────┐        ╲──────────────────── ┌────────┐
   │   ST   │ ══════════════════════════>  │   SR   │
   │Secondary│      h_ss (desired SU link) │Secondary│
   │   Tx   │                              │   Rx   │
   └────────┘                              └────────┘
```

### Node Descriptions

| Node | Full Name | Role | Power |
|------|-----------|------|-------|
| **PT** | Primary Transmitter | Licensed PU transmitter; always active | Fixed P_p = 1.0 W |
| **PR** | Primary Receiver | Licensed PU receiver; must be protected | — |
| **ST** | Secondary Transmitter | Cognitive/unlicensed transmitter | Variable P_s ∈ [0, 1] W (agent's action) |
| **SR** | Secondary Receiver | Cognitive/unlicensed receiver; wants max throughput | — |

---

## Channel Links

There are four distinct wireless links, each modeled with an independent fading channel:

| Link | Symbol | Type | Physical Meaning |
|------|--------|------|-----------------|
| PT → PR | h_pp | **Desired** | The primary user's own communication link |
| ST → SR | h_ss | **Desired** | The secondary user's own communication link |
| ST → PR | h_sp | **Interference** | ST's signal leaks into PR — this is the **key interference source** |
| PT → SR | h_ps | **Interference** | PT's signal leaks into SR — degrades SU reception |

### Why This Matters

- The **SU wants to transmit** to maximize its own throughput (R_s)
- But ST's transmission **bleeds interference into PR**
- If ST transmits too powerfully → PR's SINR drops below the threshold → **PU is harmed**
- The TD3 agent must find the sweet spot: enough power for good SU throughput, but not so much that PU suffers

---

## SINR Calculations

**SINR at Primary Receiver (PR):**

$$\text{SINR}_p = \frac{P_p \cdot |h_{pp}|^2}{P_s \cdot |h_{sp}|^2 + \sigma^2}$$

- Numerator: desired signal power from PT
- Denominator: interference from ST + noise floor
- As ST's power P_s increases → SINR_p decreases → PU is degraded

**SINR at Secondary Receiver (SR):**

$$\text{SINR}_s = \frac{P_s \cdot |h_{ss}|^2}{P_p \cdot |h_{ps}|^2 + \sigma^2}$$

- Numerator: desired signal power from ST
- Denominator: interference from PT + noise floor
- As ST's power P_s increases → SINR_s increases → better SU throughput

**Shannon Capacity (SU Throughput):**

$$R_s = \log_2(1 + \text{SINR}_s) \quad \text{[bits/s/Hz]}$$

---

## System Parameters

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Noise power | σ² | 1×10⁻³ W | Thermal noise floor at receiver |
| PT transmit power | P_p | 1.0 W | Fixed licensed power level |
| ST max power | P_max | 1.0 W | Regulatory power cap on SU |
| PU SINR threshold | γ_th | 2.0 (~3 dB) | Minimum QoS requirement for PU |

---

## Key Insight

This is a **spectrum underlay** CRN — the SU transmits in the same frequency band as the PU simultaneously. The challenge is that the SU must:

1. **Sense the channel** (observe h_pp, h_sp, h_ss, h_ps)
2. **Decide transmit power P_s** based on current channel conditions
3. **Guarantee SINR_p ≥ γ_th** at all times (protect PU)
4. **Maximise R_s** subject to the above constraint

This is exactly what the TD3 agent learns to do through repeated interaction with the environment.
