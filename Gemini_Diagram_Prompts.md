# Gemini Image Prompts for the II Review PPT

There are **three diagram placeholders** in the deck. For each, this file gives:
- The exact slide location and target box size
- Recommended Gemini aspect ratio (Gemini / Imagen generates at fixed aspect ratios — choose the closest one, then crop/scale to fit)
- The full prompt you can paste into Gemini

Style choices are kept consistent across the three diagrams so the deck reads as one document:
- Flat, vector / line-art look (no 3D, no glossy gradients, no photo-realism)
- White background, transparent if your tool supports it
- Two-colour palette: **navy `#1A365D`** for main strokes / labels, **teal `#0B7A75`** as the accent for the "secondary / learner" elements
- Thin (2–3 px) strokes, rounded line caps, sans-serif labels (Calibri / Helvetica-like)
- No emojis, no decorative shadows

---

## Diagram 1 — Reinforcement Learning Agent–Environment Loop
**Slide:** "Introduction — Reinforcement Learning" (slide 4)
**Box on slide:** 3.0 in wide × 2.6 in tall (right column)
**Render at:** 900 × 780 px (≈ 300 DPI for that box)
**Gemini aspect ratio:** **1:1** (closest match — crop a 900×780 region after generation)

### Prompt

```
Create a clean, flat vector-style diagram of the classic Reinforcement Learning loop.

Layout: two rounded rectangles side by side connected by curved arrows that form a closed loop.

Left rectangle: labeled "AGENT" in bold sans-serif. Fill #0B7A75 (teal), white text. Inside the rectangle add small subtext "Policy π(s)".

Right rectangle: labeled "ENVIRONMENT" in bold sans-serif. Fill #1A365D (navy), white text. Inside add small subtext "CRN".

A curved arrow from AGENT to ENVIRONMENT, labeled "Action  a_t" along the arrow (navy text).

A curved arrow from ENVIRONMENT back to AGENT, labeled "State  s_(t+1)  ·  Reward  r_t" (teal text).

White background, thin 2 px strokes for arrows, rounded arrowheads, generous whitespace, no shadows, no 3D, no gradients, no emojis. Sans-serif font (Calibri / Helvetica). High resolution, crisp lines, vector look.
```

---

## Diagram 2 — 4-Node Underlay CRN Topology
**Slide:** "System Model — Topology, Channels, Reward" (slide 8)
**Box on slide:** 3.3 in wide × 3.0 in tall (right column)
**Render at:** 990 × 900 px
**Gemini aspect ratio:** **1:1**

### Prompt

```
Create a clean, flat vector-style network-topology diagram of a 4-node underlay Cognitive Radio Network.

Place four nodes at the corners of a square:
- Top-left:  "PT" (Primary Transmitter)  filled in navy #1A365D
- Top-right: "PR" (Primary Receiver)     outlined in navy
- Bottom-left:  "ST" (Secondary Transmitter)  filled in teal #0B7A75
- Bottom-right: "SR" (Secondary Receiver)     outlined in teal

Draw four directed arrows between the nodes, each labeled with its channel gain:
- Solid navy arrow from PT to PR, labeled "|h_pp|²" (primary direct link)
- Solid teal arrow from ST to SR, labeled "|h_ss|²" (secondary direct link)
- Dashed gray arrow from ST to PR, labeled "|h_sp|²  (SU → PU interference)"
- Dashed gray arrow from PT to SR, labeled "|h_ps|²  (PU → SU interference)"

Show small radio-tower icons next to PT and ST (simple line art), and small antenna-receiver icons next to PR and SR.

White background, thin 2 px strokes, rounded arrowheads, generous whitespace, no shadows, no 3D, no gradients, no emojis. Sans-serif labels (Calibri / Helvetica). Vector look, high resolution.
```

---

## Diagram 3 — Proposed Algorithm End-to-End Architecture
**Slide:** "Proposed Algorithm — Architecture & Training Loop" (slide 16)
**Box on slide:** 4.6 in wide × 5.7 in tall (left column, tall vertical)
**Render at:** 1380 × 1710 px
**Gemini aspect ratio:** **3:4 (portrait)** — this is the closest standard ratio; crop to the exact box afterwards

### Prompt

```
Create a clean, flat vector-style block-diagram of a deep reinforcement learning architecture, laid out top-to-bottom as a tall portrait flow chart.

From top to bottom, place these labeled rounded rectangles, each connected by a downward arrow:

1. "CRN Environment"  — navy #1A365D fill, white text. Subtext below: "produces (s_t, r_t^tput, r_t^intf, r_t^energy)".

2. "Sequence Replay Buffer"  — light grey fill, navy text. Subtext: "stores last 8 observations per transition".

3. "GRU Belief Encoder  g_ψ"  — teal #0B7A75 fill, white text. Subtext: "2-layer GRU,  belief b_t ∈ R^16".

4. Two rectangles side-by-side connected by a horizontal arrow:
   - LEFT: "Actor  π_φ(s, b)"  — teal outline, navy text. Output arrow labeled "P_s + directional noise ε_t".
   - RIGHT: "6 Critics  Q^k(s, b, a)"  — navy outline, navy text. Subtext: "twin critics × {tput, intf, energy}".

5. "Adaptive Lagrangian  (λ₁, λ₂, λ₃)"  — gold #C88B00 fill, white text. Subtext: "dual gradient ascent,  clip[0.1, 20]".
   Show a circular feedback arrow looping back up from this block to the actor/critic pair, labeled "actor loss = −E[λ₁Q^t + λ₂Q^i + λ₃Q^e]".

6. Arrow from the actor block leading back to the top "CRN Environment" block, labeled "action a_t = P_s".

White background, thin 2 px strokes for arrows, rounded arrowheads, generous whitespace, no shadows, no 3D, no gradients, no emojis. Sans-serif labels (Calibri / Helvetica). Portrait orientation, vector look, high resolution.
```

---

## How to Drop the Images into the Deck

1. Generate each image in Gemini / Imagen at the recommended aspect ratio.
2. Crop / scale to the target pixel size above (any image editor — even PowerPoint's crop will work).
3. On the relevant slide, click the grey "[Diagram placeholder]" box → press Delete → Insert → Picture → place at the same position. The slide grid is already sized to match, so no resizing is needed if you respect the listed pixel sizes.

If a generated image looks too crowded, re-run the prompt with the line **"simplify, use fewer labels, more whitespace"** appended.
