# Running CRN TD3 vs DDPG on Kaggle

This guide shows you how to run the comparison training on a **Kaggle Notebook** (free GPU — P100 or T4). The comparison script trains both TD3 and DDPG on a Nakagami-m (m=3) Cognitive Radio Network and saves a PDF report with all plots.

---

## 1. Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. Under **Settings → Accelerator**, select **GPU T4 x2** or **GPU P100** (free tier).
3. Make sure **Internet** is turned **On** (needed for pip installs).

---

## 2. Upload Your Project Files

### Option A — Upload as a Dataset (recommended for large projects)

1. Zip your project: `crn_td3/` → `crn_td3.zip`
2. On Kaggle, go to **Datasets → New Dataset**, upload the zip, name it `crn-td3`.
3. In your notebook, add it via **Add Data → Your Datasets → crn-td3**.
4. Files will appear at `/kaggle/input/crn-td3/`.

### Option B — Paste directly into cells

Copy each `.py` file into individual notebook cells using `%%writefile filename.py`.

---

## 3. Notebook Setup

Paste the following cells into your Kaggle notebook:

### Cell 1 — Install dependencies

```python
!pip install torch numpy scipy matplotlib --quiet
```

> **Note:** PyTorch is already installed on Kaggle GPU images. This cell is a safety net.

### Cell 2 — Copy project files to working directory (if using Dataset method)

```python
import shutil, os

SRC = "/kaggle/input/crn-td3"
DST = "/kaggle/working"

files = [
    "config.py", "environment.py", "td3.py", "ddpg.py",
    "train_compare.py", "utils.py",
]

for f in files:
    src_path = os.path.join(SRC, f)
    dst_path = os.path.join(DST, f)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {f}")
    else:
        print(f"WARNING: {f} not found in dataset")

os.chdir(DST)
print("Working directory:", os.getcwd())
```

### Cell 3 — Verify GPU is available

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### Cell 4 — Run the comparison training

```python
# 500 episodes takes ~8–15 min on GPU, ~30 min on CPU
# Increase --episodes for better convergence (1000–2000 recommended for final results)

!python train_compare.py \
    --episodes 500 \
    --steps-per-ep 200 \
    --output /kaggle/working/results/crn_comparison_report.pdf \
    --seed 42
```

**Expected console output:**
```
================================================================
  CRN TD3 vs DDPG — Comparison Training & Report Generator
  Nakagami-m = 3.0  |  Episodes = 500  |  Steps/ep = 200
================================================================

[TD3] Starting training   episodes=500  steps/ep=200  Nakagami-m=3.0
  Device: cuda
  ep   50/500  reward=...  avg100=...  outage=...
  ep  100/500  ...
  ...
[TD3] Done in Xs

[DDPG] Starting training   ...
  ...

  PDF report saved: /kaggle/working/results/crn_comparison_report.pdf
```

### Cell 5 — Display the PDF inline (optional)

```python
from IPython.display import IFrame
IFrame("/kaggle/working/results/crn_comparison_report.pdf", width=900, height=600)
```

### Cell 6 — Download the PDF

```python
from IPython.display import FileLink
FileLink("results/crn_comparison_report.pdf")
```

---

## 4. Tuning for Kaggle GPU

| Parameter | Quick test | Good results | Best results |
|-----------|-----------|--------------|--------------|
| `--episodes` | 200 | 500 | 2000 |
| `--steps-per-ep` | 100 | 200 | 200 |
| Expected time (GPU) | ~3 min | ~15 min | ~60 min |
| Expected time (CPU) | ~10 min | ~45 min | ~3 hr |

**Tip:** Use `--episodes 1000` for publication-quality curves. The reward trend stabilises after ~300–500 episodes on GPU.

---

## 5. What the PDF Contains

| Page | Plot |
|------|------|
| 1 | Summary table — reward, throughput, outage, BER for TD3 vs DDPG |
| 2 | **SINR vs BER** — scatter (simulated) + theoretical BPSK curve (Nakagami-m=3) |
| 3 | **Secondary User Throughput** (bits/s/Hz) vs episodes |
| 4 | **Primary User Throughput** (bits/s/Hz) vs episodes |
| 5 | **Outage Probability** vs episodes (with 5% target line) |
| 6 | Individual reward curves for TD3 and DDPG |
| 7 | Overlaid reward comparison |

---

## 6. Increasing Episode Count Without Modifying Files

You can override from the command line:

```bash
python train_compare.py --episodes 2000 --output results/report_2000ep.pdf
```

To only run TD3 (skip DDPG) for a smoke test:

```bash
python train_compare.py --episodes 100 --no-ddpg --output results/td3_only.pdf
```

---

## 7. Saving Results as Kaggle Output

Anything written to `/kaggle/working/` is automatically available as notebook output and can be downloaded. The `results/` folder containing the PDF will be saved there.

To make it a permanent dataset output:

1. After running, click **Save Version** → **Save & Run All (Commit)**.
2. After commit completes, go to the notebook's **Output** tab.
3. Click on `results/crn_comparison_report.pdf` to download.

---

## 8. Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: config` | Make sure all `.py` files are in `/kaggle/working/` and you ran `os.chdir(DST)` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `config.py` from 128 to 64 |
| `No module named matplotlib` | Run `!pip install matplotlib` in a cell above |
| PDF is empty / only 1 page | Training finished too fast — increase `--episodes` to at least 100 |
| Very slow on CPU | Enable GPU in Settings → Accelerator |

---

## 9. Two-Cell Paste-and-Run (with inline plots)

This version imports the training modules directly so that metric objects remain
in the notebook's memory. Cell 2 then plots **BER**, **SU Throughput**, and
**Outage Probability** inline without needing to open the PDF.

### Cell A — Install, train, and generate PDF

```python
import subprocess, sys, os

# Install deps
subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "scipy", "--quiet"], check=True)

# Set working dir (adjust if using dataset)
os.chdir("/kaggle/working")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # Agg required for PDF generation inside train_compare

from train_compare import run_algorithm, generate_pdf, smooth
from config        import NAKAGAMI_M, SINR_THRESHOLD
from td3           import TD3Agent
from ddpg          import DDPGAgent

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EPISODES, STEPS = 500, 200

# Train — metric objects stay in scope for Cell B
td3_m  = run_algorithm("TD3",  TD3Agent(),  EPISODES, STEPS)
ddpg_m = run_algorithm("DDPG", DDPGAgent(), EPISODES, STEPS)

# Save full PDF report as well
generate_pdf(td3_m, ddpg_m, "results/crn_comparison_report.pdf", EPISODES, STEPS)
print("Training complete — PDF saved to results/crn_comparison_report.pdf")
```

### Cell B — Inline BER / Throughput / Outage plots

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

TD3_COLOR  = "#1f77b4"
DDPG_COLOR = "#d62728"
episodes   = np.arange(1, EPISODES + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── BER vs Episodes ──────────────────────────────────────────────────────────
ax = axes[0]
for m, c in [(td3_m, TD3_COLOR), (ddpg_m, DDPG_COLOR)]:
    raw = np.array(m.avg_bers)
    sm  = smooth(raw, window=20)
    ax.semilogy(episodes, raw, color=c, alpha=0.2, lw=0.7)
    ax.semilogy(episodes, sm,  color=c, lw=2.0, label=m.name)
ax.set_xlabel("Episode")
ax.set_ylabel("Average BER")
ax.set_title("BER vs Episodes")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# ── SU Throughput vs Episodes ────────────────────────────────────────────────
ax = axes[1]
for m, c in [(td3_m, TD3_COLOR), (ddpg_m, DDPG_COLOR)]:
    raw = np.array(m.su_throughputs)
    sm  = smooth(raw, window=20)
    ax.plot(episodes, raw, color=c, alpha=0.2, lw=0.7)
    ax.plot(episodes, sm,  color=c, lw=2.0, label=m.name)
ax.set_xlabel("Episode")
ax.set_ylabel("Throughput (bits/s/Hz)")
ax.set_title("SU Throughput vs Episodes")
ax.legend()
ax.grid(True, alpha=0.3)

# ── Outage Probability vs Episodes ───────────────────────────────────────────
ax = axes[2]
for m, c in [(td3_m, TD3_COLOR), (ddpg_m, DDPG_COLOR)]:
    raw = np.array(m.outage_probs)
    sm  = smooth(raw, window=20)
    ax.plot(episodes, raw, color=c, alpha=0.2, lw=0.7)
    ax.plot(episodes, sm,  color=c, lw=2.0, label=m.name)
ax.axhline(0.05, color="gray", ls="--", lw=1.2, label="5% target")
ax.set_xlabel("Episode")
ax.set_ylabel("Outage Probability")
ax.set_title("Outage Probability vs Episodes")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("CRN TD3 vs DDPG — Inline Performance Summary", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

> **Note:** `%matplotlib inline` in Cell B switches the backend from Agg (used for
> PDF rendering) back to the Kaggle inline renderer. The `td3_m`, `ddpg_m`,
> `EPISODES`, and `smooth` names must all be in scope from Cell A — run Cell A
> first before Cell B.

---

*Generated for: Cognitive Radio Network TD3/DDPG Mini-Project*
*Ramaiah Institute of Technology, Bangalore*
