# Running CRN TD3 vs DDPG on Google Colab

This guide shows you how to run the comparison training on **Google Colab** (free GPU — T4). The comparison script trains both TD3 and DDPG on a Nakagami-m (m=3) Cognitive Radio Network and generates all required plots.

---

## 1. Create a New Google Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com) and click **New Notebook**.
2. Under **Runtime → Change runtime type**, select **GPU** as the hardware accelerator.
3. Make sure **Connect** is showing (top right corner).

---

## 2. Upload Your Project Files

### Option A — Upload from Google Drive (recommended)

1. Zip your project: `crn_td3/` → `crn_td3.zip`
2. Upload the zip to your Google Drive (e.g., in "MyDrive").
3. In Colab, mount your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Option B — Upload directly to Colab

1. Click the **Files** icon (left sidebar) → **Upload**.
2. Upload your `crn_td3.zip`.
3. Extract it:

```python
import zipfile, os
zip_path = '/content/crn_td3.zip'  # adjust if different
extract_path = '/content/crn_td3'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
os.chdir(extract_path)
print("Working directory:", os.getcwd())
```

---

## 3. Single-Cell Version (Copy & Paste)

Copy the entire code block below into a single Colab cell and run it. This will:
- Install dependencies
- Extract your project files
- Verify GPU
- Run training (TD3 + DDPG)
- Generate all plots (BER vs SINR, Throughput, Outage)
- Save the PDF report

```python
# ============================================================================
# CRN TD3 vs DDPG — Single-Cell Colab Runner
# ============================================================================

# @title 🔽 Run Training & Generate Report
# @markdown Edit the parameters below if needed:
episodes = 3000  # @param {type:"integer"}
steps_per_ep = 200  # @param {type:"integer"}
output_pdf = "results/crn_comparison_report.pdf"  # @param {type:"string"}
seed = 42  # @param {type:"integer"}

import subprocess, sys, os, zipfile
from google.colab import drive, files

# --- 1. Mount Google Drive ---
print("=" * 60)
print("Step 1: Mounting Google Drive...")
print("=" * 60)
try:
    drive.mount('/content/drive')
    print("Drive mounted successfully.")
except Exception as e:
    print(f"Drive mount error: {e}")

# --- 2. Extract project files ---
print("\n" + "=" * 60)
print("Step 2: Extracting project files...")
print("=" * 60)

# Adjust this path to where your zip is in Drive
project_zip = '/content/drive/MyDrive/crn_td3.zip'  # <-- CHANGE THIS if needed
extract_dir = '/content/crn_td3'

if os.path.exists(project_zip):
    with zipfile.ZipFile(project_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to: {extract_dir}")
else:
    # Try alternative locations
    alt_paths = [
        '/content/crn_td3.zip',
        '/content/drive/My Drive/crn_td3.zip',
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            project_zip = alt
            with zipfile.ZipFile(alt, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted from: {alt}")
            break
    else:
        print("WARNING: Could not find zip file. Trying current directory...")

os.chdir(extract_dir)
print("Working directory:", os.getcwd())

# --- 3. Install dependencies ---
print("\n" + "=" * 60)
print("Step 3: Installing dependencies...")
print("=" * 60)
subprocess.run([sys.executable, "-m", "pip", "install", "torch", "numpy", "scipy", "matplotlib", "--quiet"], check=True)
print("Dependencies installed.")

# --- 4. Verify GPU ---
print("\n" + "=" * 60)
print("Step 4: Verifying GPU...")
print("=" * 60)
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: Running on CPU. Training will be slow.")

# --- 5. Create results directory ---
os.makedirs("results", exist_ok=True)

# --- 6. Run training ---
print("\n" + "=" * 60)
print("Step 5: Running TD3 + DDPG Training...")
print("=" * 60)
print(f"Episodes: {episodes}, Steps/Episode: {steps_per_ep}")

result = subprocess.run(
    [sys.executable, "train_compare.py",
     "--episodes", str(episodes),
     "--steps-per-ep", str(steps_per_ep),
     "--output", output_pdf,
     "--seed", str(seed)],
    capture_output=False,
)

print(f"\nTraining completed. Return code: {result.returncode}")

# --- 7. Download the PDF ---
print("\n" + "=" * 60)
print("Step 6: Downloading PDF report...")
print("=" * 60)
if os.path.exists(output_pdf):
    files.download(output_pdf)
    print(f"Downloaded: {output_pdf}")
else:
    print("ERROR: PDF not found!")
    # Try alternative paths
    alt_pdf = "crn_comparison_report.pdf"
    if os.path.exists(alt_pdf):
        files.download(alt_pdf)
        print(f"Downloaded: {alt_pdf}")

print("\n" + "=" * 60)
print("✅ All done! Check the PDF for plots.")
print("=" * 60)
```

---

## 4. What the PDF Contains

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

## 5. Tuning for Colab GPU

| Parameter | Quick test | Good results | Best results |
|-----------|-----------|--------------|--------------|
| `--episodes` | 500 | 3000 | 5000 |
| `--steps-per-ep` | 100 | 200 | 200 |
| Expected time (GPU) | ~5 min | ~25 min | ~45 min |
| Expected time (CPU) | ~20 min | ~90 min | ~3 hr |

**Tip:** Use `--episodes 3000` for good convergence. The reward trend stabilises after ~1000–2000 episodes on GPU.

---

## 6. Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: config` | Make sure all `.py` files are extracted and you ran `os.chdir(extract_dir)` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `config.py` from 128 to 64 |
| `No module named matplotlib` | Run `!pip install matplotlib` in a cell above |
| PDF is empty / only 1 page | Training finished too fast — increase `--episodes` to at least 500 |
| Very slow on CPU | Change runtime type to GPU in **Runtime → Change runtime type** |
| Zip file not found | Check the `project_zip` path in the code — adjust to match your Drive folder |

---

## 7. Quick Reference — Colab GPU Check

Run this to verify GPU is active:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Expected output:
```
CUDA available: True
Device: Tesla T4
```

---

*Generated for: Cognitive Radio Network TD3/DDPG Mini-Project*
*Ramaiah Institute of Technology, Bangalore*