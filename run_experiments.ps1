# =============================================================================
# run_experiments.ps1 — Full experiment matrix for CAMO-TD3 mini-project
#
# Three experiment groups (run independently or together):
#
#   1. Ablation study (CAMO-TD3 components individually)
#      - Baseline TD3 (no modifications)
#      - +Multi-objective critics only
#      - +Adaptive Lagrangian only (with multi-obj critics, since lambda needs them)
#      - +GRU belief encoder only
#      - +Directional noise only
#      - Full CAMO-TD3 (all four)
#
#   2. Nakagami-m sensitivity (m = 1, 2, 3) for the trio TD3/DDPG/CAMO-TD3
#
#   3. Combined headline run (m=3, all three algos, full CAMO-TD3)
#
# Usage:
#   .\run_experiments.ps1                       # everything
#   .\run_experiments.ps1 -Episodes 1500
#   .\run_experiments.ps1 -Group ablation       # just ablation
#   .\run_experiments.ps1 -Group nakagami       # just m-sweep
#   .\run_experiments.ps1 -Group headline
#   .\run_experiments.ps1 -Group ablation -Episodes 800   # quick smoke test
#
# Requirements:
#   - Activated venv with cuda-torch
#   - IMPERFECT_CSI=True in config.py (already set)
# =============================================================================

param(
    [int]    $Episodes   = 1500,
    [int]    $StepsPerEp = 200,
    [string] $Device     = "cuda",
    [string] $Group      = "all",      # all | ablation | nakagami | headline
    [int]    $Seed       = 42,
    [string] $ResultsDir = "results"
)

$ErrorActionPreference = "Stop"

# Pre-flight: check CUDA
if ($Device -eq "cuda") {
    $cudaCheck = python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'NO_CUDA')"
    if ($cudaCheck -ne "CUDA") {
        Write-Host "ERROR: --device cuda requested but torch.cuda.is_available() is False" -ForegroundColor Red
        Write-Host "       Install CUDA torch:  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
        exit 2
    }
}

if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

$startTime = Get-Date
Write-Host ""
Write-Host "================================================================="
Write-Host "  CRN Mini-Project — Experiment Orchestrator"
Write-Host "  Episodes/run   : $Episodes"
Write-Host "  Steps/episode  : $StepsPerEp"
Write-Host "  Device         : $Device"
Write-Host "  Group          : $Group"
Write-Host "  Results        : $ResultsDir/"
Write-Host "  Started        : $startTime"
Write-Host "================================================================="
Write-Host ""

# -----------------------------------------------------------------------------
# Helper: invoke train_compare.py with consistent args
# -----------------------------------------------------------------------------
function Invoke-Run {
    param(
        [string]   $Tag,
        [string[]] $ExtraArgs
    )
    $output = Join-Path $ResultsDir "$Tag.pdf"
    Write-Host ""
    Write-Host "------ [$Tag] ----------------------------------------------------"
    Write-Host "  Output: $output"

    $args = @(
        "train_compare.py",
        "--episodes",     $Episodes,
        "--steps-per-ep", $StepsPerEp,
        "--device",       $Device,
        "--seed",         $Seed,
        "--output",       $output
    ) + $ExtraArgs

    $t0 = Get-Date
    & python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: run [$Tag] failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $elapsed = (Get-Date) - $t0
    Write-Host ("  Done [$Tag] in {0:N1} min" -f $elapsed.TotalMinutes) -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# Group 1 — Ablation study (CAMO-TD3 components individually)
#
# We pair each CAMO-TD3 variant with TD3 (baseline) so each PDF directly shows
# the marginal value of that one component. Same Nakagami-m=3 throughout.
# -----------------------------------------------------------------------------
function Run-Ablation {
    Write-Host ""
    Write-Host ">>> GROUP 1: Ablation study (m=3, imperfect CSI)" -ForegroundColor Cyan

    # 1a. Baseline TD3 alone (control)
    Invoke-Run -Tag "ablation_01_baseline_td3" -ExtraArgs @(
        "--agents", "td3",
        "--nakagami-m", "3"
    )

    # 1b. TD3 vs CAMO-TD3 with ONLY multi-objective critics
    Invoke-Run -Tag "ablation_02_multi_obj_only" -ExtraArgs @(
        "--agents", "td3,camo-td3",
        "--camo-variant", "multi-obj-only",
        "--nakagami-m", "3",
        "--parallel"
    )

    # 1c. TD3 vs CAMO-TD3 with multi-obj + adaptive lambda
    Invoke-Run -Tag "ablation_03_lambda_only" -ExtraArgs @(
        "--agents", "td3,camo-td3",
        "--camo-variant", "lambda-only",
        "--nakagami-m", "3",
        "--parallel"
    )

    # 1d. TD3 vs CAMO-TD3 with ONLY GRU belief encoder
    Invoke-Run -Tag "ablation_04_gru_only" -ExtraArgs @(
        "--agents", "td3,camo-td3",
        "--camo-variant", "gru-only",
        "--nakagami-m", "3",
        "--parallel"
    )

    # 1e. TD3 vs CAMO-TD3 with ONLY directional noise
    Invoke-Run -Tag "ablation_05_directional_only" -ExtraArgs @(
        "--agents", "td3,camo-td3",
        "--camo-variant", "directional-only",
        "--nakagami-m", "3",
        "--parallel"
    )

    # 1f. Full CAMO-TD3 (all four) vs TD3 — reference
    Invoke-Run -Tag "ablation_06_full_camo" -ExtraArgs @(
        "--agents", "td3,camo-td3",
        "--camo-variant", "full",
        "--nakagami-m", "3",
        "--parallel"
    )
}

# -----------------------------------------------------------------------------
# Group 2 — Nakagami-m sensitivity sweep
# -----------------------------------------------------------------------------
function Run-NakagamiSweep {
    Write-Host ""
    Write-Host ">>> GROUP 2: Nakagami-m sweep (full trio, imperfect CSI)" -ForegroundColor Cyan

    foreach ($m in @(1, 2, 3)) {
        Invoke-Run -Tag "nakagami_m$m" -ExtraArgs @(
            "--agents", "td3,ddpg,camo-td3",
            "--camo-variant", "full",
            "--nakagami-m", $m,
            "--parallel"
        )
    }
}

# -----------------------------------------------------------------------------
# Group 3 — Headline (default) run
# -----------------------------------------------------------------------------
function Run-Headline {
    Write-Host ""
    Write-Host ">>> GROUP 3: Headline run (m=3, all three algos, full CAMO-TD3)" -ForegroundColor Cyan

    Invoke-Run -Tag "headline_imperfect_csi" -ExtraArgs @(
        "--agents", "td3,ddpg,camo-td3",
        "--camo-variant", "full",
        "--nakagami-m", "3",
        "--parallel"
    )
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
switch ($Group.ToLower()) {
    "ablation" { Run-Ablation }
    "nakagami" { Run-NakagamiSweep }
    "headline" { Run-Headline }
    "all"      {
        Run-Headline
        Run-Ablation
        Run-NakagamiSweep
    }
    default    {
        Write-Host "Unknown -Group '$Group'. Use: all | ablation | nakagami | headline" -ForegroundColor Red
        exit 1
    }
}

$totalElapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "================================================================="
Write-Host ("  All experiments complete in {0:N1} min" -f $totalElapsed.TotalMinutes) -ForegroundColor Green
Write-Host "  PDFs are in: $ResultsDir/"
Write-Host "================================================================="
