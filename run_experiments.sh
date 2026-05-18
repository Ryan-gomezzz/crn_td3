#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Full experiment matrix for CAMO-TD3 mini-project
#
# Equivalent to run_experiments.ps1 — for Linux / Colab / Kaggle.
#
# Usage:
#   ./run_experiments.sh                                # everything, 1500 ep
#   ./run_experiments.sh --episodes 1500
#   ./run_experiments.sh --group ablation
#   ./run_experiments.sh --group nakagami
#   ./run_experiments.sh --group headline
# =============================================================================
set -euo pipefail

EPISODES=1500
STEPS_PER_EP=200
DEVICE="cuda"
GROUP="all"
SEED=42
RESULTS_DIR="results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)     EPISODES="$2"; shift 2;;
        --steps-per-ep) STEPS_PER_EP="$2"; shift 2;;
        --device)       DEVICE="$2"; shift 2;;
        --group)        GROUP="$2"; shift 2;;
        --seed)         SEED="$2"; shift 2;;
        --results-dir)  RESULTS_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Pre-flight CUDA check
if [[ "$DEVICE" == "cuda" ]]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "ERROR: --device cuda requested but torch.cuda.is_available() is False"
        echo "       Install CUDA torch:"
        echo "       pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
        exit 2
    fi
fi

mkdir -p "$RESULTS_DIR"

START_TS=$(date +%s)
echo
echo "================================================================="
echo "  CRN Mini-Project — Experiment Orchestrator"
echo "  Episodes/run   : $EPISODES"
echo "  Steps/episode  : $STEPS_PER_EP"
echo "  Device         : $DEVICE"
echo "  Group          : $GROUP"
echo "  Results        : $RESULTS_DIR/"
echo "================================================================="
echo

invoke_run() {
    local tag="$1"; shift
    local output="$RESULTS_DIR/$tag.pdf"
    echo
    echo "------ [$tag] ----------------------------------------------------"
    echo "  Output: $output"
    local t0=$(date +%s)
    python train_compare.py \
        --episodes     "$EPISODES" \
        --steps-per-ep "$STEPS_PER_EP" \
        --device       "$DEVICE" \
        --seed         "$SEED" \
        --output       "$output" \
        "$@"
    local t1=$(date +%s)
    echo "  Done [$tag] in $(( (t1 - t0) / 60 )) min"
}

run_ablation() {
    echo
    echo ">>> GROUP 1: Ablation study (m=3, imperfect CSI)"

    invoke_run ablation_01_baseline_td3 \
        --agents td3 --nakagami-m 3

    invoke_run ablation_02_multi_obj_only \
        --agents td3,camo-td3 --camo-variant multi-obj-only --nakagami-m 3 --parallel

    invoke_run ablation_03_lambda_only \
        --agents td3,camo-td3 --camo-variant lambda-only --nakagami-m 3 --parallel

    invoke_run ablation_04_gru_only \
        --agents td3,camo-td3 --camo-variant gru-only --nakagami-m 3 --parallel

    invoke_run ablation_05_directional_only \
        --agents td3,camo-td3 --camo-variant directional-only --nakagami-m 3 --parallel

    invoke_run ablation_06_full_camo \
        --agents td3,camo-td3 --camo-variant full --nakagami-m 3 --parallel
}

run_nakagami() {
    echo
    echo ">>> GROUP 2: Nakagami-m sweep (full trio, imperfect CSI)"
    for m in 1 2 3; do
        invoke_run "nakagami_m${m}" \
            --agents td3,ddpg,camo-td3 --camo-variant full --nakagami-m "$m" --parallel
    done
}

run_headline() {
    echo
    echo ">>> GROUP 3: Headline run (m=3, all three algos, full CAMO-TD3)"
    invoke_run headline_imperfect_csi \
        --agents td3,ddpg,camo-td3 --camo-variant full --nakagami-m 3 --parallel
}

case "$GROUP" in
    ablation) run_ablation;;
    nakagami) run_nakagami;;
    headline) run_headline;;
    all)
        run_headline
        run_ablation
        run_nakagami
        ;;
    *)
        echo "Unknown --group '$GROUP'. Use: all | ablation | nakagami | headline"
        exit 1
        ;;
esac

END_TS=$(date +%s)
echo
echo "================================================================="
echo "  All experiments complete in $(( (END_TS - START_TS) / 60 )) min"
echo "  PDFs are in: $RESULTS_DIR/"
echo "================================================================="
