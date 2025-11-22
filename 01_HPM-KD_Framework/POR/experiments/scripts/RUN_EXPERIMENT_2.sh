#!/bin/bash
# Quick reference script for running Experiment 2: Ablation Studies

echo "=================================================="
echo "Experimento 2: Ablation Studies (RQ2)"
echo "=================================================="
echo ""
echo "Tempo estimado:"
echo "  - Quick mode: ~1 hora"
echo "  - Full mode: ~2 horas"
echo ""
echo "=================================================="
echo ""

# Default values
MODE="quick"
DATASET="MNIST"
GPU=0
OUTPUT="../results/exp02_ablation_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./RUN_EXPERIMENT_2.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode {quick,full}      Execution mode (default: quick)"
            echo "  --dataset {MNIST,CIFAR10,CIFAR100}  Dataset to use (default: MNIST)"
            echo "  --gpu {0,1,...}          GPU device ID (default: 0)"
            echo "  --output PATH            Output directory (default: auto-generated)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./RUN_EXPERIMENT_2.sh --mode quick --dataset MNIST"
            echo "  ./RUN_EXPERIMENT_2.sh --mode full --dataset CIFAR10 --gpu 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Dataset: $DATASET"
echo "  GPU: $GPU"
echo "  Output: $OUTPUT"
echo ""

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if PyTorch is installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ ERROR: PyTorch not installed!"
    echo "Install with: pip install torch torchvision"
    exit 1
fi

# Check if DeepBridge is installed
if ! python3 -c "import deepbridge" 2>/dev/null; then
    echo "⚠️  WARNING: DeepBridge not installed (optional for CNN experiments)"
fi

echo ""
echo "Starting Experiment 2..."
echo "=================================================="
echo ""

# Run experiment
python3 02_ablation_studies.py \
    --mode "$MODE" \
    --dataset "$DATASET" \
    --gpu "$GPU" \
    --output "$OUTPUT"

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
    echo "📁 Results saved to: $OUTPUT"
    echo ""
    echo "Next steps:"
    echo "  1. Check results: ls -lh $OUTPUT"
    echo "  2. View report: cat $OUTPUT/experiment_report.md"
    echo "  3. Analyze figures: open $OUTPUT/figures/*.png"
else
    echo "❌ Experiment failed with exit code: $EXIT_CODE"
    echo "Check logs for details."
fi
echo "=================================================="

exit $EXIT_CODE
