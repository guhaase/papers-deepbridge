#!/bin/bash
# Quick reference script for running Experiment 1B: Compression Ratios

echo "=================================================="
echo "Experimento 1B: Compression Ratios Maiores (CRÍTICO)"
echo "=================================================="
echo ""
echo "⚠️  EXPERIMENTO CRÍTICO PARA VALIDAR RQ1"
echo ""
echo "Tempo estimado:"
echo "  - Quick mode: ~2-3 horas"
echo "  - Full mode: ~8-10 horas"
echo ""
echo "=================================================="
echo ""

# Default values
MODE="quick"
DATASET="CIFAR10"
GPU=0
OUTPUT="../results/exp1b_$(date +%Y%m%d_%H%M%S)"
COMPRESSIONS="2.3x_ResNet18 5x_ResNet10 7x_MobileNetV2"

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
        --compressions)
            shift
            COMPRESSIONS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                COMPRESSIONS="$COMPRESSIONS $1"
                shift
            done
            ;;
        --help)
            echo "Usage: ./RUN_EXPERIMENT_1B.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode {quick,full}           Execution mode (default: quick)"
            echo "  --dataset {CIFAR10,CIFAR100}  Dataset to use (default: CIFAR10)"
            echo "  --gpu {0,1,...}               GPU device ID (default: 0)"
            echo "  --output PATH                 Output directory (default: auto-generated)"
            echo "  --compressions LIST           Compression configs (default: all)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Compression configs available:"
            echo "  - 2.3x_ResNet18     ResNet50 → ResNet18 (2.3× compression)"
            echo "  - 5x_ResNet10       ResNet50 → ResNet10 (5× compression)"
            echo "  - 7x_MobileNetV2    ResNet50 → MobileNetV2 (7× compression)"
            echo ""
            echo "Examples:"
            echo "  ./RUN_EXPERIMENT_1B.sh --mode quick --dataset CIFAR10"
            echo "  ./RUN_EXPERIMENT_1B.sh --mode full --dataset CIFAR10 --gpu 0"
            echo "  ./RUN_EXPERIMENT_1B.sh --mode full --dataset CIFAR100"
            echo "  ./RUN_EXPERIMENT_1B.sh --compressions 5x_ResNet10 7x_MobileNetV2"
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
echo "  Compressions: $COMPRESSIONS"
echo ""

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
    echo "This experiment is GPU-intensive. Expect very slow training on CPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check if PyTorch is installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ ERROR: PyTorch not installed!"
    echo "Install with: pip install torch torchvision"
    exit 1
fi

# Check if CUDA is available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ PyTorch with CUDA support detected"
else
    echo "⚠️  WARNING: PyTorch without CUDA. Training will be VERY slow!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Estimate time
if [ "$MODE" = "full" ]; then
    if [ "$DATASET" = "CIFAR100" ]; then
        ESTIMATED_TIME="12-15 hours"
    else
        ESTIMATED_TIME="8-10 hours"
    fi
else
    ESTIMATED_TIME="2-3 hours"
fi

echo ""
echo "⏱️  Estimated time: $ESTIMATED_TIME"
echo ""
read -p "Ready to start? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 0
fi

echo ""
echo "Starting Experiment 1B..."
echo "=================================================="
echo ""

# Build command
CMD="python3 01b_compression_ratios.py --mode $MODE --dataset $DATASET --gpu $GPU --output \"$OUTPUT\""

if [ -n "$COMPRESSIONS" ]; then
    CMD="$CMD --compressions $COMPRESSIONS"
fi

echo "Executing: $CMD"
echo ""

# Run experiment
eval $CMD

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment 1B completed successfully!"
    echo ""
    echo "📁 Results saved to: $OUTPUT"
    echo ""
    echo "📊 Key files:"
    echo "  - Report: $OUTPUT/experiment_report.md"
    echo "  - Data: $OUTPUT/results_compression_ratios.csv"
    echo "  - Stats: $OUTPUT/statistical_tests.csv"
    echo "  - Figures: $OUTPUT/figures/*.png"
    echo ""
    echo "Next steps:"
    echo "  1. Review report: cat $OUTPUT/experiment_report.md"
    echo "  2. Check statistics: cat $OUTPUT/statistical_tests.csv"
    echo "  3. View figures: open $OUTPUT/figures/*.png"
    echo ""
    echo "🎯 Key question answered:"
    echo "   Does HPM-KD beat Direct training with larger compression ratios?"
    echo "   Check 'hpmkd_vs_direct.png' to see the answer!"
else
    echo "❌ Experiment failed with exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Reduce batch size in the script"
    echo "  - CUDA error: Check GPU availability with 'nvidia-smi'"
    echo "  - Missing dependencies: pip install torch torchvision scipy"
    echo ""
    echo "Check logs for details."
fi
echo "=================================================="

exit $EXIT_CODE
