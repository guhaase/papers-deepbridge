#!/usr/bin/env python3
"""
HPM-KD Full MNIST Experiment
=============================

Run HPM-KD on complete MNIST dataset (70,000 samples).

Usage:
    cd /home/guhaase/projetos/DeepBridge
    python3 papers/01_HPM-KD_Framework/POR/run_full_mnist_experiment.py

Author: Gustavo Coelho Haase
Date: November 2025
"""

import sys
from pathlib import Path

# Add DeepBridge to path
deepbridge_path = Path(__file__).parent.parent.parent.parent.absolute()
if str(deepbridge_path) not in sys.path:
    sys.path.insert(0, str(deepbridge_path))

# Add current directory to path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import the experiment runner
import run_hpmkd_experiments
ExperimentConfig = run_hpmkd_experiments.ExperimentConfig
run_full_experiment_pipeline = run_hpmkd_experiments.run_full_experiment_pipeline
logger = run_hpmkd_experiments.logger


class FullMNISTConfig(ExperimentConfig):
    """Configuration for full MNIST experiment."""
    # Dataset settings
    USE_FULL_MNIST = True  # Use all 70k samples

    # Model settings - keep same as quick test
    TEACHER_ENSEMBLE_SIZE = 500
    TEACHER_DEPTH = 20
    STUDENT_DEPTH = 5

    # Experiment settings
    N_RANDOM_SEEDS = 1  # Set to 5 for paper results
    VALIDATION_SPLIT = 0.2

    # Output settings
    OUTPUT_DIR = Path(__file__).parent / "experiment_results_full"
    SAVE_MODELS = False
    VERBOSE = True


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("🚀 FULL MNIST EXPERIMENT (70,000 samples)")
    logger.info("="*80)
    logger.info("This will take significantly longer than the quick test (10k samples).")
    logger.info("Expected runtime: 5-15 minutes depending on hardware.")
    logger.info("="*80)

    config = FullMNISTConfig()

    try:
        results = run_full_experiment_pipeline(config)
        logger.info("\n🎉 Full MNIST experiment completed successfully!")
        logger.info(f"   Results saved to: {config.OUTPUT_DIR}")
        return 0
    except Exception as e:
        logger.error(f"\n❌ Full MNIST experiment failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
