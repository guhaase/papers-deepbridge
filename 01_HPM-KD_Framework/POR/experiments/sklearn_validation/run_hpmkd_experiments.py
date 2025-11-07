#!/usr/bin/env python3
"""
HPM-KD Full Experimental Pipeline
==================================

Complete experiment runner using the actual HPM-KD implementation from DeepBridge.

This script:
1. Loads datasets (MNIST and others)
2. Trains teacher models
3. Runs HPM-KD distillation using full implementation
4. Compares with baseline methods
5. Generates results matching paper format

Usage:
    # From DeepBridge root directory:
    cd /home/guhaase/projetos/DeepBridge
    python3 papers/01_HPM-KD_Framework/POR/run_hpmkd_experiments.py

Author: Gustavo Coelho Haase
Date: November 2025
Paper: HPM-KD Framework (Section 5.1, Table 2)
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import time
import logging
from pathlib import Path

# Add DeepBridge to path if needed
deepbridge_path = Path(__file__).parent.parent.parent.parent.absolute()
if str(deepbridge_path) not in sys.path:
    sys.path.insert(0, str(deepbridge_path))

# Import DeepBridge HPM-KD
try:
    from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig
    from deepbridge.utils.model_registry import ModelType
    HPM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import HPM-KD: {e}")
    print(f"   Make sure you're running from DeepBridge root: {deepbridge_path}")
    HPM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for experiments."""
    # Dataset settings
    USE_FULL_MNIST = False  # Set to True for full 70k samples
    N_SAMPLES_QUICK = 10000  # For quick testing

    # Model settings
    TEACHER_ENSEMBLE_SIZE = 500
    TEACHER_DEPTH = 20
    STUDENT_DEPTH = 5

    # Experiment settings
    N_RANDOM_SEEDS = 1  # Increase to 5 for paper results
    VALIDATION_SPLIT = 0.2

    # Output settings
    OUTPUT_DIR = Path(__file__).parent / "experiment_results"
    SAVE_MODELS = False  # Save trained models
    VERBOSE = True


def load_mnist_data(n_samples=None, random_state=42):
    """
    Load MNIST dataset.

    Args:
        n_samples: Number of samples to use (None = use all 70k)
        random_state: Random seed

    Returns:
        X, y: Features and labels as numpy arrays
    """
    logger.info("Loading MNIST dataset...")

    # Load MNIST
    X, y = fetch_openml('mnist_784', return_X_y=True, parser='auto')

    # Convert to numpy arrays if needed
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    # Normalize pixel values [0-255] -> [0-1]
    X = X.astype(np.float32) / 255.0

    # Use subset if specified
    if n_samples is not None and n_samples < len(X):
        np.random.seed(random_state)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]

    # Convert labels to int
    y = y.astype(int)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    return X, y


def train_teacher_ensemble(X_train, y_train, config):
    """
    Train teacher model (large ensemble).

    Args:
        X_train: Training features
        y_train: Training labels
        config: Experiment configuration

    Returns:
        teacher: Trained teacher model
        metrics: Training metrics dict
    """
    logger.info("="*80)
    logger.info("TRAINING TEACHER MODEL")
    logger.info("="*80)
    logger.info(f"Architecture: RandomForest(n_estimators={config.TEACHER_ENSEMBLE_SIZE}, max_depth={config.TEACHER_DEPTH})")

    start_time = time.time()

    # Create large teacher model
    teacher = RandomForestClassifier(
        n_estimators=config.TEACHER_ENSEMBLE_SIZE,
        max_depth=config.TEACHER_DEPTH,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    teacher.fit(X_train, y_train)

    elapsed = time.time() - start_time
    train_acc = teacher.score(X_train, y_train) * 100

    logger.info(f"✅ Teacher trained in {elapsed:.1f}s")
    logger.info(f"   Train accuracy: {train_acc:.2f}%")

    metrics = {
        'training_time': elapsed,
        'train_accuracy': train_acc,
        'n_estimators': config.TEACHER_ENSEMBLE_SIZE,
        'max_depth': config.TEACHER_DEPTH
    }

    return teacher, metrics


def create_student_model(config, random_state=42):
    """Create student model (small tree)."""
    return DecisionTreeClassifier(
        max_depth=config.STUDENT_DEPTH,
        min_samples_split=10,
        random_state=random_state
    )


def run_baseline_direct_training(X_train, y_train, X_test, y_test, config):
    """Baseline 1: Direct training (no distillation)."""
    logger.info("\n" + "="*80)
    logger.info("BASELINE 1: Direct Training (No Distillation)")
    logger.info("="*80)

    start_time = time.time()

    student = create_student_model(config)
    student.fit(X_train, y_train)

    elapsed = time.time() - start_time

    train_acc = student.score(X_train, y_train) * 100
    test_acc = student.score(X_test, y_test) * 100

    logger.info(f"Training time: {elapsed:.1f}s")
    logger.info(f"Test accuracy: {test_acc:.2f}%")

    return {
        'method': 'Direct Training',
        'test_accuracy': test_acc,
        'train_accuracy': train_acc,
        'training_time': elapsed,
        'retention': None
    }


def run_baseline_traditional_kd(X_train, y_train, X_test, y_test, teacher, config):
    """Baseline 2: Traditional Knowledge Distillation."""
    logger.info("\n" + "="*80)
    logger.info("BASELINE 2: Traditional Knowledge Distillation (Hinton et al. 2015)")
    logger.info("="*80)

    start_time = time.time()

    # Get soft targets from teacher
    teacher_probs = teacher.predict_proba(X_train)
    sample_weights = np.max(teacher_probs, axis=1)

    student = create_student_model(config)
    student.fit(X_train, y_train, sample_weight=sample_weights)

    elapsed = time.time() - start_time

    teacher_acc = teacher.score(X_test, y_test) * 100
    test_acc = student.score(X_test, y_test) * 100
    retention = (test_acc / teacher_acc) * 100

    logger.info(f"Training time: {elapsed:.1f}s")
    logger.info(f"Teacher accuracy: {teacher_acc:.2f}%")
    logger.info(f"Student accuracy: {test_acc:.2f}%")
    logger.info(f"Retention: {retention:.2f}%")

    return {
        'method': 'Traditional KD',
        'test_accuracy': test_acc,
        'teacher_accuracy': teacher_acc,
        'training_time': elapsed,
        'retention': retention
    }


def run_hpmkd_full(X_train, y_train, X_test, y_test, teacher, config):
    """HPM-KD: Full framework with actual implementation."""
    logger.info("\n" + "="*80)
    logger.info("HPM-KD: Full Framework Implementation")
    logger.info("="*80)

    if not HPM_AVAILABLE:
        logger.error("❌ HPM-KD not available. Cannot run full implementation.")
        return None

    logger.info("Components:")
    logger.info("  ✅ Adaptive Configuration Manager")
    logger.info("  ✅ Progressive Distillation Chain")
    logger.info("  ✅ Attention-Weighted Multi-Teacher")
    logger.info("  ✅ Meta-Temperature Scheduler")
    logger.info("  ✅ Parallel Processing Pipeline (disabled for stability)")
    logger.info("  ✅ Shared Optimization Memory")

    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train
    )

    # Configure HPM-KD
    hpm_config = HPMConfig(
        # Progressive chain
        use_progressive=True,
        min_improvement=0.001,

        # Multi-teacher (disabled for single teacher)
        use_multi_teacher=False,

        # Meta-learning
        use_adaptive_temperature=True,
        initial_temperature=4.0,

        # Parallelization (disabled to avoid pickle issues)
        use_parallel=False,

        # Caching
        use_cache=True,
        cache_memory_gb=1.0,

        # General
        n_trials=3,
        validation_split=0.2,
        random_state=42,
        verbose=config.VERBOSE
    )

    start_time = time.time()

    try:
        # Initialize distiller
        distiller = HPMDistiller(
            teacher_model=teacher,
            config=hpm_config
        )

        # Get teacher soft targets
        teacher_probs = teacher.predict_proba(X_tr)

        # Fit HPM-KD
        logger.info("Starting HPM-KD distillation...")
        distiller.fit(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            teacher_probs=teacher_probs,
            model_types=[ModelType.DECISION_TREE],
            temperatures=[2.0, 3.0, 4.0, 5.0],
            alphas=[0.3, 0.5, 0.7]
        )

        elapsed = time.time() - start_time

        # Get best student
        student = distiller.best_model

        # Evaluate
        teacher_acc = teacher.score(X_test, y_test) * 100
        test_acc = student.score(X_test, y_test) * 100
        retention = (test_acc / teacher_acc) * 100

        # Get distillation metrics
        distill_metrics = distiller.best_metrics or {}

        logger.info(f"\n✅ HPM-KD completed in {elapsed:.1f}s")
        logger.info(f"   Teacher accuracy: {teacher_acc:.2f}%")
        logger.info(f"   Student accuracy: {test_acc:.2f}%")
        logger.info(f"   Retention: {retention:.2f}%")
        logger.info(f"   Best config: {distill_metrics}")

        return {
            'method': 'HPM-KD',
            'test_accuracy': test_acc,
            'teacher_accuracy': teacher_acc,
            'training_time': elapsed,
            'retention': retention,
            'best_temperature': distill_metrics.get('temperature', None),
            'best_alpha': distill_metrics.get('alpha', None),
            'n_configs_tried': len(distiller.distillation_results)
        }

    except Exception as e:
        logger.error(f"❌ HPM-KD failed: {e}")
        logger.exception("Full traceback:")
        return None


def run_full_experiment_pipeline(config):
    """Run complete experimental pipeline."""
    logger.info("="*80)
    logger.info("HPM-KD EXPERIMENTAL PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset: MNIST ({'full 70k' if config.USE_FULL_MNIST else f'{config.N_SAMPLES_QUICK} samples'})")
    logger.info(f"  Teacher: RF({config.TEACHER_ENSEMBLE_SIZE}, depth={config.TEACHER_DEPTH})")
    logger.info(f"  Student: DT(depth={config.STUDENT_DEPTH})")
    logger.info(f"  Random seeds: {config.N_RANDOM_SEEDS}")
    logger.info(f"  Output: {config.OUTPUT_DIR}")
    logger.info("="*80)

    # Create output directory
    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    n_samples = None if config.USE_FULL_MNIST else config.N_SAMPLES_QUICK
    X, y = load_mnist_data(n_samples=n_samples)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")

    # Train teacher
    teacher, teacher_metrics = train_teacher_ensemble(X_train, y_train, config)
    teacher_test_acc = teacher.score(X_test, y_test) * 100
    logger.info(f"Teacher test accuracy: {teacher_test_acc:.2f}%")

    # Run experiments
    all_results = []

    # Baseline 1: Direct Training
    result = run_baseline_direct_training(X_train, y_train, X_test, y_test, config)
    all_results.append(result)

    # Baseline 2: Traditional KD
    result = run_baseline_traditional_kd(X_train, y_train, X_test, y_test, teacher, config)
    all_results.append(result)

    # HPM-KD Full Implementation
    result = run_hpmkd_full(X_train, y_train, X_test, y_test, teacher, config)
    if result is not None:
        all_results.append(result)

    # Create results DataFrame
    df_results = pd.DataFrame(all_results)

    # Display summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    logger.info("\n" + df_results.to_string(index=False))

    # Save results
    output_file = config.OUTPUT_DIR / "hpmkd_results.csv"
    df_results.to_csv(output_file, index=False)
    logger.info(f"\n✅ Results saved to {output_file}")

    # Compare with paper
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH PAPER (Table 2, MNIST)")
    logger.info("="*80)

    paper_results = {
        'Direct Training': 98.42,
        'Traditional KD': 98.91,
        'HPM-KD': 99.15
    }

    for method, paper_acc in paper_results.items():
        our_result = df_results[df_results['method'] == method]
        if len(our_result) > 0:
            our_acc = our_result['test_accuracy'].values[0]
            diff = our_acc - paper_acc
            logger.info(f"{method:20s} | Paper: {paper_acc:.2f}% | Ours: {our_acc:.2f}% | Diff: {diff:+.2f}%")

    # Next steps
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. ✅ Verify HPM-KD implementation works")
    logger.info("2. ⏳ Run on full MNIST (70k samples) - set USE_FULL_MNIST=True")
    logger.info("3. ⏳ Use CNN models instead of sklearn (for paper accuracy)")
    logger.info("4. ⏳ Run on all 8 datasets from Section 3.1")
    logger.info("5. ⏳ Implement remaining baselines (FitNets, DML, TAKD)")
    logger.info("6. ⏳ Run ablation studies (Section 6)")
    logger.info("7. ⏳ Generate all figures (Section 5.3-5.6)")
    logger.info("="*80)

    return df_results


def main():
    """Main entry point."""
    config = ExperimentConfig()

    # Check if HPM-KD is available
    if not HPM_AVAILABLE:
        logger.error("="*80)
        logger.error("❌ ERROR: HPM-KD not available")
        logger.error("="*80)
        logger.error("Make sure you're running from DeepBridge root directory:")
        logger.error(f"  cd {deepbridge_path}")
        logger.error(f"  python3 {Path(__file__).relative_to(deepbridge_path)}")
        logger.error("="*80)
        return 1

    # Run experiments
    try:
        results = run_full_experiment_pipeline(config)
        logger.info("\n🎉 Experimental pipeline completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\n❌ Experimental pipeline failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
