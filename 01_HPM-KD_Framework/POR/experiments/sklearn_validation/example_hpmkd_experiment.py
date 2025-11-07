#!/usr/bin/env python3
"""
HPM-KD Example Experiment
=========================

Simple example demonstrating how to run HPM-KD distillation experiment
matching the paper methodology.

This script:
1. Loads MNIST dataset (simplest benchmark from paper)
2. Trains a teacher model
3. Runs HPM-KD distillation to create student
4. Compares with baseline methods
5. Generates results matching paper format

Author: Gustavo Coelho Haase
Date: November 2025
Paper: HPM-KD Framework (Section 5.1, Table 2)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mnist_data(n_samples=10000):
    """
    Load MNIST dataset (subset for quick testing).

    Full dataset: 70,000 images (paper uses all)
    This example: 10,000 images (for quick testing)
    """
    logger.info("Loading MNIST dataset...")

    # Load MNIST
    X, y = fetch_openml('mnist_784', return_X_y=True, parser='auto')

    # Convert to numpy arrays if needed (fetch_openml may return DataFrames)
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values

    # Normalize pixel values [0-255] -> [0-1]
    X = X / 255.0

    # Use subset for quick testing (remove for full experiment)
    if n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]

    # Convert labels to int
    y = y.astype(int)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    return X, y


def train_teacher_model(X_train, y_train):
    """
    Train teacher model (large Random Forest).

    Paper specification (Table 1):
    - MNIST Teacher: 3-layer CNN with 4.2M parameters
    - For quick testing: Large Random Forest (500 trees)

    Returns trained teacher and training time.
    """
    logger.info("Training teacher model...")
    logger.info("Architecture: RandomForestClassifier(n_estimators=500, max_depth=20)")

    start_time = time.time()

    # Create large teacher model
    teacher = RandomForestClassifier(
        n_estimators=500,  # Large ensemble (high capacity)
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    teacher.fit(X_train, y_train)

    elapsed = time.time() - start_time

    # Evaluate teacher
    train_acc = teacher.score(X_train, y_train) * 100
    logger.info(f"Teacher trained in {elapsed:.1f}s")
    logger.info(f"Teacher train accuracy: {train_acc:.2f}%")

    return teacher, elapsed


def create_student_model():
    """
    Create student model (small Decision Tree).

    Paper specification (Table 1):
    - MNIST Student: 2-layer CNN with 0.4M parameters (10.5x compression)
    - For testing: Small Decision Tree (max_depth=5)

    Compression ratio: ~100x (500 trees -> 1 tree, depth 20 -> 5)
    """
    student = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    return student


def run_direct_training(X_train, y_train, X_test, y_test):
    """
    Baseline 1: Direct Training (no distillation).

    Paper Table 2, Row 1: MNIST Direct Training = 98.42%
    """
    logger.info("\n" + "="*60)
    logger.info("BASELINE 1: Direct Training (No Distillation)")
    logger.info("="*60)

    start_time = time.time()

    student = create_student_model()
    student.fit(X_train, y_train)

    elapsed = time.time() - start_time

    train_acc = student.score(X_train, y_train) * 100
    test_acc = student.score(X_test, y_test) * 100

    logger.info(f"Training time: {elapsed:.1f}s")
    logger.info(f"Train accuracy: {train_acc:.2f}%")
    logger.info(f"Test accuracy: {test_acc:.2f}%")

    return {
        'method': 'Direct Training',
        'test_accuracy': test_acc,
        'train_accuracy': train_acc,
        'time': elapsed,
        'retention': None  # No teacher comparison
    }


def run_traditional_kd(X_train, y_train, X_test, y_test, teacher):
    """
    Baseline 2: Traditional Knowledge Distillation (Hinton et al. 2015).

    Paper Table 2, Row 2: MNIST Traditional KD = 98.91% (99.63% retention)

    Simulates KD by training student on soft targets from teacher.
    """
    logger.info("\n" + "="*60)
    logger.info("BASELINE 2: Traditional Knowledge Distillation")
    logger.info("="*60)

    start_time = time.time()

    # Get teacher predictions (soft targets)
    teacher_probs = teacher.predict_proba(X_train)

    # For sklearn trees, we simulate KD by using teacher's class probabilities
    # as sample weights (simplified version of temperature-scaled softmax)
    sample_weights = np.max(teacher_probs, axis=1)

    student = create_student_model()
    student.fit(X_train, y_train, sample_weight=sample_weights)

    elapsed = time.time() - start_time

    teacher_acc = teacher.score(X_test, y_test) * 100
    test_acc = student.score(X_test, y_test) * 100
    retention = (test_acc / teacher_acc) * 100

    logger.info(f"Training time: {elapsed:.1f}s")
    logger.info(f"Teacher accuracy: {teacher_acc:.2f}%")
    logger.info(f"Student accuracy: {test_acc:.2f}%")
    logger.info(f"Accuracy retention: {retention:.2f}%")

    return {
        'method': 'Traditional KD',
        'test_accuracy': test_acc,
        'teacher_accuracy': teacher_acc,
        'retention': retention,
        'time': elapsed
    }


def run_hpmkd(X_train, y_train, X_test, y_test, teacher):
    """
    HPM-KD: Full Framework with all 6 components.

    Paper Table 2, Row 6: MNIST HPM-KD = 99.15% (99.87% retention)

    This is a simplified version. Full implementation requires:
    - deepbridge.distillation.techniques.hpm.HPMDistiller
    """
    logger.info("\n" + "="*60)
    logger.info("HPM-KD: Hierarchical Progressive Multi-Teacher KD")
    logger.info("="*60)
    logger.info("Components: Adaptive Config + Progressive Chain + Multi-Teacher")
    logger.info("          + Meta-Temperature + Parallel + Shared Memory")

    try:
        # Try to use actual HPM-KD implementation
        from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig

        logger.info("Using full HPM-KD implementation from DeepBridge")

        config = HPMConfig(
            use_progressive=True,
            use_multi_teacher=False,  # Single teacher for this example
            use_adaptive_temperature=True,
            use_parallel=False,  # Disabled for pickle issues
            use_cache=True,
            min_improvement=0.01,
            initial_temperature=4.0,
            verbose=True
        )

        start_time = time.time()

        distiller = HPMDistiller(
            teacher_model=teacher,
            config=config
        )

        # Note: HPMDistiller expects specific interfaces
        # This is a simplified call - see IMPLEMENTATION_GUIDE.md for full usage
        student = distiller.distill_to_sklearn(
            X_train=X_train,
            y_train=y_train,
            student_base=create_student_model()
        )

        elapsed = time.time() - start_time

    except (ImportError, AttributeError) as e:
        # Fallback: Simulated HPM-KD (enhanced KD)
        logger.warning(f"Full HPM-KD not available: {e}")
        logger.info("Using simulated HPM-KD (enhanced distillation)")

        start_time = time.time()

        # Simulate progressive chain: Train multiple intermediate students
        teacher_probs = teacher.predict_proba(X_train)

        # Progressive chain simulation: 3 steps with decreasing capacity
        chain_depths = [10, 7, 5]  # Decreasing tree depths
        current_teacher_probs = teacher_probs

        for i, depth in enumerate(chain_depths):
            logger.info(f"Progressive chain step {i+1}/3: depth={depth}")

            intermediate = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=10,
                random_state=42 + i
            )

            # Use current soft targets
            weights = np.max(current_teacher_probs, axis=1)
            intermediate.fit(X_train, y_train, sample_weight=weights)

            # Update soft targets for next step
            if i < len(chain_depths) - 1:
                current_teacher_probs = intermediate.predict_proba(X_train)

        student = intermediate  # Final student
        elapsed = time.time() - start_time

    # Evaluate
    teacher_acc = teacher.score(X_test, y_test) * 100
    test_acc = student.score(X_test, y_test) * 100
    retention = (test_acc / teacher_acc) * 100

    logger.info(f"Training time: {elapsed:.1f}s")
    logger.info(f"Teacher accuracy: {teacher_acc:.2f}%")
    logger.info(f"Student accuracy: {test_acc:.2f}%")
    logger.info(f"Accuracy retention: {retention:.2f}%")
    logger.info(f"Compression ratio: ~100x (500 trees depth-20 → 1 tree depth-5)")

    return {
        'method': 'HPM-KD',
        'test_accuracy': test_acc,
        'teacher_accuracy': teacher_acc,
        'retention': retention,
        'time': elapsed
    }


def main():
    """
    Main experiment pipeline.
    """
    logger.info("="*80)
    logger.info("HPM-KD Example Experiment")
    logger.info("Dataset: MNIST (subset)")
    logger.info("Paper Reference: Section 5.1, Table 2")
    logger.info("="*80)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    X, y = load_mnist_data(n_samples=10000)  # Use 10k for quick test

    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train teacher
    teacher, teacher_time = train_teacher_model(X_train, y_train)
    teacher_test_acc = teacher.score(X_test, y_test) * 100
    logger.info(f"Teacher test accuracy: {teacher_test_acc:.2f}%")

    # Run experiments
    results = []

    # Baseline 1: Direct Training
    result_direct = run_direct_training(X_train, y_train, X_test, y_test)
    results.append(result_direct)

    # Baseline 2: Traditional KD
    result_kd = run_traditional_kd(X_train, y_train, X_test, y_test, teacher)
    results.append(result_kd)

    # HPM-KD
    result_hpmkd = run_hpmkd(X_train, y_train, X_test, y_test, teacher)
    results.append(result_hpmkd)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)

    df_results = pd.DataFrame(results)
    logger.info("\n" + df_results.to_string(index=False))

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
        our_acc = df_results[df_results['method'] == method]['test_accuracy'].values
        if len(our_acc) > 0:
            our_acc = our_acc[0]
            diff = our_acc - paper_acc
            logger.info(f"{method:20s} | Paper: {paper_acc:.2f}% | Ours: {our_acc:.2f}% | Diff: {diff:+.2f}%")

    # Save results
    output_file = 'example_results.csv'
    df_results.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to {output_file}")

    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. Run on full MNIST (70k samples) - remove n_samples limit")
    logger.info("2. Run on all 8 datasets from paper (Section 3.1)")
    logger.info("3. Implement all 5 baseline methods (FitNets, DML, TAKD)")
    logger.info("4. Run ablation studies (Section 6)")
    logger.info("5. Generate figures (Section 5.3-5.6)")
    logger.info("6. See IMPLEMENTATION_GUIDE.md for complete pipeline")
    logger.info("="*80)


if __name__ == '__main__':
    main()
