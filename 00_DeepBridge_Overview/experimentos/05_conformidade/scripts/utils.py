"""
Utility Functions for Compliance Experiment

Contains helper functions for:
- Logging setup
- Compliance metrics calculation
- Confusion matrix computation
- File I/O operations
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np


def setup_logging(name: str, log_dir: Path) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        name: Logger name
        log_dir: Directory to save log files

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging initialized for {name}")
    logger.info(f"Log file: {log_file}")

    return logger


def save_results(data: Dict[str, Any], filepath: Path, logger: logging.Logger = None):
    """Save results to JSON file"""
    filepath.parent.mkdir(exist_ok=True, parents=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    if logger:
        logger.info(f"Results saved to {filepath}")


def load_results(filepath: Path) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_disparate_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray
) -> float:
    """
    Calculate Disparate Impact (DI)

    DI = P(Y=1 | protected=1) / P(Y=1 | protected=0)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attr: Protected attribute values (binary)

    Returns:
        Disparate Impact ratio
    """
    # Get unique values
    unique_vals = np.unique(protected_attr)
    if len(unique_vals) != 2:
        raise ValueError(f"Expected binary protected attribute, got {len(unique_vals)} unique values")

    # Calculate selection rates
    protected_group = protected_attr == unique_vals[1]
    reference_group = protected_attr == unique_vals[0]

    protected_rate = np.mean(y_pred[protected_group])
    reference_rate = np.mean(y_pred[reference_group])

    if reference_rate == 0:
        return float('inf')

    di = protected_rate / reference_rate
    return di


def check_eeoc_compliance(disparate_impact: float, threshold: float = 0.80) -> bool:
    """
    Check EEOC 80% rule compliance

    Args:
        disparate_impact: DI ratio
        threshold: Compliance threshold (default: 0.80)

    Returns:
        True if compliant, False if violation
    """
    return disparate_impact >= threshold


def check_question_21_compliance(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    min_representation: float = 0.02
) -> Dict[str, bool]:
    """
    Check EEOC Question 21 compliance (minimum representation)

    Args:
        y_pred: Predicted labels
        protected_attr: Protected attribute values
        min_representation: Minimum representation threshold (default: 2%)

    Returns:
        Dictionary of compliance status for each group
    """
    unique_groups = np.unique(protected_attr)
    total_selected = np.sum(y_pred == 1)

    compliance = {}
    for group in unique_groups:
        group_selected = np.sum((protected_attr == group) & (y_pred == 1))
        representation = group_selected / total_selected if total_selected > 0 else 0
        compliance[str(group)] = representation >= min_representation

    return compliance


def calculate_confusion_matrix(
    ground_truth: List[bool],
    detected: List[bool]
) -> Dict[str, int]:
    """
    Calculate confusion matrix for violation detection

    Args:
        ground_truth: List of ground truth violations (True = violation exists)
        detected: List of detected violations (True = violation detected)

    Returns:
        Dictionary with TP, FP, TN, FN counts
    """
    ground_truth = np.array(ground_truth)
    detected = np.array(detected)

    tp = np.sum((ground_truth == True) & (detected == True))
    fp = np.sum((ground_truth == False) & (detected == True))
    tn = np.sum((ground_truth == False) & (detected == False))
    fn = np.sum((ground_truth == True) & (detected == False))

    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def calculate_metrics(confusion_matrix: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate performance metrics from confusion matrix

    Args:
        confusion_matrix: Dict with tp, fp, tn, fn

    Returns:
        Dict with precision, recall, f1_score, accuracy
    """
    tp = confusion_matrix['tp']
    fp = confusion_matrix['fp']
    tn = confusion_matrix['tn']
    fn = confusion_matrix['fn']

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def calculate_feature_coverage(
    detected_attributes: List[str],
    validated_attributes: List[str]
) -> float:
    """
    Calculate feature coverage (proportion of detected attributes that were validated)

    Args:
        detected_attributes: List of detected protected attributes
        validated_attributes: List of validated protected attributes

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if len(detected_attributes) == 0:
        return 0.0

    coverage = len(validated_attributes) / len(detected_attributes)
    return coverage


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"
