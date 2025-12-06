"""
Utilities for Case Studies Experiments
"""

import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


def setup_logging(experiment_name: str, log_dir: Path) -> logging.Logger:
    """Setup logging for experiments"""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging initialized for {experiment_name}")
    logger.info(f"Log file: {log_file}")

    return logger


def save_results(results: Dict[str, Any], output_path: Path, logger: logging.Logger = None):
    """Save experiment results to JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if logger:
        logger.info(f"Results saved to {output_path}")


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load experiment results from JSON"""
    with open(input_path, 'r') as f:
        return json.load(f)


def measure_time(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str = "Operation", logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.logger:
            self.logger.info(f"{self.name} completed in {self.elapsed:.2f} seconds ({self.elapsed/60:.2f} minutes)")


def calculate_disparate_impact(y_true: np.ndarray, y_pred: np.ndarray,
                               protected_attr: np.ndarray) -> float:
    """
    Calculate Disparate Impact (DI) metric
    DI = P(Y=1|protected=1) / P(Y=1|protected=0)
    """
    protected_mask = protected_attr == 1
    privileged_mask = ~protected_mask

    protected_positive_rate = y_pred[protected_mask].mean()
    privileged_positive_rate = y_pred[privileged_mask].mean()

    if privileged_positive_rate == 0:
        return 0.0

    di = protected_positive_rate / privileged_positive_rate
    return di


def check_eeoc_compliance(di: float, threshold: float = 0.8) -> bool:
    """
    Check EEOC 80% rule compliance
    Returns True if compliant (DI >= 0.8), False otherwise
    """
    return di >= threshold


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_prob = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_prob - bin_acc)

    return ece


def format_time_breakdown(times: Dict[str, float]) -> str:
    """Format time breakdown for reporting"""
    lines = ["Time Breakdown:"]
    total = sum(times.values())

    for name, t in times.items():
        pct = (t / total * 100) if total > 0 else 0
        lines.append(f"  {name}: {t:.1f} min ({pct:.1f}%)")

    lines.append(f"  TOTAL: {total:.1f} min")
    return "\n".join(lines)


def generate_summary_report(domain: str, results: Dict[str, Any],
                           output_path: Path, logger: logging.Logger = None):
    """
    Generate a summary report for a case study

    Note: This is a placeholder. In real implementation,
    this would generate a proper PDF report.
    """
    if logger:
        logger.info(f"Generating summary report for {domain}...")

    report_lines = [
        f"=" * 80,
        f"DeepBridge Validation Report: {domain}",
        f"=" * 80,
        f"",
        f"Timestamp: {datetime.now().isoformat()}",
        f"",
        f"Dataset Statistics:",
        f"  Samples: {results.get('n_samples', 'N/A')}",
        f"  Features: {results.get('n_features', 'N/A')}",
        f"",
        f"Validation Time:",
        f"  Total: {results.get('total_time', 0):.2f} minutes",
        f"",
        f"Violations Detected: {results.get('n_violations', 0)}",
        f"",
        f"Fairness Results:",
    ]

    fairness = results.get('fairness', {})
    for metric, value in fairness.items():
        report_lines.append(f"  {metric}: {value}")

    report_lines.extend([
        f"",
        f"Uncertainty Results:",
    ])

    uncertainty = results.get('uncertainty', {})
    for metric, value in uncertainty.items():
        report_lines.append(f"  {metric}: {value}")

    report_lines.extend([
        f"",
        f"=" * 80,
    ])

    report_text = "\n".join(report_lines)

    # Save as text file (placeholder for PDF)
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w') as f:
        f.write(report_text)

    if logger:
        logger.info(f"Report saved to {txt_path}")
        logger.info(f"(PDF generation would happen here in full implementation)")


def create_latex_table_row(domain: str, n_samples: int, n_violations: int,
                           time_minutes: float, findings: str) -> str:
    """Create a LaTeX table row for the summary table"""
    return f"{domain} & {n_samples:,} & {n_violations} & {time_minutes:.0f} & {findings} \\\\"


def aggregate_case_study_results(result_files: List[Path]) -> Dict[str, Any]:
    """
    Aggregate results from multiple case studies
    """
    all_results = []
    times = []
    violations = []

    for file in result_files:
        if file.exists():
            results = load_results(file)
            all_results.append(results)
            times.append(results.get('total_time', 0))
            violations.append(results.get('n_violations', 0))

    aggregated = {
        'n_cases': len(all_results),
        'mean_time': np.mean(times) if times else 0,
        'std_time': np.std(times) if times else 0,
        'min_time': np.min(times) if times else 0,
        'max_time': np.max(times) if times else 0,
        'total_violations': sum(violations),
        'cases_with_violations': sum(1 for v in violations if v > 0),
        'detection_precision': 1.0,  # Placeholder
        'false_positives': 0,  # Placeholder
        'individual_results': all_results
    }

    return aggregated
