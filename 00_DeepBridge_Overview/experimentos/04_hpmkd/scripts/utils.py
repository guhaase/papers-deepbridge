"""
Utility functions for HPM-KD experiments
"""

import numpy as np
import pandas as pd
import pickle
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json


def setup_logging(log_dir, experiment_name: str) -> logging.Logger:
    """Setup logging for experiments"""
    log_dir = Path(log_dir)  # Ensure it's a Path object
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

    return logger


def get_model_size_mb(model) -> float:
    """Get model size in MB"""
    import pickle
    import sys

    # Serialize model to bytes
    model_bytes = pickle.dumps(model)
    size_bytes = sys.getsizeof(model_bytes)
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def measure_inference_latency(model, X, n_iterations=10, batch_size=1000) -> Dict[str, float]:
    """
    Measure inference latency

    Args:
        model: Trained model
        X: Input data
        n_iterations: Number of iterations for timing
        batch_size: Batch size for inference

    Returns:
        Dict with latency statistics
    """
    latencies = []

    for _ in range(n_iterations):
        # Take a batch
        if len(X) > batch_size:
            batch_idx = np.random.choice(len(X), batch_size, replace=False)
            X_batch = X[batch_idx]
        else:
            X_batch = X

        # Measure time
        start = time.time()
        _ = model.predict(X_batch)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        latencies.append(elapsed)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'batch_size': batch_size
    }


def calculate_compression_ratio(teacher_size_mb: float, student_size_mb: float) -> float:
    """Calculate compression ratio"""
    return teacher_size_mb / student_size_mb if student_size_mb > 0 else 0


def calculate_retention_rate(teacher_acc: float, student_acc: float) -> float:
    """Calculate knowledge retention rate"""
    return (student_acc / teacher_acc * 100) if teacher_acc > 0 else 0


def calculate_speedup(teacher_latency: float, student_latency: float) -> float:
    """Calculate inference speedup"""
    return teacher_latency / student_latency if student_latency > 0 else 0


def save_model(model, path: Path, logger: logging.Logger = None):
    """Save model to disk"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    if logger:
        logger.info(f"Model saved to {path}")


def load_model(path: Path, logger: logging.Logger = None):
    """Load model from disk"""
    with open(path, 'rb') as f:
        model = pickle.load(f)

    if logger:
        logger.info(f"Model loaded from {path}")

    return model


def save_results(results: Dict[str, Any], output_path: Path, logger: logging.Logger = None):
    """Save results to JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if logger:
        logger.info(f"Results saved to {output_path}")


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load results from JSON"""
    with open(input_path, 'r') as f:
        return json.load(f)


def aggregate_dataset_results(results_list: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across multiple datasets"""

    metrics_to_aggregate = [
        'teacher_accuracy', 'student_accuracy', 'retention_rate',
        'compression_ratio', 'latency_speedup'
    ]

    aggregated = {}

    for metric in metrics_to_aggregate:
        values = [r[metric] for r in results_list if metric in r]

        if values:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values)
            }

    return aggregated


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"


def format_size(size_mb: float) -> str:
    """Format size in MB"""
    if size_mb >= 1000:
        return f"{size_mb/1000:.2f}GB"
    else:
        return f"{size_mb:.1f}MB"


def format_latency(latency_ms: float) -> str:
    """Format latency"""
    if latency_ms < 1:
        return f"{latency_ms*1000:.0f}μs"
    else:
        return f"{latency_ms:.1f}ms"


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
            self.logger.info(f"{self.name} completed in {self.elapsed:.2f}s")


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
