"""
Utility Functions for Ablation Studies

Contains helper functions for:
- Configuration management
- Time measurement
- Statistical analysis
- File I/O operations
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np


def setup_logging(log_dir, name: str) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_dir: Directory to save log files (Path or str)
        name: Logger name

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)  # Ensure it's a Path object
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


def get_ablation_configs() -> Dict[str, Dict[str, bool]]:
    """
    Get all ablation configurations

    Returns:
        Dictionary of configuration names to settings
    """
    configs = {
        'full': {
            'unified_api': True,
            'parallel_execution': True,
            'caching': True,
            'automated_reporting': True,
            'description': 'DeepBridge completo'
        },
        'no_api': {
            'unified_api': False,
            'parallel_execution': True,
            'caching': True,
            'automated_reporting': True,
            'description': 'Sem API unificada (conversões manuais)'
        },
        'no_parallel': {
            'unified_api': True,
            'parallel_execution': False,
            'caching': True,
            'automated_reporting': True,
            'description': 'Sem paralelização (execução sequencial)'
        },
        'no_cache': {
            'unified_api': True,
            'parallel_execution': True,
            'caching': False,
            'automated_reporting': True,
            'description': 'Sem caching (recomputar predições)'
        },
        'no_auto_report': {
            'unified_api': True,
            'parallel_execution': True,
            'caching': True,
            'automated_reporting': False,
            'description': 'Sem automação de relatórios (geração manual)'
        },
        'none': {
            'unified_api': False,
            'parallel_execution': False,
            'caching': False,
            'automated_reporting': False,
            'description': 'Workflow fragmentado completo'
        }
    }

    return configs


def calculate_contribution(time_baseline: float, time_ablated: float) -> float:
    """
    Calculate absolute contribution of a component

    Args:
        time_baseline: Time with all components (baseline)
        time_ablated: Time without the component

    Returns:
        Absolute time contribution in minutes
    """
    return time_ablated - time_baseline


def calculate_contribution_percentage(contribution: float, total_gain: float) -> float:
    """
    Calculate percentage contribution of a component

    Args:
        contribution: Absolute contribution in minutes
        total_gain: Total gain from baseline to fragmented

    Returns:
        Percentage contribution (0-100)
    """
    if total_gain == 0:
        return 0.0

    return (contribution / total_gain) * 100


def calculate_statistics(times: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of times

    Args:
        times: List of execution times

    Returns:
        Dictionary with mean, std, min, max, median
    """
    times_array = np.array(times)

    return {
        'mean': float(np.mean(times_array)),
        'std': float(np.std(times_array)),
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array)),
        'median': float(np.median(times_array)),
        'n': len(times)
    }


def format_time(minutes: float) -> str:
    """Format minutes to human-readable string"""
    if minutes < 1:
        return f"{minutes*60:.1f}s"
    elif minutes < 60:
        return f"{minutes:.1f}min"
    else:
        hours = minutes / 60
        return f"{hours:.1f}h"


def calculate_speedup(time_slow: float, time_fast: float) -> float:
    """
    Calculate speedup ratio

    Args:
        time_slow: Slower execution time
        time_fast: Faster execution time

    Returns:
        Speedup ratio
    """
    if time_fast == 0:
        return float('inf')

    return time_slow / time_fast
