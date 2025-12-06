"""
Utility functions for Usability Study
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from datetime import datetime
import json


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


def calculate_sus_score(responses: List[int]) -> float:
    """
    Calculate System Usability Scale (SUS) score

    Args:
        responses: List of 10 responses (1-5 scale)

    Returns:
        SUS score (0-100)

    Reference:
        Brooke, J. (1996). SUS: A "quick and dirty" usability scale.
    """
    if len(responses) != 10:
        raise ValueError("SUS requires exactly 10 responses")

    if not all(1 <= r <= 5 for r in responses):
        raise ValueError("All responses must be between 1 and 5")

    score = 0
    for i, response in enumerate(responses):
        if i % 2 == 0:  # Odd items (0-indexed: 0,2,4,6,8)
            score += (response - 1)
        else:  # Even items (1,3,5,7,9)
            score += (5 - response)

    return score * 2.5


def interpret_sus_score(score: float) -> Dict[str, str]:
    """
    Interpret SUS score

    Returns:
        Dictionary with interpretation details
    """
    if score < 50:
        grade = 'F'
        adjective = 'Poor'
        acceptability = 'Not Acceptable'
    elif score < 68:
        grade = 'D'
        adjective = 'OK'
        acceptability = 'Marginal'
    elif score < 80:
        grade = 'C'
        adjective = 'Good'
        acceptability = 'Acceptable'
    elif score < 85:
        grade = 'B'
        adjective = 'Good'
        acceptability = 'Acceptable'
    elif score < 90:
        grade = 'A'
        adjective = 'Excellent'
        acceptability = 'Acceptable'
        percentile = 'Top 10%'
    else:
        grade = 'A+'
        adjective = 'Best Imaginable'
        acceptability = 'Acceptable'
        percentile = 'Top 5%'

    result = {
        'score': score,
        'grade': grade,
        'adjective': adjective,
        'acceptability': acceptability
    }

    if score >= 85:
        result['percentile'] = percentile if score >= 85 else None

    return result


def calculate_nasa_tlx(dimensions: Dict[str, float]) -> float:
    """
    Calculate NASA Task Load Index (TLX)

    Args:
        dimensions: Dictionary with 6 dimensions (0-100 scale):
            - mental_demand
            - physical_demand
            - temporal_demand
            - performance
            - effort
            - frustration

    Returns:
        Overall TLX score (0-100)

    Reference:
        Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX.
    """
    required_dims = ['mental_demand', 'physical_demand', 'temporal_demand',
                     'performance', 'effort', 'frustration']

    if not all(dim in dimensions for dim in required_dims):
        raise ValueError(f"Missing dimensions. Required: {required_dims}")

    if not all(0 <= v <= 100 for v in dimensions.values()):
        raise ValueError("All dimension values must be between 0 and 100")

    return sum(dimensions.values()) / 6


def interpret_nasa_tlx(score: float) -> str:
    """Interpret NASA TLX score"""
    if score < 20:
        return "Very Low Workload"
    elif score < 40:
        return "Low Workload"
    elif score < 60:
        return "Moderate Workload"
    elif score < 80:
        return "High Workload"
    else:
        return "Very High Workload"


def calculate_success_rate(successes: List[bool]) -> Dict[str, float]:
    """
    Calculate success rate

    Args:
        successes: List of boolean values (True = success, False = failure)

    Returns:
        Dictionary with success rate statistics
    """
    n_total = len(successes)
    n_success = sum(successes)
    n_failure = n_total - n_success

    success_rate = (n_success / n_total) * 100 if n_total > 0 else 0

    # Calculate 95% confidence interval (Wilson score interval)
    if n_total > 0:
        p = n_success / n_total
        z = 1.96  # 95% CI
        denominator = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
        ci_lower = max(0, (center - margin) * 100)
        ci_upper = min(100, (center + margin) * 100)
    else:
        ci_lower = ci_upper = 0

    return {
        'n_total': n_total,
        'n_success': n_success,
        'n_failure': n_failure,
        'success_rate': success_rate,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper
    }


def calculate_time_statistics(times: List[float]) -> Dict[str, float]:
    """
    Calculate time statistics

    Args:
        times: List of completion times in minutes

    Returns:
        Dictionary with time statistics
    """
    if not times:
        return {
            'n': 0,
            'mean': 0,
            'std': 0,
            'median': 0,
            'min': 0,
            'max': 0,
            'q25': 0,
            'q75': 0
        }

    return {
        'n': len(times),
        'mean': float(np.mean(times)),
        'std': float(np.std(times, ddof=1)),
        'median': float(np.median(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'q25': float(np.percentile(times, 25)),
        'q75': float(np.percentile(times, 75))
    }


def save_results(results: Dict, output_path: Path, logger: logging.Logger = None):
    """Save results to JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if logger:
        logger.info(f"Results saved to {output_path}")


def load_results(input_path: Path) -> Dict:
    """Load results from JSON"""
    with open(input_path, 'r') as f:
        return json.load(f)


def generate_participant_id(index: int, prefix: str = "P") -> str:
    """Generate anonymous participant ID"""
    return f"{prefix}{index:03d}"


def categorize_experience(years: float) -> str:
    """Categorize years of experience"""
    if years < 2:
        return "Beginner"
    elif years < 5:
        return "Intermediate"
    elif years < 8:
        return "Advanced"
    else:
        return "Expert"


def categorize_role(role: str) -> str:
    """Normalize role categories"""
    role_lower = role.lower()
    if 'scientist' in role_lower or 'data' in role_lower:
        return "Data Scientist"
    elif 'engineer' in role_lower or 'ml' in role_lower:
        return "ML Engineer"
    elif 'research' in role_lower:
        return "Researcher"
    else:
        return "Other"


def format_time(minutes: float) -> str:
    """Format time in minutes to human-readable format"""
    if minutes < 1:
        return f"{minutes * 60:.0f} sec"
    elif minutes < 60:
        return f"{minutes:.1f} min"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}min"
