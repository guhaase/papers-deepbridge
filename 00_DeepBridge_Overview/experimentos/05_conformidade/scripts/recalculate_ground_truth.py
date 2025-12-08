"""
Recalculate Ground Truth Based on Actual Data

Scans all generated test cases and calculates the ACTUAL violations
(including marginal ones caused by random generation).

This ensures ground truth matches reality, not just intended injections.

Autor: DeepBridge Team
Data: 2025-12-07
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"


def calculate_disparate_impact(df: pd.DataFrame, protected_attr: str, target: str) -> dict:
    """
    Calculate disparate impact for all groups

    Args:
        df: DataFrame with data
        protected_attr: Protected attribute name
        target: Target variable name

    Returns:
        Dictionary with DI metrics per group
    """
    results = {}
    groups = df[protected_attr].unique()

    # Calculate approval rates
    approval_rates = {}
    for group in groups:
        group_data = df[df[protected_attr] == group]
        approval_rate = group_data[target].mean()
        approval_rates[group] = approval_rate

    # Find reference group (highest approval rate)
    reference_group = max(approval_rates.items(), key=lambda x: x[1])[0]
    reference_rate = approval_rates[reference_group]

    # Calculate DI for each group
    for group in groups:
        if group == reference_group:
            continue

        group_rate = approval_rates[group]
        di = group_rate / reference_rate if reference_rate > 0 else 1.0

        results[f"{protected_attr}_{group}"] = {
            'disparate_impact': di,
            'approval_rate': group_rate,
            'reference_rate': reference_rate,
            'compliant': di >= 0.80
        }

    return results


def analyze_case(case_id: int, logger) -> dict:
    """
    Analyze a single case and determine actual violations

    Args:
        case_id: Case identifier
        logger: Logger instance

    Returns:
        Dictionary with ground truth for this case
    """
    # Load dataset
    case_file = DATA_DIR / f"case_{case_id:02d}.csv"
    df = pd.read_csv(case_file)

    # Drop case_id column if present
    if 'case_id' in df.columns:
        df = df.drop('case_id', axis=1)

    violations = {}

    # Check gender
    gender_results = calculate_disparate_impact(df, 'gender', 'approved')
    for group, metrics in gender_results.items():
        if not metrics['compliant']:
            violations[group] = {
                'disparate_impact': metrics['disparate_impact'],
                'eeoc_compliant': False,
                'reason': 'DI < 0.80'
            }

    # Check race
    race_results = calculate_disparate_impact(df, 'race', 'approved')
    for group, metrics in race_results.items():
        if not metrics['compliant']:
            violations[group] = {
                'disparate_impact': metrics['disparate_impact'],
                'eeoc_compliant': False,
                'reason': 'DI < 0.80'
            }

    has_violation = len(violations) > 0

    ground_truth = {
        'case_id': case_id,
        'has_violation': has_violation,
        'total_violations': len(violations),
        'violations': violations,
        'n_samples': len(df),
        'protected_attributes': ['gender', 'race']
    }

    return ground_truth


def recalculate_all_ground_truth(n_cases: int = 50) -> list:
    """
    Recalculate ground truth for all cases

    Args:
        n_cases: Total number of cases

    Returns:
        List of ground truth dictionaries
    """
    global logger

    logger.info(f"Recalculating ground truth for {n_cases} cases...")
    logger.info("This will capture ALL violations, including marginal ones")

    all_ground_truths = []

    for case_id in range(1, n_cases + 1):
        try:
            gt = analyze_case(case_id, logger)
            all_ground_truths.append(gt)

            if case_id % 10 == 0:
                logger.info(f"  Analyzed {case_id}/{n_cases} cases")

        except Exception as e:
            logger.error(f"  Error analyzing case {case_id}: {str(e)}")

    logger.info(f"✓ All {n_cases} cases analyzed")

    return all_ground_truths


def compare_with_original(new_gt: list, original_file: Path) -> None:
    """
    Compare new ground truth with original

    Args:
        new_gt: New ground truth list
        original_file: Path to original ground truth file
    """
    global logger

    logger.info("\nComparing with original ground truth...")

    with open(original_file, 'r') as f:
        original = json.load(f)

    original_cases = {case['case_id']: case for case in original['cases']}

    differences = []

    for new_case in new_gt:
        case_id = new_case['case_id']
        original_case = original_cases[case_id]

        # Check if violation status changed
        if new_case['has_violation'] != original_case['has_violation']:
            differences.append({
                'case_id': case_id,
                'original_violation': original_case['has_violation'],
                'new_violation': new_case['has_violation'],
                'new_violations_found': new_case['violations']
            })

        # Check if number of violations changed
        elif new_case['total_violations'] != original_case['total_violations']:
            differences.append({
                'case_id': case_id,
                'original_count': original_case['total_violations'],
                'new_count': new_case['total_violations'],
                'new_violations_found': new_case['violations']
            })

    logger.info(f"Found {len(differences)} cases with differences:")
    for diff in differences:
        logger.info(f"  Case {diff['case_id']}: {diff}")

    return differences


def main():
    """Main execution"""
    global logger
    logger = setup_logging("recalculate_ground_truth", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("RECALCULATING GROUND TRUTH FROM ACTUAL DATA")
    logger.info("=" * 80)
    logger.info("This ensures ground truth reflects ALL violations, not just injected ones")
    logger.info("")

    # Load original ground truth
    original_gt_file = RESULTS_DIR / "compliance_ground_truth.json"

    # Recalculate ground truth from actual data
    new_ground_truths = recalculate_all_ground_truth(n_cases=50)

    # Compare with original
    differences = compare_with_original(new_ground_truths, original_gt_file)

    # Save updated ground truth
    ground_truth_output = {
        'total_cases': len(new_ground_truths),
        'violation_cases': sum(1 for gt in new_ground_truths if gt['has_violation']),
        'clean_cases': sum(1 for gt in new_ground_truths if not gt['has_violation']),
        'cases': new_ground_truths,
        'note': 'Ground truth recalculated from actual data to include all violations (including marginal ones)'
    }

    # Backup original
    backup_file = RESULTS_DIR / "compliance_ground_truth_ORIGINAL.json"
    if original_gt_file.exists() and not backup_file.exists():
        import shutil
        shutil.copy(original_gt_file, backup_file)
        logger.info(f"Original ground truth backed up to: {backup_file}")

    # Save new ground truth
    new_gt_file = RESULTS_DIR / "compliance_ground_truth.json"
    save_results(ground_truth_output, new_gt_file, logger)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total cases: {len(new_ground_truths)}")
    logger.info(f"Cases with violations: {ground_truth_output['violation_cases']}")
    logger.info(f"Cases without violations: {ground_truth_output['clean_cases']}")
    logger.info(f"Cases with differences from original: {len(differences)}")
    logger.info(f"Updated ground truth saved to: {new_gt_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
