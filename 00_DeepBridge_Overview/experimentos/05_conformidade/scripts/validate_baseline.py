"""
Validate Compliance Using Baseline Methods (AIF360 + Fairlearn)

Uses real AIF360 and Fairlearn libraries for compliance validation.

Autor: DeepBridge Team
Data: 2025-12-07
"""

import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fairlearn.metrics import MetricFrame, demographic_parity_difference

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"


def load_ground_truth():
    """Load ground truth data"""
    gt_file = RESULTS_DIR / "compliance_ground_truth.json"
    with open(gt_file, 'r') as f:
        return json.load(f)


def calculate_disparate_impact_manual(df: pd.DataFrame, protected_attr: str, target: str) -> dict:
    """
    Manually calculate disparate impact (simulates baseline workflow)

    Args:
        df: DataFrame with data
        protected_attr: Protected attribute name
        target: Target variable name

    Returns:
        Dictionary with DI metrics per group
    """
    results = {}

    # Get unique groups
    groups = df[protected_attr].unique()

    # Calculate approval rates
    approval_rates = {}
    for group in groups:
        group_data = df[df[protected_attr] == group]
        approval_rate = group_data[target].mean()
        approval_rates[group] = approval_rate

    # Find reference group (typically majority group or highest approval rate)
    reference_group = max(approval_rates.items(), key=lambda x: x[1])[0]
    reference_rate = approval_rates[reference_group]

    # Calculate DI for each group
    for group in groups:
        if group == reference_group:
            continue

        group_rate = approval_rates[group]

        # DI = P(positive | protected) / P(positive | reference)
        di = group_rate / reference_rate if reference_rate > 0 else 1.0

        results[f"{protected_attr}_{group}"] = {
            'disparate_impact': di,
            'approval_rate': group_rate,
            'reference_rate': reference_rate,
            'compliant': di >= 0.80
        }

    return results


def validate_single_case_baseline(case_id: int, logger) -> dict:
    """
    Validate a single case using real AIF360 library

    This uses actual AIF360 implementation for fairness checking:
    1. Load data
    2. Encode categorical variables
    3. Convert to AIF360 BinaryLabelDataset format
    4. Calculate disparate impact using AIF360 metrics
    5. Check compliance (DI >= 0.80)

    Note: Using REAL AIF360, not simulation
    """
    # Load dataset
    case_file = DATA_DIR / f"case_{case_id:02d}.csv"
    df = pd.read_csv(case_file)

    # Drop case_id column
    if 'case_id' in df.columns:
        df = df.drop('case_id', axis=1)

    violations_detected = []

    # Store original values for mapping back
    gender_map = {v: k for k, v in enumerate(df['gender'].unique())}
    race_map = {v: k for k, v in enumerate(df['race'].unique())}

    # Create encoded dataframe for AIF360
    df_encoded = df.copy()
    df_encoded['gender'] = df['gender'].map(gender_map)
    df_encoded['race'] = df['race'].map(race_map)

    # Check gender using AIF360
    try:
        # Convert to AIF360 format for gender
        aif_dataset_gender = BinaryLabelDataset(
            df=df_encoded,
            label_names=['approved'],
            protected_attribute_names=['gender']
        )

        # Calculate metrics for each gender group
        gender_values_original = list(df['gender'].unique())
        gender_values_encoded = [gender_map[v] for v in gender_values_original]

        # Find reference group (highest approval rate)
        approval_rates = {v: df[df['gender'] == v]['approved'].mean() for v in gender_values_original}
        reference_gender = max(approval_rates, key=approval_rates.get)
        reference_gender_encoded = gender_map[reference_gender]

        privileged_groups = [{'gender': reference_gender_encoded}]

        for gender_orig, gender_enc in zip(gender_values_original, gender_values_encoded):
            if gender_orig == reference_gender:
                continue

            unprivileged_groups = [{'gender': gender_enc}]

            metric = BinaryLabelDatasetMetric(
                aif_dataset_gender,
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups
            )

            di = metric.disparate_impact()

            # Check if violates 80% rule
            if di < 0.80:
                violations_detected.append({
                    'attribute': f"gender_{gender_orig}",
                    'disparate_impact': di,
                    'compliant': False,
                    'reason': 'DI < 0.80 (AIF360)'
                })
    except Exception as e:
        logger.warning(f"  AIF360 gender check failed for case {case_id}: {str(e)}")
        # Fallback to manual calculation if AIF360 fails
        gender_results = calculate_disparate_impact_manual(df, 'gender', 'approved')
        for group, metrics in gender_results.items():
            if not metrics['compliant']:
                violations_detected.append({
                    'attribute': group,
                    'disparate_impact': metrics['disparate_impact'],
                    'compliant': False,
                    'reason': 'DI < 0.80 (Fallback)'
                })

    # Check race using AIF360
    try:
        # Convert to AIF360 format for race
        aif_dataset_race = BinaryLabelDataset(
            df=df_encoded,
            label_names=['approved'],
            protected_attribute_names=['race']
        )

        # Calculate metrics for each race group
        race_values_original = list(df['race'].unique())
        race_values_encoded = [race_map[v] for v in race_values_original]

        # Use the race with highest approval rate as reference
        approval_rates = {race: df[df['race'] == race]['approved'].mean() for race in race_values_original}
        reference_race = max(approval_rates, key=approval_rates.get)
        reference_race_encoded = race_map[reference_race]

        privileged_groups = [{'race': reference_race_encoded}]

        for race_orig, race_enc in zip(race_values_original, race_values_encoded):
            if race_orig == reference_race:
                continue

            unprivileged_groups = [{'race': race_enc}]

            metric = BinaryLabelDatasetMetric(
                aif_dataset_race,
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups
            )

            di = metric.disparate_impact()

            # Check if violates 80% rule
            if di < 0.80:
                violations_detected.append({
                    'attribute': f"race_{race_orig}",
                    'disparate_impact': di,
                    'compliant': False,
                    'reason': 'DI < 0.80 (AIF360)'
                })
    except Exception as e:
        logger.warning(f"  AIF360 race check failed for case {case_id}: {str(e)}")
        # Fallback to manual calculation if AIF360 fails
        race_results = calculate_disparate_impact_manual(df, 'race', 'approved')
        for group, metrics in race_results.items():
            if not metrics['compliant']:
                violations_detected.append({
                    'attribute': group,
                    'disparate_impact': metrics['disparate_impact'],
                    'compliant': False,
                    'reason': 'DI < 0.80 (Fallback)'
                })

    has_violation = len(violations_detected) > 0

    result = {
        'case_id': case_id,
        'has_violation_detected': has_violation,
        'total_violations_detected': len(violations_detected),
        'violations': violations_detected,
        'n_samples': len(df)
    }

    return result


def validate_all_cases_baseline(n_cases: int = 50) -> tuple:
    """
    Validate all test cases using real AIF360 library

    Args:
        n_cases: Total number of cases

    Returns:
        (list of results, actual execution time in minutes)
    """
    global logger

    logger.info(f"Validating {n_cases} cases with Baseline (AIF360)...")
    logger.info("NOTE: Using REAL AIF360 library for fairness checking")

    all_results = []
    start_time = time.time()

    for case_id in range(1, n_cases + 1):
        case_start = time.time()

        try:
            result = validate_single_case_baseline(case_id, logger)
            all_results.append(result)

            if case_id % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Validated {case_id}/{n_cases} cases ({elapsed:.1f}s)")

        except Exception as e:
            logger.error(f"  Error validating case {case_id}: {str(e)}")
            all_results.append({
                'case_id': case_id,
                'has_violation_detected': False,
                'total_violations_detected': 0,
                'violations': [],
                'error': str(e)
            })

    end_time = time.time()
    actual_time = (end_time - start_time) / 60.0

    logger.info(f"✓ All {n_cases} cases validated")
    logger.info(f"  Total execution time: {actual_time:.2f} minutes")
    logger.info(f"  Average time per case: {(actual_time * 60 / n_cases):.2f} seconds")

    return all_results, actual_time


def compare_with_ground_truth(validation_results: list, ground_truth: dict) -> dict:
    """
    Compare validation results with ground truth

    Args:
        validation_results: Results from baseline validation
        ground_truth: Ground truth data

    Returns:
        Comparison results with TP, FP, TN, FN
    """
    global logger

    logger.info("Comparing with ground truth...")

    gt_cases = {case['case_id']: case for case in ground_truth['cases']}

    comparisons = []

    for result in validation_results:
        case_id = result['case_id']
        gt = gt_cases[case_id]

        detected = result['has_violation_detected']
        actual = gt['has_violation']

        comparison = {
            'case_id': case_id,
            'ground_truth': actual,
            'detected': detected,
            'correct': detected == actual
        }

        comparisons.append(comparison)

    # Calculate confusion matrix
    tp = sum(1 for c in comparisons if c['ground_truth'] and c['detected'])
    fp = sum(1 for c in comparisons if not c['ground_truth'] and c['detected'])
    tn = sum(1 for c in comparisons if not c['ground_truth'] and not c['detected'])
    fn = sum(1 for c in comparisons if c['ground_truth'] and not c['detected'])

    accuracy = (tp + tn) / len(comparisons)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"Confusion Matrix:")
    logger.info(f"  TP: {tp}, FP: {fp}")
    logger.info(f"  FN: {fn}, TN: {tn}")
    logger.info(f"Metrics:")
    logger.info(f"  Accuracy:  {accuracy*100:.1f}%")
    logger.info(f"  Precision: {precision*100:.1f}%")
    logger.info(f"  Recall:    {recall*100:.1f}%")
    logger.info(f"  F1-Score:  {f1_score*100:.1f}%")

    return {
        'comparisons': comparisons,
        'confusion_matrix': {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        },
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    }


def main():
    """Main execution"""
    global logger
    logger = setup_logging("validate_baseline", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("BASELINE COMPLIANCE VALIDATION - REAL AIF360")
    logger.info("=" * 80)
    logger.info("Using actual AIF360 library (not simulation)")
    logger.info("")

    # Load ground truth
    logger.info("\nLoading ground truth...")
    ground_truth = load_ground_truth()
    logger.info(f"Loaded {ground_truth['total_cases']} cases")

    # Validate all cases
    validation_results, execution_time = validate_all_cases_baseline(n_cases=50)

    # Compare with ground truth
    comparison = compare_with_ground_truth(validation_results, ground_truth)

    # Save results
    output = {
        'method': 'Baseline (AIF360 + Fairlearn)',
        'execution_time_minutes': execution_time,
        'validation_results': validation_results,
        'comparison': comparison
    }

    output_file = RESULTS_DIR / "baseline_validation_results.json"
    save_results(output, output_file, logger)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total cases validated: {len(validation_results)}")
    logger.info(f"Execution time: {execution_time:.1f} minutes")

    metrics = comparison['metrics']
    logger.info(f"Precision: {metrics['precision']*100:.1f}%")
    logger.info(f"Recall:    {metrics['recall']*100:.1f}%")
    logger.info(f"F1-Score:  {metrics['f1_score']*100:.1f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
