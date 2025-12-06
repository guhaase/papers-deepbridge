"""
Generate Ground Truth Dataset for Compliance Testing

Creates 50 test cases with known violations:
- 25 cases WITH violations
- 25 cases WITHOUT violations

Each case includes:
- Synthetic dataset
- Ground truth compliance status
- Expected violations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def generate_case(
    case_id: int,
    has_violation: bool,
    n_samples: int = 1000,
    seed: int = None
) -> tuple:
    """
    Generate a single test case

    Args:
        case_id: Case identifier
        has_violation: Whether to inject a violation
        n_samples: Number of samples
        seed: Random seed

    Returns:
        (dataframe, ground_truth_dict)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.lognormal(10, 1, n_samples)
    credit_score = np.random.normal(700, 100, n_samples).clip(300, 850)

    # Protected attributes
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5])
    race = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian'],
        n_samples,
        p=[0.60, 0.15, 0.15, 0.10]
    )

    if has_violation:
        # INJECT VIOLATION: Disparate Impact in gender
        # P(approve | M) = 0.60, P(approve | F) = 0.40
        # DI = 0.40 / 0.60 = 0.67 < 0.80
        prob_approve = np.where(
            gender == 'M',
            0.60 + 0.2 * (credit_score - 700) / 100,
            0.40 + 0.2 * (credit_score - 700) / 100
        )
        prob_approve = np.clip(prob_approve, 0, 1)

        # Expected DI
        expected_di_gender = 0.67

        # Also inject race violation (Black group)
        race_bias = np.where(race == 'Black', -0.10, 0.0)
        prob_approve = np.clip(prob_approve + race_bias, 0, 1)
        expected_di_race_black = 0.75

        violations = {
            'gender': {
                'disparate_impact': expected_di_gender,
                'eeoc_compliant': False,
                'reason': 'DI < 0.80'
            },
            'race_black': {
                'disparate_impact': expected_di_race_black,
                'eeoc_compliant': False,
                'reason': 'DI < 0.80'
            }
        }
        total_violations = 2

    else:
        # NO VIOLATION: Equal opportunity
        prob_approve = 0.50 + 0.2 * (credit_score - 700) / 100
        prob_approve = np.clip(prob_approve, 0, 1)

        violations = {}
        total_violations = 0

    # Generate decisions
    approved = np.random.binomial(1, prob_approve)

    # Create DataFrame
    df = pd.DataFrame({
        'case_id': case_id,
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'gender': gender,
        'race': race,
        'approved': approved
    })

    # Ground truth
    ground_truth = {
        'case_id': case_id,
        'has_violation': has_violation,
        'total_violations': total_violations,
        'violations': violations,
        'n_samples': n_samples,
        'protected_attributes': ['gender', 'race']
    }

    return df, ground_truth


def generate_all_cases(n_cases: int = 50, seed: int = 42) -> tuple:
    """
    Generate all test cases

    Args:
        n_cases: Total number of cases (must be even)
        seed: Random seed

    Returns:
        (list_of_dataframes, list_of_ground_truths)
    """
    assert n_cases % 2 == 0, "n_cases must be even"

    n_violations = n_cases // 2
    n_clean = n_cases // 2

    all_cases = []
    all_ground_truths = []

    logger.info(f"Generating {n_cases} test cases...")
    logger.info(f"  - {n_violations} cases WITH violations")
    logger.info(f"  - {n_clean} cases WITHOUT violations")

    # Generate violation cases
    for i in range(n_violations):
        case_id = i + 1
        df, gt = generate_case(
            case_id=case_id,
            has_violation=True,
            seed=seed + case_id
        )
        all_cases.append(df)
        all_ground_truths.append(gt)

        if (i + 1) % 5 == 0:
            logger.info(f"  Generated {i+1}/{n_violations} violation cases")

    # Generate clean cases
    for i in range(n_clean):
        case_id = n_violations + i + 1
        df, gt = generate_case(
            case_id=case_id,
            has_violation=False,
            seed=seed + case_id
        )
        all_cases.append(df)
        all_ground_truths.append(gt)

        if (i + 1) % 5 == 0:
            logger.info(f"  Generated {i+1}/{n_clean} clean cases")

    logger.info(f"✓ All {n_cases} cases generated successfully")

    return all_cases, all_ground_truths


def main():
    """Main execution"""
    global logger
    logger = setup_logging("generate_ground_truth", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("GENERATING COMPLIANCE GROUND TRUTH DATASET")
    logger.info("=" * 80)

    # Generate all cases
    all_cases, all_ground_truths = generate_all_cases(n_cases=50, seed=42)

    # Save individual datasets
    logger.info("\nSaving individual case datasets...")
    for i, df in enumerate(all_cases):
        case_id = i + 1
        output_file = DATA_DIR / f"case_{case_id:02d}.csv"
        df.to_csv(output_file, index=False)

        if (i + 1) % 10 == 0:
            logger.info(f"  Saved {i+1}/50 datasets")

    logger.info(f"✓ All datasets saved to {DATA_DIR}")

    # Save ground truth
    ground_truth_output = {
        'total_cases': len(all_ground_truths),
        'violation_cases': sum(1 for gt in all_ground_truths if gt['has_violation']),
        'clean_cases': sum(1 for gt in all_ground_truths if not gt['has_violation']),
        'cases': all_ground_truths
    }

    gt_file = RESULTS_DIR / "compliance_ground_truth.json"
    save_results(ground_truth_output, gt_file, logger)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total cases: {len(all_ground_truths)}")
    logger.info(f"Cases with violations: {ground_truth_output['violation_cases']}")
    logger.info(f"Cases without violations: {ground_truth_output['clean_cases']}")
    logger.info(f"Datasets saved to: {DATA_DIR}")
    logger.info(f"Ground truth saved to: {gt_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
