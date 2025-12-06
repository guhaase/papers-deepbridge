"""
Run All Case Studies

Executes all 6 case studies sequentially and collects results
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

# Import all case study modules
import case_study_credit
import case_study_hiring
import case_study_healthcare
import case_study_mortgage
import case_study_insurance
import case_study_fraud

from utils import setup_logging

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"


def main():
    """Run all case studies"""
    logger = setup_logging("run_all_cases", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("RUNNING ALL CASE STUDIES")
    logger.info("=" * 80)

    start_time = time.time()

    case_studies = [
        ("Credit Scoring", case_study_credit),
        ("Hiring", case_study_hiring),
        ("Healthcare", case_study_healthcare),
        ("Mortgage", case_study_mortgage),
        ("Insurance", case_study_insurance),
        ("Fraud Detection", case_study_fraud),
    ]

    results_summary = []

    for i, (name, module) in enumerate(case_studies, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"CASE STUDY {i}/6: {name}")
        logger.info("=" * 80)

        case_start = time.time()

        try:
            result = module.main()
            case_time = (time.time() - case_start) / 60

            results_summary.append({
                'name': name,
                'status': 'SUCCESS',
                'time': case_time,
                'violations': result['n_violations'],
                'samples': result['n_samples']
            })

            logger.info(f"✓ {name} completed in {case_time:.2f} minutes")

        except Exception as e:
            case_time = (time.time() - case_start) / 60
            logger.error(f"✗ {name} failed: {str(e)}")

            results_summary.append({
                'name': name,
                'status': 'FAILED',
                'time': case_time,
                'error': str(e)
            })

    total_time = (time.time() - start_time) / 60

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY OF ALL CASE STUDIES")
    logger.info("=" * 80)

    for result in results_summary:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        logger.info(f"{status_symbol} {result['name']:20s} - {result['time']:6.2f} min - "
                   f"Violations: {result.get('violations', 'N/A')}")

    logger.info("")
    logger.info(f"Total execution time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")

    successful = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
    logger.info(f"Successful: {successful}/6")

    if successful == 6:
        logger.info("")
        logger.info("✓ All case studies completed successfully!")
        logger.info("")
        logger.info("Next step: Run aggregate_analysis.py to generate tables and figures")
    else:
        logger.warning(f"⚠ {6 - successful} case studies failed")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
