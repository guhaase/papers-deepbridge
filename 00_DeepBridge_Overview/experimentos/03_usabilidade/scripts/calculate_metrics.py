"""
Calculate Usability Metrics

Loads raw data and calculates all usability metrics:
- SUS scores and interpretation
- NASA TLX scores and workload levels
- Success rates
- Completion times
- Error rates
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging, save_results,
    interpret_sus_score, interpret_nasa_tlx,
    calculate_success_rate, calculate_time_statistics
)

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"


def calculate_all_metrics(logger):
    """Calculate all usability metrics"""

    # Load data
    logger.info("Loading data...")
    sus_df = pd.read_csv(RESULTS_DIR / "03_usability_sus_scores.csv")
    tlx_df = pd.read_csv(RESULTS_DIR / "03_usability_nasa_tlx.csv")
    times_df = pd.read_csv(RESULTS_DIR / "03_usability_task_times.csv")
    errors_df = pd.read_csv(RESULTS_DIR / "03_usability_errors.csv")

    logger.info(f"Loaded data for {len(sus_df)} participants")

    # SUS Metrics
    logger.info("Calculating SUS metrics...")
    sus_scores = sus_df['sus_score'].values

    sus_metrics = {
        'mean': float(np.mean(sus_scores)),
        'std': float(np.std(sus_scores, ddof=1)),
        'median': float(np.median(sus_scores)),
        'min': float(np.min(sus_scores)),
        'max': float(np.max(sus_scores)),
        'n': len(sus_scores),
        'scores': sus_scores.tolist()
    }

    # Interpret mean SUS score
    sus_interpretation = interpret_sus_score(sus_metrics['mean'])
    sus_metrics['interpretation'] = sus_interpretation

    logger.info(f"  Mean SUS: {sus_metrics['mean']:.2f} ± {sus_metrics['std']:.2f}")
    logger.info(f"  Interpretation: {sus_interpretation['adjective']} ({sus_interpretation['grade']})")

    # NASA TLX Metrics
    logger.info("Calculating NASA TLX metrics...")
    tlx_scores = tlx_df['tlx_overall'].values

    tlx_metrics = {
        'overall': {
            'mean': float(np.mean(tlx_scores)),
            'std': float(np.std(tlx_scores, ddof=1)),
            'median': float(np.median(tlx_scores)),
            'min': float(np.min(tlx_scores)),
            'max': float(np.max(tlx_scores)),
        },
        'dimensions': {
            'mental_demand': {
                'mean': float(tlx_df['mental_demand'].mean()),
                'std': float(tlx_df['mental_demand'].std())
            },
            'physical_demand': {
                'mean': float(tlx_df['physical_demand'].mean()),
                'std': float(tlx_df['physical_demand'].std())
            },
            'temporal_demand': {
                'mean': float(tlx_df['temporal_demand'].mean()),
                'std': float(tlx_df['temporal_demand'].std())
            },
            'performance': {
                'mean': float(tlx_df['performance'].mean()),
                'std': float(tlx_df['performance'].std())
            },
            'effort': {
                'mean': float(tlx_df['effort'].mean()),
                'std': float(tlx_df['effort'].std())
            },
            'frustration': {
                'mean': float(tlx_df['frustration'].mean()),
                'std': float(tlx_df['frustration'].std())
            }
        },
        'interpretation': interpret_nasa_tlx(np.mean(tlx_scores))
    }

    logger.info(f"  Mean TLX: {tlx_metrics['overall']['mean']:.2f} ± {tlx_metrics['overall']['std']:.2f}")
    logger.info(f"  Interpretation: {tlx_metrics['interpretation']}")

    # Success Rate Metrics
    logger.info("Calculating success rates...")

    overall_success = calculate_success_rate(times_df['completed_all_tasks'].tolist())
    task1_success = calculate_success_rate(times_df['task1_success'].tolist())
    task2_success = calculate_success_rate(times_df['task2_success'].tolist())
    task3_success = calculate_success_rate(times_df['task3_success'].tolist())

    success_metrics = {
        'overall': overall_success,
        'by_task': {
            'task1_fairness': task1_success,
            'task2_report': task2_success,
            'task3_cicd': task3_success
        }
    }

    logger.info(f"  Overall Success Rate: {overall_success['success_rate']:.1f}% ({overall_success['n_success']}/{overall_success['n_total']})")
    logger.info(f"  Task 1: {task1_success['success_rate']:.1f}%")
    logger.info(f"  Task 2: {task2_success['success_rate']:.1f}%")
    logger.info(f"  Task 3: {task3_success['success_rate']:.1f}%")

    # Time Metrics
    logger.info("Calculating time metrics...")

    # Filter successful completions for time analysis
    successful_times = times_df[times_df['completed_all_tasks'] == True]

    time_metrics = {
        'total': calculate_time_statistics(successful_times['total_time'].tolist()),
        'task1_fairness': calculate_time_statistics(times_df['task1_fairness_time'].dropna().tolist()),
        'task2_report': calculate_time_statistics(times_df['task2_report_time'].dropna().tolist()),
        'task3_cicd': calculate_time_statistics(times_df['task3_cicd_time'].dropna().tolist())
    }

    logger.info(f"  Mean Total Time: {time_metrics['total']['mean']:.2f} ± {time_metrics['total']['std']:.2f} min")
    logger.info(f"  Task 1: {time_metrics['task1_fairness']['mean']:.2f} ± {time_metrics['task1_fairness']['std']:.2f} min")
    logger.info(f"  Task 2: {time_metrics['task2_report']['mean']:.2f} ± {time_metrics['task2_report']['std']:.2f} min")
    logger.info(f"  Task 3: {time_metrics['task3_cicd']['mean']:.2f} ± {time_metrics['task3_cicd']['std']:.2f} min")

    # Error Metrics
    logger.info("Calculating error metrics...")

    error_metrics = {
        'total': calculate_time_statistics(errors_df['total_errors'].tolist()),
        'by_category': {
            'syntax': calculate_time_statistics(errors_df['syntax_errors'].tolist()),
            'api': calculate_time_statistics(errors_df['api_errors'].tolist()),
            'conceptual': calculate_time_statistics(errors_df['conceptual_errors'].tolist()),
            'other': calculate_time_statistics(errors_df['other_errors'].tolist())
        }
    }

    logger.info(f"  Mean Total Errors: {error_metrics['total']['mean']:.2f} ± {error_metrics['total']['std']:.2f}")

    # Compile all metrics
    all_metrics = {
        'sus': sus_metrics,
        'nasa_tlx': tlx_metrics,
        'success_rate': success_metrics,
        'completion_time': time_metrics,
        'errors': error_metrics,
        'n_participants': len(sus_df)
    }

    return all_metrics


def main():
    """Main execution"""
    logger = setup_logging("calculate_metrics", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("CALCULATING USABILITY METRICS")
    logger.info("=" * 80)

    # Calculate metrics
    metrics = calculate_all_metrics(logger)

    # Save results
    output_file = RESULTS_DIR / "03_usability_metrics.json"
    save_results(metrics, output_file, logger)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("METRICS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"SUS Score: {metrics['sus']['mean']:.2f} ± {metrics['sus']['std']:.2f} ({metrics['sus']['interpretation']['adjective']})")
    logger.info(f"NASA TLX: {metrics['nasa_tlx']['overall']['mean']:.2f} ± {metrics['nasa_tlx']['overall']['std']:.2f} ({metrics['nasa_tlx']['interpretation']})")
    logger.info(f"Success Rate: {metrics['success_rate']['overall']['success_rate']:.1f}%")
    logger.info(f"Mean Time: {metrics['completion_time']['total']['mean']:.2f} min")
    logger.info(f"Mean Errors: {metrics['errors']['total']['mean']:.2f}")
    logger.info("=" * 80)

    # Compare to targets
    logger.info("")
    logger.info("COMPARISON TO TARGETS")
    logger.info("=" * 80)

    targets = {
        'SUS Score': (85, metrics['sus']['mean']),
        'NASA TLX': (30, metrics['nasa_tlx']['overall']['mean']),
        'Success Rate': (90, metrics['success_rate']['overall']['success_rate']),
        'Mean Time': (15, metrics['completion_time']['total']['mean']),
        'Mean Errors': (2, metrics['errors']['total']['mean'])
    }

    for metric, (target, actual) in targets.items():
        if metric in ['SUS Score', 'Success Rate']:
            # Higher is better
            status = "✓" if actual >= target else "✗"
        else:
            # Lower is better
            status = "✓" if actual <= target else "✗"

        logger.info(f"{status} {metric}: Target {target}, Actual {actual:.2f}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
