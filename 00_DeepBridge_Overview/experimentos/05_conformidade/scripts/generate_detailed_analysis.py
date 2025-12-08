"""
Generate Detailed Analysis and Visualizations for Corrected Experiment 5

Creates comprehensive visualizations and statistical analysis comparing
DeepBridge with real AIF360 baseline.

Autor: DeepBridge Team
Data: 2025-12-07
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

FIGURES_DIR.mkdir(exist_ok=True)

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results():
    """Load validation results"""
    with open(RESULTS_DIR / "deepbridge_validation_results.json", 'r') as f:
        deepbridge_results = json.load(f)

    with open(RESULTS_DIR / "baseline_validation_results.json", 'r') as f:
        baseline_results = json.load(f)

    with open(RESULTS_DIR / "compliance_ground_truth.json", 'r') as f:
        ground_truth = json.load(f)

    return deepbridge_results, baseline_results, ground_truth


def analyze_violation_patterns(deepbridge_results, baseline_results, ground_truth, logger):
    """Analyze patterns in violations detected"""
    logger.info("Analyzing violation patterns...")

    gt_cases = {case['case_id']: case for case in ground_truth['cases']}

    # Analyze violation types
    violation_types_db = {}
    violation_types_bl = {}

    for result in deepbridge_results['validation_results']:
        case_id = result['case_id']
        for violation in result['violations']:
            attr = violation['attribute']
            if attr not in violation_types_db:
                violation_types_db[attr] = 0
            violation_types_db[attr] += 1

    for result in baseline_results['validation_results']:
        case_id = result['case_id']
        for violation in result['violations']:
            attr = violation['attribute']
            if attr not in violation_types_bl:
                violation_types_bl[attr] = 0
            violation_types_bl[attr] += 1

    logger.info(f"DeepBridge detected violations: {violation_types_db}")
    logger.info(f"Baseline detected violations: {violation_types_bl}")

    return violation_types_db, violation_types_bl


def plot_violation_distribution(violation_types_db, violation_types_bl):
    """Plot distribution of violations by type"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # DeepBridge
    if violation_types_db:
        attrs_db = list(violation_types_db.keys())
        counts_db = list(violation_types_db.values())
        ax1.bar(range(len(attrs_db)), counts_db, color='#2ecc71', alpha=0.7)
        ax1.set_xticks(range(len(attrs_db)))
        ax1.set_xticklabels(attrs_db, rotation=45, ha='right')
        ax1.set_ylabel('Number of Cases', fontsize=12)
        ax1.set_title('DeepBridge - Violations by Attribute', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

    # Baseline
    if violation_types_bl:
        attrs_bl = list(violation_types_bl.keys())
        counts_bl = list(violation_types_bl.values())
        ax2.bar(range(len(attrs_bl)), counts_bl, color='#3498db', alpha=0.7)
        ax2.set_xticks(range(len(attrs_bl)))
        ax2.set_xticklabels(attrs_bl, rotation=45, ha='right')
        ax2.set_ylabel('Number of Cases', fontsize=12)
        ax2.set_title('AIF360 Baseline - Violations by Attribute', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'violation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_disparate_impact_comparison(deepbridge_results, baseline_results):
    """Plot DI values comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all DI values
    di_values_db = []
    di_values_bl = []
    case_labels = []

    for result_db, result_bl in zip(
        deepbridge_results['validation_results'],
        baseline_results['validation_results']
    ):
        case_id = result_db['case_id']

        if result_db['violations']:
            for violation in result_db['violations']:
                di_values_db.append(violation['disparate_impact'])

                # Find matching violation in baseline
                matching = [v for v in result_bl['violations']
                           if v['attribute'] == violation['attribute']]
                if matching:
                    di_values_bl.append(matching[0]['disparate_impact'])
                else:
                    di_values_bl.append(0.80)  # No violation

                case_labels.append(f"Case {case_id}\n{violation['attribute']}")

    if di_values_db:
        x = np.arange(len(case_labels))
        width = 0.35

        ax.bar(x - width/2, di_values_db, width, label='DeepBridge', color='#2ecc71', alpha=0.7)
        ax.bar(x + width/2, di_values_bl, width, label='AIF360', color='#3498db', alpha=0.7)

        # Threshold line
        ax.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='EEOC Threshold (0.80)')

        ax.set_xlabel('Violation Cases', fontsize=12)
        ax.set_ylabel('Disparate Impact', fontsize=12)
        ax.set_title('Disparate Impact Values - DeepBridge vs AIF360', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'disparate_impact_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_detection_accuracy_by_case(deepbridge_results, baseline_results, ground_truth):
    """Plot detection accuracy for each case"""
    gt_cases = {case['case_id']: case for case in ground_truth['cases']}

    case_ids = []
    db_correct = []
    bl_correct = []
    has_violation = []

    for result_db, result_bl in zip(
        deepbridge_results['validation_results'],
        baseline_results['validation_results']
    ):
        case_id = result_db['case_id']
        gt = gt_cases[case_id]

        case_ids.append(case_id)
        db_correct.append(1 if result_db['has_violation_detected'] == gt['has_violation'] else 0)
        bl_correct.append(1 if result_bl['has_violation_detected'] == gt['has_violation'] else 0)
        has_violation.append(gt['has_violation'])

    fig, ax = plt.subplots(figsize=(15, 5))

    x = np.arange(len(case_ids))

    # Color by violation presence
    colors_db = ['#2ecc71' if correct else '#e74c3c' for correct in db_correct]
    colors_bl = ['#3498db' if correct else '#e74c3c' for correct in bl_correct]

    width = 0.35
    ax.bar(x - width/2, db_correct, width, label='DeepBridge', color=colors_db, alpha=0.7)
    ax.bar(x + width/2, bl_correct, width, label='AIF360', color=colors_bl, alpha=0.7)

    # Mark cases with violations
    for i, has_v in enumerate(has_violation):
        if has_v:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color='yellow')

    ax.set_xlabel('Case ID', fontsize=12)
    ax.set_ylabel('Correct Detection (1=Yes, 0=No)', fontsize=12)
    ax.set_title('Detection Accuracy by Case (Yellow = Has Violation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(case_ids[::2])
    ax.legend(fontsize=10)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'detection_accuracy_by_case.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_execution_time_detailed(deepbridge_results, baseline_results):
    """Plot detailed execution time comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['DeepBridge', 'AIF360']
    times = [
        deepbridge_results['execution_time_minutes'] * 60,  # Convert to seconds
        baseline_results['execution_time_minutes'] * 60
    ]
    colors = ['#2ecc71', '#3498db']

    # Bar plot
    bars = ax1.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Total Execution Time (50 cases)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Per-case time
    times_per_case = [t / 50 for t in times]
    bars2 = ax2.bar(methods, times_per_case, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Time per Case (seconds)', fontsize=12)
    ax2.set_title('Average Time per Case', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, time in zip(bars2, times_per_case):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speedup annotation
    speedup = times[1] / times[0]
    ax2.text(0.5, max(times_per_case) * 0.8,
            f'Speedup: {speedup:.1f}×',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'execution_time_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_heatmap(deepbridge_results, baseline_results):
    """Plot confusion matrices as heatmaps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # DeepBridge
    cm_db = deepbridge_results['comparison']['confusion_matrix']
    cm_db_array = np.array([
        [cm_db['tn'], cm_db['fp']],
        [cm_db['fn'], cm_db['tp']]
    ])

    sns.heatmap(cm_db_array, annot=True, fmt='d', cmap='Greens', ax=ax1,
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('DeepBridge - Confusion Matrix', fontsize=14, fontweight='bold')

    # Baseline
    cm_bl = baseline_results['comparison']['confusion_matrix']
    cm_bl_array = np.array([
        [cm_bl['tn'], cm_bl['fp']],
        [cm_bl['fn'], cm_bl['tp']]
    ])

    sns.heatmap(cm_bl_array, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'],
                cbar_kws={'label': 'Count'})
    ax2.set_title('AIF360 - Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_radar(deepbridge_results, baseline_results):
    """Plot metrics comparison as radar chart"""
    metrics_db = deepbridge_results['comparison']['metrics']
    metrics_bl = baseline_results['comparison']['metrics']

    categories = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values_db = [
        metrics_db['precision'],
        metrics_db['recall'],
        metrics_db['f1_score'],
        metrics_db['accuracy']
    ]
    values_bl = [
        metrics_bl['precision'],
        metrics_bl['recall'],
        metrics_bl['f1_score'],
        metrics_bl['accuracy']
    ]

    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_db += values_db[:1]
    values_bl += values_bl[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, values_db, 'o-', linewidth=2, label='DeepBridge', color='#2ecc71')
    ax.fill(angles, values_db, alpha=0.25, color='#2ecc71')

    ax.plot(angles, values_bl, 'o-', linewidth=2, label='AIF360', color='#3498db')
    ax.fill(angles, values_bl, alpha=0.25, color='#3498db')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_table(deepbridge_results, baseline_results, ground_truth):
    """Generate summary statistics table"""
    summary = {
        'Total Cases': len(ground_truth['cases']),
        'Cases with Violations': ground_truth['violation_cases'],
        'Cases without Violations': ground_truth['clean_cases'],
        '': '',
        'DeepBridge Metrics': {
            'Precision': f"{deepbridge_results['comparison']['metrics']['precision']*100:.1f}%",
            'Recall': f"{deepbridge_results['comparison']['metrics']['recall']*100:.1f}%",
            'F1-Score': f"{deepbridge_results['comparison']['metrics']['f1_score']*100:.1f}%",
            'Execution Time': f"{deepbridge_results['execution_time_minutes']*60:.2f}s",
        },
        'AIF360 Metrics': {
            'Precision': f"{baseline_results['comparison']['metrics']['precision']*100:.1f}%",
            'Recall': f"{baseline_results['comparison']['metrics']['recall']*100:.1f}%",
            'F1-Score': f"{baseline_results['comparison']['metrics']['f1_score']*100:.1f}%",
            'Execution Time': f"{baseline_results['execution_time_minutes']*60:.2f}s",
        }
    }

    return summary


def main():
    """Main execution"""
    global logger
    logger = setup_logging("detailed_analysis", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("GENERATING DETAILED ANALYSIS AND VISUALIZATIONS")
    logger.info("=" * 80)

    # Load results
    logger.info("\nLoading results...")
    deepbridge_results, baseline_results, ground_truth = load_results()
    logger.info("✓ Results loaded")

    # Analyze violation patterns
    violation_types_db, violation_types_bl = analyze_violation_patterns(
        deepbridge_results, baseline_results, ground_truth, logger
    )

    # Generate visualizations
    logger.info("\nGenerating detailed visualizations...")

    logger.info("  1. Violation distribution...")
    plot_violation_distribution(violation_types_db, violation_types_bl)

    logger.info("  2. Disparate Impact comparison...")
    plot_disparate_impact_comparison(deepbridge_results, baseline_results)

    logger.info("  3. Detection accuracy by case...")
    plot_detection_accuracy_by_case(deepbridge_results, baseline_results, ground_truth)

    logger.info("  4. Execution time detailed...")
    plot_execution_time_detailed(deepbridge_results, baseline_results)

    logger.info("  5. Confusion matrix heatmap...")
    plot_confusion_matrix_heatmap(deepbridge_results, baseline_results)

    logger.info("  6. Metrics radar chart...")
    plot_metrics_radar(deepbridge_results, baseline_results)

    logger.info(f"✓ All visualizations saved to {FIGURES_DIR}")

    # Generate summary
    logger.info("\nGenerating summary table...")
    summary = generate_summary_table(deepbridge_results, baseline_results, ground_truth)

    summary_file = RESULTS_DIR / "detailed_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved to {summary_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total cases: {summary['Total Cases']}")
    logger.info(f"Cases with violations: {summary['Cases with Violations']}")
    logger.info(f"Cases without violations: {summary['Cases without Violations']}")
    logger.info("")
    logger.info("DeepBridge:")
    for key, value in summary['DeepBridge Metrics'].items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    logger.info("AIF360:")
    for key, value in summary['AIF360 Metrics'].items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    logger.info("\n✓ Detailed analysis complete!")


if __name__ == "__main__":
    main()
