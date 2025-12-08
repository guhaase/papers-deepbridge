"""
Analyze and Visualize Compliance Validation Results

Compares DeepBridge vs Baseline and generates:
- Comparison tables (LaTeX)
- Visualizations (confusion matrices, metrics comparison)
- Statistical tests

Autor: DeepBridge Team
Data: 2025-12-07
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
LOGS_DIR = BASE_DIR / "logs"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def load_validation_results():
    """Load validation results from DeepBridge and Baseline"""
    deepbridge_file = RESULTS_DIR / "deepbridge_validation_results.json"
    baseline_file = RESULTS_DIR / "baseline_validation_results.json"

    with open(deepbridge_file, 'r') as f:
        deepbridge = json.load(f)

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    return deepbridge, baseline


def generate_comparison_table(db_results, bl_results, output_file):
    """Generate LaTeX comparison table"""
    db_metrics = db_results['comparison']['metrics']
    bl_metrics = bl_results['comparison']['metrics']
    db_cm = db_results['comparison']['confusion_matrix']
    bl_cm = bl_results['comparison']['confusion_matrix']

    db_time = db_results['execution_time_minutes']
    bl_time = bl_results['execution_time_minutes']

    time_reduction = (bl_time - db_time) / bl_time * 100

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Resultados de Conformidade Regulatória - Comparação DeepBridge vs Baseline}")
    latex.append("\\label{tab:compliance_results}")
    latex.append("\\begin{tabular}{lrrrrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Método} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\")
    latex.append("\\midrule")

    # DeepBridge row
    latex.append(
        f"\\textbf{{DeepBridge}} & "
        f"{db_cm['tp']} & {db_cm['fp']} & {db_cm['fn']} & "
        f"\\textbf{{{db_metrics['precision']*100:.1f}\\%}} & "
        f"\\textbf{{{db_metrics['recall']*100:.1f}\\%}} & "
        f"\\textbf{{{db_metrics['f1_score']*100:.1f}\\%}} \\\\"
    )

    # Baseline row
    latex.append(
        f"Baseline & "
        f"{bl_cm['tp']} & {bl_cm['fp']} & {bl_cm['fn']} & "
        f"{bl_metrics['precision']*100:.1f}\\% & "
        f"{bl_metrics['recall']*100:.1f}\\% & "
        f"{bl_metrics['f1_score']*100:.1f}\\% \\\\"
    )

    latex.append("\\midrule")
    latex.append(
        f"Melhoria & - & - & - & "
        f"+{(db_metrics['precision']-bl_metrics['precision'])*100:.1f}pp & "
        f"+{(db_metrics['recall']-bl_metrics['recall'])*100:.1f}pp & "
        f"+{(db_metrics['f1_score']-bl_metrics['f1_score'])*100:.1f}pp \\\\"
    )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Time comparison table
    latex.append("")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Tempo de Auditoria de Conformidade}")
    latex.append("\\label{tab:compliance_time}")
    latex.append("\\begin{tabular}{lrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Método} & \\textbf{Tempo (min)} & \\textbf{Redução} \\\\")
    latex.append("\\midrule")
    latex.append(f"\\textbf{{DeepBridge}} & \\textbf{{{db_time:.1f}}} & - \\\\")
    latex.append(f"Baseline (Manual) & {bl_time:.1f} & - \\\\")
    latex.append("\\midrule")
    latex.append(f"Redução Temporal & - & \\textbf{{{time_reduction:.0f}\\%}} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    with open(output_file, 'w') as f:
        f.write(latex_content)

    return latex_content


def plot_confusion_matrices(db_results, bl_results, output_file):
    """Plot confusion matrices side by side"""
    db_cm = db_results['comparison']['confusion_matrix']
    bl_cm = bl_results['comparison']['confusion_matrix']

    # Create confusion matrices as 2x2 arrays
    db_matrix = np.array([[db_cm['tn'], db_cm['fp']],
                          [db_cm['fn'], db_cm['tp']]])

    bl_matrix = np.array([[bl_cm['tn'], bl_cm['fp']],
                          [bl_cm['fn'], bl_cm['tp']]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # DeepBridge confusion matrix
    sns.heatmap(db_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('DeepBridge - Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # Baseline confusion matrix
    sns.heatmap(bl_matrix, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'],
                cbar_kws={'label': 'Count'})
    ax2.set_title('Baseline - Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(db_results, bl_results, output_file):
    """Plot metrics comparison bar chart"""
    db_metrics = db_results['comparison']['metrics']
    bl_metrics = bl_results['comparison']['metrics']

    metrics = ['Precision', 'Recall', 'F1-Score']
    db_values = [db_metrics['precision'], db_metrics['recall'], db_metrics['f1_score']]
    bl_values = [bl_metrics['precision'], bl_metrics['recall'], bl_metrics['f1_score']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, [v*100 for v in db_values], width, label='DeepBridge',
                    color='#2E86AB', edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, [v*100 for v in bl_values], width, label='Baseline',
                    color='#A23B72', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Compliance Detection Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_comparison(db_results, bl_results, output_file):
    """Plot execution time comparison"""
    db_time = db_results['execution_time_minutes']
    bl_time = bl_results['execution_time_minutes']

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['DeepBridge', 'Baseline\n(Manual)']
    times = [db_time, bl_time]
    colors = ['#06A77D', '#D5573B']

    bars = ax.barh(methods, times, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Compliance Audit Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height()/2,
                f'{time:.1f} min',
                ha='left', va='center',
                fontsize=11, fontweight='bold')

    # Add reduction annotation
    reduction = (bl_time - db_time) / bl_time * 100
    ax.text(bl_time/2, 0.5, f'{reduction:.0f}% faster',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def perform_statistical_tests(db_results, bl_results):
    """Perform statistical tests comparing methods"""
    global logger

    logger.info("Performing statistical tests...")

    db_cm = db_results['comparison']['confusion_matrix']
    bl_cm = bl_results['comparison']['confusion_matrix']

    # McNemar's test for paired nominal data
    # Contingency table: [[both_correct, db_correct_bl_wrong],
    #                     [db_wrong_bl_correct, both_wrong]]

    n_cases = 50
    db_correct = db_cm['tp'] + db_cm['tn']
    bl_correct = bl_cm['tp'] + bl_cm['tn']

    # Simplified: test if error rates are different
    db_errors = db_cm['fp'] + db_cm['fn']
    bl_errors = bl_cm['fp'] + bl_cm['fn']

    logger.info(f"DeepBridge errors: {db_errors}/{n_cases}")
    logger.info(f"Baseline errors: {bl_errors}/{n_cases}")

    # Proportion test
    from statsmodels.stats.proportion import proportions_ztest

    count = np.array([db_errors, bl_errors])
    nobs = np.array([n_cases, n_cases])

    z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

    logger.info(f"Proportions Z-test:")
    logger.info(f"  Z-statistic: {z_stat:.4f}")
    logger.info(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        logger.info(f"  Result: Significant difference (p < 0.05)")
    else:
        logger.info(f"  Result: No significant difference (p >= 0.05)")

    return {
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def print_summary(db_results, bl_results, stats_results):
    """Print final summary"""
    print("\n" + "=" * 80)
    print("COMPLIANCE VALIDATION - FINAL RESULTS")
    print("=" * 80)
    print()

    # Metrics
    print("PERFORMANCE METRICS:")
    print("-" * 80)

    db_metrics = db_results['comparison']['metrics']
    bl_metrics = bl_results['comparison']['metrics']

    print(f"{'Metric':<15} {'DeepBridge':>12} {'Baseline':>12} {'Improvement':>15}")
    print("-" * 80)
    print(f"{'Precision':<15} {db_metrics['precision']*100:>11.1f}% {bl_metrics['precision']*100:>11.1f}% "
          f"{(db_metrics['precision']-bl_metrics['precision'])*100:>14.1f}pp")
    print(f"{'Recall':<15} {db_metrics['recall']*100:>11.1f}% {bl_metrics['recall']*100:>11.1f}% "
          f"{(db_metrics['recall']-bl_metrics['recall'])*100:>14.1f}pp")
    print(f"{'F1-Score':<15} {db_metrics['f1_score']*100:>11.1f}% {bl_metrics['f1_score']*100:>11.1f}% "
          f"{(db_metrics['f1_score']-bl_metrics['f1_score'])*100:>14.1f}pp")

    # Time
    print()
    print("EXECUTION TIME:")
    print("-" * 80)
    db_time = db_results['execution_time_minutes']
    bl_time = bl_results['execution_time_minutes']
    reduction = (bl_time - db_time) / bl_time * 100

    print(f"DeepBridge: {db_time:.1f} minutes")
    print(f"Baseline:   {bl_time:.1f} minutes")
    print(f"Reduction:  {reduction:.0f}%")

    # Statistical significance
    print()
    print("STATISTICAL TESTS:")
    print("-" * 80)
    print(f"Z-statistic: {stats_results['z_statistic']:.4f}")
    print(f"P-value:     {stats_results['p_value']:.4f}")
    print(f"Significant: {'Yes' if stats_results['significant'] else 'No'}")

    print()
    print("=" * 80)
    print()


def main():
    """Main execution"""
    global logger
    logger = setup_logging("analyze_and_visualize", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("COMPLIANCE VALIDATION - ANALYSIS AND VISUALIZATION")
    logger.info("=" * 80)

    # Load results
    logger.info("\nLoading validation results...")
    db_results, bl_results = load_validation_results()

    # Generate LaTeX tables
    logger.info("Generating LaTeX tables...")
    table_file = TABLES_DIR / "compliance_comparison.tex"
    generate_comparison_table(db_results, bl_results, table_file)
    logger.info(f"✓ Tables saved to {table_file}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    logger.info("  - Confusion matrices...")
    plot_confusion_matrices(db_results, bl_results,
                           FIGURES_DIR / "compliance_confusion_matrices.pdf")

    logger.info("  - Metrics comparison...")
    plot_metrics_comparison(db_results, bl_results,
                           FIGURES_DIR / "compliance_metrics_comparison.pdf")

    logger.info("  - Time comparison...")
    plot_time_comparison(db_results, bl_results,
                        FIGURES_DIR / "compliance_time_comparison.pdf")

    logger.info(f"✓ Figures saved to {FIGURES_DIR}")

    # Statistical tests
    logger.info("\nPerforming statistical analysis...")
    stats_results = perform_statistical_tests(db_results, bl_results)

    # Save analysis results
    analysis_output = {
        'deepbridge': db_results,
        'baseline': bl_results,
        'statistical_tests': stats_results
    }

    output_file = RESULTS_DIR / "compliance_analysis.json"
    save_results(analysis_output, output_file, logger)

    # Print summary
    print_summary(db_results, bl_results, stats_results)

    logger.info("=" * 80)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"Figures: {FIGURES_DIR}")
    logger.info(f"Tables:  {TABLES_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
