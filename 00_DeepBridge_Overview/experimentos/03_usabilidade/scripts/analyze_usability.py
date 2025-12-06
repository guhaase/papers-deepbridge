"""
Main Usability Analysis Script

Orchestrates the complete usability analysis pipeline:
1. Generate mock data (or load real data)
2. Calculate metrics
3. Statistical analysis
4. Generate visualizations
5. Create summary report
6. Generate LaTeX table
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import generate_mock_data
import calculate_metrics
import statistical_analysis
import generate_visualizations

from utils import setup_logging, load_results, save_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

TABLES_DIR.mkdir(exist_ok=True)


def generate_latex_table(logger):
    """Generate LaTeX table summarizing results"""
    if logger:
        logger.info("Generating LaTeX table...")

    # Load metrics
    metrics = load_results(RESULTS_DIR / "03_usability_metrics.json")

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Resultados do Estudo de Usabilidade}")
    latex.append("\\label{tab:usability_results}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Métrica} & \\textbf{Meta} & \\textbf{Resultado} & \\textbf{Status} \\\\")
    latex.append("\\midrule")

    # Data rows
    rows = [
        ("SUS Score", "≥ 85", f"{metrics['sus']['mean']:.1f} ± {metrics['sus']['std']:.1f}",
         "✓" if metrics['sus']['mean'] >= 85 else "✗"),

        ("NASA TLX", "≤ 30", f"{metrics['nasa_tlx']['overall']['mean']:.1f} ± {metrics['nasa_tlx']['overall']['std']:.1f}",
         "✓" if metrics['nasa_tlx']['overall']['mean'] <= 30 else "✗"),

        ("Taxa de Sucesso", "≥ 90\\%", f"{metrics['success_rate']['overall']['success_rate']:.1f}\\%",
         "✓" if metrics['success_rate']['overall']['success_rate'] >= 90 else "✗"),

        ("Tempo Médio", "≤ 15 min", f"{metrics['completion_time']['total']['mean']:.1f} ± {metrics['completion_time']['total']['std']:.1f} min",
         "✓" if metrics['completion_time']['total']['mean'] <= 15 else "✗"),

        ("Erros Médios", "≤ 2", f"{metrics['errors']['total']['mean']:.1f} ± {metrics['errors']['total']['std']:.1f}",
         "✓" if metrics['errors']['total']['mean'] <= 2 else "✗"),
    ]

    for metric, target, result, status in rows:
        latex.append(f"{metric} & {target} & {result} & {status} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    output_file = TABLES_DIR / "usability_summary.tex"
    with open(output_file, 'w') as f:
        f.write(latex_content)

    if logger:
        logger.info(f"  LaTeX table saved to {output_file}")

    return latex_content


def generate_summary_report(logger):
    """Generate comprehensive summary report"""
    if logger:
        logger.info("Generating summary report...")

    metrics = load_results(RESULTS_DIR / "03_usability_metrics.json")
    stats = load_results(RESULTS_DIR / "03_usability_statistical_analysis.json")

    report = []
    report.append("=" * 80)
    report.append("DEEPBRIDGE USABILITY STUDY - SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")

    # Participants
    report.append("PARTICIPANTS")
    report.append("-" * 80)
    report.append(f"Total: {metrics['n_participants']}")
    report.append("")

    # SUS Score
    report.append("SYSTEM USABILITY SCALE (SUS)")
    report.append("-" * 80)
    report.append(f"Score: {metrics['sus']['mean']:.2f} ± {metrics['sus']['std']:.2f}")
    report.append(f"Interpretation: {metrics['sus']['interpretation']['adjective']} ({metrics['sus']['interpretation']['grade']})")
    if 'percentile' in metrics['sus']['interpretation']:
        report.append(f"Percentile: {metrics['sus']['interpretation']['percentile']}")
    report.append(f"Range: [{metrics['sus']['min']:.1f}, {metrics['sus']['max']:.1f}]")
    report.append("")
    report.append(f"Statistical Test (vs. global average of 68):")
    report.append(f"  t-statistic: {stats['sus_ttest']['t_statistic']:.3f}")
    report.append(f"  p-value: {stats['sus_ttest']['p_value_one_sided']:.4f}")
    report.append(f"  Cohen's d: {stats['sus_ttest']['cohens_d']:.3f} ({stats['sus_ttest']['effect_size_interpretation']})")
    report.append(f"  Result: {'SIGNIFICANT' if stats['sus_ttest']['significant'] else 'NOT SIGNIFICANT'}")
    report.append("")

    # NASA TLX
    report.append("NASA TASK LOAD INDEX (TLX)")
    report.append("-" * 80)
    report.append(f"Overall: {metrics['nasa_tlx']['overall']['mean']:.2f} ± {metrics['nasa_tlx']['overall']['std']:.2f}")
    report.append(f"Interpretation: {metrics['nasa_tlx']['interpretation']}")
    report.append("")
    report.append("Dimensions:")
    for dim, values in metrics['nasa_tlx']['dimensions'].items():
        dim_name = dim.replace('_', ' ').title()
        report.append(f"  {dim_name:20s}: {values['mean']:.1f} ± {values['std']:.1f}")
    report.append("")

    # Success Rate
    report.append("SUCCESS RATE")
    report.append("-" * 80)
    overall = metrics['success_rate']['overall']
    report.append(f"Overall: {overall['success_rate']:.1f}% ({overall['n_success']}/{overall['n_total']})")
    report.append(f"95% CI: [{overall['ci_95_lower']:.1f}%, {overall['ci_95_upper']:.1f}%]")
    report.append("")
    report.append("By Task:")
    for task_name, task_data in metrics['success_rate']['by_task'].items():
        report.append(f"  {task_name:20s}: {task_data['success_rate']:.1f}% ({task_data['n_success']}/{task_data['n_total']})")
    report.append("")

    # Completion Times
    report.append("COMPLETION TIMES")
    report.append("-" * 80)
    total_time = metrics['completion_time']['total']
    report.append(f"Total Time: {total_time['mean']:.2f} ± {total_time['std']:.2f} min")
    report.append(f"Median: {total_time['median']:.2f} min")
    report.append(f"Range: [{total_time['min']:.2f}, {total_time['max']:.2f}] min")
    report.append("")
    report.append("By Task:")
    for task_name, task_data in metrics['completion_time'].items():
        if task_name != 'total':
            report.append(f"  {task_name:20s}: {task_data['mean']:.2f} ± {task_data['std']:.2f} min")
    report.append("")

    # Errors
    report.append("ERRORS")
    report.append("-" * 80)
    total_errors = metrics['errors']['total']
    report.append(f"Mean Errors: {total_errors['mean']:.2f} ± {total_errors['std']:.2f}")
    report.append(f"Median: {total_errors['median']:.0f}")
    report.append(f"Range: [{total_errors['min']:.0f}, {total_errors['max']:.0f}]")
    report.append("")

    # Summary
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)

    targets_met = 0
    total_targets = 5

    if metrics['sus']['mean'] >= 85:
        report.append("✓ SUS Score target met (≥ 85)")
        targets_met += 1
    else:
        report.append("✗ SUS Score target not met")

    if metrics['nasa_tlx']['overall']['mean'] <= 30:
        report.append("✓ NASA TLX target met (≤ 30)")
        targets_met += 1
    else:
        report.append("✗ NASA TLX target not met")

    if metrics['success_rate']['overall']['success_rate'] >= 90:
        report.append("✓ Success Rate target met (≥ 90%)")
        targets_met += 1
    else:
        report.append("✗ Success Rate target not met")

    if metrics['completion_time']['total']['mean'] <= 15:
        report.append("✓ Completion Time target met (≤ 15 min)")
        targets_met += 1
    else:
        report.append("✗ Completion Time target not met")

    if metrics['errors']['total']['mean'] <= 2:
        report.append("✓ Error Rate target met (≤ 2)")
        targets_met += 1
    else:
        report.append("✗ Error Rate target not met")

    report.append("")
    report.append(f"Targets Met: {targets_met}/{total_targets} ({targets_met/total_targets*100:.0f}%)")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # Save report
    output_file = RESULTS_DIR / "03_usability_summary_report.txt"
    with open(output_file, 'w') as f:
        f.write(report_text)

    if logger:
        logger.info(f"  Summary report saved to {output_file}")

    # Also print to console
    print("\n" + report_text)

    return report_text


def main():
    """Main execution pipeline"""
    logger = setup_logging("analyze_usability", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("USABILITY STUDY ANALYSIS PIPELINE")
    logger.info("=" * 80)

    # Step 1: Generate mock data
    logger.info("\n1. GENERATING MOCK DATA")
    logger.info("-" * 80)
    generate_mock_data.main()

    # Step 2: Calculate metrics
    logger.info("\n2. CALCULATING METRICS")
    logger.info("-" * 80)
    calculate_metrics.main()

    # Step 3: Statistical analysis
    logger.info("\n3. STATISTICAL ANALYSIS")
    logger.info("-" * 80)
    statistical_analysis.main()

    # Step 4: Generate visualizations
    logger.info("\n4. GENERATING VISUALIZATIONS")
    logger.info("-" * 80)
    generate_visualizations.main()

    # Step 5: Generate LaTeX table
    logger.info("\n5. GENERATING LATEX TABLE")
    logger.info("-" * 80)
    generate_latex_table(logger)

    # Step 6: Generate summary report
    logger.info("\n6. GENERATING SUMMARY REPORT")
    logger.info("-" * 80)
    generate_summary_report(logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("USABILITY ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("\nOutputs:")
    logger.info(f"  Results: {RESULTS_DIR}")
    logger.info(f"  Figures: {FIGURES_DIR}")
    logger.info(f"  Tables: {TABLES_DIR}")
    logger.info(f"  Logs: {LOGS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
