"""
Aggregate Analysis of All Case Studies

Loads results from all case studies and generates:
- LaTeX table for the paper
- Statistical analysis
- Visualizations (time comparison, violations summary)
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, load_results, aggregate_case_study_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def generate_latex_table(case_data: list, output_file: Path, logger):
    """Generate LaTeX table for the paper"""

    logger.info("Generating LaTeX table...")

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Resultados dos Estudos de Caso}")
    latex.append("\\label{tab:case_studies}")
    latex.append("\\begin{tabular}{lrrrl}")
    latex.append("\\toprule")
    latex.append("\\textbf{Domínio} & \\textbf{Amostras} & \\textbf{Violações} & \\textbf{Tempo (min)} & \\textbf{Achado Principal} \\\\")
    latex.append("\\midrule")

    for case in case_data:
        domain = case['domain']
        samples = f"{case['samples']:,}"
        violations = case['violations']
        time_min = f"{case['time']:.0f}"
        finding = case['finding']

        latex.append(f"{domain} & {samples} & {violations} & {time_min} & {finding} \\\\")

    latex.append("\\midrule")

    # Add summary row
    total_samples = sum(c['samples'] for c in case_data)
    total_violations = sum(c['violations'] for c in case_data)
    avg_time = np.mean([c['time'] for c in case_data])

    latex.append(f"\\textbf{{Total/Média}} & {total_samples:,} & {total_violations} & {avg_time:.1f} & -- \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    with open(output_file, 'w') as f:
        f.write(latex_content)

    logger.info(f"LaTeX table saved to {output_file}")
    return latex_content


def generate_time_comparison_plot(case_data: list, output_file: Path, logger):
    """Generate time comparison bar plot"""

    logger.info("Generating time comparison plot...")

    plt.figure(figsize=(10, 6))

    domains = [c['domain'] for c in case_data]
    times = [c['time'] for c in case_data]
    expected_times = [c.get('expected_time', c['time']) for c in case_data]

    x = np.arange(len(domains))
    width = 0.35

    plt.bar(x - width/2, times, width, label='Actual', color='steelblue')
    plt.bar(x + width/2, expected_times, width, label='Expected', color='lightcoral')

    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.title('Validation Time by Domain', fontsize=14, fontweight='bold')
    plt.xticks(x, domains, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Time comparison plot saved to {output_file}")
    plt.close()


def generate_violations_plot(case_data: list, output_file: Path, logger):
    """Generate violations summary plot"""

    logger.info("Generating violations plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Violations by domain
    domains = [c['domain'] for c in case_data]
    violations = [c['violations'] for c in case_data]

    colors = ['red' if v > 0 else 'green' for v in violations]
    ax1.barh(domains, violations, color=colors, alpha=0.7)
    ax1.set_xlabel('Number of Violations', fontsize=12)
    ax1.set_title('Violations Detected by Domain', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Summary pie chart
    has_violations = sum(1 for v in violations if v > 0)
    no_violations = len(violations) - has_violations

    ax2.pie([has_violations, no_violations],
            labels=[f'With Violations\n({has_violations})', f'No Violations\n({no_violations})'],
            colors=['lightcoral', 'lightgreen'],
            autopct='%1.0f%%',
            startangle=90)
    ax2.set_title('Case Studies Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Violations plot saved to {output_file}")
    plt.close()


def generate_statistical_analysis(case_data: list, output_file: Path, logger):
    """Generate statistical analysis JSON"""

    logger.info("Generating statistical analysis...")

    times = [c['time'] for c in case_data]
    samples = [c['samples'] for c in case_data]
    violations = [c['violations'] for c in case_data]

    analysis = {
        'time_statistics': {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
        },
        'sample_statistics': {
            'total': int(np.sum(samples)),
            'mean': float(np.mean(samples)),
            'min': int(np.min(samples)),
            'max': int(np.max(samples)),
        },
        'violation_statistics': {
            'total': int(np.sum(violations)),
            'cases_with_violations': int(sum(1 for v in violations if v > 0)),
            'cases_without_violations': int(sum(1 for v in violations if v == 0)),
            'detection_rate': float(sum(1 for v in violations if v > 0) / len(violations)),
        },
        'expected_vs_actual': {
            'expected_mean_time': 27.7,
            'actual_mean_time': float(np.mean(times)),
            'time_difference': float(np.mean(times) - 27.7),
            'expected_total_violations': 4,
            'actual_total_violations': int(np.sum(violations)),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Statistical analysis saved to {output_file}")

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("STATISTICAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Mean validation time: {analysis['time_statistics']['mean']:.2f} min (expected: 27.7 min)")
    logger.info(f"Total violations: {analysis['violation_statistics']['total']} (expected: 4)")
    logger.info(f"Cases with violations: {analysis['violation_statistics']['cases_with_violations']}/6")
    logger.info(f"Total samples processed: {analysis['sample_statistics']['total']:,}")
    logger.info("=" * 80)

    return analysis


def main():
    """Main function"""
    logger = setup_logging("aggregate_analysis", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("AGGREGATE ANALYSIS OF CASE STUDIES")
    logger.info("=" * 80)

    # Define expected results and load actual results
    case_definitions = [
        {
            'name': 'credit',
            'domain': 'Crédito',
            'expected_time': 17,
            'expected_violations': 2,
            'finding': 'DI=0.74 (gênero)'
        },
        {
            'name': 'hiring',
            'domain': 'Contratação',
            'expected_time': 12,
            'expected_violations': 1,
            'finding': 'DI=0.59 (raça)'
        },
        {
            'name': 'healthcare',
            'domain': 'Saúde',
            'expected_time': 23,
            'expected_violations': 0,
            'finding': 'Bem calibrado'
        },
        {
            'name': 'mortgage',
            'domain': 'Hipoteca',
            'expected_time': 45,
            'expected_violations': 1,
            'finding': 'Violação ECOA'
        },
        {
            'name': 'insurance',
            'domain': 'Seguros',
            'expected_time': 38,
            'expected_violations': 0,
            'finding': 'Passa todos testes'
        },
        {
            'name': 'fraud',
            'domain': 'Fraude',
            'expected_time': 31,
            'expected_violations': 0,
            'finding': 'Alta resiliência'
        },
    ]

    case_data = []

    for case_def in case_definitions:
        result_file = RESULTS_DIR / f"case_study_{case_def['name']}_results.json"

        if result_file.exists():
            result = load_results(result_file)

            case_data.append({
                'domain': case_def['domain'],
                'samples': result['n_samples'],
                'violations': result['n_violations'],
                'time': result['total_time'],
                'expected_time': case_def['expected_time'],
                'finding': case_def['finding']
            })

            logger.info(f"✓ Loaded results for {case_def['domain']}")
        else:
            logger.warning(f"✗ Results not found for {case_def['name']}: {result_file}")

    if len(case_data) == 0:
        logger.error("No results found! Please run the case studies first.")
        return

    logger.info(f"Loaded {len(case_data)}/6 case studies")
    logger.info("")

    # Generate outputs
    latex_file = TABLES_DIR / "case_studies_summary.tex"
    generate_latex_table(case_data, latex_file, logger)

    time_plot = FIGURES_DIR / "case_studies_times.pdf"
    generate_time_comparison_plot(case_data, time_plot, logger)

    violations_plot = FIGURES_DIR / "case_studies_violations.pdf"
    generate_violations_plot(case_data, violations_plot, logger)

    analysis_file = RESULTS_DIR / "case_studies_analysis.json"
    analysis = generate_statistical_analysis(case_data, analysis_file, logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("AGGREGATE ANALYSIS COMPLETED")
    logger.info("=" * 80)
    logger.info(f"LaTeX table: {latex_file}")
    logger.info(f"Time plot: {time_plot}")
    logger.info(f"Violations plot: {violations_plot}")
    logger.info(f"Analysis JSON: {analysis_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
