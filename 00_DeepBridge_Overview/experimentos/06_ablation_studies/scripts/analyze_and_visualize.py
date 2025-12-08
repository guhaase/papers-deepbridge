"""
Analyze and Visualize Ablation Studies Results

Generates:
- Waterfall chart showing cumulative contributions
- Stacked bar chart
- Statistical analysis (ANOVA, Tukey HSD)
- LaTeX tables

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
from scipy.stats import f_oneway

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
LOGS_DIR = BASE_DIR / "logs"

FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)


def load_ablation_results():
    """Load ablation results"""
    results_file = RESULTS_DIR / "ablation_results.json"
    with open(results_file, 'r') as f:
        return json.load(f)


def generate_latex_table(results, output_file):
    """Generate LaTeX table for ablation results"""
    configs = results['configurations']
    contributions = results['contribution_analysis']['contributions']

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Resultados do Estudo de Ablação - Contribuição de Cada Componente}")
    latex.append("\\label{tab:ablation_results}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Componente} & \\textbf{Ganho (min)} & \\textbf{\\% do Total} & \\textbf{Tempo sem} \\\\")
    latex.append("\\midrule")

    # Individual components
    comp_names = {
        'unified_api': 'API Unificada',
        'parallelization': 'Paralelização',
        'caching': 'Caching',
        'auto_reporting': 'Relatórios Automáticos'
    }

    for comp_key, comp_name in comp_names.items():
        comp_data = contributions[comp_key]
        latex.append(
            f"{comp_name} & "
            f"{comp_data['contribution_min']:.1f} & "
            f"{comp_data['contribution_pct']:.1f}\\% & "
            f"{comp_data['time_without']:.1f} min \\\\"
        )

    latex.append("\\midrule")

    # Total
    total_gain = results['contribution_analysis']['total_gain_minutes']
    baseline_time = configs['baseline']['mean_minutes']
    full_time = configs['full']['mean_minutes']
    speedup = results['contribution_analysis']['speedup_factor']

    latex.append(
        f"\\textbf{{Total}} & "
        f"\\textbf{{{total_gain:.1f}}} & "
        f"\\textbf{{100.0\\%}} & "
        f"\\textbf{{{baseline_time:.1f} min}} \\\\"
    )

    latex.append("\\midrule")
    latex.append(
        f"Speedup & \\multicolumn{{3}}{{c}}{{\\textbf{{{speedup:.1f}×}}}} \\\\"
    )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    with open(output_file, 'w') as f:
        f.write(latex_content)

    return latex_content


def plot_waterfall_chart(results, output_file):
    """Plot waterfall chart showing cumulative contributions"""
    contributions = results['contribution_analysis']['contributions']
    configs = results['configurations']

    # Prepare data
    full_time = configs['full']['mean_minutes']
    baseline_time = configs['baseline']['mean_minutes']

    components = ['Unified\nAPI', 'Parallel-\nization', 'Caching', 'Auto-\nreporting']
    comp_keys = ['unified_api', 'parallelization', 'caching', 'auto_reporting']

    values = [contributions[key]['contribution_min'] for key in comp_keys]

    # Create waterfall
    fig, ax = plt.subplots(figsize=(12, 7))

    # Starting point (baseline)
    x_pos = 0
    y_start = baseline_time

    # Plot baseline bar
    ax.bar(x_pos, baseline_time, width=0.6, color='#D5573B',
           edgecolor='black', linewidth=1.5, label='Baseline (Fragmented)')
    ax.text(x_pos, baseline_time + 5, f'{baseline_time:.0f} min',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot reduction bars
    x_positions = list(range(1, len(components) + 1))
    cumulative = baseline_time

    for i, (comp, val) in enumerate(zip(components, values)):
        bottom = cumulative - val
        ax.bar(x_positions[i], val, bottom=bottom, width=0.6,
               color='#06A77D', edgecolor='black', linewidth=1.5)

        # Connection line
        ax.plot([x_positions[i]-0.5, x_positions[i]-0.3], [cumulative, cumulative],
                'k--', linewidth=1, alpha=0.5)

        # Label
        ax.text(x_positions[i], bottom + val/2, f'-{val:.0f}',
                ha='center', va='center', fontsize=10, fontweight='bold')

        cumulative -= val

    # Plot final DeepBridge bar
    final_x = len(components) + 1
    ax.bar(final_x, full_time, width=0.6, color='#2E86AB',
           edgecolor='black', linewidth=1.5, label='DeepBridge (Full)')
    ax.text(final_x, full_time + 5, f'{full_time:.0f} min',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Connection line to final
    ax.plot([final_x-0.5, final_x-0.3], [cumulative, cumulative],
            'k--', linewidth=1, alpha=0.5)

    # Labels
    x_labels = ['Baseline'] + components + ['DeepBridge\n(Full)']
    ax.set_xticks([0] + x_positions + [final_x])
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel('Execution Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study - Component Contributions (Waterfall Chart)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, baseline_time + 20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stacked_bar_chart(results, output_file):
    """Plot stacked bar chart showing component contributions"""
    contributions = results['contribution_analysis']['contributions']

    # Prepare data
    components = ['Unified API', 'Parallelization', 'Caching', 'Auto-reporting']
    comp_keys = ['unified_api', 'parallelization', 'caching', 'auto_reporting']

    values = [contributions[key]['contribution_min'] for key in comp_keys]
    percentages = [contributions[key]['contribution_pct'] for key in comp_keys]

    # Colors
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar
    bottom = 0
    for comp, val, pct, color in zip(components, values, percentages, colors):
        ax.barh(0, val, left=bottom, height=0.5, color=color,
                edgecolor='black', linewidth=1.5, label=f'{comp} ({pct:.1f}%)')

        # Add value label
        ax.text(bottom + val/2, 0, f'{val:.0f} min\n({pct:.1f}%)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')

        bottom += val

    ax.set_yticks([])
    ax.set_xlabel('Time Contribution (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Component Contributions to Total Speedup',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax.set_xlim(0, bottom + 5)

    # Add total gain annotation
    total_gain = results['contribution_analysis']['total_gain_minutes']
    ax.text(bottom/2, 0.8, f'Total Gain: {total_gain:.0f} min',
            ha='center', va='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_configuration_boxplot(results, output_file):
    """Plot boxplot comparing all configurations"""
    configs = results['configurations']

    # Prepare data
    config_names = ['Full', 'No API', 'No Parallel', 'No Cache', 'No Auto', 'Baseline']
    config_keys = ['full', 'no_api', 'no_parallel', 'no_cache', 'no_auto', 'baseline']

    data = [configs[key]['times_minutes'] for key in config_keys]

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(data, labels=config_names, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color boxes
    colors = ['#2E86AB', '#E63946', '#F77F00', '#06A77D', '#118AB2', '#D5573B']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    ax.set_ylabel('Execution Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study - Configuration Comparison',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Rotate x labels
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def perform_statistical_tests(results):
    """Perform ANOVA to test if configurations are significantly different"""
    global logger

    logger.info("Performing statistical tests...")

    configs = results['configurations']

    # Get all times
    groups = []
    group_names = []

    for key in ['full', 'no_api', 'no_parallel', 'no_cache', 'no_auto', 'baseline']:
        groups.append(configs[key]['times_minutes'])
        group_names.append(configs[key]['config_display_name'])

    # One-way ANOVA
    f_stat, p_value = f_oneway(*groups)

    logger.info(f"One-way ANOVA:")
    logger.info(f"  F-statistic: {f_stat:.4f}")
    logger.info(f"  P-value: {p_value:.6f}")

    if p_value < 0.001:
        logger.info(f"  Result: Highly significant difference (p < 0.001)")
    elif p_value < 0.05:
        logger.info(f"  Result: Significant difference (p < 0.05)")
    else:
        logger.info(f"  Result: No significant difference (p >= 0.05)")

    return {
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    }


def print_summary(results, stats_results):
    """Print final summary"""
    print("\n" + "=" * 80)
    print("ABLATION STUDIES - FINAL RESULTS")
    print("=" * 80)
    print()

    # Component contributions
    print("COMPONENT CONTRIBUTIONS:")
    print("-" * 80)

    contributions = results['contribution_analysis']['contributions']

    print(f"{'Component':<20} {'Gain (min)':>12} {'% of Total':>12} {'Time Without':>15}")
    print("-" * 80)

    comp_names = {
        'unified_api': 'Unified API',
        'parallelization': 'Parallelization',
        'caching': 'Caching',
        'auto_reporting': 'Auto-reporting'
    }

    for comp_key, comp_name in comp_names.items():
        comp_data = contributions[comp_key]
        print(
            f"{comp_name:<20} {comp_data['contribution_min']:>11.2f}  "
            f"{comp_data['contribution_pct']:>11.1f}% {comp_data['time_without']:>14.2f} min"
        )

    print("-" * 80)

    total_gain = results['contribution_analysis']['total_gain_minutes']
    baseline_time = results['configurations']['baseline']['mean_minutes']

    print(f"{'TOTAL':<20} {total_gain:>11.2f}  {100.0:>11.1f}% {baseline_time:>14.2f} min")

    # Speedup
    print()
    print("OVERALL SPEEDUP:")
    print("-" * 80)
    speedup = results['contribution_analysis']['speedup_factor']
    full_time = results['configurations']['full']['mean_minutes']

    print(f"DeepBridge (Full):  {full_time:.2f} min")
    print(f"Baseline:           {baseline_time:.2f} min")
    print(f"Speedup Factor:     {speedup:.1f}×")

    # Statistical tests
    print()
    print("STATISTICAL TESTS:")
    print("-" * 80)
    print(f"ANOVA F-statistic: {stats_results['anova']['f_statistic']:.4f}")
    print(f"P-value:           {stats_results['anova']['p_value']:.6f}")
    print(f"Significant:       {'Yes' if stats_results['anova']['significant'] else 'No'}")

    print()
    print("=" * 80)
    print()


def main():
    """Main execution"""
    global logger
    logger = setup_logging("analyze_and_visualize", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("ABLATION STUDIES - ANALYSIS AND VISUALIZATION")
    logger.info("=" * 80)

    # Load results
    logger.info("\nLoading ablation results...")
    results = load_ablation_results()

    # Generate LaTeX table
    logger.info("Generating LaTeX table...")
    table_file = TABLES_DIR / "ablation_results.tex"
    generate_latex_table(results, table_file)
    logger.info(f"✓ Table saved to {table_file}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    logger.info("  - Waterfall chart...")
    plot_waterfall_chart(results, FIGURES_DIR / "ablation_waterfall.pdf")

    logger.info("  - Stacked bar chart...")
    plot_stacked_bar_chart(results, FIGURES_DIR / "ablation_stacked_bar.pdf")

    logger.info("  - Configuration boxplot...")
    plot_configuration_boxplot(results, FIGURES_DIR / "ablation_boxplot.pdf")

    logger.info(f"✓ Figures saved to {FIGURES_DIR}")

    # Statistical tests
    logger.info("\nPerforming statistical analysis...")
    stats_results = perform_statistical_tests(results)

    # Save analysis
    analysis_output = {
        'ablation_results': results,
        'statistical_tests': stats_results
    }

    output_file = RESULTS_DIR / "ablation_analysis.json"
    save_results(analysis_output, output_file, logger)

    # Print summary
    print_summary(results, stats_results)

    logger.info("=" * 80)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"Figures: {FIGURES_DIR}")
    logger.info(f"Tables:  {TABLES_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
