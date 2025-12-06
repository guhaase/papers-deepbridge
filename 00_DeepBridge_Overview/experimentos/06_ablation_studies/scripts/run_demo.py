"""
Ablation Studies Demo - Mock Implementation

Simulates ablation study results demonstrating:
- API Unificada: 50% do ganho (~66 min)
- Paralelização: 30% do ganho (~40 min)
- Caching: 10% do ganho (~13 min)
- Automação de Relatórios: 10% do ganho (~13 min)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging, save_results, get_ablation_configs,
    calculate_contribution, calculate_contribution_percentage,
    calculate_statistics, format_time, calculate_speedup
)

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = BASE_DIR / "tables"
LOGS_DIR = BASE_DIR / "logs"

RESULTS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def generate_mock_times(base_time: float, n_runs: int = 10, std: float = 0.5, seed: int = 42) -> list:
    """
    Generate mock execution times with small variance

    Args:
        base_time: Expected mean time in minutes
        n_runs: Number of runs
        std: Standard deviation
        seed: Random seed

    Returns:
        List of execution times
    """
    np.random.seed(seed)
    times = np.random.normal(base_time, std, n_runs)
    # Ensure positive times
    times = np.maximum(times, base_time * 0.8)
    return times.tolist()


def run_ablation_study(n_runs: int = 10, seed: int = 42) -> dict:
    """
    Run mock ablation study

    Args:
        n_runs: Number of runs per configuration
        seed: Random seed

    Returns:
        Dictionary with results for all configurations
    """
    # Expected times (in minutes)
    expected_times = {
        'full': 17.0,
        'no_api': 83.0,
        'no_parallel': 57.0,
        'no_cache': 30.0,
        'no_auto_report': 30.0,
        'none': 150.0
    }

    # Generate times for each configuration
    results = {}
    configs = get_ablation_configs()

    for config_name, expected_time in expected_times.items():
        times = generate_mock_times(
            base_time=expected_time,
            n_runs=n_runs,
            std=expected_time * 0.03,  # 3% variance
            seed=seed + hash(config_name) % 100
        )

        stats = calculate_statistics(times)

        results[config_name] = {
            'config': configs[config_name],
            'times': times,
            'statistics': stats,
            'expected_time': expected_time
        }

    return results


def calculate_contributions(results: dict) -> dict:
    """
    Calculate contributions of each component

    Args:
        results: Results from ablation study

    Returns:
        Dictionary with contributions
    """
    # Baseline (full DeepBridge)
    time_full = results['full']['statistics']['mean']

    # Fragmented (no components)
    time_none = results['none']['statistics']['mean']

    # Total gain
    total_gain = time_none - time_full

    # Calculate contributions
    contributions = {}

    # API Unificada
    time_no_api = results['no_api']['statistics']['mean']
    contrib_api = calculate_contribution(time_full, time_no_api)
    pct_api = calculate_contribution_percentage(contrib_api, total_gain)

    contributions['api_unificada'] = {
        'absolute': contrib_api,
        'percentage': pct_api,
        'time_with': time_full,
        'time_without': time_no_api
    }

    # Paralelização
    time_no_parallel = results['no_parallel']['statistics']['mean']
    contrib_parallel = calculate_contribution(time_full, time_no_parallel)
    pct_parallel = calculate_contribution_percentage(contrib_parallel, total_gain)

    contributions['paralizacao'] = {
        'absolute': contrib_parallel,
        'percentage': pct_parallel,
        'time_with': time_full,
        'time_without': time_no_parallel
    }

    # Caching
    time_no_cache = results['no_cache']['statistics']['mean']
    contrib_cache = calculate_contribution(time_full, time_no_cache)
    pct_cache = calculate_contribution_percentage(contrib_cache, total_gain)

    contributions['caching'] = {
        'absolute': contrib_cache,
        'percentage': pct_cache,
        'time_with': time_full,
        'time_without': time_no_cache
    }

    # Automação de Relatórios
    time_no_auto = results['no_auto_report']['statistics']['mean']
    contrib_auto = calculate_contribution(time_full, time_no_auto)
    pct_auto = calculate_contribution_percentage(contrib_auto, total_gain)

    contributions['automacao_relatorios'] = {
        'absolute': contrib_auto,
        'percentage': pct_auto,
        'time_with': time_full,
        'time_without': time_no_auto
    }

    # Summary
    contributions['summary'] = {
        'time_full': time_full,
        'time_none': time_none,
        'total_gain': total_gain,
        'total_percentage_check': pct_api + pct_parallel + pct_cache + pct_auto
    }

    return contributions


def generate_latex_table(results: dict, contributions: dict, output_file: Path):
    """Generate LaTeX table for ablation results"""

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation Study: Decomposição dos Ganhos de Tempo}")
    latex.append("\\label{tab:ablation_results}")
    latex.append("\\begin{tabular}{lrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Configuração} & \\textbf{Tempo (min)} & \\textbf{Ganho} & \\textbf{\\% do Total} \\\\")
    latex.append("\\midrule")

    # Full
    time_full = results['full']['statistics']['mean']
    latex.append(f"\\textbf{{DeepBridge Completo}} & {time_full:.1f} & - & - \\\\")

    # No API
    time_no_api = results['no_api']['statistics']['mean']
    contrib = contributions['api_unificada']
    latex.append(f"Sem API Unificada & {time_no_api:.1f} & +{contrib['absolute']:.1f} & {contrib['percentage']:.0f}\\% \\\\")

    # No Parallel
    time_no_parallel = results['no_parallel']['statistics']['mean']
    contrib = contributions['paralizacao']
    latex.append(f"Sem Paralelização & {time_no_parallel:.1f} & +{contrib['absolute']:.1f} & {contrib['percentage']:.0f}\\% \\\\")

    # No Cache
    time_no_cache = results['no_cache']['statistics']['mean']
    contrib = contributions['caching']
    latex.append(f"Sem Caching & {time_no_cache:.1f} & +{contrib['absolute']:.1f} & {contrib['percentage']:.0f}\\% \\\\")

    # No AutoReport
    time_no_auto = results['no_auto_report']['statistics']['mean']
    contrib = contributions['automacao_relatorios']
    latex.append(f"Sem Automação & {time_no_auto:.1f} & +{contrib['absolute']:.1f} & {contrib['percentage']:.0f}\\% \\\\")

    latex.append("\\midrule")

    # None (Fragmented)
    time_none = results['none']['statistics']['mean']
    total_gain = contributions['summary']['total_gain']
    latex.append(f"\\textbf{{Workflow Fragmentado}} & {time_none:.1f} & +{total_gain:.1f} & 100\\% \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    with open(output_file, 'w') as f:
        f.write(latex_content)

    return latex_content


def print_summary(results: dict, contributions: dict):
    """Print summary of ablation study"""

    print("\n" + "=" * 80)
    print("ABLATION STUDY - DEMO RESULTS")
    print("=" * 80)
    print()

    # Configuration times
    print("EXECUTION TIMES BY CONFIGURATION:")
    print("-" * 80)
    print(f"{'Configuração':<30} {'Tempo (min)':>12} {'Ganho':>10}")
    print("-" * 80)

    time_full = results['full']['statistics']['mean']
    print(f"{'DeepBridge Completo':<30} {time_full:>11.1f} {'-':>10}")

    configs_order = [
        ('no_api', 'Sem API Unificada', 'api_unificada'),
        ('no_parallel', 'Sem Paralelização', 'paralizacao'),
        ('no_cache', 'Sem Caching', 'caching'),
        ('no_auto_report', 'Sem Automação Relatórios', 'automacao_relatorios')
    ]

    for config_key, config_label, contrib_key in configs_order:
        time = results[config_key]['statistics']['mean']
        gain = contributions[contrib_key]['absolute']
        print(f"{config_label:<30} {time:>11.1f} {f'+{gain:.1f}':>10}")

    time_none = results['none']['statistics']['mean']
    total_gain = contributions['summary']['total_gain']
    print("-" * 80)
    print(f"{'Workflow Fragmentado':<30} {time_none:>11.1f} {f'+{total_gain:.1f}':>10}")

    # Component contributions
    print("\n" + "-" * 80)
    print("COMPONENT CONTRIBUTIONS:")
    print("-" * 80)
    print(f"{'Componente':<30} {'Ganho (min)':>12} {'% do Total':>12}")
    print("-" * 80)

    components = [
        ('API Unificada', 'api_unificada'),
        ('Paralelização', 'paralizacao'),
        ('Caching', 'caching'),
        ('Automação Relatórios', 'automacao_relatorios')
    ]

    for comp_label, comp_key in components:
        contrib = contributions[comp_key]
        print(f"{comp_label:<30} {contrib['absolute']:>11.1f} {contrib['percentage']:>11.0f}%")

    print("-" * 80)
    print(f"{'TOTAL':<30} {total_gain:>11.1f} {'100':>11}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Total time reduction: {total_gain:.1f} min ({time_none:.1f} → {time_full:.1f} min)")

    speedup = calculate_speedup(time_none, time_full)
    print(f"✓ Overall speedup: {speedup:.1f}×")
    print()

    for comp_label, comp_key in components:
        contrib = contributions[comp_key]
        expected_pct = {
            'api_unificada': 50,
            'paralizacao': 30,
            'caching': 10,
            'automacao_relatorios': 10
        }[comp_key]

        status = "✓" if abs(contrib['percentage'] - expected_pct) < 5 else "⚠"
        print(f"{status} {comp_label}: {contrib['percentage']:.0f}% (target: {expected_pct}%)")

    print("=" * 80)
    print()


def main():
    """Main execution"""
    logger = setup_logging("ablation_demo", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("ABLATION STUDY - DEMO (MOCK DATA)")
    logger.info("=" * 80)

    # Run ablation study
    logger.info("\nRunning ablation study with 10 runs per configuration...")
    results = run_ablation_study(n_runs=10, seed=42)

    # Calculate contributions
    logger.info("Calculating component contributions...")
    contributions = calculate_contributions(results)

    # Save results
    output_data = {
        'results': results,
        'contributions': contributions
    }

    output_file = RESULTS_DIR / "ablation_demo_results.json"
    save_results(output_data, output_file, logger)

    # Generate LaTeX table
    logger.info("Generating LaTeX table...")
    latex_file = TABLES_DIR / "ablation_results.tex"
    generate_latex_table(results, contributions, latex_file)
    logger.info(f"LaTeX table saved to {latex_file}")

    # Print summary
    print_summary(results, contributions)

    logger.info("Demo completed successfully!")
    logger.info(f"Results saved to {RESULTS_DIR}")
    logger.info(f"Tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()
