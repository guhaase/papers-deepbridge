"""
Comparação e Análise Estatística dos Resultados

Compara tempos DeepBridge vs. Fragmentado e realiza testes estatísticos.

Autor: DeepBridge Team
Data: 2025-12-05
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Any

from utils import (
    load_config,
    ExperimentLogger,
    save_results,
    create_results_summary
)


class BenchmarkAnalysis:
    """Análise comparativa dos benchmarks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.results_dir = Path(__file__).parent.parent / config['outputs']['results_dir']
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_benchmark_results(self) -> tuple:
        """
        Carrega resultados dos benchmarks

        Returns:
            (deepbridge_results, fragmented_results)
        """
        self.logger.info("Loading benchmark results...")

        # DeepBridge
        deepbridge_path = self.results_dir / 'deepbridge_times.json'
        if not deepbridge_path.exists():
            raise FileNotFoundError(f"DeepBridge results not found: {deepbridge_path}")

        with open(deepbridge_path, 'r') as f:
            deepbridge_results = json.load(f)

        # Fragmented
        fragmented_path = self.results_dir / 'fragmented_times.json'
        if not fragmented_path.exists():
            raise FileNotFoundError(f"Fragmented results not found: {fragmented_path}")

        with open(fragmented_path, 'r') as f:
            fragmented_results = json.load(f)

        self.logger.info("Results loaded successfully")

        return deepbridge_results, fragmented_results

    def calculate_comparison_metrics(
        self,
        deepbridge_results: Dict,
        fragmented_results: Dict
    ) -> pd.DataFrame:
        """
        Calcula métricas de comparação

        Returns:
            DataFrame com comparação por tarefa
        """
        self.logger.info("Calculating comparison metrics...")

        comparisons = []

        tasks = [t for t in deepbridge_results.keys() if t != 'total']
        tasks.append('total')

        for task in tasks:
            db_stats = deepbridge_results[task]
            frag_stats = fragmented_results[task]

            db_mean_min = db_stats['mean_minutes']
            frag_mean_min = frag_stats['mean_minutes']

            speedup = frag_mean_min / db_mean_min
            reduction_abs = frag_mean_min - db_mean_min
            reduction_pct = (reduction_abs / frag_mean_min) * 100

            comparisons.append({
                'Task': task.capitalize(),
                'DeepBridge_Mean_Min': round(db_mean_min, 2),
                'DeepBridge_Std_Min': round(db_stats['std_minutes'], 2),
                'Fragmented_Mean_Min': round(frag_mean_min, 2),
                'Fragmented_Std_Min': round(frag_stats['std_minutes'], 2),
                'Speedup': round(speedup, 2),
                'Reduction_Min': round(reduction_abs, 2),
                'Reduction_Pct': round(reduction_pct, 2)
            })

        df = pd.DataFrame(comparisons)

        self.logger.info("\nComparison Table:")
        self.logger.info("\n" + df.to_string(index=False))

        return df

    def perform_statistical_tests(
        self,
        deepbridge_results: Dict,
        fragmented_results: Dict
    ) -> Dict[str, Any]:
        """
        Realiza testes estatísticos (paired t-test, Wilcoxon)

        Returns:
            Dict com resultados dos testes
        """
        self.logger.info("\nPerforming statistical tests...")

        test_results = {}

        tasks = [t for t in deepbridge_results.keys() if t != 'total']
        tasks.append('total')

        for task in tasks:
            db_times = np.array(deepbridge_results[task]['all_times_seconds'])
            frag_times = np.array(fragmented_results[task]['all_times_seconds'])

            # Paired t-test
            t_stat, p_value_t = stats.ttest_rel(frag_times, db_times)

            # Wilcoxon signed-rank test (non-parametric)
            w_stat, p_value_w = stats.wilcoxon(frag_times, db_times)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(db_times)**2 + np.std(frag_times)**2) / 2)
            cohens_d = (np.mean(frag_times) - np.mean(db_times)) / pooled_std

            test_results[task] = {
                'paired_ttest': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value_t),
                    'significant': p_value_t < 0.05
                },
                'wilcoxon': {
                    'w_statistic': float(w_stat),
                    'p_value': float(p_value_w),
                    'significant': p_value_w < 0.05
                },
                'effect_size': {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
            }

            self.logger.info(f"\n{task.upper()}:")
            self.logger.info(f"  Paired t-test: t={t_stat:.2f}, p={p_value_t:.4f} "
                           f"{'***' if p_value_t < 0.001 else '**' if p_value_t < 0.01 else '*' if p_value_t < 0.05 else 'ns'}")
            self.logger.info(f"  Wilcoxon: W={w_stat:.2f}, p={p_value_w:.4f} "
                           f"{'***' if p_value_w < 0.001 else '**' if p_value_w < 0.01 else '*' if p_value_w < 0.05 else 'ns'}")
            self.logger.info(f"  Cohen's d: {cohens_d:.2f} ({self._interpret_cohens_d(cohens_d)})")

        return test_results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpreta Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def perform_anova(
        self,
        deepbridge_results: Dict,
        fragmented_results: Dict
    ) -> Dict[str, Any]:
        """
        ANOVA para comparar todos os tempos de tarefas

        Returns:
            Resultados do ANOVA
        """
        self.logger.info("\nPerforming ANOVA...")

        # Coletar todos os tempos por tarefa
        all_times = []
        all_labels = []

        tasks = [t for t in deepbridge_results.keys() if t != 'total']

        for task in tasks:
            # DeepBridge
            db_times = deepbridge_results[task]['all_times_seconds']
            all_times.extend(db_times)
            all_labels.extend([f"{task}_deepbridge"] * len(db_times))

            # Fragmented
            frag_times = fragmented_results[task]['all_times_seconds']
            all_times.extend(frag_times)
            all_labels.extend([f"{task}_fragmented"] * len(frag_times))

        # One-way ANOVA
        groups = {}
        for time_val, label in zip(all_times, all_labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(time_val)

        f_stat, p_value = stats.f_oneway(*groups.values())

        anova_results = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'num_groups': len(groups),
            'total_observations': len(all_times)
        }

        self.logger.info(f"  F-statistic: {f_stat:.2f}")
        self.logger.info(f"  p-value: {p_value:.4f} "
                       f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

        return anova_results

    def create_latex_table(self, comparison_df: pd.DataFrame) -> str:
        """
        Gera tabela em LaTeX

        Args:
            comparison_df: DataFrame com comparação

        Returns:
            String com código LaTeX
        """
        self.logger.info("\nGenerating LaTeX table...")

        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Benchmarks de Tempo: DeepBridge vs. Ferramentas Fragmentadas}\n"
        latex += "\\label{tab:time_benchmarks}\n"
        latex += "\\small\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\toprule\n"
        latex += "\\textbf{Tarefa} & \\textbf{DeepBridge} & \\textbf{Fragmentado} & \\textbf{Speedup} & \\textbf{Redução} \\\\\n"
        latex += "\\midrule\n"

        for _, row in comparison_df.iterrows():
            if row['Task'].upper() == 'TOTAL':
                latex += "\\midrule\n"
                task_name = "\\textbf{Total}"
            else:
                task_name = row['Task']

            latex += f"{task_name} & {row['DeepBridge_Mean_Min']:.1f} min & "
            latex += f"{row['Fragmented_Mean_Min']:.1f} min & "
            latex += f"{row['Speedup']:.1f}× & "
            latex += f"{row['Reduction_Pct']:.1f}\\% \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        # Salvar
        latex_path = Path(__file__).parent.parent / 'tables' / 'time_benchmarks.tex'
        latex_path.parent.mkdir(parents=True, exist_ok=True)

        with open(latex_path, 'w') as f:
            f.write(latex)

        self.logger.info(f"LaTeX table saved: {latex_path}")

        return latex

    def generate_summary_report(
        self,
        comparison_df: pd.DataFrame,
        statistical_tests: Dict,
        anova_results: Dict
    ):
        """Gera relatório resumo em texto"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SUMMARY REPORT")
        self.logger.info("=" * 60)

        # Principais resultados
        total_row = comparison_df[comparison_df['Task'] == 'Total'].iloc[0]

        self.logger.info("\nMAIN FINDINGS:")
        self.logger.info(f"  Total time DeepBridge: {total_row['DeepBridge_Mean_Min']:.1f} min")
        self.logger.info(f"  Total time Fragmented: {total_row['Fragmented_Mean_Min']:.1f} min")
        self.logger.info(f"  Overall Speedup: {total_row['Speedup']:.1f}×")
        self.logger.info(f"  Overall Reduction: {total_row['Reduction_Pct']:.1f}%")

        # Verificar se atingiu meta (89% redução)
        target_reduction = 89.0
        actual_reduction = total_row['Reduction_Pct']

        self.logger.info(f"\nTARGET CHECK:")
        self.logger.info(f"  Target reduction: {target_reduction}%")
        self.logger.info(f"  Actual reduction: {actual_reduction:.1f}%")

        if actual_reduction >= target_reduction:
            self.logger.info(f"  ✓ TARGET MET!")
        else:
            self.logger.info(f"  ✗ Target not met (difference: {target_reduction - actual_reduction:.1f}%)")

        # Significância estatística
        self.logger.info(f"\nSTATISTICAL SIGNIFICANCE:")
        total_tests = statistical_tests['total']

        if total_tests['paired_ttest']['significant']:
            self.logger.info(f"  ✓ Difference is statistically significant (p < 0.05)")
        else:
            self.logger.info(f"  ✗ Difference is NOT statistically significant")

        self.logger.info(f"  Effect size: {total_tests['effect_size']['cohens_d']:.2f} "
                       f"({total_tests['effect_size']['interpretation']})")

    def run_analysis(self):
        """Executa análise completa"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("BENCHMARK ANALYSIS")
        self.logger.info(f"{'=' * 60}\n")

        # 1. Carregar resultados
        deepbridge_results, fragmented_results = self.load_benchmark_results()

        # 2. Calcular métricas de comparação
        comparison_df = self.calculate_comparison_metrics(
            deepbridge_results,
            fragmented_results
        )

        # 3. Testes estatísticos
        statistical_tests = self.perform_statistical_tests(
            deepbridge_results,
            fragmented_results
        )

        # 4. ANOVA
        anova_results = self.perform_anova(
            deepbridge_results,
            fragmented_results
        )

        # 5. Gerar tabela LaTeX
        latex_table = self.create_latex_table(comparison_df)

        # 6. Relatório resumo
        self.generate_summary_report(
            comparison_df,
            statistical_tests,
            anova_results
        )

        # 7. Salvar tudo
        analysis_results = {
            'comparison': comparison_df.to_dict(orient='records'),
            'statistical_tests': statistical_tests,
            'anova': anova_results,
            'latex_table': latex_table
        }

        output_path = self.results_dir / 'analysis_results'
        save_results(analysis_results, output_path, formats=['json'])

        # Salvar CSV da comparação
        comparison_df.to_csv(
            self.results_dir / 'comparison_summary.csv',
            index=False
        )

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Analysis completed successfully!")
        self.logger.info(f"{'=' * 60}\n")

        return analysis_results


def main():
    """Função principal"""
    # Configurar logging
    log_dir = Path(__file__).parent.parent / 'logs'
    exp_logger = ExperimentLogger(log_dir, name='analysis')
    logger = exp_logger.get_logger()

    # Carregar config
    config = load_config()

    # Executar análise
    analysis = BenchmarkAnalysis(config)
    results = analysis.run_analysis()

    logger.info("\nAll results saved!")

    return results


if __name__ == "__main__":
    main()
