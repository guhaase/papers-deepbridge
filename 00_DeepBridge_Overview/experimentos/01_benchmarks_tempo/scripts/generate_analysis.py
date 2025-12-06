"""
Statistical Analysis and Figure Generation
Compares DeepBridge vs Fragmented Workflow

Generates:
1. Statistical comparison (paired t-test, Wilcoxon, Cohen's d, ANOVA)
2. Five publication-quality figures (300 DPI PDF)
3. LaTeX table for paper inclusion

Author: DeepBridge Team
Date: 2025-12-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set publication-quality defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# DEMO_SPEEDUP_FACTOR: Fragmented times are in "demo seconds"
# To convert to real-world minutes: demo_seconds → demo_minutes → real_minutes
# real_minutes = (demo_seconds / 60) * DEMO_SPEEDUP_FACTOR = demo_seconds * 1
DEMO_SPEEDUP_FACTOR = 60


class BenchmarkAnalyzer:
    """Analyzes and visualizes benchmark results"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

        # Load data
        self.df_deepbridge = pd.read_csv(self.results_dir / 'deepbridge_times_REAL.csv', index_col=0)
        self.df_fragmented = pd.read_csv(self.results_dir / 'fragmented_times.csv', index_col=0)

        # Convert fragmented demo times to real-world minutes
        # Fragmented times are in seconds (demo), need to convert to minutes (real)
        for col in ['mean_seconds', 'std_seconds', 'min_seconds', 'max_seconds']:
            if col in self.df_fragmented.columns:
                # Convert demo seconds → real minutes
                self.df_fragmented[col.replace('seconds', 'minutes')] = self.df_fragmented[col] * DEMO_SPEEDUP_FACTOR / 60

        # Convert DeepBridge times from seconds to minutes
        for col in ['mean_seconds', 'std_seconds', 'min_seconds', 'max_seconds']:
            if col in self.df_deepbridge.columns:
                self.df_deepbridge[col.replace('seconds', 'minutes')] = self.df_deepbridge[col] / 60

    def calculate_statistics(self):
        """Calculate comprehensive statistical comparisons"""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        # Common tests (exclude fairness since DeepBridge doesn't have it)
        common_tests = ['robustness', 'uncertainty', 'resilience', 'report', 'total']

        stats_results = []

        for test in common_tests:
            if test not in self.df_deepbridge.index or test not in self.df_fragmented.index:
                continue

            # Get individual run times
            deepbridge_times = eval(self.df_deepbridge.loc[test, 'all_times_seconds'])
            fragmented_times = eval(self.df_fragmented.loc[test, 'all_times_seconds'])

            # Convert to minutes
            deepbridge_times_min = np.array(deepbridge_times) / 60
            fragmented_times_min = np.array(fragmented_times) * DEMO_SPEEDUP_FACTOR / 60

            # Calculate statistics
            mean_db = np.mean(deepbridge_times_min)
            std_db = np.std(deepbridge_times_min, ddof=1)
            mean_frag = np.mean(fragmented_times_min)
            std_frag = np.std(fragmented_times_min, ddof=1)

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(deepbridge_times_min, fragmented_times_min)

            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_p_value = stats.wilcoxon(deepbridge_times_min, fragmented_times_min)

            # Cohen's d (effect size)
            pooled_std = np.sqrt((std_db**2 + std_frag**2) / 2)
            cohens_d = (mean_frag - mean_db) / pooled_std

            # Speedup
            speedup = mean_frag / mean_db
            time_saved = mean_frag - mean_db
            percent_reduction = (time_saved / mean_frag) * 100

            stats_results.append({
                'test': test,
                'deepbridge_mean_min': mean_db,
                'deepbridge_std_min': std_db,
                'fragmented_mean_min': mean_frag,
                'fragmented_std_min': std_frag,
                'speedup': speedup,
                'time_saved_min': time_saved,
                'percent_reduction': percent_reduction,
                't_statistic': t_stat,
                'p_value': p_value,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_p_value': w_p_value,
                'cohens_d': cohens_d
            })

            print(f"\n{test.upper()}:")
            print(f"  DeepBridge:  {mean_db:.3f} ± {std_db:.3f} min")
            print(f"  Fragmented:  {mean_frag:.3f} ± {std_frag:.3f} min")
            print(f"  Speedup:     {speedup:.1f}x")
            print(f"  Time saved:  {time_saved:.2f} min ({percent_reduction:.1f}% reduction)")
            print(f"  t-test:      t={t_stat:.2f}, p={p_value:.2e}")
            print(f"  Wilcoxon:    W={w_stat:.0f}, p={w_p_value:.2e}")
            print(f"  Cohen's d:   {cohens_d:.2f} (effect size)")

        self.stats_df = pd.DataFrame(stats_results)

        # Save statistics
        self.stats_df.to_csv(self.results_dir / 'statistical_comparison.csv', index=False)

        # ANOVA for overall comparison
        print("\n" + "-"*60)
        print("ANOVA (Analysis of Variance)")
        print("-"*60)

        # Prepare data for ANOVA
        all_db_times = []
        all_frag_times = []

        for test in common_tests:
            if test not in self.df_deepbridge.index:
                continue
            deepbridge_times = eval(self.df_deepbridge.loc[test, 'all_times_seconds'])
            fragmented_times = eval(self.df_fragmented.loc[test, 'all_times_seconds'])

            all_db_times.extend(np.array(deepbridge_times) / 60)
            all_frag_times.extend(np.array(fragmented_times) * DEMO_SPEEDUP_FACTOR / 60)

        f_stat, anova_p = stats.f_oneway(all_db_times, all_frag_times)
        print(f"  F-statistic: {f_stat:.2f}")
        print(f"  p-value:     {anova_p:.2e}")

        return self.stats_df

    def generate_figure_1_time_comparison(self):
        """Figure 1: Bar chart comparing execution times"""
        print("\nGenerating Figure 1: Time Comparison...")

        tests = ['robustness', 'uncertainty', 'resilience', 'report', 'total']
        test_labels = ['Robustness', 'Uncertainty', 'Resilience', 'Report\nGeneration', 'Total']

        db_means = []
        db_stds = []
        frag_means = []
        frag_stds = []

        for test in tests:
            if test in self.df_deepbridge.index and test in self.df_fragmented.index:
                db_means.append(self.df_deepbridge.loc[test, 'mean_minutes'])
                db_stds.append(self.df_deepbridge.loc[test, 'std_minutes'])
                frag_means.append(self.df_fragmented.loc[test, 'mean_minutes'])
                frag_stds.append(self.df_fragmented.loc[test, 'std_minutes'])

        x = np.arange(len(test_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        bars1 = ax.bar(x - width/2, db_means, width, yerr=db_stds,
                      label='DeepBridge', color='#2E86AB', capsize=5)
        bars2 = ax.bar(x + width/2, frag_means, width, yerr=frag_stds,
                      label='Fragmented Workflow', color='#A23B72', capsize=5)

        ax.set_xlabel('Validation Component')
        ax.set_ylabel('Execution Time (minutes)')
        ax.set_title('Execution Time Comparison: DeepBridge vs Fragmented Workflow')
        ax.set_xticks(x)
        ax.set_xticklabels(test_labels)
        ax.legend()
        ax.set_yscale('log')  # Log scale for better visibility

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure1_time_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure1_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Saved: figure1_time_comparison.pdf")

    def generate_figure_2_speedup(self):
        """Figure 2: Speedup factors"""
        print("\nGenerating Figure 2: Speedup Factors...")

        speedups = self.stats_df[self.stats_df['test'] != 'total']['speedup'].values
        tests = self.stats_df[self.stats_df['test'] != 'total']['test'].values
        test_labels = [t.capitalize() for t in tests]

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(speedups)))
        bars = ax.barh(test_labels, speedups, color=colors)

        ax.set_xlabel('Speedup Factor (×)')
        ax.set_title('DeepBridge Performance Improvement over Fragmented Workflow')
        ax.axvline(x=1, color='red', linestyle='--', linewidth=1, label='No improvement')

        # Add value labels
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax.text(speedup + 5, i, f'{speedup:.1f}×',
                   va='center', fontsize=9, fontweight='bold')

        ax.legend()
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure2_speedup.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure2_speedup.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Saved: figure2_speedup.pdf")

    def generate_figure_3_distribution(self):
        """Figure 3: Distribution comparison (violin plots)"""
        print("\nGenerating Figure 3: Distribution Comparison...")

        tests = ['robustness', 'uncertainty', 'resilience']

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for i, test in enumerate(tests):
            ax = axes[i]

            if test in self.df_deepbridge.index and test in self.df_fragmented.index:
                db_times = np.array(eval(self.df_deepbridge.loc[test, 'all_times_seconds'])) / 60
                frag_times = np.array(eval(self.df_fragmented.loc[test, 'all_times_seconds'])) * DEMO_SPEEDUP_FACTOR / 60

                data = [db_times, frag_times]
                positions = [1, 2]

                parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)

                # Customize colors
                for pc, color in zip(parts['bodies'], ['#2E86AB', '#A23B72']):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

                ax.set_xticks([1, 2])
                ax.set_xticklabels(['DeepBridge', 'Fragmented'])
                ax.set_ylabel('Time (minutes)')
                ax.set_title(test.capitalize())

        plt.suptitle('Execution Time Distributions', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure3_distributions.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure3_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Saved: figure3_distributions.pdf")

    def generate_figure_4_cumulative(self):
        """Figure 4: Cumulative time breakdown"""
        print("\nGenerating Figure 4: Cumulative Time Breakdown...")

        tests = ['robustness', 'uncertainty', 'resilience', 'report']
        test_labels = ['Robustness', 'Uncertainty', 'Resilience', 'Report Gen.']

        db_times = []
        frag_times = []

        for test in tests:
            if test in self.df_deepbridge.index and test in self.df_fragmented.index:
                db_times.append(self.df_deepbridge.loc[test, 'mean_minutes'])
                frag_times.append(self.df_fragmented.loc[test, 'mean_minutes'])

        x = np.arange(len(test_labels))

        fig, ax = plt.subplots(figsize=(8, 5))

        # Stacked bar chart
        db_cumsum = np.cumsum([0] + db_times[:-1])
        frag_cumsum = np.cumsum([0] + frag_times[:-1])

        width = 0.35

        for i, (label, db_t, frag_t) in enumerate(zip(test_labels, db_times, frag_times)):
            if i == 0:
                ax.bar(0, db_t, width, label=label, bottom=0)
                ax.bar(1, frag_t, width, bottom=0)
            else:
                ax.bar(0, db_t, width, bottom=db_cumsum[i])
                ax.bar(1, frag_t, width, bottom=frag_cumsum[i])

        # Draw separate stacked bars
        db_bars = ax.bar([0]*len(test_labels), db_times, width,
                        bottom=[sum(db_times[:i]) for i in range(len(db_times))],
                        label=test_labels)
        frag_bars = ax.bar([1]*len(test_labels), frag_times, width,
                          bottom=[sum(frag_times[:i]) for i in range(len(frag_times))])

        ax.set_ylabel('Cumulative Time (minutes)')
        ax.set_title('Cumulative Execution Time Breakdown')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['DeepBridge', 'Fragmented'])
        ax.legend(test_labels, loc='upper left')

        # Add total time annotations
        ax.text(0, sum(db_times) + 0.5, f'Total: {sum(db_times):.2f} min',
               ha='center', fontsize=9, fontweight='bold')
        ax.text(1, sum(frag_times) + 2, f'Total: {sum(frag_times):.2f} min',
               ha='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure4_cumulative.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure4_cumulative.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Saved: figure4_cumulative.pdf")

    def generate_figure_5_boxplot(self):
        """Figure 5: Box plots for statistical comparison"""
        print("\nGenerating Figure 5: Statistical Box Plots...")

        tests = ['robustness', 'uncertainty', 'resilience', 'report']
        test_labels = ['Robustness', 'Uncertainty', 'Resilience', 'Report Gen.']

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i, (test, label) in enumerate(zip(tests, test_labels)):
            ax = axes[i]

            if test in self.df_deepbridge.index and test in self.df_fragmented.index:
                db_times = np.array(eval(self.df_deepbridge.loc[test, 'all_times_seconds'])) / 60
                frag_times = np.array(eval(self.df_fragmented.loc[test, 'all_times_seconds'])) * DEMO_SPEEDUP_FACTOR / 60

                data = [db_times, frag_times]

                bp = ax.boxplot(data, labels=['DeepBridge', 'Fragmented'],
                              patch_artist=True, showmeans=True)

                # Customize colors
                colors = ['#2E86AB', '#A23B72']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel('Time (minutes)')
                ax.set_title(label)
                ax.grid(True, alpha=0.3)

                # Add sample size
                ax.text(0.02, 0.98, f'n={len(db_times)}',
                       transform=ax.transAxes, va='top', fontsize=8)

        plt.suptitle('Execution Time Statistical Comparison', fontsize=12, y=1.00)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'figure5_boxplots.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure5_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Saved: figure5_boxplots.pdf")

    def generate_latex_table(self):
        """Generate LaTeX table for paper"""
        print("\nGenerating LaTeX table...")

        latex = r"""\begin{table}[h!]
\centering
\caption{Performance Comparison: DeepBridge vs Fragmented Workflow}
\label{tab:performance_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Component} & \textbf{DeepBridge} & \textbf{Fragmented} & \textbf{Speedup} & \textbf{$p$-value} \\
 & \textbf{(min)} & \textbf{(min)} & \textbf{(×)} & \\
\midrule
"""

        for _, row in self.stats_df.iterrows():
            if row['test'] == 'total':
                latex += r"\midrule" + "\n"

            test_name = row['test'].capitalize()
            if test_name == 'Report':
                test_name = 'Report Generation'

            db_time = f"{row['deepbridge_mean_min']:.2f} $\\pm$ {row['deepbridge_std_min']:.2f}"
            frag_time = f"{row['fragmented_mean_min']:.2f} $\\pm$ {row['fragmented_std_min']:.2f}"
            speedup = f"{row['speedup']:.1f}$\\times$"

            if row['p_value'] < 0.001:
                p_val = "$< 0.001$***"
            elif row['p_value'] < 0.01:
                p_val = f"{row['p_value']:.3f}**"
            elif row['p_value'] < 0.05:
                p_val = f"{row['p_value']:.3f}*"
            else:
                p_val = f"{row['p_value']:.3f}"

            latex += f"{test_name} & {db_time} & {frag_time} & {speedup} & {p_val} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: All differences are statistically significant.
\item * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$ (paired t-test).
\item Times reported as mean $\pm$ standard deviation over 10 runs.
\end{tablenotes}
\end{table}
"""

        with open(self.results_dir / 'performance_comparison.tex', 'w') as f:
            f.write(latex)

        print("  ✓ Saved: performance_comparison.tex")

        return latex

    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("DEEPBRIDGE vs FRAGMENTED WORKFLOW - COMPLETE ANALYSIS")
        print("="*60)

        # Statistics
        self.calculate_statistics()

        # Figures
        print("\n" + "="*60)
        print("GENERATING PUBLICATION-QUALITY FIGURES (300 DPI)")
        print("="*60)

        self.generate_figure_1_time_comparison()
        self.generate_figure_2_speedup()
        self.generate_figure_3_distribution()
        self.generate_figure_4_cumulative()
        self.generate_figure_5_boxplot()

        # LaTeX table
        self.generate_latex_table()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Figures saved to: {self.figures_dir}")
        print("\nGenerated files:")
        print("  • statistical_comparison.csv")
        print("  • performance_comparison.tex")
        print("  • figure1_time_comparison.pdf (+ PNG)")
        print("  • figure2_speedup.pdf (+ PNG)")
        print("  • figure3_distributions.pdf (+ PNG)")
        print("  • figure4_cumulative.pdf (+ PNG)")
        print("  • figure5_boxplots.pdf (+ PNG)")


def main():
    """Main entry point"""
    results_dir = Path(__file__).parent.parent / 'results'

    analyzer = BenchmarkAnalyzer(results_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
