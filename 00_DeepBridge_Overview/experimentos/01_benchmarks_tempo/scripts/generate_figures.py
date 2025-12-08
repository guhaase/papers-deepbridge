"""
Geração de Figuras para o Paper

Cria visualizações dos resultados dos benchmarks.

Autor: DeepBridge Team
Data: 2025-12-05
"""

import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

from utils import (
    load_config,
    ExperimentLogger
)


# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)


class FigureGenerator:
    """Gerador de figuras para o paper"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.results_dir = Path(__file__).parent.parent / config['outputs']['results_dir']
        self.figures_dir = Path(__file__).parent.parent / config['outputs']['figures_dir']
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = config['outputs']['figure_dpi']
        self.format = config['outputs']['figure_format']

    def load_results(self) -> tuple:
        """Carrega resultados"""
        self.logger.info("Loading results...")

        # DeepBridge
        with open(self.results_dir / 'deepbridge_times_REAL.json', 'r') as f:
            deepbridge = json.load(f)

        # Fragmented
        with open(self.results_dir / 'fragmented_times.json', 'r') as f:
            fragmented = json.load(f)

        # Comparison
        comparison_df = pd.read_csv(self.results_dir / 'comparison_summary.csv')

        return deepbridge, fragmented, comparison_df

    def plot_time_comparison_barplot(self, comparison_df: pd.DataFrame):
        """
        Gráfico de barras comparando tempos

        Salva: time_comparison_barplot.pdf
        """
        self.logger.info("Creating time comparison barplot...")

        # Filtrar total
        df = comparison_df[comparison_df['Task'] != 'Total'].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(df))
        width = 0.35

        # Barras
        bars1 = ax.bar(
            x - width/2,
            df['DeepBridge_Mean_Min'],
            width,
            label='DeepBridge',
            color='#2ecc71',
            edgecolor='black',
            linewidth=0.7
        )

        bars2 = ax.bar(
            x + width/2,
            df['Fragmented_Mean_Min'],
            width,
            label='Fragmentado',
            color='#e74c3c',
            edgecolor='black',
            linewidth=0.7
        )

        # Labels
        ax.set_xlabel('Tarefa', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tempo (minutos)', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Tempo: DeepBridge vs. Workflow Fragmentado',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Task'], rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper left')

        # Grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        plt.tight_layout()

        # Salvar
        output_path = self.figures_dir / f'time_comparison_barplot.{self.format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved: {output_path}")

    def plot_speedup_by_task(self, comparison_df: pd.DataFrame):
        """
        Gráfico de speedup por tarefa

        Salva: speedup_by_task.pdf
        """
        self.logger.info("Creating speedup plot...")

        df = comparison_df[comparison_df['Task'] != 'Total'].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Barras horizontais
        y_pos = np.arange(len(df))

        bars = ax.barh(
            y_pos,
            df['Speedup'],
            color='#3498db',
            edgecolor='black',
            linewidth=0.7
        )

        # Linha de referência em 1x (sem speedup)
        ax.axvline(1, color='red', linestyle='--', linewidth=1.5, label='Sem speedup')

        # Labels
        ax.set_xlabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax.set_title('Speedup por Tarefa (Fragmentado / DeepBridge)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Task'])
        ax.legend(fontsize=11)

        # Grid
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Adicionar valores
        for i, (bar, speedup) in enumerate(zip(bars, df['Speedup'])):
            width = bar.get_width()
            ax.text(
                width + 0.2,
                bar.get_y() + bar.get_height()/2,
                f'{speedup:.1f}×',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )

        plt.tight_layout()

        output_path = self.figures_dir / f'speedup_by_task.{self.format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved: {output_path}")

    def plot_reduction_percentage(self, comparison_df: pd.DataFrame):
        """
        Gráfico de redução percentual

        Salva: reduction_percentage.pdf
        """
        self.logger.info("Creating reduction percentage plot...")

        # Incluir total
        df = comparison_df.copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2ecc71' if task != 'Total' else '#27ae60' for task in df['Task']]

        bars = ax.bar(
            df['Task'],
            df['Reduction_Pct'],
            color=colors,
            edgecolor='black',
            linewidth=0.7
        )

        # Linha de meta (89%)
        ax.axhline(89, color='red', linestyle='--', linewidth=2, label='Meta (89%)')

        # Labels
        ax.set_xlabel('Tarefa', fontsize=12, fontweight='bold')
        ax.set_ylabel('Redução de Tempo (%)', fontsize=12, fontweight='bold')
        ax.set_title('Redução Percentual de Tempo com DeepBridge',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(df['Task'], rotation=45, ha='right')
        ax.legend(fontsize=11)

        # Grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Adicionar valores
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        plt.tight_layout()

        output_path = self.figures_dir / f'reduction_percentage.{self.format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved: {output_path}")

    def plot_boxplot_comparison(self, deepbridge: Dict, fragmented: Dict):
        """
        Boxplot comparando distribuições de tempo

        Salva: boxplot_comparison.pdf
        """
        self.logger.info("Creating boxplot comparison...")

        tasks = [t for t in deepbridge.keys() if t != 'total']

        data = []
        labels = []

        for task in tasks:
            # DeepBridge
            db_times = [t/60 for t in deepbridge[task]['all_times_seconds']]  # converter para min
            data.append(db_times)
            labels.append(f"{task.capitalize()}\n(DB)")

            # Fragmented
            frag_times = [t/60 for t in fragmented[task]['all_times_seconds']]
            data.append(frag_times)
            labels.append(f"{task.capitalize()}\n(Frag)")

        fig, ax = plt.subplots(figsize=(14, 7))

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True
        )

        # Colorir: verde para DB, vermelho para Frag
        colors = []
        for i in range(len(tasks)):
            colors.extend(['#2ecc71', '#e74c3c'])

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Labels
        ax.set_ylabel('Tempo (minutos)', fontsize=12, fontweight='bold')
        ax.set_title('Distribuição de Tempos: DeepBridge (DB) vs. Fragmentado (Frag)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

        # Grid
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.6, edgecolor='black', label='DeepBridge'),
            Patch(facecolor='#e74c3c', alpha=0.6, edgecolor='black', label='Fragmentado')
        ]
        ax.legend(handles=legend_elements, fontsize=11, loc='upper left')

        plt.tight_layout()

        output_path = self.figures_dir / f'boxplot_comparison.{self.format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved: {output_path}")

    def plot_total_time_breakdown(self, deepbridge: Dict, fragmented: Dict):
        """
        Stacked bar chart mostrando breakdown do tempo total

        Salva: total_time_breakdown.pdf
        """
        self.logger.info("Creating total time breakdown...")

        tasks = ['fairness', 'robustness', 'uncertainty', 'resilience', 'report']

        # DeepBridge
        db_times = [deepbridge[t]['mean_minutes'] for t in tasks]

        # Fragmented
        frag_times = [fragmented[t]['mean_minutes'] for t in tasks]

        fig, ax = plt.subplots(figsize=(8, 6))

        x = ['DeepBridge', 'Fragmentado']
        width = 0.6

        # Colors
        colors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c', '#f39c12']

        # Stacked bars
        bottom_db = 0
        bottom_frag = 0

        for i, task in enumerate(tasks):
            # DeepBridge
            ax.bar(
                0,
                db_times[i],
                width,
                bottom=bottom_db,
                color=colors[i],
                edgecolor='black',
                linewidth=0.7,
                label=task.capitalize()
            )
            bottom_db += db_times[i]

            # Fragmented
            ax.bar(
                1,
                frag_times[i],
                width,
                bottom=bottom_frag,
                color=colors[i],
                edgecolor='black',
                linewidth=0.7
            )
            bottom_frag += frag_times[i]

        # Labels
        ax.set_ylabel('Tempo Total (minutos)', fontsize=12, fontweight='bold')
        ax.set_title('Breakdown do Tempo Total por Tarefa',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(x)
        ax.legend(fontsize=10, loc='upper right')

        # Adicionar totais
        ax.text(0, bottom_db + 2, f'{bottom_db:.1f} min', ha='center', fontsize=11, fontweight='bold')
        ax.text(1, bottom_frag + 2, f'{bottom_frag:.1f} min', ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()

        output_path = self.figures_dir / f'total_time_breakdown.{self.format}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved: {output_path}")

    def generate_all_figures(self):
        """Gera todas as figuras"""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("FIGURE GENERATION")
        self.logger.info(f"{'=' * 60}\n")

        # Carregar resultados
        deepbridge, fragmented, comparison_df = self.load_results()

        # Gerar figuras
        self.plot_time_comparison_barplot(comparison_df)
        self.plot_speedup_by_task(comparison_df)
        self.plot_reduction_percentage(comparison_df)
        self.plot_boxplot_comparison(deepbridge, fragmented)
        self.plot_total_time_breakdown(deepbridge, fragmented)

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("All figures generated successfully!")
        self.logger.info(f"Figures saved in: {self.figures_dir}")
        self.logger.info(f"{'=' * 60}\n")


def main():
    """Função principal"""
    # Configurar logging
    log_dir = Path(__file__).parent.parent / 'logs'
    exp_logger = ExperimentLogger(log_dir, name='figures')
    logger = exp_logger.get_logger()

    # Carregar config
    config = load_config()

    # Gerar figuras
    generator = FigureGenerator(config)
    generator.generate_all_figures()

    logger.info("\nAll figures generated!")


if __name__ == "__main__":
    main()
