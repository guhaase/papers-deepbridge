#!/usr/bin/env python3
"""
HPM-KD Experiment Report Generator
===================================

Gera relatórios automáticos em Markdown para experimentos de Knowledge Distillation.

Features:
- Log automático de métricas, configurações e observações
- Geração de figuras (training curves, confusion matrix, etc.)
- Template-based Markdown reports
- Export para JSON, CSV
- Integração com Google Colab

Author: Gustavo Coelho Haase
Date: November 2025
"""

import json
import yaml
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ExperimentReporter:
    """
    Gerador automático de relatórios Markdown para experimentos.

    Usage:
        reporter = ExperimentReporter('03_cnn_mnist_teacher', output_dir='results/')
        reporter.log_metrics({'accuracy': 0.9942})
        reporter.log_config({'epochs': 20, 'lr': 0.1})
        reporter.plot_training_curves(history)
        reporter.generate_markdown_report()
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path] = 'results/',
        description: str = '',
        gpu_name: Optional[str] = None
    ):
        """
        Inicializa o reporter.

        Args:
            experiment_name: Nome do experimento (ex: '03_cnn_mnist_teacher')
            output_dir: Diretório base para salvar resultados
            description: Descrição curta do experimento
            gpu_name: Nome da GPU utilizada (auto-detectado se None)
        """
        self.experiment_name = experiment_name
        self.description = description
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

        self.start_time = datetime.now()
        self.metrics = {}
        self.config = {}
        self.observations = []
        self.saved_files = []

        # Auto-detect GPU
        if gpu_name is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_name = torch.cuda.get_device_name(0)
                else:
                    self.gpu_name = 'CPU'
            except ImportError:
                self.gpu_name = 'Unknown'
        else:
            self.gpu_name = gpu_name

        # Configurar estilo de plots
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300

    def log_metrics(self, metrics_dict: Dict[str, Any]):
        """
        Log métricas do experimento.

        Args:
            metrics_dict: Dicionário com métricas (ex: {'accuracy': 0.99})
        """
        self.metrics.update(metrics_dict)

    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuração do experimento.

        Args:
            config_dict: Dicionário com configurações (ex: {'epochs': 20})
        """
        self.config.update(config_dict)

    def add_observation(self, observation: str):
        """
        Adicionar observação textual ao relatório.

        Args:
            observation: Texto da observação
        """
        self.observations.append(observation)

    def save_model(self, model, filename: str):
        """
        Salvar modelo treinado.

        Args:
            model: Modelo PyTorch
            filename: Nome do arquivo (ex: 'teacher_model.pth')
        """
        import torch
        path = self.output_dir / filename
        torch.save(model.state_dict(), path)

        # Log file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        self.saved_files.append({
            'name': filename,
            'path': str(path),
            'size_mb': f'{file_size_mb:.1f}'
        })
        print(f"✅ Modelo salvo: {path} ({file_size_mb:.1f} MB)")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = 'training_curves.png'
    ):
        """
        Gerar plot de curvas de treinamento.

        Args:
            history: Dicionário com histórico (ex: {'train_acc': [...], 'val_acc': [...]})
            filename: Nome do arquivo de saída
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        if 'train_acc' in history and 'val_acc' in history:
            epochs = range(1, len(history['train_acc']) + 1)
            ax1.plot(epochs, history['train_acc'], 'o-', label='Train', linewidth=2)
            ax1.plot(epochs, history['val_acc'], 's-', label='Validation', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

        # Loss
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax2.plot(epochs, history['train_loss'], 'o-', label='Train', linewidth=2)
            ax2.plot(epochs, history['val_loss'], 's-', label='Validation', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.figures_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.saved_files.append({'name': filename, 'path': str(save_path), 'type': 'figure'})
        print(f"✅ Curvas de treinamento salvas: {save_path}")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        filename: str = 'confusion_matrix.png'
    ):
        """
        Gerar plot de confusion matrix.

        Args:
            cm: Confusion matrix (numpy array)
            class_names: Nomes das classes
            filename: Nome do arquivo de saída
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalizar para percentuais
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot heatmap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            ax=ax,
            xticklabels=class_names if class_names else range(cm.shape[0]),
            yticklabels=class_names if class_names else range(cm.shape[0]),
            cbar_kws={'label': 'Percentage (%)'}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = self.figures_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.saved_files.append({'name': filename, 'path': str(save_path), 'type': 'figure'})
        print(f"✅ Confusion matrix salva: {save_path}")

    def plot_comparison_bar(
        self,
        comparison_data: Dict[str, float],
        metric_name: str = 'Accuracy',
        filename: str = 'comparison.png',
        ylabel: str = 'Accuracy (%)'
    ):
        """
        Gerar gráfico de barras para comparação de métodos.

        Args:
            comparison_data: Dict com {method: value} (ex: {'Direct': 0.98, 'KD': 0.99})
            metric_name: Nome da métrica
            filename: Nome do arquivo
            ylabel: Label do eixo Y
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(comparison_data.keys())
        values = [v * 100 if v < 1.5 else v for v in comparison_data.values()]  # Convert to %

        bars = ax.bar(methods, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])

        # Adicionar valores no topo das barras
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.2f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(values) * 1.1])
        ax.grid(True, axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = self.figures_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.saved_files.append({'name': filename, 'path': str(save_path), 'type': 'figure'})
        print(f"✅ Gráfico de comparação salvo: {save_path}")

    def generate_markdown_report(self) -> Path:
        """
        Gerar relatório completo em Markdown.

        Returns:
            Path do relatório gerado
        """
        # Calcular duração
        duration = datetime.now() - self.start_time
        duration_str = self._format_duration(duration)

        # Construir relatório
        report_lines = []

        # Header
        report_lines.append(f"# Relatório de Experimento: {self.experiment_name}\n")
        report_lines.append(f"**Data de Execução:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Duração Total:** {duration_str}")
        report_lines.append(f"**GPU Utilizada:** {self.gpu_name}\n")

        if self.description:
            report_lines.append(f"**Descrição:** {self.description}\n")

        report_lines.append("---\n")

        # Configuração
        if self.config:
            report_lines.append("## 📋 Configuração do Experimento\n")
            report_lines.append("| Parâmetro | Valor |")
            report_lines.append("|-----------|-------|")
            for key, value in self.config.items():
                report_lines.append(f"| {key} | {value} |")
            report_lines.append("\n---\n")

        # Resultados
        if self.metrics:
            report_lines.append("## 📈 Resultados Principais\n")
            report_lines.append("### Performance Final\n")
            report_lines.append("| Métrica | Valor |")
            report_lines.append("|---------|-------|")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"| {key} | {value:.4f} |")
                else:
                    report_lines.append(f"| {key} | {value} |")
            report_lines.append("\n---\n")

        # Visualizações
        figure_files = [f for f in self.saved_files if f.get('type') == 'figure']
        if figure_files:
            report_lines.append("## 📊 Visualizações\n")
            for fig in figure_files:
                fig_name = fig['name'].replace('_', ' ').replace('.png', '').title()
                report_lines.append(f"### {fig_name}")
                report_lines.append(f"![{fig_name}](figures/{fig['name']})\n")
            report_lines.append("---\n")

        # Observações
        if self.observations:
            report_lines.append("## 🔍 Análise e Observações\n")
            for obs in self.observations:
                report_lines.append(f"- {obs}")
            report_lines.append("\n---\n")

        # Arquivos salvos
        if self.saved_files:
            report_lines.append("## 💾 Arquivos Salvos\n")
            for file_info in self.saved_files:
                name = file_info['name']
                size = file_info.get('size_mb', '')
                size_str = f" ({size} MB)" if size else ""
                report_lines.append(f"- ✅ `{name}`{size_str}")
            report_lines.append("\n---\n")

        # Footer
        report_lines.append("## 📌 Notas Adicionais\n")
        report_lines.append(f"**Gerado automaticamente por:** ExperimentReporter v1.0")
        report_lines.append(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Salvar relatório
        report_path = self.output_dir / 'report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # Salvar métricas e config em JSON
        self._save_json(self.metrics, 'metrics.json')
        self._save_json(self.config, 'config.json')

        # Salvar resultados em CSV (se tiver métricas numéricas)
        self._save_csv()

        print(f"\n✅ Relatório gerado: {report_path}")
        return report_path

    def _save_json(self, data: Dict, filename: str):
        """Salvar dados em JSON"""
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_csv(self):
        """Salvar métricas em CSV"""
        if not self.metrics:
            return

        # Tentar converter métricas para DataFrame
        try:
            df = pd.DataFrame([self.metrics])
            csv_path = self.output_dir / 'results.csv'
            df.to_csv(csv_path, index=False)
            print(f"✅ Resultados salvos: {csv_path}")
        except Exception as e:
            print(f"⚠️ Não foi possível salvar CSV: {e}")

    def _format_duration(self, duration: timedelta) -> str:
        """Formatar duração como string legível"""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def display_summary(self):
        """
        Exibir resumo no notebook (Jupyter/Colab).
        """
        try:
            from IPython.display import Markdown, display

            summary = f"""
## ✅ Experimento Concluído: {self.experiment_name}

**Duração:** {self._format_duration(datetime.now() - self.start_time)}
**GPU:** {self.gpu_name}

### Métricas Principais
{self._format_metrics_table()}

### Arquivos Salvos
- 📄 Relatório: `{self.output_dir / 'report.md'}`
- 📊 Métricas: `{self.output_dir / 'metrics.json'}`
- 📁 Figuras: `{self.figures_dir}/` ({len([f for f in self.saved_files if f.get('type') == 'figure'])} arquivos)
            """

            display(Markdown(summary))

        except ImportError:
            # Fallback se não estiver em notebook
            print(f"\n{'='*60}")
            print(f"✅ Experimento Concluído: {self.experiment_name}")
            print(f"{'='*60}")
            print(f"Duração: {self._format_duration(datetime.now() - self.start_time)}")
            print(f"Relatório: {self.output_dir / 'report.md'}")
            print(f"{'='*60}\n")

    def _format_metrics_table(self) -> str:
        """Formatar métricas como tabela MD"""
        if not self.metrics:
            return "_Nenhuma métrica registrada_"

        rows = ["| Métrica | Valor |", "|---------|-------|"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            rows.append(f"| {key} | {value_str} |")

        return "\n".join(rows)


class FinalReportGenerator:
    """
    Gerador de relatório final consolidando todos os experimentos.

    Usage:
        generator = FinalReportGenerator(results_dir='results/', output_dir='paper_final/')
        generator.consolidate_results()
        generator.generate_final_report()
    """

    def __init__(self, results_dir: Union[str, Path], output_dir: Union[str, Path]):
        """
        Args:
            results_dir: Diretório com resultados de todos os experimentos
            output_dir: Diretório para salvar relatório final
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = {}

    def consolidate_results(self):
        """Consolidar resultados de todos os experimentos"""
        print("🔄 Consolidando resultados de todos os experimentos...")

        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            metrics_file = exp_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file) as f:
                    self.all_results[exp_dir.name] = json.load(f)

        print(f"✅ {len(self.all_results)} experimentos consolidados")

    def generate_comparison_table(self, filename: str = 'table_comparison.csv'):
        """Gerar tabela de comparação de todos os métodos"""
        if not self.all_results:
            print("⚠️ Nenhum resultado para consolidar")
            return

        df = pd.DataFrame(self.all_results).T
        csv_path = self.output_dir / filename
        df.to_csv(csv_path)

        print(f"✅ Tabela de comparação salva: {csv_path}")
        return csv_path

    def generate_final_report(self):
        """Gerar relatório final consolidado"""
        report_lines = []

        report_lines.append("# Relatório Final: HPM-KD Framework - Todos os Experimentos\n")
        report_lines.append(f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append("---\n")

        report_lines.append("## 📊 Resumo de Todos os Experimentos\n")
        report_lines.append(f"**Total de Experimentos:** {len(self.all_results)}\n")

        # Tabela consolidada
        if self.all_results:
            report_lines.append("## 📈 Tabela Consolidada de Resultados\n")
            df = pd.DataFrame(self.all_results).T
            report_lines.append(df.to_markdown())
            report_lines.append("\n---\n")

        # Salvar
        report_path = self.output_dir / 'FINAL_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"✅ Relatório final gerado: {report_path}")
        return report_path


# ============================================
# EXEMPLO DE USO
# ============================================
if __name__ == '__main__':
    # Criar reporter
    reporter = ExperimentReporter(
        experiment_name='example_test',
        description='Teste do sistema de relatórios'
    )

    # Log de configuração
    reporter.log_config({
        'dataset': 'MNIST',
        'model': 'ResNet18',
        'epochs': 20,
        'batch_size': 128,
        'lr': 0.1,
        'seed': 42
    })

    # Log de métricas
    reporter.log_metrics({
        'test_accuracy': 0.9942,
        'train_accuracy': 0.9987,
        'best_epoch': 18,
        'final_loss': 0.0234
    })

    # Simular curvas de treinamento
    history = {
        'train_acc': [0.8 + i * 0.02 for i in range(20)],
        'val_acc': [0.75 + i * 0.022 for i in range(20)],
        'train_loss': [2.0 - i * 0.09 for i in range(20)],
        'val_loss': [2.1 - i * 0.085 for i in range(20)]
    }
    reporter.plot_training_curves(history)

    # Adicionar observações
    reporter.add_observation("Modelo convergiu rapidamente (epoch 12)")
    reporter.add_observation("Nenhum overfitting detectado")
    reporter.add_observation("GPU utilization: 95%")

    # Gerar relatório
    reporter.generate_markdown_report()
    reporter.display_summary()

    print("\n✅ Exemplo concluído! Verifique o relatório em: results/example_test/report.md")
