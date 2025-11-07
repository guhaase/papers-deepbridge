# Plano de Reorganização dos Experimentos HPM-KD
## Otimizado para Google Colab + Geração Automática de Relatórios MD

**Data:** 07 de Novembro de 2025
**Objetivo:** Estrutura modular para execução em GPU (Google Colab) com relatórios automáticos em Markdown

---

## 🎯 Objetivos da Reorganização

1. **Execução em Google Colab**: Todos os experimentos rodáveis em Colab (GPU)
2. **Relatórios Automáticos**: Cada experimento gera um `.md` com resultados
3. **Modularidade**: Notebooks independentes que podem rodar separadamente
4. **Reprodutibilidade**: Seeds fixos, configurações documentadas
5. **Rastreabilidade**: Cada resultado salvo com timestamp e configurações

---

## 📂 Nova Estrutura Proposta

```
experiments/
├── 📓 notebooks/                          # Notebooks Colab (execução principal)
│   ├── 00_setup.ipynb                     # Setup inicial (instalar DeepBridge, configs)
│   │
│   ├── 01_sklearn_baselines.ipynb         # Exp 1: Sklearn baselines (quick)
│   ├── 02_sklearn_hpmkd.ipynb             # Exp 2: HPM-KD com sklearn
│   │
│   ├── 03_cnn_mnist_teacher.ipynb         # Exp 3: Treinar teacher CNN (MNIST)
│   ├── 04_cnn_mnist_baselines.ipynb       # Exp 4: Baselines CNN (Direct, KD, FitNets)
│   ├── 05_cnn_mnist_hpmkd.ipynb           # Exp 5: HPM-KD CNN (full framework)
│   │
│   ├── 06_cifar10_experiments.ipynb       # Exp 6: CIFAR-10 (teacher + baselines + HPM-KD)
│   ├── 07_ablation_studies.ipynb          # Exp 7: Ablation (remover componentes)
│   │
│   ├── 08_compression_analysis.ipynb      # Exp 8: Análise de compression ratios
│   ├── 09_multi_dataset.ipynb             # Exp 9: UCI datasets (tabular)
│   │
│   └── 10_generate_paper_results.ipynb    # Exp 10: Gerar todas as tabelas/figuras do paper
│
├── 🐍 scripts/                            # Scripts Python (funções reusáveis)
│   ├── __init__.py
│   ├── models.py                          # Definições de modelos (CNN, ResNet, etc)
│   ├── training.py                        # Funções de treinamento
│   ├── evaluation.py                      # Funções de avaliação
│   ├── hpmkd.py                           # HPM-KD framework wrapper
│   ├── data_loaders.py                    # Carregamento de datasets
│   ├── baselines.py                       # Implementações de baselines (KD, FitNets, etc)
│   └── report_generator.py                # 🌟 GERADOR DE RELATÓRIOS MD
│
├── 📊 results/                            # Resultados organizados por experimento
│   ├── 01_sklearn_baselines/
│   │   ├── report.md                      # 🌟 RELATÓRIO GERADO
│   │   ├── metrics.json
│   │   ├── results.csv
│   │   └── figures/
│   │       ├── accuracy_comparison.png
│   │       └── training_curves.png
│   │
│   ├── 02_sklearn_hpmkd/
│   │   ├── report.md
│   │   ├── metrics.json
│   │   ├── hpmkd_config.json
│   │   └── figures/
│   │
│   ├── 03_cnn_mnist_teacher/
│   │   ├── report.md
│   │   ├── teacher_model.pth
│   │   ├── training_log.json
│   │   └── figures/
│   │
│   ├── 04_cnn_mnist_baselines/
│   │   ├── report.md
│   │   ├── comparison_table.csv
│   │   └── figures/
│   │
│   ├── 05_cnn_mnist_hpmkd/
│   │   ├── report.md
│   │   ├── hpmkd_results.json
│   │   ├── student_model.pth
│   │   └── figures/
│   │
│   ├── 06_cifar10_experiments/
│   ├── 07_ablation_studies/
│   ├── 08_compression_analysis/
│   ├── 09_multi_dataset/
│   │
│   └── paper_final/                       # 🌟 Resultados finais para o paper
│       ├── FINAL_REPORT.md                # Relatório consolidado
│       ├── table1_compression_results.csv
│       ├── table2_ablation_results.csv
│       ├── figure1_performance.pdf
│       ├── figure2_retention.pdf
│       └── ...
│
├── 🔧 configs/                            # Configurações experimentais
│   ├── mnist_config.yaml
│   ├── cifar10_config.yaml
│   ├── ablation_config.yaml
│   └── hpmkd_defaults.yaml
│
├── 📖 templates/                          # Templates de relatórios
│   ├── experiment_report.md.j2           # Template Jinja2 para relatórios
│   ├── final_report.md.j2                # Template para relatório final
│   └── table_templates/
│       ├── table_compression.md.j2
│       ├── table_ablation.md.j2
│       └── table_comparison.md.j2
│
└── 📚 docs/
    ├── README.md                          # Documentação geral
    ├── COLAB_SETUP.md                     # Guia de setup no Colab
    ├── EXPERIMENT_GUIDE.md                # Guia de execução de experimentos
    └── RESULTS_INTERPRETATION.md          # Como interpretar os resultados
```

---

## 🚀 Fluxo de Execução no Google Colab

### Passo 1: Setup Inicial (Uma vez)

```python
# No notebook: 00_setup.ipynb

# 1. Clone repositório
!git clone https://github.com/DeepBridge-Validation/DeepBridge.git
%cd DeepBridge

# 2. Instalar dependências
!pip install -e .
!pip install jinja2 pyyaml

# 3. Verificar GPU
import torch
print(f"GPU disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 4. Setup Google Drive (para salvar resultados)
from google.colab import drive
drive.mount('/content/drive')

# 5. Criar estrutura de diretórios
!mkdir -p /content/drive/MyDrive/HPM-KD-Results
```

### Passo 2: Executar Experimentos (Um por vez ou todos)

Cada notebook segue este template:

```python
# Exemplo: 03_cnn_mnist_teacher.ipynb

# ============================================
# 1. IMPORTS E SETUP
# ============================================
import sys
sys.path.append('/content/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments')

from scripts.models import create_teacher_cnn
from scripts.training import train_model
from scripts.evaluation import evaluate_model
from scripts.report_generator import ExperimentReporter

# ============================================
# 2. CONFIGURAÇÃO DO EXPERIMENTO
# ============================================
config = {
    'experiment_name': '03_cnn_mnist_teacher',
    'dataset': 'MNIST',
    'model': 'ResNet18',
    'epochs': 20,
    'batch_size': 128,
    'lr': 0.1,
    'seed': 42,
    'device': 'cuda'
}

# ============================================
# 3. EXECUTAR EXPERIMENTO
# ============================================
reporter = ExperimentReporter(
    experiment_name=config['experiment_name'],
    output_dir='/content/drive/MyDrive/HPM-KD-Results'
)

# 3.1. Treinar modelo
model, history = train_model(config)

# 3.2. Avaliar modelo
metrics = evaluate_model(model, config)

# 3.3. Salvar resultados
reporter.log_metrics(metrics)
reporter.log_config(config)
reporter.save_model(model, 'teacher_model.pth')
reporter.plot_training_curves(history)

# ============================================
# 4. GERAR RELATÓRIO MD
# ============================================
reporter.generate_markdown_report()

# ============================================
# 5. EXIBIR RESUMO
# ============================================
reporter.display_summary()
```

### Passo 3: Geração de Relatório Final

```python
# No notebook: 10_generate_paper_results.ipynb

from scripts.report_generator import FinalReportGenerator

generator = FinalReportGenerator(
    results_dir='/content/drive/MyDrive/HPM-KD-Results',
    output_dir='/content/drive/MyDrive/HPM-KD-Results/paper_final'
)

# Consolidar todos os experimentos
generator.consolidate_results()

# Gerar tabelas do paper
generator.generate_table1_compression()
generator.generate_table2_ablation()
generator.generate_table3_comparison()

# Gerar figuras do paper
generator.generate_figure1_performance()
generator.generate_figure2_retention()
generator.generate_figure3_ablation()

# Gerar relatório final
generator.generate_final_report()

print("✅ Relatório final gerado em: paper_final/FINAL_REPORT.md")
```

---

## 📊 Template de Relatório Markdown Gerado

Cada experimento gera um `report.md` neste formato:

```markdown
# Relatório de Experimento: 03_cnn_mnist_teacher

**Data de Execução:** 2025-11-07 14:32:15
**Duração Total:** 18m 45s
**GPU Utilizada:** Tesla T4

---

## 📋 Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| Dataset | MNIST |
| Modelo | ResNet18 |
| Epochs | 20 |
| Batch Size | 128 |
| Learning Rate | 0.1 |
| Optimizer | SGD (momentum=0.9) |
| Seed | 42 |

---

## 📈 Resultados Principais

### Performance Final

| Métrica | Valor |
|---------|-------|
| Test Accuracy | 99.42% |
| Train Accuracy | 99.87% |
| Best Epoch | 18 |
| Final Loss | 0.0234 |

### Comparação com Baseline

| Modelo | Accuracy | Parâmetros | Compression |
|--------|----------|------------|-------------|
| Teacher (este) | 99.42% | 11.2M | 1× |
| Direct Student | 98.12% | 1.1M | 10.2× |

**Melhoria sobre baseline:** +1.30 pp

---

## 📊 Visualizações

### Curvas de Treinamento
![Training Curves](figures/training_curves.png)

### Confusion Matrix
![Confusion Matrix](figures/confusion_matrix.png)

### Accuracy por Classe
![Per-Class Accuracy](figures/per_class_accuracy.png)

---

## 💾 Arquivos Salvos

- ✅ `teacher_model.pth` (42.3 MB)
- ✅ `training_log.json` (15.2 KB)
- ✅ `metrics.json` (2.1 KB)
- ✅ `config.json` (1.3 KB)
- ✅ Figuras: 3 arquivos PNG

---

## 🔍 Análise e Observações

### Convergência
- Modelo convergiu rapidamente (epoch 12)
- Nenhum overfitting detectado
- Learning rate decay funcionou bem

### Performance
- Accuracy superior a 99% em todas as classes
- Melhor performance: Classe 1 (99.8%)
- Pior performance: Classe 8 (98.9%)

### Recursos Computacionais
- Tempo por epoch: ~56 segundos
- GPU memory usage: 3.2 GB / 15 GB
- Training efficiency: 95% GPU utilization

---

## ✅ Checklist de Validação

- [x] Accuracy > 99% (Target: 99.3-99.5%)
- [x] Modelo salvo corretamente
- [x] Todas as figuras geradas
- [x] Métricas registradas
- [x] Reprodutível (seed fixado)

---

## 🔄 Próximos Passos

1. **Experimento 04:** Treinar baselines (Direct, KD, FitNets)
2. **Experimento 05:** Rodar HPM-KD completo
3. **Comparação:** Gerar tabela comparativa

---

## 📌 Notas Adicionais

- Teacher model pronto para distillation
- Performance dentro do esperado para o paper
- Todos os checkpoints salvos para reprodução

---

**Gerado automaticamente por:** ExperimentReporter v1.0
**Notebook:** `03_cnn_mnist_teacher.ipynb`
```

---

## 🔧 Sistema de Geração de Relatórios

### Classe Principal: `ExperimentReporter`

```python
# scripts/report_generator.py

import json
import yaml
from pathlib import Path
from datetime import datetime
from jinja2 import Template
import matplotlib.pyplot as plt
import pandas as pd

class ExperimentReporter:
    """
    Gerador automático de relatórios Markdown para experimentos.

    Usage:
        reporter = ExperimentReporter('03_cnn_mnist_teacher', output_dir='results/')
        reporter.log_metrics({'accuracy': 0.9942})
        reporter.log_config({'epochs': 20, 'lr': 0.1})
        reporter.generate_markdown_report()
    """

    def __init__(self, experiment_name, output_dir='results/'):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)

        self.start_time = datetime.now()
        self.metrics = {}
        self.config = {}
        self.observations = []

    def log_metrics(self, metrics_dict):
        """Log métricas do experimento"""
        self.metrics.update(metrics_dict)

    def log_config(self, config_dict):
        """Log configuração do experimento"""
        self.config.update(config_dict)

    def save_model(self, model, filename):
        """Salvar modelo treinado"""
        import torch
        path = self.output_dir / filename
        torch.save(model.state_dict(), path)
        self.log_metrics({'model_saved': str(path)})

    def plot_training_curves(self, history):
        """Gerar plot de curvas de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        ax1.plot(history['train_acc'], label='Train')
        ax1.plot(history['val_acc'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(history['train_loss'], label='Train')
        ax2.plot(history['val_loss'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'training_curves.png', dpi=300)
        plt.close()

    def add_observation(self, observation):
        """Adicionar observação textual"""
        self.observations.append(observation)

    def generate_markdown_report(self):
        """Gerar relatório completo em Markdown"""

        # Calcular duração
        duration = datetime.now() - self.start_time

        # Carregar template
        template_path = Path(__file__).parent.parent / 'templates' / 'experiment_report.md.j2'
        with open(template_path) as f:
            template = Template(f.read())

        # Renderizar template
        report = template.render(
            experiment_name=self.experiment_name,
            timestamp=self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            duration=str(duration),
            config=self.config,
            metrics=self.metrics,
            observations=self.observations,
            figures_dir='figures/'
        )

        # Salvar relatório
        report_path = self.output_dir / 'report.md'
        with open(report_path, 'w') as f:
            f.write(report)

        # Salvar métricas e config em JSON
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ Relatório gerado: {report_path}")
        return report_path

    def display_summary(self):
        """Exibir resumo no notebook"""
        from IPython.display import Markdown, display

        summary = f"""
        ## ✅ Experimento Concluído: {self.experiment_name}

        **Duração:** {datetime.now() - self.start_time}

        ### Métricas Principais
        {self._format_metrics_table()}

        ### Arquivos Salvos
        - Relatório: `{self.output_dir / 'report.md'}`
        - Métricas: `{self.output_dir / 'metrics.json'}`
        - Figuras: `{self.figures_dir}/`
        """

        display(Markdown(summary))

    def _format_metrics_table(self):
        """Formatar métricas como tabela MD"""
        rows = []
        for key, value in self.metrics.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            rows.append(f"| {key} | {value} |")

        return "| Métrica | Valor |\n|---------|-------|\n" + "\n".join(rows)
```

---

## 📦 Template Jinja2 para Relatórios

```jinja2
{# templates/experiment_report.md.j2 #}

# Relatório de Experimento: {{ experiment_name }}

**Data de Execução:** {{ timestamp }}
**Duração Total:** {{ duration }}

---

## 📋 Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
{% for key, value in config.items() %}
| {{ key }} | {{ value }} |
{% endfor %}

---

## 📈 Resultados Principais

### Performance Final

| Métrica | Valor |
|---------|-------|
{% for key, value in metrics.items() %}
{% if value is number %}
| {{ key }} | {{ "%.4f"|format(value) }} |
{% else %}
| {{ key }} | {{ value }} |
{% endif %}
{% endfor %}

---

## 📊 Visualizações

{% if figures_exist %}
### Curvas de Treinamento
![Training Curves]({{ figures_dir }}/training_curves.png)

### Confusion Matrix
![Confusion Matrix]({{ figures_dir }}/confusion_matrix.png)
{% endif %}

---

## 🔍 Análise e Observações

{% for obs in observations %}
- {{ obs }}
{% endfor %}

---

## 📌 Notas Adicionais

**Gerado automaticamente por:** ExperimentReporter v1.0
**Timestamp:** {{ timestamp }}
```

---

## 🎯 Experimentos Prioritários (Sequência Sugerida)

### Fase 1: Validação Rápida (1-2 horas)
1. ✅ **01_sklearn_baselines.ipynb** → Baseline sklearn (10 min)
2. ✅ **02_sklearn_hpmkd.ipynb** → HPM-KD sklearn (15 min)

### Fase 2: CNN MNIST (3-4 horas)
3. 🔄 **03_cnn_mnist_teacher.ipynb** → Teacher ResNet18 (30 min)
4. 🔄 **04_cnn_mnist_baselines.ipynb** → Direct, KD, FitNets (45 min cada)
5. 🔄 **05_cnn_mnist_hpmkd.ipynb** → HPM-KD completo (60 min)

### Fase 3: CIFAR-10 (4-6 horas)
6. ⏳ **06_cifar10_experiments.ipynb** → Teacher + Baselines + HPM-KD (2-3 horas)

### Fase 4: Análises (2-3 horas)
7. ⏳ **07_ablation_studies.ipynb** → Remover componentes (1 hora)
8. ⏳ **08_compression_analysis.ipynb** → Diferentes compression ratios (1 hora)
9. ⏳ **09_multi_dataset.ipynb** → UCI datasets (30 min)

### Fase 5: Paper Final (1 hora)
10. ⏳ **10_generate_paper_results.ipynb** → Consolidar tudo (1 hora)

**Tempo Total Estimado:** 12-16 horas de GPU

---

## 📌 Checklist de Implementação

### Estrutura de Diretórios
- [ ] Criar `notebooks/` com 10 notebooks
- [ ] Criar `scripts/` com módulos Python
- [ ] Criar `templates/` com templates Jinja2
- [ ] Criar `configs/` com arquivos YAML
- [ ] Reorganizar `results/` por experimento

### Scripts Python
- [ ] `models.py` → Definições de modelos
- [ ] `training.py` → Funções de treinamento
- [ ] `evaluation.py` → Funções de avaliação
- [ ] `hpmkd.py` → HPM-KD wrapper
- [ ] `data_loaders.py` → Datasets
- [ ] `baselines.py` → Implementações de baselines
- [ ] `report_generator.py` → Gerador de relatórios MD

### Templates
- [ ] `experiment_report.md.j2` → Template de experimento
- [ ] `final_report.md.j2` → Template de relatório final
- [ ] Templates de tabelas (3 tipos)

### Notebooks
- [ ] `00_setup.ipynb`
- [ ] `01-10`: 10 notebooks de experimentos

### Documentação
- [ ] `COLAB_SETUP.md` → Guia de setup
- [ ] `EXPERIMENT_GUIDE.md` → Guia de execução
- [ ] `RESULTS_INTERPRETATION.md` → Interpretação de resultados

---

## ✅ Benefícios da Nova Estrutura

1. **✅ Modularidade**: Cada experimento é independente
2. **✅ Reprodutibilidade**: Seeds fixos, configs documentadas
3. **✅ Rastreabilidade**: Cada resultado tem timestamp e configuração
4. **✅ Automatização**: Relatórios MD gerados automaticamente
5. **✅ Google Colab Ready**: Notebooks prontos para GPU
6. **✅ Incremental**: Pode rodar um experimento por vez
7. **✅ Organização**: Resultados centralizados por experimento
8. **✅ Paper-Ready**: Geração automática de tabelas e figuras do paper

---

## 🚀 Próximos Passos

1. **Revisar e aprovar** esta proposta
2. **Migrar código existente** para nova estrutura
3. **Criar templates** de notebooks e relatórios
4. **Testar no Colab** (experimento piloto)
5. **Executar todos os experimentos** sequencialmente
6. **Gerar relatório final** para o paper

---

**Autor:** Claude (Anthropic)
**Data:** 07/11/2025
**Versão:** 1.0
