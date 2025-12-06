# Estrutura Sugerida para Organização dos Experimentos

## Diretórios Recomendados

```
experimentos/
├── README.md                          # Visão geral (já criado)
├── ESTRUTURA_SUGERIDA.md             # Este arquivo
├── 01_benchmarks_tempo.md            # Documentação (já criado)
├── 02_estudos_de_caso.md             # Documentação (já criado)
├── 03_usabilidade.md                 # Documentação (já criado)
├── 04_hpmkd.md                       # Documentação (já criado)
├── 05_conformidade.md                # Documentação (já criado)
├── 06_ablation_studies.md            # Documentação (já criado)
│
├── datasets/                          # Datasets utilizados
│   ├── README.md
│   ├── raw/                          # Dados brutos baixados
│   │   ├── adult_income.csv
│   │   ├── german_credit.csv
│   │   ├── compas.csv
│   │   └── ...
│   ├── processed/                    # Dados processados
│   │   ├── adult_income_train.csv
│   │   ├── adult_income_test.csv
│   │   └── ...
│   └── synthetic/                    # Dados sintéticos gerados
│       ├── credit_scoring_ground_truth.csv
│       ├── hiring_ground_truth.csv
│       └── ...
│
├── models/                           # Modelos treinados
│   ├── README.md
│   ├── teachers/                     # Teacher models (HPM-KD)
│   │   ├── adult_xgb.pkl
│   │   ├── adult_lgbm.pkl
│   │   ├── adult_catboost.pkl
│   │   └── ...
│   ├── students/                     # Student models (HPM-KD)
│   │   ├── adult_hpmkd.pkl
│   │   ├── adult_vanilla_kd.pkl
│   │   └── ...
│   └── case_studies/                 # Modelos dos estudos de caso
│       ├── credit_scoring_xgb.pkl
│       ├── hiring_rf.pkl
│       └── ...
│
├── scripts/                          # Scripts Python
│   ├── README.md
│   ├── common/                       # Utilidades comuns
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   ├── 01_benchmarks_tempo/
│   │   ├── run_deepbridge.py
│   │   ├── run_fragmented.py
│   │   ├── compare_times.py
│   │   └── generate_figures.py
│   │
│   ├── 02_estudos_de_caso/
│   │   ├── 01_credit_scoring.py
│   │   ├── 02_hiring.py
│   │   ├── 03_healthcare.py
│   │   ├── 04_mortgage.py
│   │   ├── 05_insurance.py
│   │   ├── 06_fraud.py
│   │   └── aggregate_results.py
│   │
│   ├── 03_usabilidade/
│   │   ├── setup_study.py
│   │   ├── analyze_sus.py
│   │   ├── analyze_nasa_tlx.py
│   │   └── generate_report.py
│   │
│   ├── 04_hpmkd/
│   │   ├── train_teachers.py
│   │   ├── train_vanilla_kd.py
│   │   ├── train_takd.py
│   │   ├── train_auto_kd.py
│   │   ├── train_hpmkd.py
│   │   ├── evaluate_all.py
│   │   ├── ablation_study.py
│   │   └── generate_figures.py
│   │
│   ├── 05_conformidade/
│   │   ├── generate_ground_truth.py
│   │   ├── run_deepbridge.py
│   │   ├── run_baselines.py
│   │   ├── calculate_metrics.py
│   │   └── generate_figures.py
│   │
│   └── 06_ablation_studies/
│       ├── run_all_configs.py
│       ├── analyze_contributions.py
│       └── generate_figures.py
│
├── notebooks/                        # Jupyter notebooks para análise
│   ├── README.md
│   ├── 01_benchmarks_tempo_analysis.ipynb
│   ├── 02_case_studies_analysis.ipynb
│   ├── 03_usabilidade_analysis.ipynb
│   ├── 04_hpmkd_analysis.ipynb
│   ├── 05_conformidade_analysis.ipynb
│   ├── 06_ablation_analysis.ipynb
│   └── exploratory/                  # Análises exploratórias
│       └── ...
│
├── results/                          # Resultados dos experimentos
│   ├── README.md
│   ├── 01_benchmarks_tempo/
│   │   ├── deepbridge_times.csv
│   │   ├── fragmented_times.csv
│   │   ├── statistical_analysis.json
│   │   └── summary.json
│   │
│   ├── 02_estudos_de_caso/
│   │   ├── credit_scoring_results.json
│   │   ├── hiring_results.json
│   │   ├── healthcare_results.json
│   │   ├── mortgage_results.json
│   │   ├── insurance_results.json
│   │   ├── fraud_results.json
│   │   └── aggregated_stats.csv
│   │
│   ├── 03_usabilidade/
│   │   ├── sus_scores.csv
│   │   ├── nasa_tlx.csv
│   │   ├── task_times.csv
│   │   ├── errors.csv
│   │   ├── feedback.json
│   │   └── statistical_analysis.json
│   │
│   ├── 04_hpmkd/
│   │   ├── teacher_results.csv
│   │   ├── vanilla_kd_results.csv
│   │   ├── takd_results.csv
│   │   ├── auto_kd_results.csv
│   │   ├── hpmkd_results.csv
│   │   ├── ablation_results.csv
│   │   └── aggregated_stats.json
│   │
│   ├── 05_conformidade/
│   │   ├── ground_truth.csv
│   │   ├── deepbridge_results.json
│   │   ├── baseline_results.json
│   │   ├── confusion_matrix.json
│   │   └── statistical_tests.json
│   │
│   └── 06_ablation_studies/
│       ├── all_configs.csv
│       ├── contributions.json
│       ├── anova_results.json
│       └── summary.json
│
├── figures/                          # Visualizações para o paper
│   ├── README.md
│   ├── 01_benchmarks_tempo/
│   │   ├── time_comparison_barplot.pdf
│   │   ├── speedup_by_task.pdf
│   │   └── ablation_study.pdf
│   │
│   ├── 02_estudos_de_caso/
│   │   ├── case_studies_times.pdf
│   │   ├── case_studies_violations.pdf
│   │   └── aggregate_metrics.pdf
│   │
│   ├── 03_usabilidade/
│   │   ├── sus_score_distribution.pdf
│   │   ├── nasa_tlx_dimensions.pdf
│   │   ├── task_completion_times.pdf
│   │   └── success_rate_by_task.pdf
│   │
│   ├── 04_hpmkd/
│   │   ├── accuracy_comparison.pdf
│   │   ├── retention_rates.pdf
│   │   ├── compression_latency.pdf
│   │   └── ablation_study.pdf
│   │
│   ├── 05_conformidade/
│   │   ├── confusion_matrix.pdf
│   │   ├── precision_recall.pdf
│   │   ├── feature_coverage.pdf
│   │   └── audit_time.pdf
│   │
│   └── 06_ablation_studies/
│       ├── waterfall.pdf
│       ├── stacked_bar.pdf
│       ├── boxplot.pdf
│       └── contributions_pie.pdf
│
├── tables/                           # Tabelas formatadas em LaTeX
│   ├── README.md
│   ├── 01_time_benchmarks.tex
│   ├── 02_case_studies_summary.tex
│   ├── 03_usability_summary.tex
│   ├── 04_hpmkd_results.tex
│   ├── 05_compliance_results.tex
│   └── 06_ablation_results.tex
│
├── logs/                             # Logs de execução
│   ├── README.md
│   ├── 01_benchmarks_tempo/
│   ├── 02_estudos_de_caso/
│   ├── 03_usabilidade/
│   ├── 04_hpmkd/
│   ├── 05_conformidade/
│   └── 06_ablation_studies/
│
└── environment/                      # Configuração de ambiente
    ├── requirements.txt
    ├── environment.yml               # Conda environment
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    └── README.md
```

## Como Usar Esta Estrutura

### 1. Criar Diretórios

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos

# Criar estrutura de diretórios
mkdir -p datasets/{raw,processed,synthetic}
mkdir -p models/{teachers,students,case_studies}
mkdir -p scripts/{common,01_benchmarks_tempo,02_estudos_de_caso,03_usabilidade,04_hpmkd,05_conformidade,06_ablation_studies}
mkdir -p notebooks/exploratory
mkdir -p results/{01_benchmarks_tempo,02_estudos_de_caso,03_usabilidade,04_hpmkd,05_conformidade,06_ablation_studies}
mkdir -p figures/{01_benchmarks_tempo,02_estudos_de_caso,03_usabilidade,04_hpmkd,05_conformidade,06_ablation_studies}
mkdir -p tables
mkdir -p logs/{01_benchmarks_tempo,02_estudos_de_caso,03_usabilidade,04_hpmkd,05_conformidade,06_ablation_studies}
mkdir -p environment/docker
```

### 2. Criar requirements.txt

```bash
cat > environment/requirements.txt << 'EOF'
# Core
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# ML Frameworks
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.2.0
torch>=2.0.0

# Fairness Tools (baselines)
aif360>=0.5.0
fairlearn>=0.9.0

# Robustness Tools (baselines)
alibi-detect>=0.11.4

# Uncertainty Tools (baselines)
uq360>=0.3.0

# Drift Detection (baselines)
evidently>=0.4.0

# DeepBridge (quando disponível)
# deepbridge>=0.1.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Analysis
jupyter>=1.0.0
jupyterlab>=4.0.0
ipywidgets>=8.0.0

# Statistics
statsmodels>=0.14.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
python-dotenv>=1.0.0

# Testing
pytest>=7.3.0
pytest-cov>=4.1.0

# Reporting
fpdf2>=2.7.0
jinja2>=3.1.2

# Parallel Processing
joblib>=1.3.0
dask[complete]>=2023.5.0
EOF
```

### 3. Fluxo de Trabalho Sugerido

#### Fase 1: Preparação de Dados
```bash
# 1. Baixar datasets
python scripts/common/download_datasets.py

# 2. Processar datasets
python scripts/common/preprocess_datasets.py

# 3. Treinar modelos necessários
python scripts/common/train_baseline_models.py
```

#### Fase 2: Execução de Experimentos
```bash
# Executar cada experimento
python scripts/01_benchmarks_tempo/run_deepbridge.py
python scripts/01_benchmarks_tempo/run_fragmented.py

python scripts/02_estudos_de_caso/01_credit_scoring.py
# ... etc

python scripts/03_usabilidade/setup_study.py
# ... etc

python scripts/04_hpmkd/train_hpmkd.py
# ... etc

python scripts/05_conformidade/run_deepbridge.py
# ... etc

python scripts/06_ablation_studies/run_all_configs.py
```

#### Fase 3: Análise
```bash
# Analisar resultados em notebooks
jupyter lab notebooks/

# Gerar figuras
python scripts/01_benchmarks_tempo/generate_figures.py
python scripts/02_estudos_de_caso/aggregate_results.py
# ... etc

# Gerar tabelas LaTeX
python scripts/common/generate_latex_tables.py
```

### 4. Boas Práticas

#### Reprodutibilidade
- **Seeds fixas**: Usar `random_state=42` em todos os experimentos
- **Versionamento**: Git para código, DVC para dados/modelos grandes
- **Logs**: Registrar todas as execuções com timestamps
- **Environment**: Documentar versões exatas de bibliotecas

#### Organização de Código
```python
# Exemplo de estrutura de script
"""
Script: 01_credit_scoring.py
Descrição: Estudo de caso - Credit Scoring
Autor: [nome]
Data: 2025-12-05
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    filename='logs/02_estudos_de_caso/credit_scoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'processed'
RESULTS_DIR = BASE_DIR / 'results' / '02_estudos_de_caso'

def main():
    logging.info("Starting credit scoring experiment...")

    # Load data
    df = pd.read_csv(DATA_DIR / 'german_credit_test.csv')

    # Run experiment
    # ...

    # Save results
    results.to_json(RESULTS_DIR / 'credit_scoring_results.json')

    logging.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
```

#### Documentação de Resultados
Cada arquivo de resultado deve incluir metadados:

```json
{
  "experiment": "02_credit_scoring",
  "date": "2025-12-05T22:30:00",
  "dataset": "german_credit",
  "model": "xgboost",
  "environment": {
    "python": "3.10.12",
    "deepbridge": "0.1.0",
    "xgboost": "1.7.6"
  },
  "seed": 42,
  "results": {
    "time_seconds": 1020,
    "violations_detected": 2,
    "disparate_impact_gender": 0.74
  }
}
```

## Checklist de Implementação

### Infraestrutura
- [ ] Criar todos os diretórios
- [ ] Criar requirements.txt
- [ ] Criar environment.yml (conda)
- [ ] Configurar logging
- [ ] Criar .gitignore apropriado
- [ ] Configurar DVC para datasets grandes (opcional)

### Utilitários Comuns
- [ ] `data_loader.py`: Funções para carregar datasets
- [ ] `metrics.py`: Cálculo de métricas comuns
- [ ] `visualization.py`: Funções de visualização padrão
- [ ] `latex_tables.py`: Geração de tabelas LaTeX
- [ ] `download_datasets.py`: Download de datasets públicos

### README em cada diretório
- [ ] `datasets/README.md`
- [ ] `models/README.md`
- [ ] `scripts/README.md`
- [ ] `notebooks/README.md`
- [ ] `results/README.md`
- [ ] `figures/README.md`
- [ ] `tables/README.md`
- [ ] `logs/README.md`
- [ ] `environment/README.md`

## Notas Finais

1. **Versionamento de Dados**: Considere usar DVC (Data Version Control) para datasets e modelos grandes
2. **Paralelização**: Muitos experimentos podem rodar em paralelo (diferentes datasets, configurações)
3. **Monitoramento**: Use ferramentas como `tqdm` para progress bars e logging extensivo
4. **Backup**: Fazer backup regular de `results/`, `models/` e `figures/`
5. **Documentação Contínua**: Atualizar READMEs conforme experimentos avançam
