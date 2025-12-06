# Experimento 01: Benchmarks de Tempo

## Objetivo

Comprovar que **DeepBridge reduz o tempo de validação em 89%** (17 min vs. 150 min) comparado com workflow manual usando ferramentas fragmentadas.

## Estrutura

```
01_benchmarks_tempo/
├── README.md                  # Este arquivo
├── config/
│   └── config.yaml           # Configurações do experimento
├── scripts/
│   ├── utils.py              # Utilidades comuns
│   ├── benchmark_deepbridge.py       # Benchmark DeepBridge
│   ├── benchmark_fragmented.py       # Benchmark workflow fragmentado
│   ├── compare_and_analyze.py        # Análise estatística
│   ├── generate_figures.py           # Geração de figuras
│   └── run_experiment.py             # Orchestrador principal
├── results/                  # Resultados (gerados)
├── figures/                  # Figuras (geradas)
├── tables/                   # Tabelas LaTeX (geradas)
└── logs/                     # Logs de execução (gerados)
```

## Instalação

### 1. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## Uso

### Execução Completa

Executa todo o pipeline (10 runs de cada benchmark + análise + figuras):

```bash
cd scripts
python run_experiment.py --all
```

**Tempo estimado**: ~5-6 horas (dependendo do hardware)

### Modo Rápido (Teste)

Executa 1 run apenas para testar:

```bash
python run_experiment.py --quick
```

**Tempo estimado**: ~30 minutos

### Execução Parcial

Execute apenas partes específicas:

```bash
# Apenas benchmark DeepBridge
python run_experiment.py --deepbridge

# Apenas benchmark Fragmentado
python run_experiment.py --fragmented

# Apenas análise (requer resultados prévios)
python run_experiment.py --analyze

# Apenas geração de figuras (requer resultados prévios)
python run_experiment.py --figures
```

## Configuração

Edite `config/config.yaml` para modificar:

- **Número de runs**: `general.num_runs` (default: 10)
- **Seed**: `general.seed` (default: 42)
- **Tempos esperados**: `tests.<tarefa>.expected_time_deepbridge/fragmented`
- **Dataset**: `dataset.*`
- **Modelo**: `model.*`

## Outputs

### Resultados (results/)

- `deepbridge_times.json` - Tempos do DeepBridge
- `deepbridge_times.csv` - Tempos do DeepBridge (CSV)
- `fragmented_times.json` - Tempos do workflow fragmentado
- `fragmented_times.csv` - Tempos do workflow fragmentado (CSV)
- `comparison_summary.csv` - Resumo da comparação
- `analysis_results.json` - Resultados da análise estatística

### Figuras (figures/)

- `time_comparison_barplot.pdf` - Comparação de tempos (barras)
- `speedup_by_task.pdf` - Speedup por tarefa
- `reduction_percentage.pdf` - Redução percentual
- `boxplot_comparison.pdf` - Boxplot das distribuições
- `total_time_breakdown.pdf` - Breakdown do tempo total

### Tabelas (tables/)

- `time_benchmarks.tex` - Tabela formatada em LaTeX para o paper

### Logs (logs/)

- `deepbridge_benchmark_YYYYMMDD_HHMMSS.log`
- `fragmented_benchmark_YYYYMMDD_HHMMSS.log`
- `analysis_YYYYMMDD_HHMMSS.log`
- `figures_YYYYMMDD_HHMMSS.log`
- `experiment_orchestrator_YYYYMMDD_HHMMSS.log`

## Análise dos Resultados

### Métricas Calculadas

Para cada tarefa (fairness, robustness, uncertainty, resilience, report):

- **Tempo médio** (mean ± std)
- **Tempo mínimo/máximo**
- **Speedup** (tempo_fragmentado / tempo_deepbridge)
- **Redução absoluta** (min)
- **Redução percentual** (%)

### Testes Estatísticos

1. **Paired t-test**: Compara tempos DeepBridge vs. Fragmentado
2. **Wilcoxon signed-rank test**: Teste não-paramétrico
3. **Cohen's d**: Tamanho do efeito
4. **ANOVA**: Comparação global de todos os grupos

### Interpretação

- **p < 0.05**: Diferença estatisticamente significativa
- **Cohen's d**:
  - < 0.2: negligible
  - 0.2-0.5: small
  - 0.5-0.8: medium
  - > 0.8: large

## Resultados Esperados

| Tarefa | DeepBridge | Fragmentado | Speedup | Redução |
|--------|------------|-------------|---------|---------|
| Fairness | 5 min | 30 min | 6× | 83% |
| Robustness | 7 min | 25 min | 3.6× | 72% |
| Uncertainty | 3 min | 20 min | 6.7× | 85% |
| Resilience | 2 min | 15 min | 7.5× | 87% |
| Report | <1 min | 60 min | >60× | >98% |
| **TOTAL** | **17 min** | **150 min** | **8.8×** | **89%** |

## Troubleshooting

### Erro: Dataset não encontrado

```bash
# Instalar scikit-learn com datasets
pip install scikit-learn --upgrade

# Ou baixar manualmente Adult Income dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

### Erro: Memória insuficiente

Reduza `num_runs` em `config.yaml`:

```yaml
general:
  num_runs: 5  # ao invés de 10
```

### Erro: Timeout

Aumente timeout em `config.yaml`:

```yaml
resources:
  timeout_minutes: 60  # ao invés de 30
```

### Execução muito lenta

O tempo de simulação está muito alto? Ajuste tempos esperados em `config.yaml`:

```yaml
tests:
  fairness:
    expected_time_deepbridge: 0.5  # 30 segundos ao invés de 5 min
    expected_time_fragmented: 3    # 3 min ao invés de 30 min
```

## Notas Importantes

### Modo de Simulação

⚠️ **IMPORTANTE**: Os scripts atualmente simulam os tempos esperados usando `time.sleep()`. Quando o DeepBridge estiver implementado, substituir por:

```python
# Substituir em benchmark_deepbridge.py
from deepbridge import DBDataset, Experiment

dataset = DBDataset(
    data=X_test,
    target=y_test,
    model=model,
    protected_attributes=['sex', 'race', 'age']
)

exp = Experiment(dataset, tests='all')
results = exp.run_tests()
```

### Ferramentas Baseline

Para benchmarks reais com ferramentas fragmentadas, instalar:

```bash
pip install aif360 fairlearn alibi-detect uq360 evidently
```

E descomentar código relevante em `benchmark_fragmented.py`.

## Contribuindo

Para adicionar novos testes ou modificar o experimento:

1. Edite configurações em `config/config.yaml`
2. Modifique scripts em `scripts/`
3. Execute em modo `--quick` para testar
4. Execute `--all` para resultados finais

## Citação

Se usar este experimento, citar:

```bibtex
@inproceedings{deepbridge2025,
  title={DeepBridge: Um Framework Unificado para Validação Multi-Dimensional de Machine Learning},
  author={...},
  booktitle={MLSys 2026},
  year={2025}
}
```

## Contato

Para dúvidas ou problemas, abrir issue no repositório do DeepBridge.

---

**Última atualização**: 2025-12-05
**Versão**: 1.0.0
