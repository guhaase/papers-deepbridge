# Experimento 1: Benchmarks de Tempo

## Objetivo

Comprovar a afirmação do paper de que **DeepBridge reduz o tempo de validação em 89%** (17 min vs. 150 min) comparado com workflow manual usando ferramentas fragmentadas.

## Afirmações a Comprovar

### Tabela de Benchmarks (Seção 6.3)

| Tarefa | DeepBridge | Fragmentado | Afirmação |
|--------|------------|-------------|-----------|
| Fairness (15 métricas) | 5 min | 30 min | ⏳ A comprovar |
| Robustez | 7 min | 25 min | ⏳ A comprovar |
| Incerteza | 3 min | 20 min | ⏳ A comprovar |
| Resiliência | 2 min | 15 min | ⏳ A comprovar |
| Geração de relatório | <1 min | 60 min | ⏳ A comprovar |
| **Total** | **17 min** | **150 min** | **89% redução** |

### Decomposição dos Ganhos (Seção 6.3)

- API unificada: 50% do ganho
- Paralelização: 30% do ganho
- Caching: 10% do ganho
- Automação de relatórios: 10% do ganho

## Metodologia

### 1. Setup do Experimento

#### Dataset de Teste
- **Nome**: Adult Income Dataset (UCI)
- **Tamanho**: 48,842 amostras
- **Features**: 14 features (6 numéricas, 8 categóricas)
- **Target**: Binário (income >50K ou ≤50K)
- **Atributos Protegidos**: gender, race, age

#### Modelo
- **Algoritmo**: XGBoost
- **Hiperparâmetros**:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
- **Split**: 80% treino, 20% teste
- **Seed**: 42 (para reprodutibilidade)

### 2. Workflow DeepBridge

```python
import time
from deepbridge import DBDataset, Experiment

# Timer para cada etapa
times_deepbridge = {}

# 1. Criar dataset (incluído no tempo)
start = time.time()
dataset = DBDataset(
    data=df_test,
    target_column='income',
    model=xgb_model,
    protected_attributes=['gender', 'race', 'age']
)
times_deepbridge['setup'] = time.time() - start

# 2. Fairness (15 métricas)
start = time.time()
exp_fairness = Experiment(dataset, tests=['fairness'])
results_fairness = exp_fairness.run_tests()
times_deepbridge['fairness'] = time.time() - start

# 3. Robustness
start = time.time()
exp_robustness = Experiment(dataset, tests=['robustness'])
results_robustness = exp_robustness.run_tests()
times_deepbridge['robustness'] = time.time() - start

# 4. Uncertainty
start = time.time()
exp_uncertainty = Experiment(dataset, tests=['uncertainty'])
results_uncertainty = exp_uncertainty.run_tests()
times_deepbridge['uncertainty'] = time.time() - start

# 5. Resilience
start = time.time()
exp_resilience = Experiment(dataset, tests=['resilience'])
results_resilience = exp_resilience.run_tests()
times_deepbridge['resilience'] = time.time() - start

# 6. Report Generation
start = time.time()
exp_all = Experiment(dataset, tests='all')
exp_all.save_pdf('report.pdf')
times_deepbridge['report'] = time.time() - start

# Total time
times_deepbridge['total'] = sum(times_deepbridge.values())
```

### 3. Workflow Fragmentado (Baseline)

```python
import time
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from fairlearn.metrics import demographic_parity_difference
from alibi_detect.cd import TabularDrift
from uq360.algorithms.posthocuq import PosthocUQ
import matplotlib.pyplot as plt
from fpdf import FPDF

times_fragmented = {}

# 1. Fairness com AIF360 + Fairlearn (15 métricas)
start = time.time()

# Converter para formato AIF360
aif_dataset = BinaryLabelDataset(
    df=df_test,
    label_names=['income'],
    protected_attribute_names=['gender', 'race', 'age']
)

# Calcular 15 métricas manualmente
# - Demographic Parity
# - Equal Opportunity
# - Equalized Odds
# - Disparate Impact
# ... (11 métricas adicionais)

# Para cada métrica, precisa:
# 1. Preparar dados no formato correto
# 2. Calcular métrica
# 3. Armazenar resultado

times_fragmented['fairness'] = time.time() - start

# 2. Robustness com Alibi Detect
start = time.time()

# Converter para NumPy
X_test_np = df_test.drop('income', axis=1).values

# Testar perturbações
# - Noise injection
# - Feature permutation
# - Adversarial examples (se disponível)

times_fragmented['robustness'] = time.time() - start

# 3. Uncertainty com UQ360
start = time.time()

# Converter para formato UQ360
# Calcular calibração, intervalos de predição, etc.

times_fragmented['uncertainty'] = time.time() - start

# 4. Resilience com Evidently AI ou custom
start = time.time()

# Calcular drift metrics (PSI, KL, etc.)

times_fragmented['resilience'] = time.time() - start

# 5. Report Generation Manual
start = time.time()

# Criar PDF manualmente
pdf = FPDF()
pdf.add_page()

# Para cada métrica:
# 1. Criar visualização com matplotlib
# 2. Salvar como imagem
# 3. Adicionar ao PDF
# 4. Adicionar texto explicativo

# Isso deve levar ~60 minutos para relatório completo

times_fragmented['report'] = time.time() - start

# Total time
times_fragmented['total'] = sum(times_fragmented.values())
```

### 4. Medições

**Número de Runs**: 10 execuções independentes
**Seed**: Variável entre runs (42, 43, 44, ..., 51)
**Métricas**:
- Tempo médio (segundos)
- Desvio padrão
- Tempo mínimo
- Tempo máximo

## Resultados Esperados

### Tempos DeepBridge (minutos)

| Tarefa | Média | Std | Min | Max |
|--------|-------|-----|-----|-----|
| Fairness | 5.0 | 0.3 | 4.7 | 5.4 |
| Robustez | 7.0 | 0.4 | 6.5 | 7.6 |
| Incerteza | 3.0 | 0.2 | 2.8 | 3.3 |
| Resiliência | 2.0 | 0.1 | 1.9 | 2.2 |
| Relatório | 0.8 | 0.1 | 0.7 | 1.0 |
| **Total** | **17.8** | **0.8** | **16.6** | **19.5** |

### Tempos Fragmentado (minutos)

| Tarefa | Média | Std | Min | Max |
|--------|-------|-----|-----|-----|
| Fairness | 30.0 | 2.5 | 27.0 | 33.0 |
| Robustez | 25.0 | 2.0 | 22.5 | 27.5 |
| Incerteza | 20.0 | 1.8 | 18.0 | 22.0 |
| Resiliência | 15.0 | 1.5 | 13.0 | 17.0 |
| Relatório | 60.0 | 5.0 | 55.0 | 65.0 |
| **Total** | **150.0** | **10.0** | **135.5** | **164.5** |

### Speedup

- **Speedup Global**: 150 / 17 = **8.8×**
- **Redução Percentual**: (150 - 17) / 150 = **89%**

## Análise Estatística

### Teste de Hipótese

**H0**: Não há diferença significativa entre DeepBridge e workflow fragmentado
**H1**: DeepBridge é significativamente mais rápido

**Teste**: Paired t-test (duas caudas)
**Nível de Significância**: α = 0.05

```python
from scipy import stats

# Paired t-test para cada tarefa
for task in ['fairness', 'robustness', 'uncertainty', 'resilience', 'report']:
    t_stat, p_value = stats.ttest_rel(
        times_deepbridge[task],  # 10 runs
        times_fragmented[task]   # 10 runs
    )
    print(f"{task}: t={t_stat:.2f}, p={p_value:.4f}")

# Esperado: p < 0.001 para todas as tarefas
```

## Validação dos Ganhos por Componente

Para comprovar que:
- API unificada contribui 50%
- Paralelização contribui 30%
- Caching contribui 10%
- Automação relatórios contribui 10%

### Ablation Study

```python
# Baseline: DeepBridge completo
time_full = 17 min

# Ablation 1: Sem API unificada (usar conversões manuais)
time_no_api = measure_time_without_unified_api()
gain_api = (time_no_api - time_full) / (150 - 17)  # % do ganho total
# Esperado: ~50%

# Ablation 2: Sem paralelização
time_no_parallel = measure_time_sequential()
gain_parallel = (time_no_parallel - time_full) / (150 - 17)
# Esperado: ~30%

# Ablation 3: Sem caching
time_no_cache = measure_time_without_cache()
gain_cache = (time_no_cache - time_full) / (150 - 17)
# Esperado: ~10%

# Ablation 4: Relatórios manuais
time_manual_report = measure_time_manual_reporting()
gain_report = (time_manual_report - time_full) / (150 - 17)
# Esperado: ~10%
```

## Ambiente de Execução

### Hardware
- **CPU**: Intel i7-12700K (12 cores, 20 threads) ou similar
- **RAM**: 32GB DDR4
- **Storage**: SSD NVMe
- **GPU**: Não necessária para este experimento

### Software
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10
- **DeepBridge**: versão atual
- **Bibliotecas de Comparação**:
  - aif360==0.5.0
  - fairlearn==0.9.0
  - alibi-detect==0.11.4
  - uq360==0.3.0
  - evidently==0.4.0

## Scripts

### Script Principal
`/experimentos/scripts/01_time_benchmarks.py`

### Análise de Resultados
`/experimentos/notebooks/01_time_benchmarks_analysis.ipynb`

### Geração de Figuras
`/experimentos/scripts/01_generate_figures.py`

## Outputs Esperados

1. **CSV com Tempos Brutos**:
   - `results/01_deepbridge_times.csv`
   - `results/01_fragmented_times.csv`

2. **Figuras para Paper**:
   - `figures/time_comparison_barplot.pdf`
   - `figures/speedup_by_task.pdf`
   - `figures/ablation_study.pdf`

3. **Tabela LaTeX**:
   - `tables/time_benchmarks.tex`

4. **Análise Estatística**:
   - `results/statistical_analysis.json`

## Checklist

- [ ] Implementar script de benchmark DeepBridge
- [ ] Implementar script de benchmark fragmentado
- [ ] Executar 10 runs para cada workflow
- [ ] Calcular estatísticas (média, std, min, max)
- [ ] Realizar teste t pareado
- [ ] Implementar ablation study
- [ ] Gerar visualizações
- [ ] Formatar tabela em LaTeX
- [ ] Documentar ambiente de execução
- [ ] Validar reprodutibilidade

## Prioridade

🔴 **ALTA** - Este é um dos resultados centrais do paper

## Tempo Estimado

**2-3 semanas**:
- Semana 1: Implementação dos scripts
- Semana 2: Execução dos experimentos e coleta de dados
- Semana 3: Análise estatística e geração de visualizações
