# Experimento 6: Ablation Studies

## Objetivo

Comprovar a decomposição dos ganhos de tempo (Seção 6.3) através de estudos de ablação sistemáticos:

- **API unificada**: 50% do ganho
- **Paralelização**: 30% do ganho
- **Caching**: 10% do ganho
- **Automação de relatórios**: 10% do ganho

## Afirmações a Comprovar

### Decomposição dos Ganhos (Seção 6.3)

| Componente | Contribuição | Ganho Absoluto | Status |
|------------|--------------|----------------|--------|
| API Unificada | 50% | ~66 min | ⏳ Pendente |
| Paralelização | 30% | ~40 min | ⏳ Pendente |
| Caching | 10% | ~13 min | ⏳ Pendente |
| Automação Relatórios | 10% | ~13 min | ⏳ Pendente |
| **Total** | **100%** | **~133 min** | - |

**Ganho Total**: 150 min (fragmentado) - 17 min (DeepBridge) = **133 min**

## Metodologia

### 1. Configurações de Ablação

Criar versões do DeepBridge com componentes desabilitados:

```python
# Configuração 0: DeepBridge COMPLETO (baseline)
config_full = {
    'unified_api': True,
    'parallel_execution': True,
    'caching': True,
    'automated_reporting': True
}
# Tempo esperado: 17 min

# Configuração 1: SEM API unificada
config_no_api = {
    'unified_api': False,  # Usar conversões manuais
    'parallel_execution': True,
    'caching': True,
    'automated_reporting': True
}
# Tempo esperado: 17 + 66 = 83 min

# Configuração 2: SEM paralelização
config_no_parallel = {
    'unified_api': True,
    'parallel_execution': False,  # Execução sequencial
    'caching': True,
    'automated_reporting': True
}
# Tempo esperado: 17 + 40 = 57 min

# Configuração 3: SEM caching
config_no_cache = {
    'unified_api': True,
    'parallel_execution': True,
    'caching': False,  # Recomputar predições
    'automated_reporting': True
}
# Tempo esperado: 17 + 13 = 30 min

# Configuração 4: SEM automação de relatórios
config_no_auto_report = {
    'unified_api': True,
    'parallel_execution': True,
    'caching': True,
    'automated_reporting': False  # Geração manual
}
# Tempo esperado: 17 + 13 = 30 min

# Configuração 5: NADA (fragmentado completo)
config_none = {
    'unified_api': False,
    'parallel_execution': False,
    'caching': False,
    'automated_reporting': False
}
# Tempo esperado: ~150 min
```

### 2. Medição de Tempo por Configuração

Para cada configuração, executar validação completa:

```python
import time
from deepbridge import DBDataset, Experiment

def measure_ablation_time(config, dataset, num_runs=10):
    times = []

    for run in range(num_runs):
        start = time.time()

        if config['unified_api']:
            # DeepBridge API unificada
            exp = Experiment(dataset, tests='all', config=config)
            results = exp.run_tests()
        else:
            # Workflow fragmentado (conversões manuais)
            results = run_fragmented_workflow(dataset)

        if config['automated_reporting']:
            exp.save_pdf('report.pdf')
        else:
            generate_manual_report(results)

        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }
```

### 3. Dataset de Teste

Usar dataset padrão para todas configurações:

**Adult Income Dataset**
- Amostras: 48.842
- Features: 14
- Protected attributes: gender, race, age

### 4. Cálculo de Contribuições

```python
# Baseline: DeepBridge completo
time_full = 17  # min

# Sem API unificada
time_no_api = measure_ablation_time(config_no_api)
contribution_api = time_no_api - time_full

# Sem paralelização
time_no_parallel = measure_ablation_time(config_no_parallel)
contribution_parallel = time_no_parallel - time_full

# Sem caching
time_no_cache = measure_ablation_time(config_no_cache)
contribution_cache = time_no_cache - time_full

# Sem automação relatórios
time_no_auto_report = measure_ablation_time(config_no_auto_report)
contribution_report = time_no_auto_report - time_full

# Total
total_gain = 150 - 17  # 133 min

# Percentuais
pct_api = (contribution_api / total_gain) * 100
pct_parallel = (contribution_parallel / total_gain) * 100
pct_cache = (contribution_cache / total_gain) * 100
pct_report = (contribution_report / total_gain) * 100
```

## Ablação 1: API Unificada

### Afirmação
**50% do ganho** (~66 min de 133 min)

### Implementação

**Com API Unificada** (DeepBridge):
```python
# Criar uma vez, usar em qualquer lugar
dataset = DBDataset(df, target='approved', model=model)

# Reutilizar em todas validações
fairness_results = run_fairness(dataset)  # ~5 min
robustness_results = run_robustness(dataset)  # ~7 min
uncertainty_results = run_uncertainty(dataset)  # ~3 min
```

**Sem API Unificada** (workflow fragmentado):
```python
# Conversão para AIF360
aif_dataset = BinaryLabelDataset(df=df, ...)  # 5 min
fairness_results = run_fairness_aif360(aif_dataset)  # 30 min

# Conversão para Alibi Detect
alibi_data = df.values.astype(np.float32)  # 3 min
robustness_results = run_robustness_alibi(alibi_data)  # 25 min

# Conversão para UQ360
uq_data = Dataset(df, ...)  # 4 min
uncertainty_results = run_uncertainty_uq360(uq_data)  # 20 min

# Total conversões: ~12 min
# Total execução: ~75 min
# Total: ~87 min vs. 15 min (DeepBridge sem relatório)
# Ganho: 87 - 15 = 72 min ≈ 66 min (expectativa)
```

### Resultados Esperados

| Métrica | Com API | Sem API | Ganho |
|---------|---------|---------|-------|
| Tempo (min) | 17 | 83 | 66 min |
| % do Ganho Total | - | - | 50% |

## Ablação 2: Paralelização

### Afirmação
**30% do ganho** (~40 min de 133 min)

### Implementação

**Com Paralelização**:
```python
from concurrent.futures import ThreadPoolExecutor

# Executar testes em paralelo
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        'fairness': executor.submit(run_fairness, dataset),
        'robustness': executor.submit(run_robustness, dataset),
        'uncertainty': executor.submit(run_uncertainty, dataset),
        'resilience': executor.submit(run_resilience, dataset)
    }

    results = {name: future.result() for name, future in futures.items()}

# Tempo total: max(5, 7, 3, 2) ≈ 7 min (overlap)
```

**Sem Paralelização** (sequencial):
```python
# Executar testes sequencialmente
results_fairness = run_fairness(dataset)  # 5 min
results_robustness = run_robustness(dataset)  # 7 min
results_uncertainty = run_uncertainty(dataset)  # 3 min
results_resilience = run_resilience(dataset)  # 2 min

# Tempo total: 5 + 7 + 3 + 2 = 17 min
```

**Ganho**: 17 - 7 = 10 min por execução
**Escalado**: Com todas otimizações, ganho ≈ 40 min

### Medição de Speedup

```python
import time

# Medir tempo sequencial
start = time.time()
run_tests_sequential(dataset)
time_sequential = time.time() - start

# Medir tempo paralelo
start = time.time()
run_tests_parallel(dataset, n_workers=4)
time_parallel = time.time() - start

# Speedup
speedup = time_sequential / time_parallel
# Esperado: ~2-3× (não linear devido a I/O, sincronização)
```

### Resultados Esperados

| Métrica | Paralelo | Sequencial | Ganho |
|---------|----------|------------|-------|
| Tempo (min) | 17 | 57 | 40 min |
| % do Ganho Total | - | - | 30% |
| Speedup | 3.4× | 1× | - |

## Ablação 3: Caching

### Afirmação
**10% do ganho** (~13 min de 133 min)

### Implementação

**Com Caching**:
```python
class DBDataset:
    def __init__(self, data, model, ...):
        self._predictions_cache = None

    @property
    def predictions(self):
        if self._predictions_cache is None:
            self._predictions_cache = self.model.predict(self.data)
        return self._predictions_cache

    @property
    def prediction_probabilities(self):
        if self._proba_cache is None:
            self._proba_cache = self.model.predict_proba(self.data)
        return self._proba_cache

# Predições computadas UMA VEZ e reutilizadas
dataset.predictions  # Computa e cacheia
dataset.predictions  # Retorna do cache (instantâneo)
```

**Sem Caching**:
```python
# Recomputar predições a cada teste
fairness_preds = model.predict(data)  # 2 min
robustness_preds = model.predict(data)  # 2 min
uncertainty_preds = model.predict_proba(data)  # 2 min
resilience_preds = model.predict(data)  # 2 min

# Tempo total desperdiçado: ~8 min
# Com overhead adicional: ~13 min
```

### Medição

```python
import time

# Com caching
start = time.time()
dataset = DBDataset(data, model, caching=True)
for _ in range(10):
    preds = dataset.predictions  # Cache hit após primeira
time_with_cache = time.time() - start

# Sem caching
start = time.time()
dataset = DBDataset(data, model, caching=False)
for _ in range(10):
    preds = model.predict(data)  # Recomputa sempre
time_without_cache = time.time() - start

# Ganho
gain = time_without_cache - time_with_cache
```

### Resultados Esperados

| Métrica | Com Cache | Sem Cache | Ganho |
|---------|-----------|-----------|-------|
| Tempo (min) | 17 | 30 | 13 min |
| % do Ganho Total | - | - | 10% |

## Ablação 4: Automação de Relatórios

### Afirmação
**10% do ganho** (~13 min de 133 min)

### Implementação

**Com Automação**:
```python
# Geração automática de relatório PDF
exp = Experiment(dataset, tests='all')
results = exp.run_tests()
exp.save_pdf('report.pdf')  # <1 min

# Template-driven, visualizações automáticas
```

**Sem Automação** (manual):
```python
from fpdf import FPDF
import matplotlib.pyplot as plt

# Criar PDF manualmente
pdf = FPDF()
pdf.add_page()

# Para CADA métrica:
# 1. Criar visualização
fig, ax = plt.subplots()
ax.plot(...)  # 2 min por gráfico
plt.savefig('temp_fig.png')

# 2. Adicionar ao PDF
pdf.image('temp_fig.png')  # 0.5 min

# 3. Adicionar texto explicativo
pdf.cell(0, 10, 'Análise...', ln=True)  # 1 min

# Para 15 métricas + 10 visualizações + formatação:
# ~60 minutos total

pdf.output('report.pdf')
```

### Medição

```python
import time

# Automação
start = time.time()
exp.save_pdf('report_auto.pdf')
time_auto = time.time() - start

# Manual
start = time.time()
generate_manual_report(results)
time_manual = time.time() - start

# Ganho
gain = time_manual - time_auto
```

### Resultados Esperados

| Métrica | Automação | Manual | Ganho |
|---------|-----------|--------|-------|
| Tempo (min) | <1 | 60 | ~60 min |
| % do Ganho (relatório) | - | - | 98% |
| % do Ganho Total | - | - | ~10% |

## Análise Combinada

### Todas Ablações

```python
configs = {
    'Full': config_full,
    'No API': config_no_api,
    'No Parallel': config_no_parallel,
    'No Cache': config_no_cache,
    'No AutoReport': config_no_auto_report,
    'None (Fragmented)': config_none
}

results = {}
for name, config in configs.items():
    results[name] = measure_ablation_time(config, dataset, num_runs=10)
```

### Resultados Esperados

| Configuração | Tempo (min) | Ganho vs. Full | % do Ganho Total |
|--------------|-------------|----------------|------------------|
| Full (DeepBridge) | 17 | 0 | - |
| No API | 83 | +66 | 50% |
| No Parallel | 57 | +40 | 30% |
| No Cache | 30 | +13 | 10% |
| No AutoReport | 30 | +13 | 10% |
| None (Fragmentado) | 150 | +133 | 100% |

### Verificação de Aditividade

Idealmente, contribuições devem ser aproximadamente aditivas:

```python
# Configuração com NENHUM componente
time_none = 150

# Somar contribuições
estimated_time_none = (
    time_full +
    contribution_api +
    contribution_parallel +
    contribution_cache +
    contribution_report
)

# Verificar
difference = abs(time_none - estimated_time_none)
# Esperado: difference < 10 min (efeitos de interação)
```

## Análise Estatística

### ANOVA

Testar se diferenças entre configurações são significativas:

```python
from scipy import stats

# Tempos para cada configuração (10 runs cada)
times_full = [17.2, 16.8, 17.5, ...]
times_no_api = [82.5, 83.1, 84.2, ...]
times_no_parallel = [56.8, 57.3, 58.1, ...]
# etc.

# One-way ANOVA
f_stat, p_value = stats.f_oneway(
    times_full,
    times_no_api,
    times_no_parallel,
    times_no_cache,
    times_no_auto_report,
    times_none
)

# Esperado: p < 0.001 (diferenças altamente significativas)
```

### Post-hoc (Tukey HSD)

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Comparações pareadas
tukey = pairwise_tukeyhsd(all_times, all_groups)
print(tukey)

# Esperado: todas comparações significativas (p < 0.05)
```

## Visualizações

### Waterfall Chart

Mostrar contribuição acumulada de cada componente:

```python
import matplotlib.pyplot as plt

components = ['Fragmentado', 'API', 'Paralelo', 'Cache', 'AutoReport', 'Full']
times = [150, 83, 57, 30, 30, 17]
contributions = [0, -67, -26, -27, 0, -13]

# Waterfall chart
# ...
```

### Stacked Bar Chart

```python
contributions = {
    'API Unificada': 66,
    'Paralelização': 40,
    'Caching': 13,
    'Automação Relatórios': 13
}

# Stacked bar
```

## Scripts

### Principal
`/experimentos/scripts/06_ablation_main.py`

### Por Componente
`/experimentos/scripts/06_ablation_api.py`
`/experimentos/scripts/06_ablation_parallel.py`
`/experimentos/scripts/06_ablation_cache.py`
`/experimentos/scripts/06_ablation_report.py`

### Análise
`/experimentos/notebooks/06_ablation_analysis.ipynb`

## Outputs

### Dados
- `results/06_ablation_all_configs.csv`
- `results/06_ablation_contributions.json`
- `results/06_ablation_anova.json`

### Figuras
- `figures/ablation_waterfall.pdf`
- `figures/ablation_stacked_bar.pdf`
- `figures/ablation_boxplot.pdf`
- `figures/ablation_contributions_pie.pdf`

### Tabelas
- `tables/ablation_results.tex`

## Checklist

- [ ] Implementar configuração Full
- [ ] Implementar configuração No API
- [ ] Implementar configuração No Parallel
- [ ] Implementar configuração No Cache
- [ ] Implementar configuração No AutoReport
- [ ] Implementar configuração None (Fragmentado)
- [ ] Executar 10 runs para cada configuração
- [ ] Calcular contribuições absolutas
- [ ] Calcular contribuições percentuais
- [ ] Verificar aditividade
- [ ] Executar ANOVA
- [ ] Executar Tukey HSD
- [ ] Gerar waterfall chart
- [ ] Gerar stacked bar chart
- [ ] Formatar tabela LaTeX

## Prioridade

🟢 **BAIXA** - Útil para entender componentes, mas não crítico

## Tempo Estimado

**1-2 semanas**:
- Semana 1: Implementação das configurações e execução
- Semana 2: Análise estatística e visualizações
