# Roadmap de Correções - DeepBridge Experiments

**Objetivo**: Tornar os experimentos publicáveis em conferência de qualidade
**Prazo sugerido**: 4-6 semanas
**Prioridade**: Por impacto na validade científica

---

## 🔴 CRÍTICO - Semana 1-2

### 1.1 Experimento 5: Implementar Baseline Real

**Problema**: Baseline atual é simulado, não usa ferramentas reais
**Impacto**: Invalida completamente as conclusões
**Tempo**: 4-5 dias

**Tarefas**:

```bash
# 1. Instalar dependências
pip install aif360 fairlearn

# 2. Criar script validate_baseline_REAL.py
```

**Implementação**:

```python
# experimentos/05_conformidade/scripts/validate_baseline_real.py

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

def validate_with_aif360(case_df):
    """Validação real com AIF360"""
    # Converter para formato AIF360
    dataset = BinaryLabelDataset(
        df=case_df,
        label_names=['approved'],
        protected_attribute_names=['gender', 'race'],
        favorable_label=1,
        unfavorable_label=0
    )

    # Calcular métricas reais
    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{'gender': 0}],
        privileged_groups=[{'gender': 1}]
    )

    # Disparate Impact real
    di = metric.disparate_impact()

    return {
        'disparate_impact': di,
        'compliant': di >= 0.80
    }

# Executar para 50 casos e comparar com DeepBridge
```

**Validação**:
- [ ] AIF360 instalado e funcionando
- [ ] Validação executada nos 50 casos
- [ ] Resultados salvos em `baseline_real_results.json`
- [ ] Comparação estatística atualizada
- [ ] Tabelas LaTeX atualizadas

**Resultado esperado**: p-value < 0.01 (mais robusto que 0.0499)

---

### 1.2 Experimento 6: Decisão Crítica

**Problema**: Tempos são completamente simulados
**Impacto**: Experimento inteiro é inválido
**Tempo**: Decidir em 1 dia, implementar em 1-2 semanas

**Opção A: Remover Experimento** ⏱️ 1 hora
```bash
# Mais rápido e honesto
# Remove Experimento 6 do paper
# Foca nos experimentos válidos
```

**Vantagens**:
- Rápido
- Honesto
- Evita trabalho massivo

**Desvantagens**:
- Perde análise de ablação
- Paper fica com menos experimentos

**Opção B: Implementar Ablação Real** ⏱️ 1-2 semanas
```python
# experimentos/06_ablation_studies/scripts/run_ablation_REAL.py

class DeepBridgeAblation:
    """Versões reais com componentes desabilitados"""

    def run_without_unified_api(self, X, y, model):
        """Simula workflow fragmentado REAL"""
        start = time.time()

        # CONVERSÃO 1: Para AIF360
        conversion_start = time.time()
        aif_data = self._convert_to_aif360(X, y)
        conversion_time_1 = time.time() - conversion_start

        # CONVERSÃO 2: Para Alibi
        conversion_start = time.time()
        alibi_data = self._convert_to_alibi(X)
        conversion_time_2 = time.time() - conversion_start

        # CONVERSÃO 3: Para UQ360
        conversion_start = time.time()
        uq_data = self._convert_to_uq360(X, y)
        conversion_time_3 = time.time() - conversion_start

        # Executar validações
        fairness_results = self._run_fairness_aif360(aif_data)
        robustness_results = self._run_robustness_alibi(alibi_data)
        uncertainty_results = self._run_uncertainty_uq360(uq_data)

        total_time = time.time() - start

        return {
            'total_time': total_time,
            'conversion_overhead': conversion_time_1 + conversion_time_2 + conversion_time_3
        }

    def run_without_parallelization(self, X, y, model):
        """Execução sequencial REAL"""
        start = time.time()

        # Forçar execução serial (não paralela)
        fairness = self._run_fairness_sequential(X, y, model)
        robustness = self._run_robustness_sequential(X, y, model)
        uncertainty = self._run_uncertainty_sequential(X, y, model)
        resilience = self._run_resilience_sequential(X, y, model)

        total_time = time.time() - start
        return {'total_time': total_time}

    def run_without_caching(self, X, y, model):
        """Sem cache - recomputa predições"""
        start = time.time()

        # Desabilitar cache
        model.predict.cache_clear()  # Se usar functools.lru_cache

        # Cada validação chama predict() novamente
        for _ in range(4):  # 4 validações
            _ = model.predict(X)  # Sem cache!

        total_time = time.time() - start
        return {'total_time': total_time}
```

**Validação**:
- [ ] Implementadas 4 versões ablation
- [ ] Executados 30 runs por versão (6 × 30 = 180 runs)
- [ ] Tempos medidos com timer apropriado
- [ ] Análise estatística (ANOVA, Tukey HSD)
- [ ] Intervalos de confiança reportados

**Resultado esperado**:
- Unified API: 40-70 min overhead (não 66 min fixo)
- Parallelization: 20-40 min ganho
- Caching: 5-15 min ganho
- **Total speedup**: 4-7× (mais realista que 8.9×)

**Recomendação**: ✅ **Opção A** (remover) se prazo curto, **Opção B** (implementar) se prazo longo

---

### 1.3 Experimento 1: Completar Fairness Benchmark

**Problema**: Fairness test retornou 0.0s (sem dados)
**Impacto**: Benchmark incompleto, speedup não validado
**Tempo**: 2 dias

**Tarefas**:

```python
# experimentos/01_benchmarks_tempo/scripts/benchmark_deepbridge.py

def run_fairness_test_REAL(dataset, model):
    """Executar teste de fairness real"""
    start = time.time()

    # Detectar atributos protegidos
    protected_attrs = dataset.detect_protected_attributes()

    # Calcular métricas de fairness
    fairness_results = {}
    for attr in protected_attrs:
        di = calculate_disparate_impact(dataset, attr)
        fairness_results[attr] = {
            'disparate_impact': di,
            'compliant': di >= 0.80
        }

    elapsed = time.time() - start

    return {
        'execution_time': elapsed,
        'results': fairness_results
    }

# Executar para Adult Income dataset
# Salvar em deepbridge_times_REAL.json
```

**Validação**:
- [ ] Fairness test executado com sucesso
- [ ] Tempo medido > 0 (esperado: 5-15 segundos)
- [ ] Resultados salvos corretamente
- [ ] Total atualizado (esperado: 40-60 segundos total)

---

## 🟡 IMPORTANTE - Semana 3-4

### 2.1 Adicionar Mais Datasets

**Problema**: Apenas 1 dataset (Adult Income)
**Impacto**: Generalização questionável
**Tempo**: 3-4 dias

**Datasets sugeridos**:

1. **COMPAS** (Criminal recidivism)
   - Protected: race, gender
   - Target: recidivism
   - Fonte: ProPublica

2. **German Credit**
   - Protected: age, gender
   - Target: credit approval
   - Fonte: UCI ML Repository

3. **Law School Admissions**
   - Protected: race, gender
   - Target: admission decision
   - Fonte: Fair ML datasets

**Implementação**:

```bash
# experimentos/01_benchmarks_tempo/data/
├── adult_income/      # Existente
├── compas/            # Novo
├── german_credit/     # Novo
└── law_school/        # Novo

# Executar benchmarks em todos
for dataset in adult_income compas german_credit law_school; do
    python scripts/run_experiment.py --dataset $dataset
done

# Agregar resultados
python scripts/aggregate_multidata.py
```

**Validação**:
- [ ] 3 datasets adicionais processados
- [ ] Benchmarks executados em todos
- [ ] Resultados consistentes (speedup 5-9×)
- [ ] Tabela comparativa criada

**Resultado esperado**:
```
Dataset         DeepBridge   Fragmented   Speedup
--------------------------------------------------
Adult Income    23.4s        150s         6.4×
COMPAS          18.2s        125s         6.9×
German Credit   15.1s        95s          6.3×
Law School      21.3s        140s         6.6×
--------------------------------------------------
MÉDIA           19.5s        127.5s       6.5× ± 0.3×
```

---

### 2.2 Comparação com Ferramentas Existentes

**Problema**: Não compara com ferramentas além do baseline fragmentado
**Impacto**: Falta contexto, reviewers perguntarão
**Tempo**: 2-3 dias

**Ferramentas para comparar**:

1. **Fairlearn** (Microsoft)
2. **AIF360** (IBM)
3. **What-If Tool** (Google)
4. **Alibi** (Seldon)

**Implementação**:

```python
# experimentos/07_comparacao_ferramentas/ (NOVO)

def compare_with_fairlearn(dataset):
    """Comparar com Fairlearn"""
    from fairlearn.metrics import MetricFrame

    start = time.time()

    # Setup Fairlearn
    metric_frame = MetricFrame(
        metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
        y_true=dataset.y,
        y_pred=dataset.predictions,
        sensitive_features=dataset.protected_attributes
    )

    # Calcular disparate impact
    results = metric_frame.by_group

    elapsed = time.time() - start

    return {
        'tool': 'Fairlearn',
        'time': elapsed,
        'results': results
    }

# Comparar: DeepBridge vs Fairlearn vs AIF360 vs Alibi
```

**Validação**:
- [ ] 3-4 ferramentas integradas
- [ ] Benchmarks executados
- [ ] Tabela comparativa criada
- [ ] Discussão de tradeoffs

**Resultado esperado**:
```
Ferramenta    Tempo   Métricas Cobertas   Facilidade Uso
------------------------------------------------------------
DeepBridge    23s     Fairness, Rob, Unc  ⭐⭐⭐⭐⭐
Fairlearn     45s     Fairness only       ⭐⭐⭐⭐
AIF360        60s     Fairness only       ⭐⭐⭐
Alibi         40s     Robustness only     ⭐⭐⭐
```

---

## 🟢 OPCIONAL - Semana 5-6

### 3.1 Survey de Usabilidade Real

**Problema**: Exp 3 usa dados mock
**Impacto**: Baixo (usabilidade é secundária)
**Tempo**: 1 semana

**Implementação**:

1. Recrutar 10-15 participantes (colegas, estudantes)
2. Tarefa: "Validar modelo de crédito para fairness"
3. Grupos:
   - Grupo A: DeepBridge
   - Grupo B: Fairlearn
4. Medir:
   - Tempo para completar
   - Linhas de código
   - Satisfação (escala Likert)

**Validação**:
- [ ] ≥10 participantes
- [ ] Diferença significativa (t-test)
- [ ] Questionário de satisfação

---

### 3.2 Análise de Sensibilidade

**Problema**: Falta robustez das conclusões
**Impacto**: Médio
**Tempo**: 2-3 dias

**Implementação**:

```python
# Testar diferentes thresholds de DI
for threshold in [0.75, 0.80, 0.85, 0.90]:
    precision, recall = evaluate_compliance(threshold)
    plot(threshold, precision, recall)

# Testar com diferentes tamanhos de amostra
for n_samples in [100, 500, 1000, 5000]:
    speedup = run_benchmark(n_samples)
    plot(n_samples, speedup)
```

**Validação**:
- [ ] Resultados consistentes entre thresholds
- [ ] Speedup escala com tamanho
- [ ] Gráficos de sensibilidade

---

## 📊 Checklist de Validação Final

Antes de submeter, garantir:

### Validade Interna
- [ ] Baseline real (não simulado)
- [ ] Tempos medidos (não simulados)
- [ ] Métricas corretas
- [ ] Análise estatística apropriada

### Validade Externa
- [ ] ≥3 datasets testados
- [ ] Comparação com ≥2 ferramentas
- [ ] Resultados consistentes

### Reprodutibilidade
- [ ] Código público (GitHub)
- [ ] README com instruções claras
- [ ] Dados disponíveis (ou script para gerar)
- [ ] Seeds fixos

### Escrita
- [ ] Seção de limitações honesta
- [ ] Claims suportados por dados
- [ ] Tabelas e figuras claras
- [ ] Ameaças à validade discutidas

---

## 📅 Timeline Sugerido

### Semana 1-2 (CRÍTICO)
- Dias 1-5: Exp 5 - Baseline real
- Dias 6-10: Exp 6 - Decisão e ação

### Semana 3-4 (IMPORTANTE)
- Dias 11-14: Adicionar datasets
- Dias 15-17: Comparação com ferramentas
- Dia 18: Completar Exp 1 fairness

### Semana 5 (ANÁLISE)
- Dias 19-21: Análise estatística completa
- Dias 22-23: Atualizar visualizações

### Semana 6 (ESCRITA)
- Dias 24-26: Atualizar paper
- Dias 27-28: Revisão interna
- Dia 29: Submeter para review interno
- Dia 30: Correções finais

**SUBMISSÃO**: Dia 35-40

---

## 💰 Estimativa de Esforço

| Tarefa | Prioridade | Tempo | Pessoa-dias |
|--------|-----------|-------|-------------|
| Exp 5 - Baseline real | P0 | 4-5 dias | 5 |
| Exp 6 - Opção A (remover) | P0 | 1h | 0.1 |
| Exp 6 - Opção B (implementar) | P0 | 1-2 semanas | 10 |
| Exp 1 - Fairness | P0 | 2 dias | 2 |
| Adicionar datasets | P1 | 3-4 dias | 4 |
| Comparar ferramentas | P1 | 2-3 dias | 3 |
| Survey usabilidade | P2 | 1 semana | 5 |
| Análise sensibilidade | P2 | 2-3 dias | 3 |
| Escrita/revisão | - | 1 semana | 5 |

**TOTAL (Cenário mínimo)**: 10-12 pessoa-dias (2-3 semanas)
**TOTAL (Cenário completo)**: 30-35 pessoa-dias (6-7 semanas)

---

## 🎯 Meta Final

**Paper publicável em conferência Tier 2** com potencial para Tier 1 se execução for excelente.

**Investimento**: 4-6 semanas de trabalho focado
**Retorno**: Paper sólido, cientificamente válido, publicável

**Alternativa**: Não corrigir = rejeição certa

**Decisão**: Nos próximos 2 dias
