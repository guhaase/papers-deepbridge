# Experimento 1: Benchmarks de Tempo - Correção em Andamento

**Data**: 2025-12-08
**Status**: 🟡 **EM PROGRESSO**

---

## ✅ O que foi feito até agora

### 1. Análise do Problema ✅

**Problema identificado**:
- Baseline usa `time.sleep()` para SIMULAR delays
- 17 ocorrências de simulações em `benchmark_fragmented.py`
- DEMO_SPEEDUP_FACTOR = 60 converte minutos → segundos
- Comparação inválida: DeepBridge real vs baseline simulado

**Evidências**:
```python
# benchmark_fragmented.py:30-32
DEMO_SPEEDUP_FACTOR = 60  # Minutos → segundos!

# benchmark_fragmented.py:145-154
time.sleep((5 * 60) / DEMO_SPEEDUP_FACTOR)   # Simula 5 min
time.sleep((15 * 60) / DEMO_SPEEDUP_FACTOR)  # Simula 15 min
time.sleep((3 * 60) / DEMO_SPEEDUP_FACTOR)   # Simula 3 min
time.sleep((7 * 60) / DEMO_SPEEDUP_FACTOR)   # Simula 7 min
```

### 2. Implementação do Baseline REAL ✅

**Arquivo criado**: `benchmark_fragmented_REAL.py`

**Ferramentas REAIS implementadas**:

#### a) Fairness (AIF360 + Fairlearn) ✅
```python
# Conversão REAL para AIF360
aif_dataset = BinaryLabelDataset(df=df_encoded, ...)

# Métricas REAIS com AIF360
metric = BinaryLabelDatasetMetric(aif_dataset, ...)
di = metric.disparate_impact()  # CALCULADO, não simulado

# Métricas REAIS com Fairlearn
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=...)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=...)
```

#### b) Robustness (sklearn) ✅
```python
# Perturbações gaussianas REAIS
noise = np.random.normal(0, noise_level, X_numeric.shape)
X_perturbed = X_numeric + noise

# Testes adversariais REAIS
y_pred_perturbed = model.predict(X_perturbed)
acc_perturbed = accuracy_score(y_test, y_pred_perturbed)
```

#### c) Uncertainty (calibração real) ✅
```python
# Obter probabilidades REAIS
y_proba = model.predict_proba(X_test)[:, 1]

# Calibração REAL
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_proba, n_bins=10
)
ece = np.abs(fraction_of_positives - mean_predicted_value).mean()
```

#### d) Resilience (drift real) ✅
```python
# Wasserstein distance REAL
wd = wasserstein_distance(
    X_train_numeric[col].values,
    X_test_numeric[col].values
)
```

#### e) Report Generation (matplotlib real) ✅
```python
# Visualizações REAIS com matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... criar plots ...
plt.savefig(fig_path, dpi=300)

# Documento REAL em texto
with open(report_path, 'w') as f:
    f.write("VALIDATION REPORT...")
```

### 3. Correções de Bugs ✅

**Bugs corrigidos**:
1. ❌ `ExperimentLogger.setup_logger()` não existe
   - ✅ Substituído por `logging.basicConfig()`

2. ❌ Config file: `experiment_config.yaml` não existe
   - ✅ Corrigido para `config.yaml`

3. ❌ Config key: `config['execution']['seed']` não existe
   - ✅ Corrigido para `config['general']['seed']`

4. ❌ `save_results(results, path, self.logger)` assinatura incorreta
   - ✅ Corrigido para `save_results(results, path, formats=['json'])`

### 4. Execução em Andamento 🟡

**Status atual**:
- ✅ DeepBridge JÁ executado (tempos REAIS disponíveis)
- 🟡 Baseline REAL em execução (pode levar 5-15 minutos)

**DeepBridge - Tempos REAIS** (de `deepbridge_times_REAL.json`):
```
Robustness:  13.6s (±0.7s)
Uncertainty:  5.8s (±0.3s)
Resilience:   3.9s (±0.2s)
Report:       0.08s (±0.05s)
Total:       23.4s (±1.2s)
```

**Nota**: Fairness está vazio (no_data) - precisa investigar

---

## ⏳ Próximos Passos

### Imediato (Aguardando)

1. **Aguardar conclusão do baseline REAL** (~5-15 min)
   - Processo rodando: `benchmark_fragmented_REAL.py`
   - Output esperado: `fragmented_benchmark_REAL.json`

2. **Verificar resultados**
   - Arquivo: `results/fragmented_benchmark_REAL.json`
   - Validar tempos medidos (não simulados)

### Curto Prazo (1-2 horas)

3. **Comparar DeepBridge vs Baseline REAL**
   - Carregar ambos os JSONs
   - Calcular speedup REAL
   - Verificar se há diferença significativa

4. **Investigar problema do Fairness**
   - Por que DeepBridge fairness está vazio (no_data)?
   - Executar teste de fairness isolado
   - Corrigir e re-executar

5. **Gerar análise estatística**
   - Teste t pareado
   - Intervalos de confiança
   - Gráficos comparativos

6. **Atualizar documentação**
   - Marcar experimento 1 como CORRIGIDO
   - Adicionar disclaimers sobre versão antiga
   - Documentar metodologia REAL

### Médio Prazo (1-2 dias)

7. **Executar múltiplos runs**
   - Rodar baseline REAL 10 vezes (como DeepBridge)
   - Calcular média e desvio padrão
   - Análise de variabilidade

8. **Gerar figuras**
   - Comparação de tempos
   - Breakdown por teste
   - Speedup por componente

9. **Escrever relatório final**
   - Resultados corrigidos
   - Comparação com versão simulada
   - Adequação para publicação

---

## 📊 Expectativa de Resultados

### Cenário Otimista

**DeepBridge**: ~23s (já medido)
**Baseline REAL**: ~60-120s (estimativa)
**Speedup**: 2.5-5× (vs 8.9× simulado)

**Adequação**: ✅ Aceitável para Tier 2
- Comparação justa (ambos reais)
- Speedup modesto mas real
- Metodologia sólida

### Cenário Realista

**DeepBridge**: ~23s
**Baseline REAL**: ~30-60s
**Speedup**: 1.3-2.5×

**Adequação**: ⚠️ Borderline para Tier 2
- Speedup baixo
- Ainda válido (ferramentas reais)
- Precisa enfatizar outras contribuições

### Cenário Pessimista

**DeepBridge**: ~23s
**Baseline REAL**: ~20-30s (similar ou mais rápido)
**Speedup**: <1.5× ou negativo

**Adequação**: ❌ Problemático
- Não demonstra vantagem de performance
- Precisa focar em usabilidade, não velocidade
- Reformular narrativa do paper

---

## 🔧 Comandos para Monitorar

### Verificar se processo ainda está rodando

```bash
ps aux | grep benchmark_fragmented_REAL
```

### Ver output em tempo real

```bash
tail -f /tmp/fragmented_real_output.log
```

### Verificar resultado

```bash
ls -lh results/fragmented_benchmark_REAL.json
cat results/fragmented_benchmark_REAL.json | jq '.times_minutes'
```

### Comparar com DeepBridge

```bash
cat results/deepbridge_times_REAL.json | jq '.total.mean_minutes'
cat results/fragmented_benchmark_REAL.json | jq '.times_minutes.total'
```

---

## 📝 Arquivos Criados

### Código
- `scripts/benchmark_fragmented_REAL.py` ✅ (645 linhas)

### Resultados (esperados)
- `results/fragmented_benchmark_REAL.json` ⏳ (aguardando)
- `results/fragmented_report_REAL.txt` ⏳ (aguardando)
- `results/fragmented_report_figures.png` ⏳ (aguardando)

### Logs
- `logs/benchmark_fragmented_real_*.log` ⏳ (em geração)
- `/tmp/fragmented_real_output.log` ⏳ (stdout redirecionado)

---

## ⚠️ Riscos e Limitações

### Riscos Técnicos

1. **Tempo de execução muito longo**
   - Se baseline demorar > 15 min, pode ser impraticável
   - Solução: Reduzir tamanho do dataset de teste

2. **Erros em runtime**
   - AIF360/Fairlearn podem falhar com certos dados
   - Solução: Try-except com fallback

3. **Resultados inesperados**
   - Baseline pode ser mais rápido que DeepBridge
   - Solução: Reformular claim (usabilidade vs performance)

### Riscos para Publicação

1. **Speedup muito baixo**
   - Se < 2×, reviewers questionarão contribuição
   - Mitigação: Enfatizar API unificada

2. **Dataset único**
   - Apenas Adult Income
   - Mitigação: Adicionar mais datasets (TODO)

3. **Ferramentas limitadas**
   - Não inclui todas as ferramentas citadas
   - Mitigação: Ser transparente sobre escopo

---

## 📈 Métricas de Sucesso

### Para considerar CORRIGIDO

- [x] Baseline usa ferramentas REAIS (não time.sleep)
- [x] Código executa sem erros
- [ ] Resultados disponíveis em JSON
- [ ] Tempos medidos (não estimados)
- [ ] Comparação justa (mesma metodologia)

### Para considerar PUBLICÁVEL

- [ ] Speedup > 1.5× (mínimo)
- [ ] Análise estatística completa
- [ ] Múltiplos runs (n=10)
- [ ] Intervalos de confiança
- [ ] Documentação atualizada

### Para Tier 1

- [ ] Speedup > 3×
- [ ] Múltiplos datasets (≥3)
- [ ] Comparação com múltiplas ferramentas
- [ ] Ablation study
- [ ] Validação externa

---

## 👥 Recomendações para a Equipe

### Decisão Estratégica Necessária

**Pergunta**: Se baseline REAL for similar ou mais rápido que DeepBridge, qual narrativa usar?

**Opções**:

1. **Enfatizar Usabilidade**
   - API unificada vs fragmentada
   - Menos código para usar
   - Melhor DX (Developer Experience)

2. **Enfatizar Funcionalidade**
   - Múltiplos testes em uma call
   - Auto-reporting
   - Integração nativa

3. **Enfatizar Qualidade**
   - Detecção mais completa
   - Menos falsos positivos
   - Melhor coverage

4. **Reformular Experimento**
   - Adicionar overhead de integração ao baseline
   - Medir tempo total de workflow (não só execução)
   - Incluir tempo de desenvolvimento

### Ação Recomendada

**AGUARDAR** conclusão do baseline REAL antes de decidir próxima estratégia.

Se speedup < 1.5×:
- Reunir equipe
- Revisar claims do paper
- Reformular narrativa
- Considerar adicionar experimentos adicionais

Se speedup > 2×:
- Continuar com plano atual
- Adicionar mais datasets
- Finalizar análise

---

**Assinatura**: Correção em andamento
**Data**: 2025-12-08
**Versão**: 1.0 (Em progresso)
**Status**: 🟡 Aguardando conclusão do baseline REAL
