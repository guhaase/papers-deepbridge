# Comparação Visual dos Experimentos - DeepBridge

## Tabela Comparativa Completa

| Experimento | Dados | Baseline | Tempos | Execução Real | Score | Tier 1 | Tier 2 | Status |
|-------------|-------|----------|--------|---------------|-------|--------|--------|--------|
| **1. Benchmarks** | ✅ Real | ❌ Simulado | ⚠️ Parcial | ✅ DeepBridge Sim<br>❌ Baseline Não | 4/10 | ❌ | 🟡 | Corrigível |
| **2. Casos de Uso** | ❌ Sintético | ❌ Ausente | ❌ Simulado | ❌ Mock | 3/10 | ❌ | ❌ | Demo |
| **3. Usabilidade** | ❌ Mock | ❌ Ausente | ❌ Gerado | ❌ Mock | 5/10 | ❌ | 🟡 | Pilot |
| **4. HPM-KD** | ❌ Sintético | ❌ Ausente | ➖ N/A | ❌ Mock | 1/10 | ❌ | ❌ | Remover |
| **5. Conformidade** | ⚠️ Sintético | ✅ Real | ⚠️ Suspeito | ✅ Parcial | 6/10 | ❌ | 🟡 | Corrigível |
| **6. Ablation** | ❌ Sintético | ❌ Simulado | ❌ Simulado | ❌ Nada | 0/10 | ❌ | ❌ | Remover |

**Legenda**:
- ✅ Adequado
- ⚠️ Problemático mas corrigível
- ❌ Inadequado/Ausente
- 🟡 Borderline
- ➖ Não aplicável

---

## Matriz de Problemas

### Simulações com time.sleep()

| Experimento | Ocorrências | Localização | Propósito | Crítico? |
|-------------|-------------|-------------|-----------|----------|
| **Exp 1** | ~20 | `benchmark_fragmented.py:135-160` | Simular delays de conversão | ✅ SIM |
| **Exp 2** | ~12 | `case_study_*.py:124-147` | Simular validação | ✅ SIM |
| **Exp 3** | 0 | N/A | N/A | ❌ Não |
| **Exp 4** | 0 | N/A | N/A | ❌ Não |
| **Exp 5** | 0 | N/A | N/A | ❌ Não |
| **Exp 6** | ~8 | `run_ablation.py:167-180` | Simular "trabalho real" | ✅ SIM |

**Total**: 61 ocorrências de `time.sleep()` em código experimental

---

### Dados Mock vs Real

| Experimento | Tipo | Origem | Tamanho | Realismo | Aceitável? |
|-------------|------|--------|---------|----------|------------|
| **Exp 1** | Real | `fetch_openml('adult')` | 48k samples | Alto | ✅ Sim |
| **Exp 2** | Sintético | `np.random` | 1k samples | Médio | ⚠️ Com justificativa |
| **Exp 3** | Mock | `generate_mock_data.py` | 20 users | Baixo | ⚠️ Se pilot study |
| **Exp 4** | Sintético | `make_classification` | 36k samples | Baixo | ❌ Não |
| **Exp 5** | Sintético | `generate_ground_truth.py` | 50 casos | Médio | ⚠️ Com validação |
| **Exp 6** | Sintético | `make_classification` | 7k samples | Baixo | ❌ Não |

---

### Baselines: Real vs Simulado

| Experimento | Baseline | Ferramentas | Execução | Problema |
|-------------|----------|-------------|----------|----------|
| **Exp 1** | Workflow Fragmentado | AIF360, Fairlearn, Alibi | ❌ **time.sleep()** | Compara real vs simulado |
| **Exp 2** | Nenhum | N/A | N/A | Sem comparação |
| **Exp 3** | Nenhum | N/A | N/A | Sem comparação |
| **Exp 4** | Vanilla KD, TAKD, Auto-KD | N/A | ❌ **np.random.normal()** | Baselines inventados |
| **Exp 5** | AIF360 + Fairlearn | AIF360 | ✅ **Real** | Tempo estimado, não medido |
| **Exp 6** | Configs sem componentes | N/A | ❌ **Hardcoded times** | Todas configs simuladas |

**Único baseline real**: Experimento 5

---

## Análise de Validade Científica

### Critérios de Avaliação

| Critério | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 |
|----------|-------|-------|-------|-------|-------|-------|
| **Dados reais** | ✅ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| **Baseline real** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Tempos medidos** | ⚠️ | ❌ | ❌ | ➖ | ⚠️ | ❌ |
| **N adequado** | ✅ | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ |
| **Estatística robusta** | ✅ | ❌ | ⚠️ | ❌ | ⚠️ | ❌ |
| **Reprodutível** | ✅ | ✅ | ⚠️ | ❌ | ✅ | ❌ |
| **Transparente** | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ |

**Notas**:
- ✅ Sim (atende critério)
- ⚠️ Parcial (atende com limitações)
- ❌ Não (não atende)
- ➖ Não aplicável

---

## Roadmap de Correções (Priorizado)

### 🔴 Prioridade CRÍTICA (Semana 1-2)

| Tarefa | Experimento | Tempo | Impacto | Viabilidade |
|--------|-------------|-------|---------|-------------|
| Implementar baseline real | Exp 1 | 1-2 sem | Alto | Alta |
| Corrigir ground truth | Exp 5 | 2-3 dias | Médio | Alta |
| Investigar tempo suspeito | Exp 5 | 1-2 dias | Médio | Alta |
| **REMOVER** Exp 4 do paper | Exp 4 | 1 hora | Alto | Imediata |
| **REMOVER** Exp 6 do paper | Exp 6 | 1 hora | Alto | Imediata |

### 🟡 Prioridade MÉDIA (Semana 3-4)

| Tarefa | Experimento | Tempo | Impacto | Viabilidade |
|--------|-------------|-------|---------|-------------|
| Usar dados reais | Exp 2 | 2-3 sem | Médio | Média |
| Executar DeepBridge real | Exp 2 | 1 sem | Alto | Alta |
| Conduzir estudo real | Exp 3 | 2-3 sem | Médio | Média |
| Aumentar N para 100+ | Exp 5 | 1 sem | Médio | Alta |

### ⚪ Prioridade BAIXA (Opcional)

| Tarefa | Experimento | Tempo | Impacto | Viabilidade |
|--------|-------------|-------|---------|-------------|
| Implementar HPM-KD real | Exp 4 | 4-6 sem | Baixo | Baixa |
| Implementar ablation real | Exp 6 | 2-4 sem | Médio | Média |
| Adicionar mais casos | Exp 2 | 1-2 sem | Baixo | Média |

---

## Cenários de Publicação

### Cenário A: Mínimo Viável (4-6 semanas)

**Experimentos incluídos**: 3
- ✅ Exp 1 (corrigido)
- ✅ Exp 5 (corrigido)
- ⚠️ Exp 3 (com disclaimer)

**Trabalho**:
- Corrigir Exp 1: baseline real (1-2 semanas)
- Corrigir Exp 5: GT + tempo (3-5 dias)
- Remover Exp 4, 6 (1 hora)
- Adicionar disclaimer Exp 3 (1 hora)

**Resultado**:
- Adequação: Tier 2 borderline
- Risco rejeição: 40-50%
- Contribuição: Moderada

---

### Cenário B: Robusto (8-10 semanas)

**Experimentos incluídos**: 4
- ✅ Exp 1 (corrigido)
- ✅ Exp 2 (corrigido)
- ✅ Exp 3 (estudo real)
- ✅ Exp 5 (corrigido)

**Trabalho**:
- Corrigir Exp 1: baseline real (1-2 semanas)
- Corrigir Exp 2: dados + execução real (2-3 semanas)
- Corrigir Exp 3: estudo real (2-3 semanas)
- Corrigir Exp 5: GT + tempo + N (1-2 semanas)
- Remover Exp 4, 6 (1 hora)

**Resultado**:
- Adequação: Tier 2 forte / Tier 1 borderline
- Risco rejeição: 25-35%
- Contribuição: Alta

---

### Cenário C: Completo (12-16 semanas)

**Experimentos incluídos**: 6
- Todos corrigidos e validados

**Trabalho**:
- Cenário B +
- Implementar HPM-KD real (4-6 semanas)
- Implementar ablation real (2-4 semanas)

**Resultado**:
- Adequação: Tier 1
- Risco rejeição: 15-25%
- Contribuição: Muito Alta

---

## Matriz de Decisão

### Se deadline < 6 semanas
→ **Cenário A** (mínimo viável)
- Foca em Exp 1 e 5
- Remove Exp 4 e 6
- Aceita Exp 3 como pilot

### Se deadline 6-10 semanas
→ **Cenário B** (robusto)
- Corrige Exp 1, 2, 3, 5
- Remove Exp 4 e 6
- Paper forte para Tier 2

### Se deadline > 10 semanas
→ **Cenário C** (completo)
- Corrige todos
- Implementa tudo do zero
- Paper competitivo para Tier 1

---

## Evidências de Problemas (para Reference)

### Experimento 1
```python
# benchmark_fragmented.py:30-32
DEMO_SPEEDUP_FACTOR = 60  # Converte minutos → segundos!

# benchmark_fragmented.py:135-150
time.sleep((5 * 60 + np.random.normal(0, 30)) / DEMO_SPEEDUP_FACTOR)  # 5 min → 5s
time.sleep((15 * 60 + np.random.normal(0, 30)) / DEMO_SPEEDUP_FACTOR) # 15 min → 15s
```

### Experimento 2
```python
# case_study_credit.py:39
def load_german_credit_data():
    """In real implementation, this would load from UCI repository.
    For now, we generate synthetic data"""
    np.random.seed(42)
    # ... gera tudo fake ...

# case_study_credit.py:124
with Timer("Fairness Tests", logger) as t:
    time.sleep(5)  # Simulate computation
```

### Experimento 6
```python
# run_ablation.py:156-157
base_time = config['expected_time_min'] * 60  # Hardcoded!
simulated_time = max(base_time + variation, 0)

# run_ablation.py:186
execution_times.append(simulated_time / 60.0)  # Usa simulação!
```

---

## Checklist Pré-Submissão

### Code Quality
- [ ] Remover TODOS os `time.sleep()` de código experimental
- [ ] Substituir dados mock por reais onde aplicável
- [ ] Implementar baselines reais
- [ ] Medir tempos de verdade
- [ ] Código passar em code review independente

### Paper Quality
- [ ] Claims consistentes com código
- [ ] Limitações claramente descritas
- [ ] Disclaimers onde necessário
- [ ] Seção de ameaças à validade honesta
- [ ] Material suplementar com código completo

### Validação
- [ ] Reprodutibilidade verificada
- [ ] Estatísticas robustas (p < 0.01, não 0.0499)
- [ ] N adequado (100+ para estatística)
- [ ] Reviewer mock aprovar

### Ética
- [ ] Nenhuma simulação disfarçada de real
- [ ] Transparência total sobre mock data
- [ ] Código e paper 100% consistentes
- [ ] Nenhuma claim exagerada

---

## Conclusão

**Status Atual**: 4 de 6 experimentos (67%) são inválidos

**Com Correções Mínimas**: 3 experimentos válidos (Tier 2 borderline)

**Com Correções Completas**: 6 experimentos válidos (Tier 1)

**Recomendação**: Executar **Cenário A ou B** dependendo do deadline. Cenário C é ideal mas pode não ser viável.

**CRÍTICO**: NÃO submeter no estado atual. Risco de rejeição é 90%+ e pode danificar reputação.

---

**Próximos Passos**:
1. Reunião de equipe para decidir cenário
2. Começar correções imediatamente
3. Code review contínuo
4. Validação externa antes de submissão

---

*Relatório gerado por Claude Code - Análise Crítica de Experimentos*
*Data: 2025-12-07*
*Arquivos JSON completos disponíveis em: `AVALIACAO_COMPLETA_EXPERIMENTOS.json`*
