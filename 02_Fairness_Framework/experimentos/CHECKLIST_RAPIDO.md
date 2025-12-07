# ✅ Checklist Rápido de Experimentos

## 🎯 Experimentos Críticos (Prioridade MÁXIMA)

### 1. Auto-Detecção (500 datasets)
- [ ] Coletar 500 datasets (Kaggle + UCI + OpenML)
- [ ] Anotar ground truth (2 especialistas, Kappa > 0.85)
- [ ] Executar auto-detecção
- [ ] **Meta**: F1 ≥ 0.90 (Precision: 0.92, Recall: 0.89)

### 2. Verificação EEOC/ECOA
- [ ] Testar regra 80% (5 casos controlados)
- [ ] Testar Question 21 (4 casos controlados)
- [ ] Validar em case studies
- [ ] **Meta**: 100% precisão, 0 falsos positivos

### 3. Case Studies (COMPAS, Credit, Adult, Healthcare)
- [ ] COMPAS: Tempo ≤ 10 min, FPR reduction ≥ 60%
- [ ] German Credit: Tempo ≤ 8 min, DI violação detectada
- [ ] Adult Income: Tempo ≤ 15 min, DI Female 0.40-0.46
- [ ] Healthcare: Tempo ≤ 12 min, DI Black 1.35-1.50

### 4. Usabilidade - SUS Score
- [ ] Recrutar 20 participantes (2-8 anos exp ML)
- [ ] Executar 3 tarefas (Setup, Detection, Threshold)
- [ ] Aplicar SUS + TLX
- [ ] **Meta**: SUS ≥ 85, Taxa sucesso ≥ 95%

### 5. Performance - Speedup
- [ ] Small (1K): ≥ 3.5x speedup
- [ ] Medium (50K): ≥ 2.5x speedup
- [ ] Large (500K): ≥ 2.0x speedup
- [ ] **Meta**: Speedup médio ≥ 2.9x

### 6. Comparação com Ferramentas
- [ ] Testar AIF360, Fairlearn, Aequitas
- [ ] Validar feature matrix
- [ ] Comparar acurácia de métricas
- [ ] **Meta**: DeepBridge única com auto-detecção + EEOC + threshold opt

---

## 📊 Validação de Claims Principais

| Claim | Experimento | Target | Status |
|-------|-------------|--------|--------|
| Auto-detecção F1=0.90 | 1.1 | ≥0.85 | ⬜ |
| 100% acurácia case studies | 1.2 | 100% | ⬜ |
| 15 métricas (4+11) | 2.1 | 15 | ⬜ |
| 100% precisão EEOC | 3.1 | 100% | ⬜ |
| SUS 85.2 | 5.1 | ≥75 | ⬜ |
| Speedup 2.9x | 6.1 | ≥2.0x | ⬜ |
| COMPAS 79% economia | 4.1 | ≥75% | ⬜ |
| Credit 77% economia | 4.2 | ≥75% | ⬜ |
| Adult 75% economia | 4.3 | ≥70% | ⬜ |
| Healthcare 77% economia | 4.4 | ≥75% | ⬜ |

---

## 🚨 Red Flags (Parar e Revisar)

- [ ] F1-Score < 0.80 → Revisar algoritmo de detecção
- [ ] SUS < 70 → Melhorar UX/documentação
- [ ] Speedup < 1.5x → Otimizar código
- [ ] Taxa sucesso < 80% → Simplificar API
- [ ] EEOC precision < 100% → BUG CRÍTICO

---

## 📅 Timeline Resumido

| Semana | Atividade | Deliverable |
|--------|-----------|-------------|
| 1-2 | Setup + Coleta de dados | 500 datasets prontos |
| 3-4 | Auto-detecção | Exp 1.1, 1.2 completos |
| 5-6 | Métricas + EEOC | Exp 2.1, 3.1-3.3 completos |
| 7-9 | Case Studies | Exp 4.1-4.4 completos |
| 10-12 | Usabilidade | Exp 5.1-5.5 completos |
| 13-14 | Performance | Exp 6.1-6.3, 7.1-7.2 completos |
| 15 | Comparação | Exp 8.1-8.2 completos |
| 16 | Robustness | Exp 9.1 completo |
| 17-18 | Finalização | Paper submission ready |

---

## 📦 Artefatos Essenciais

### Dados:
- [ ] `data/ground_truth.csv` - 500 datasets anotados
- [ ] `data/case_studies/compas.csv`
- [ ] `data/case_studies/german_credit.csv`
- [ ] `data/case_studies/adult.csv`
- [ ] `data/case_studies/healthcare.csv`

### Resultados:
- [ ] `results/auto_detection_500_datasets.csv`
- [ ] `results/eeoc_validation.csv`
- [ ] `results/case_studies_summary.csv`
- [ ] `results/sus_scores.csv`
- [ ] `results/performance_benchmarks.csv`
- [ ] `results/tool_comparison_matrix.csv`

### Scripts:
- [ ] `scripts/exp1_auto_detection.py`
- [ ] `scripts/exp3_eeoc_validation.py`
- [ ] `scripts/exp4_case_studies.py`
- [ ] `scripts/exp5_usability_analysis.py`
- [ ] `scripts/exp6_performance.py`

### Relatórios:
- [ ] `reports/experiment_summary.pdf`
- [ ] `reports/reproduction_guide.md`

---

## 🎯 Critérios de Aceitação (FAccT 2026)

### ✅ DEVE TER (Deal-breakers):
1. 100% precisão em EEOC/ECOA ← CRÍTICO
2. SUS ≥ 75
3. Speedup ≥ 2.0x
4. Case studies completos (4/4)
5. N ≥ 15 participantes em usabilidade

### ⭐ BOM TER (Fortalece):
1. F1 auto-detecção ≥ 0.90
2. N = 20 participantes
3. 500 datasets
4. Comparação com 3 ferramentas
5. Reproduction package

### 🚀 EXCELENTE TER (Top-tier):
1. Todos claims validados ±10%
2. Open-source dataset annotations
3. Live demo
4. Industry adoption cases

---

## 📝 Notas de Execução

### Priorização se tempo/recursos limitados:

**Opção 1: Mínimo Viável (8 semanas)**
- Auto-detecção: 100 datasets (não 500)
- Usabilidade: 10 participantes (não 20)
- Case studies: 4 completos
- Performance: Small + Medium (não Large)
- Comparação: 2 ferramentas (AIF360 + Fairlearn)

**Opção 2: Balanceado (12 semanas)**
- Auto-detecção: 300 datasets
- Usabilidade: 15 participantes
- Case studies: 4 completos
- Performance: Todos tamanhos
- Comparação: 3 ferramentas

**Opção 3: Completo (18 semanas)**
- Tudo conforme plano original

---

## ⚠️ Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Datasets insuficientes | Média | Alto | Gerar sintéticos adicionais |
| Baixo recrutamento (N<15) | Média | Médio | Incentivos + prazo estendido |
| SUS < 75 | Baixa | Alto | Melhorar docs + tutoriais |
| Speedup < 2x | Baixa | Médio | Otimizar threshold opt |
| EEOC bugs | Baixa | CRÍTICO | Testes exaustivos + revisão |

---

**Última atualização**: 2025-12-06
**Status**: ⬜ Não iniciado | 🔄 Em progresso | ✅ Completo
