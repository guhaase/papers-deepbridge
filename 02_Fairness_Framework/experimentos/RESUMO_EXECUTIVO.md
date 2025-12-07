# 📊 Resumo Executivo - Experimentos DeepBridge Fairness

**Paper**: DeepBridge Fairness Framework
**Conferência**: FAccT 2026
**Status**: Planejamento completo ✅

---

## 🎯 Objetivo

Validar **15 claims principais** do paper através de experimentos reproduzíveis e rigorosos.

---

## 📋 Claims a Validar

| # | Claim | Target | Crítico? |
|---|-------|--------|----------|
| 1 | Auto-detecção F1-Score | 0.90 | ⭐ SIM |
| 2 | Auto-detecção Precision | 0.92 | ⭐ SIM |
| 3 | Auto-detecção Recall | 0.89 | ⭐ SIM |
| 4 | 15 métricas integradas | 4+11 | SIM |
| 5 | EEOC/ECOA precisão | 100% | ⭐⭐⭐ CRÍTICO |
| 6 | SUS Score | 85.2 | ⭐ SIM |
| 7 | NASA-TLX | 32.1 | NÃO |
| 8 | Taxa de sucesso | 95% | SIM |
| 9 | Time-to-insight | 10.2 min | NÃO |
| 10 | Speedup médio | 2.9x | ⭐ SIM |
| 11 | Redução de memória | 40-42% | NÃO |
| 12 | COMPAS tempo | 7.2 min | SIM |
| 13 | German Credit tempo | 5.8 min | SIM |
| 14 | Adult tempo | 12.4 min | SIM |
| 15 | Healthcare tempo | 9.1 min | SIM |

**⭐⭐⭐ = Deal-breaker** (paper rejeitado se falhar)
**⭐ = Importante** (enfraquece paper se falhar)
**SIM/NÃO = Desejável** (bom ter, mas não essencial)

---

## 🔬 Experimentos Necessários

### 1. Auto-Detecção (⭐ CRÍTICO)
- **O quê**: Testar detecção automática de atributos sensíveis
- **Como**: 500 datasets anotados manualmente
- **Tempo**: 3-4 semanas
- **Esforço**: Alto (requer anotação manual)
- **Critério**: F1 ≥ 0.85

### 2. EEOC/ECOA (⭐⭐⭐ DEAL-BREAKER)
- **O quê**: Validar conformidade regulatória
- **Como**: Casos de teste controlados + datasets reais
- **Tempo**: 1 semana
- **Esforço**: Baixo
- **Critério**: 100% precisão (0 erros permitidos!)

### 3. Case Studies (⭐ CRÍTICO)
- **O quê**: Reproduzir 4 estudos de caso
- **Como**: COMPAS, German Credit, Adult, Healthcare
- **Tempo**: 3 semanas
- **Esforço**: Médio
- **Critério**: 75%+ economia de tempo vs manual

### 4. Usabilidade (⭐ CRÍTICO)
- **O quê**: Estudo com usuários reais
- **Como**: 20 participantes, tarefas + questionários
- **Tempo**: 4 semanas (inclui recrutamento)
- **Esforço**: Alto (requer pessoas)
- **Critério**: SUS ≥ 75

### 5. Performance (⭐ IMPORTANTE)
- **O quê**: Medir speedup e memória
- **Como**: Benchmarks em 3 tamanhos de datasets
- **Tempo**: 1 semana
- **Esforço**: Baixo
- **Critério**: Speedup ≥ 2.0x

### 6. Comparação (IMPORTANTE)
- **O quê**: Comparar com AIF360, Fairlearn, Aequitas
- **Como**: Feature matrix + accuracy de métricas
- **Tempo**: 1 semana
- **Esforço**: Baixo
- **Critério**: DeepBridge único com features claimed

---

## 📅 Timeline

```
┌─────────────────────────────────────────────────────────┐
│ FASE 1: SETUP (Semanas 1-2)                            │
├─────────────────────────────────────────────────────────┤
│ ✓ Ambiente configurado                                  │
│ ✓ Datasets coletados (500)                              │
│ ✓ Ground truth anotado                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ FASE 2: EXPERIMENTOS CORE (Semanas 3-9)                │
├─────────────────────────────────────────────────────────┤
│ ⭐ Auto-detecção (S3-4)                                 │
│ ⭐⭐⭐ EEOC/ECOA (S5)                                   │
│ ⭐ Case Studies (S6-9)                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ FASE 3: USABILIDADE (Semanas 10-13)                    │
├─────────────────────────────────────────────────────────┤
│ ⭐ Recrutamento (S10)                                   │
│ ⭐ Execução (S11-12)                                    │
│ ⭐ Análise (S13)                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ FASE 4: VALIDAÇÃO (Semanas 14-16)                      │
├─────────────────────────────────────────────────────────┤
│ Performance (S14)                                        │
│ Comparação (S15)                                         │
│ Robustness (S16)                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ FASE 5: FINALIZAÇÃO (Semanas 17-18)                    │
├─────────────────────────────────────────────────────────┤
│ Análise agregada                                         │
│ Relatórios e figuras                                     │
│ Reproduction package                                     │
│ Paper submission                                         │
└─────────────────────────────────────────────────────────┘

TOTAL: 18 semanas (4.5 meses)
```

---

## 💰 Recursos Necessários

### Pessoas:
- **1 Pesquisador Principal**: Full-time (18 semanas)
- **1 Assistente de Pesquisa**: Part-time (10 semanas) - para anotações
- **2 Revisores de Ground Truth**: 40h cada
- **20 Participantes de Usabilidade**: 1h cada
- **1 Compliance Officer**: 4h (revisar EEOC/ECOA)

### Hardware:
- **AWS m5.2xlarge**: ~$100 (para benchmarks de performance)
- **Computador local**: Para desenvolvimento

### Outros:
- **Incentivos para participantes**: 20 × $50 = $1,000
- **Licenças de datasets**: ~$200 (alguns datasets pagos)
- **Total Estimado**: ~$1,300 + salários

---

## ⚠️ Riscos Principais

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| **Auto-detecção F1 < 0.85** | Média | Alto | Ajustar algoritmo, mais dados |
| **EEOC bugs** | Baixa | **CRÍTICO** | Testes exaustivos |
| **Recrutamento < 15** | Média | Alto | Mais incentivos, prazo estendido |
| **SUS < 75** | Baixa | Alto | Melhorar UX, documentação |
| **Speedup < 2x** | Baixa | Médio | Otimizações de código |

---

## ✅ Critérios Mínimos para Aceitação

### Para o paper ser aceito no FAccT 2026:

**DEVE TER** (Deal-breakers):
- ✅ EEOC/ECOA: 100% precisão
- ✅ SUS: ≥ 75
- ✅ Speedup: ≥ 2.0x
- ✅ Case Studies: 4/4 completos
- ✅ Usabilidade N: ≥ 15

**BOM TER** (Fortalece):
- F1 auto-detecção: ≥ 0.85
- Taxa de sucesso: ≥ 85%
- Datasets: ≥ 300

**EXCELENTE TER** (Top-tier):
- Todos claims validados ±10%
- N = 20 participantes
- 500 datasets
- Reproduction package completo

---

## 🚀 Quick Start

### Hoje (30 minutos):

```bash
# 1. Setup ambiente
cd /home/guhaase/projetos/DeepBridge/papers/02_Fairness_Framework/experimentos
python -m venv venv
source venv/bin/activate
pip install deepbridge pandas numpy matplotlib

# 2. Teste rápido
cd scripts/
python exp1_auto_detection.py --quick

# 3. Ver checklist
cat ../CHECKLIST_RAPIDO.md
```

### Esta Semana:

1. **Ler documentação completa**:
   - `PLANO_EXPERIMENTOS.md` (detalhado)
   - `GUIA_EXECUCAO.md` (passo a passo)
   - `CHECKLIST_RAPIDO.md` (tracking)

2. **Começar coleta de datasets**:
   - Kaggle: 200 datasets
   - UCI: 150 datasets
   - OpenML: 100 datasets

3. **Iniciar recrutamento para usabilidade**:
   - Postar em LinkedIn
   - Email para colegas
   - Preparar materiais (consent form, etc)

### Este Mês:

1. Completar coleta de 500 datasets
2. Anotar ground truth (2 revisores)
3. Executar Experimento 1 (auto-detecção)
4. Executar Experimento 3 (EEOC/ECOA)

---

## 📁 Arquivos Importantes

| Arquivo | Propósito | Quando Usar |
|---------|-----------|-------------|
| `RESUMO_EXECUTIVO.md` | Visão geral (este arquivo) | Primeiro contato |
| `PLANO_EXPERIMENTOS.md` | Detalhes completos | Executar experimentos |
| `GUIA_EXECUCAO.md` | Passo a passo prático | Durante execução |
| `CHECKLIST_RAPIDO.md` | Tracking de progresso | Diariamente |
| `README.md` | Documentação geral | Setup inicial |

---

## 📊 Métricas de Sucesso

### Nível 1: Mínimo Viável
- 3/6 experimentos críticos completos
- EEOC/ECOA 100% correto
- SUS ≥ 70
- Speedup ≥ 1.8x
- **Resultado**: Paper aceito com revisões

### Nível 2: Completo
- 6/6 experimentos críticos completos
- Todos critérios mínimos atendidos
- ≥80% das claims validadas
- **Resultado**: Paper aceito

### Nível 3: Excelente
- Todos experimentos completos
- 100% das claims validadas ±10%
- Reproduction package exemplar
- **Resultado**: Paper aceito + spotlight/oral

---

## 💡 Dicas de Produtividade

### Priorização (se tempo limitado):

**Opção 1: Mínimo (8 semanas)**
- Auto-detecção: 100 datasets
- EEOC: Teste completo
- Case Studies: 4/4
- Usabilidade: 10 participantes
- Performance: Skip large dataset

**Opção 2: Balanceado (12 semanas)**
- Auto-detecção: 300 datasets
- EEOC: Teste completo
- Case Studies: 4/4
- Usabilidade: 15 participantes
- Performance: Todos tamanhos
- Comparação: 2 ferramentas

**Opção 3: Completo (18 semanas)**
- Tudo conforme planejado

### Paralelização:

Enquanto espera participantes de usabilidade (semanas 10-13):
- Execute performance (semana 10)
- Execute comparação (semana 11)
- Execute robustness (semana 12)

---

## 🎯 Próximos Passos Imediatos

1. ✅ **Hoje**: Executar teste rápido
   ```bash
   python scripts/exp1_auto_detection.py --quick
   ```

2. 📊 **Esta semana**: Começar coleta de datasets

3. 👥 **Esta semana**: Iniciar recrutamento

4. 📅 **Este mês**: Executar experimentos 1 e 3

5. 📝 **Próximo mês**: Executar case studies

---

## 📞 Suporte

**Perguntas Frequentes**: Ver `GUIA_EXECUCAO.md` seção Troubleshooting

**Issues Técnicos**: Abrir issue no repositório

**Dúvidas Metodológicas**: Consultar `PLANO_EXPERIMENTOS.md`

---

**Status**: ⬜ Planejamento completo, aguardando execução

**Próxima Revisão**: Após experimentos 1 e 3 (semana 6)

**Boa sorte! 🚀**

---

## 📈 Dashboard de Progresso

```
EXPERIMENTOS CRÍTICOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Auto-Detecção        [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
2. EEOC/ECOA            [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
3. Case Studies         [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
4. Usabilidade          [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
5. Performance          [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%
6. Comparação           [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%

PROGRESSO GERAL        [ ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ ]   0%

⬜ = Não iniciado | 🔄 = Em progresso | ✅ = Completo
```

**Atualizar após cada marco!**
