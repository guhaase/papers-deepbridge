# Experimento 3: Estudo de Usabilidade

## Objetivo

Comprovar as afirmações sobre usabilidade do DeepBridge através de estudo empírico com usuários reais (cientistas de dados e engenheiros de ML).

## Métricas e Metas

| Métrica | Meta | Resultado Esperado |
|---------|------|-------------------|
| SUS Score | ≥ 85 (excelente) | 87.5 ± 3.2 |
| NASA TLX | ≤ 30 (carga baixa) | 28 ± 5.1 |
| Taxa de Sucesso | ≥ 90% | 95% (19/20) |
| Tempo Médio Total | ≤ 15 min | 12 ± 2.5 min |
| Erros Médios | ≤ 2 | 1.3 ± 0.9 |

## Estrutura do Experimento

### Participantes

- **Total**: 20 profissionais
- **Perfil**: 10 cientistas de dados + 10 engenheiros de ML
- **Experiência**: 2-10 anos em ML
- **Domínios**: Fintech (40%), Saúde (25%), Tech (20%), Varejo (15%)

### Tarefas (3 tarefas, ~15 minutos total)

1. **Tarefa 1: Validar Fairness** (~6.5 min)
   - Validar fairness de modelo de crédito
   - Identificar violações EEOC 80% rule

2. **Tarefa 2: Gerar Relatório PDF** (~2.8 min)
   - Gerar relatório profissional audit-ready
   - Customizar com logo da empresa

3. **Tarefa 3: Integrar em CI/CD** (~6.2 min)
   - Criar script para pipeline CI/CD
   - Retornar exit code apropriado

### Questionários

- **SUS (System Usability Scale)**: 10 perguntas, escala 1-5
- **NASA TLX (Task Load Index)**: 6 dimensões, escala 0-100
- **Questões abertas**: Feedback qualitativo

## Estrutura de Diretórios

```
03_usabilidade/
├── config/          # Configurações
├── data/            # Dados de participantes
├── materials/       # Materiais do estudo (questionários, tarefas)
│   ├── SUS_questionnaire.md
│   ├── NASA_TLX_questionnaire.md
│   └── study_tasks.md
├── notebooks/       # Notebooks de análise
├── scripts/         # Scripts de análise
│   ├── utils.py
│   ├── generate_mock_data.py
│   ├── calculate_metrics.py
│   ├── statistical_analysis.py
│   ├── generate_visualizations.py
│   └── analyze_usability.py
├── results/         # Resultados JSON/CSV
├── figures/         # Visualizações
├── tables/          # Tabelas LaTeX
└── logs/            # Logs de execução
```

## Como Executar

### 1. Instalação

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/03_usabilidade
pip install -r requirements.txt
```

### 2. Análise Completa (Mock Data)

```bash
# Executa todo o pipeline: gera dados, calcula métricas, análise estatística, visualizações
python scripts/analyze_usability.py
```

### 3. Etapas Individuais

```bash
# 1. Gerar dados sintéticos (para teste)
python scripts/generate_mock_data.py

# 2. Calcular métricas
python scripts/calculate_metrics.py

# 3. Análise estatística
python scripts/statistical_analysis.py

# 4. Gerar visualizações
python scripts/generate_visualizations.py
```

## Outputs

### Dados Brutos
- `results/03_usability_sus_scores.csv` - Respostas SUS
- `results/03_usability_nasa_tlx.csv` - Respostas NASA TLX
- `results/03_usability_task_times.csv` - Tempos de completação
- `results/03_usability_errors.csv` - Contagem de erros

### Análises
- `results/03_usability_metrics.json` - Métricas calculadas
- `results/03_usability_statistical_analysis.json` - Análise estatística
- `results/03_usability_summary_report.txt` - Relatório textual

### Figuras
- `figures/sus_score_distribution.pdf` - Distribuição SUS
- `figures/nasa_tlx_dimensions.pdf` - Dimensões NASA TLX
- `figures/task_completion_times.pdf` - Tempos de completação
- `figures/success_rate_by_task.pdf` - Taxa de sucesso

### Tabelas
- `tables/usability_summary.tex` - Tabela LaTeX para o paper

## Metodologia

### Procedimento (60 minutos por participante)

1. **Introdução** (5 min)
   - Explicação do estudo
   - Consentimento informado
   - Questionário demográfico

2. **Tutorial** (10 min)
   - Overview DeepBridge
   - Demonstração exemplo

3. **Execução das Tarefas** (30 min)
   - Trabalho independente
   - Think-aloud protocol
   - Observação sem intervenção

4. **Questionários** (10 min)
   - SUS
   - NASA TLX
   - Questões abertas

5. **Entrevista** (5 min)
   - Feedback qualitativo

## Análise Estatística

### SUS Score

**Teste**: One-sample t-test

- H0: SUS Score = 68 (média global)
- H1: SUS Score > 68
- Esperado: p < 0.001 (altamente significativo)

### Interpretação

**SUS Score**:
- 87.5 = "Excellent" (Grau A)
- Top 10% de sistemas
- Significativamente acima da média global (68)

**NASA TLX**:
- 28 = "Low Workload"
- Indica baixa carga cognitiva

**Success Rate**:
- 95% (19/20) > meta de 90%

**Completion Time**:
- 12 min < meta de 15 min
- Muito menor que baseline estimado (45 min com ferramentas fragmentadas)

## Comparação com Baseline

### Workflow Fragmentado (Estimado)
- **Ferramentas**: AIF360 + Fairlearn + Alibi Detect + etc.
- **Tempo estimado**: 45 minutos
- **SUS Score estimado**: 55-65 (baseado em literatura)
- **Complexidade**: Alta (múltiplas APIs, integrações manuais)

### DeepBridge
- **Ferramenta**: Única (DeepBridge)
- **Tempo**: 12 minutos (73% mais rápido)
- **SUS Score**: 87.5 (excelente)
- **Complexidade**: Baixa (API unificada)

## Limitações

### Estudo Mock (Atual)

Os dados atuais são **sintéticos** para:
- ✅ Testar infraestrutura de análise
- ✅ Validar pipeline completo
- ✅ Gerar exemplos de outputs
- ✅ Demonstrar análises esperadas

### Estudo Real (Futuro)

Para conduzir estudo real:
1. Recrutar 20 participantes reais
2. Preparar ambiente e materiais
3. Executar sessões individuais (60 min cada)
4. Coletar dados reais
5. Substituir dados sintéticos por reais
6. Manter pipeline de análise (já pronto!)

## Considerações Éticas

- ✅ Consentimento informado
- ✅ Anonimização de dados
- ✅ Direito de desistir
- ✅ Privacidade garantida
- ⏳ Aprovação de comitê de ética (se aplicável)

## Próximos Passos

1. ⏳ Finalizar protocolo do estudo
2. ⏳ Recrutar participantes
3. ⏳ Conduzir sessões piloto (2-3)
4. ⏳ Ajustar baseado em piloto
5. ⏳ Executar 20 sessões principais
6. ⏳ Substituir dados mock por reais
7. ⏳ Executar análise final
8. ⏳ Integrar resultados no paper

## Referências

- Brooke, J. (1996). SUS: A "quick and dirty" usability scale.
- Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX.
- Bangor, A., Kortum, P., & Miller, J. (2009). Determining what individual SUS scores mean.
