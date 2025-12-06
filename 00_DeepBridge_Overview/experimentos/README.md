# Experimentos para Comprovação do Paper DeepBridge

Este diretório contém a documentação de todos os experimentos necessários para comprovar as afirmações e números apresentados no paper "DeepBridge: Um Framework Unificado e Pronto para Produção para Validação Multi-Dimensional de Machine Learning".

## Visão Geral

O paper apresenta resultados quantificados em 4 dimensões principais:
1. **Economia de Tempo**: Redução de 81-89% no tempo de validação
2. **Economia de Custo**: Redução de 10× via HPM-KD
3. **Conformidade Regulatória**: 100% de precisão na detecção de violações
4. **Usabilidade**: SUS Score 87.5 (top 10%)

## Estrutura dos Experimentos

### 1. Benchmarks de Tempo
**Arquivo**: `01_benchmarks_tempo.md`
- Comparação DeepBridge vs. ferramentas fragmentadas
- Medição de tempo por dimensão de validação
- Total: 17 min vs. 150 min (89% redução)

### 2. Estudos de Caso
**Arquivo**: `02_estudos_de_caso.md`
- 6 domínios: Crédito, Contratação, Saúde, Hipoteca, Seguros, Fraude
- Tempo médio: 27.7 minutos
- Detecção de violações de conformidade

### 3. Estudo de Usabilidade
**Arquivo**: `03_usabilidade.md`
- 20 participantes (cientistas de dados + engenheiros ML)
- SUS Score: 87.5
- Taxa de sucesso: 95% (19/20)
- NASA TLX: 28/100

### 4. HPM-KD Framework
**Arquivo**: `04_hpmkd.md`
- 20 datasets UCI/OpenML
- Retenção de acurácia: 98.4%
- Compressão: 10.3×
- Speedup de latência: 10.4×

### 5. Conformidade Regulatória
**Arquivo**: `05_conformidade.md`
- Precisão de detecção: 100%
- Falsos positivos: 0
- Cobertura de features: 10/10 vs. 2/10 (ferramentas existentes)

### 6. Estudos de Ablação
**Arquivo**: `06_ablation_studies.md`
- Contribuição da API unificada: 50%
- Contribuição da paralelização: 30%
- Contribuição do caching: 10%
- Contribuição da automação de relatórios: 10%

## Status dos Experimentos

| Experimento | Status | Prioridade | Tempo Estimado |
|-------------|--------|------------|----------------|
| Benchmarks de Tempo | ⏳ Pendente | 🔴 Alta | 2-3 semanas |
| Estudos de Caso | ⏳ Pendente | 🔴 Alta | 4-6 semanas |
| Usabilidade | ⏳ Pendente | 🟡 Média | 3-4 semanas |
| HPM-KD | ⏳ Pendente | 🔴 Alta | 3-4 semanas |
| Conformidade | ⏳ Pendente | 🟡 Média | 1-2 semanas |
| Ablation Studies | ⏳ Pendente | 🟢 Baixa | 1-2 semanas |

## Recursos Necessários

### Datasets
- UCI Machine Learning Repository (20 datasets tabulares)
- OpenML (datasets complementares)
- Datasets sintéticos para casos de uso (crédito, contratação, saúde, etc.)

### Infraestrutura
- Máquina com GPU (para HPM-KD)
- CPU multi-core (para testes de paralelização)
- Mínimo 32GB RAM

### Ferramentas de Comparação
- AIF360 (fairness)
- Fairlearn (fairness)
- Alibi Detect (robustness)
- UQ360 (uncertainty)
- Evidently AI (drift detection)

### Participantes (Usabilidade)
- 10 cientistas de dados
- 10 engenheiros de ML
- Experiência: 2-10 anos
- Domínios: fintech, saúde, tech, varejo

## Cronograma Sugerido

### Fase 1: Experimentos Técnicos (8-10 semanas)
1. Semanas 1-3: Benchmarks de Tempo
2. Semanas 3-6: HPM-KD
3. Semanas 6-10: Estudos de Caso

### Fase 2: Experimentos com Usuários (4 semanas)
4. Semanas 11-14: Estudo de Usabilidade

### Fase 3: Validação Final (2 semanas)
5. Semanas 15-16: Conformidade e Ablation Studies

## Entregáveis

Para cada experimento:
- [ ] Script Python reproduzível
- [ ] Datasets utilizados (ou instruções para obtenção)
- [ ] Resultados brutos (CSV/JSON)
- [ ] Análise estatística (notebook Jupyter)
- [ ] Visualizações (figuras para o paper)
- [ ] Tabelas formatadas em LaTeX
- [ ] Documentação de metodologia

## Notas Importantes

1. **Reprodutibilidade**: Todos os experimentos devem incluir seeds fixas e instruções detalhadas
2. **Significância Estatística**: Usar testes apropriados (t-test, ANOVA, etc.) com p < 0.05
3. **Múltiplas Execuções**: Cada experimento deve ter mínimo 5 runs para calcular médias e desvio padrão
4. **Documentação**: Registrar versões de todas as bibliotecas usadas

## Referências

- Paper: `/home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/POR/V2/main.pdf`
- Código DeepBridge: (adicionar path quando disponível)
