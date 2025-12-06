# Experimento 3: Estudo de Usabilidade

## Objetivo

Comprovar as afirmações sobre usabilidade do DeepBridge através de estudo empírico com usuários reais (cientistas de dados e engenheiros de ML).

## Afirmações a Comprovar

| Métrica | Valor Afirmado | Status |
|---------|----------------|--------|
| SUS Score | 87.5 (top 10%, "excelente") | ⏳ Pendente |
| Taxa de Sucesso | 95% (19/20 completaram) | ⏳ Pendente |
| Tempo para Primeira Validação | 12 min (vs. 45 min estimado) | ⏳ Pendente |
| NASA TLX (carga cognitiva) | 28/100 (baixa) | ⏳ Pendente |
| Adoção em Produção | 6 organizações, 3 domínios | ⏳ Pendente |

## Metodologia

### 1. Participantes

**Total**: 20 profissionais

**Perfil**:
- 10 cientistas de dados
- 10 engenheiros de ML

**Experiência**: 2-10 anos em ML

**Distribuição por Domínio**:
- Fintech: 8 participantes (40%)
- Saúde: 5 participantes (25%)
- Tech: 4 participantes (20%)
- Varejo: 3 participantes (15%)

**Critérios de Inclusão**:
- Experiência mínima de 2 anos com Python
- Familiaridade com scikit-learn
- Experiência com deployment de modelos ML
- Não ter usado DeepBridge anteriormente

**Critérios de Exclusão**:
- Desenvolvedores do DeepBridge
- Pessoas com conflito de interesse

### 2. Tarefas

Cada participante deve completar 3 tarefas:

#### Tarefa 1: Validar Fairness de Modelo
**Descrição**: Dado um modelo de crédito treinado e um dataset de teste, validar fairness em relação a gênero e raça.

**Entregável**:
- Executar 15 métricas de fairness
- Identificar violações da regra 80% EEOC
- Interpretar resultados

**Tempo Estimado**: 5-8 minutos

**Critérios de Sucesso**:
- Conseguiu criar DBDataset
- Executou experimento de fairness
- Identificou corretamente as violações

#### Tarefa 2: Gerar Relatório PDF Audit-Ready
**Descrição**: Gerar relatório PDF completo com todos os resultados de validação.

**Entregável**:
- PDF profissional com visualizações
- Incluir fairness, robustez, incerteza
- Personalizar com logo da empresa (fornecido)

**Tempo Estimado**: 2-4 minutos

**Critérios de Sucesso**:
- Relatório gerado com sucesso
- Inclui todas as seções esperadas
- Customização aplicada corretamente

#### Tarefa 3: Integrar Validação em Pipeline CI/CD
**Descrição**: Criar script Python que integra validação DeepBridge em pipeline de CI/CD.

**Entregável**:
- Script que executa validação automaticamente
- Retorna exit code apropriado se violações detectadas
- Salva relatório em diretório específico

**Tempo Estimado**: 5-8 minutos

**Critérios de Sucesso**:
- Script funciona corretamente
- Detecta violações e retorna exit code != 0
- Relatório salvo no local correto

### 3. Procedimento

#### Preparação (Antes da Sessão)
1. **Enviar Material Antecipadamente**:
   - Instruções de instalação do DeepBridge
   - Link para documentação
   - Descrição do estudo e consentimento informado

2. **Setup do Ambiente**:
   - Ambiente virtual Python com DeepBridge instalado
   - Datasets e modelos pré-carregados
   - Jupyter notebook com células template

#### Durante a Sessão (60 minutos)
1. **Introdução (5 min)**:
   - Explicação do estudo
   - Consentimento informado
   - Questionário demográfico

2. **Tutorial (10 min)**:
   - Overview rápido do DeepBridge (5 min)
   - Demonstração de exemplo simples (5 min)

3. **Execução das Tarefas (30 min)**:
   - Participante trabalha de forma independente
   - Observador toma notas (sem intervenção)
   - Think-aloud protocol (participante verbaliza pensamento)

4. **Questionários (10 min)**:
   - SUS (System Usability Scale)
   - NASA TLX (Task Load Index)
   - Questões abertas sobre experiência

5. **Entrevista Semi-Estruturada (5 min)**:
   - Pontos positivos
   - Pontos negativos
   - Sugestões de melhoria

#### Pós-Sessão
- Análise das gravações (se consentido)
- Compilação de métricas
- Análise qualitativa de feedback

### 4. Instrumentos de Medição

#### System Usability Scale (SUS)

Questionário de 10 itens (escala Likert 1-5):

1. Acho que gostaria de usar este sistema frequentemente
2. Achei o sistema desnecessariamente complexo
3. Achei o sistema fácil de usar
4. Acho que precisaria de suporte técnico para usar este sistema
5. Achei que as várias funções neste sistema estavam bem integradas
6. Achei que havia muita inconsistência neste sistema
7. Imagino que a maioria das pessoas aprenderia a usar este sistema rapidamente
8. Achei o sistema muito complicado de usar
9. Senti-me muito confiante usando o sistema
10. Precisei aprender muitas coisas antes de começar a usar este sistema

**Cálculo**:
```python
def calculate_sus_score(responses):
    # responses: lista de 10 respostas (1-5)
    # Itens ímpares (1,3,5,7,9): contribuição = resposta - 1
    # Itens pares (2,4,6,8,10): contribuição = 5 - resposta

    score = 0
    for i, response in enumerate(responses):
        if i % 2 == 0:  # ímpar (0-indexed)
            score += (response - 1)
        else:  # par
            score += (5 - response)

    return score * 2.5  # Escala 0-100
```

**Interpretação**:
- < 50: Abaixo da média (poor)
- 50-70: Média (ok)
- 70-85: Boa (good)
- 85-90: Excelente (excellent) - **Top 10%**
- > 90: Melhor imaginável (best imaginable)

**Meta**: SUS Score ≥ 85 (excelente)

#### NASA Task Load Index (TLX)

Avalia carga cognitiva em 6 dimensões (escala 0-100):

1. **Mental Demand**: Quão mentalmente exigente foi a tarefa?
2. **Physical Demand**: Quão fisicamente exigente foi a tarefa?
3. **Temporal Demand**: Quão apressado você se sentiu?
4. **Performance**: Quão bem sucedido você acha que foi?
5. **Effort**: Quanto esforço foi necessário?
6. **Frustration**: Quão frustrado você se sentiu?

**Cálculo**:
```python
def calculate_nasa_tlx(dimensions):
    # dimensions: dict com 6 valores (0-100)
    return sum(dimensions.values()) / 6
```

**Interpretação**:
- < 20: Carga muito baixa
- 20-40: Carga baixa
- 40-60: Carga moderada
- 60-80: Carga alta
- > 80: Carga muito alta

**Meta**: NASA TLX ≤ 30 (carga baixa)

### 5. Métricas Objetivas

#### Taxa de Sucesso
```python
success_rate = (participantes_que_completaram_todas_tarefas / total_participantes) * 100
```
**Meta**: ≥ 90%

#### Tempo para Completar
```python
# Por tarefa
time_task_1 = [tempo_participante_1, ..., tempo_participante_20]
time_task_2 = [...]
time_task_3 = [...]

# Total
time_total = [sum([t1, t2, t3]) for t1, t2, t3 in zip(time_task_1, time_task_2, time_task_3)]
```

**Meta**: Tempo médio ≤ 15 minutos (vs. 45 min estimado com ferramentas fragmentadas)

#### Erros Cometidos
Categorias:
- Erro de sintaxe Python
- Erro de API (uso incorreto do DeepBridge)
- Erro conceitual (interpretação incorreta de métrica)
- Outro

**Meta**: Média ≤ 2 erros por participante

## Resultados Esperados

### Quantitativos

| Métrica | Meta | Resultado Esperado |
|---------|------|-------------------|
| SUS Score | ≥ 85 | 87.5 ± 3.2 |
| NASA TLX | ≤ 30 | 28 ± 5.1 |
| Taxa de Sucesso | ≥ 90% | 95% (19/20) |
| Tempo Médio Total | ≤ 15 min | 12 ± 2.5 min |
| Tempo Tarefa 1 | ≤ 8 min | 6.5 ± 1.2 min |
| Tempo Tarefa 2 | ≤ 4 min | 2.8 ± 0.8 min |
| Tempo Tarefa 3 | ≤ 8 min | 6.2 ± 1.5 min |
| Erros Médios | ≤ 2 | 1.3 ± 0.9 |

### Qualitativos

**Feedback Positivo Esperado** (% de participantes):
- "API intuitiva, similar ao scikit-learn": 75% (15/20)
- "Relatórios profissionais sem esforço": 90% (18/20)
- "Conformidade automática é revolucionária": 60% (12/20)
- "Documentação clara e completa": 70% (14/20)
- "Fácil de integrar no workflow existente": 65% (13/20)

**Feedback Negativo Esperado** (% de participantes):
- "Instalação inicial lenta (muitas dependências)": 40% (8/20)
- "Desejo mais templates de relatório": 25% (5/20)
- "Tempo de execução poderia ser mais rápido": 15% (3/20)
- "Alguns erros difíceis de debugar": 10% (2/20)

## Análise Estatística

### SUS Score

**Teste**: One-sample t-test
**H0**: SUS Score = 68 (média global para ferramentas de software)
**H1**: SUS Score > 68

```python
from scipy import stats

sus_scores = [87, 89, 85, ...]  # 20 scores
t_stat, p_value = stats.ttest_1samp(sus_scores, 68, alternative='greater')

# Esperado: p < 0.001
```

### Comparação com Baseline

**Baseline**: Workflow fragmentado (AIF360 + Fairlearn + etc.)
- Tempo estimado: 45 minutos
- SUS Score estimado: 55-65 (baseado em literatura)

**Teste**: Independent t-test
```python
# DeepBridge vs. Baseline (se tivermos grupo controle)
t_stat, p_value = stats.ttest_ind(times_deepbridge, times_baseline)
```

## Recrutamento

### Estratégias
1. **LinkedIn**: Postagem em grupos de Data Science/ML
2. **Meetups**: Apresentação em meetups locais
3. **Empresas Parceiras**: Solicitar indicações
4. **Plataformas**: User Testing, UserBrain

### Incentivos
- Compensação: $50-100 por participante (60 min)
- Certificado de participação
- Early access ao DeepBridge
- Relatório de resultados do estudo

## Considerações Éticas

### Consentimento Informado
- Explicação clara do propósito do estudo
- Direito de desistir a qualquer momento
- Anonimização dos dados
- Uso dos dados apenas para pesquisa

### Privacidade
- Dados pessoais anonimizados
- Gravações (se houver) armazenadas de forma segura
- Destruição após análise (se consentido)

### Aprovação
- Submeter protocolo para comitê de ética (se instituição acadêmica)
- Obter consentimento por escrito

## Outputs

### Dados Brutos
- `results/03_usability_sus_scores.csv`
- `results/03_usability_nasa_tlx.csv`
- `results/03_usability_task_times.csv`
- `results/03_usability_errors.csv`
- `results/03_usability_feedback.json`

### Análise
- `results/03_usability_statistical_analysis.json`
- `notebooks/03_usability_analysis.ipynb`

### Figuras
- `figures/sus_score_distribution.pdf`
- `figures/nasa_tlx_dimensions.pdf`
- `figures/task_completion_times.pdf`
- `figures/success_rate_by_task.pdf`

### Tabelas
- `tables/usability_summary.tex`

## Cronograma

**Total: 3-4 semanas**

### Semana 1: Preparação
- Finalizar protocolo
- Criar materiais (tutorial, tarefas, questionários)
- Recrutar participantes
- Obter aprovação ética (se necessário)

### Semana 2-3: Coleta de Dados
- Executar sessões com 20 participantes
- 2-3 sessões por dia
- Transcrever notas e feedbacks

### Semana 4: Análise
- Calcular métricas quantitativas
- Análise qualitativa de feedback
- Análise estatística
- Gerar visualizações e tabelas

## Checklist

- [ ] Finalizar protocolo do estudo
- [ ] Criar tutorial e materiais de treinamento
- [ ] Preparar tarefas e datasets
- [ ] Preparar questionários (SUS, NASA TLX)
- [ ] Criar roteiro de entrevista semi-estruturada
- [ ] Recrutar 20 participantes
- [ ] Obter consentimento informado
- [ ] Executar sessões piloto (2-3)
- [ ] Ajustar protocolo baseado em piloto
- [ ] Executar 20 sessões principais
- [ ] Transcrever notas e feedbacks
- [ ] Calcular SUS scores
- [ ] Calcular NASA TLX scores
- [ ] Analisar tempos de completação
- [ ] Análise qualitativa de feedback
- [ ] Análise estatística
- [ ] Gerar visualizações
- [ ] Formatar tabelas LaTeX
- [ ] Documentar metodologia completa

## Prioridade

🟡 **MÉDIA** - Importante para demonstrar usabilidade, mas não crítico para funcionalidade

## Tempo Estimado

**3-4 semanas**

## Referências

- Brooke, J. (1996). SUS: A "quick and dirty" usability scale.
- Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX.
- Bangor, A., Kortum, P., & Miller, J. (2009). Determining what individual SUS scores mean: Adding an adjective rating scale.
