# Prompts para Pesquisa de Trabalhos Relacionados
# HPM-KD e DiXtill Frameworks

**Data de Criação**: 2025-12-07
**Objetivo**: Verificar se já existem trabalhos similares publicados antes de submeter os papers
**Frameworks**: HPM-KD (Paper 1) e DiXtill (Paper 16)

---

## 📋 Instruções de Uso

1. Use cada prompt em ferramentas de IA como ChatGPT, Claude, Perplexity, ou Gemini
2. Também pesquise em bases acadêmicas: Google Scholar, arXiv, Semantic Scholar, ACM Digital Library, IEEE Xplore
3. Anote os resultados encontrados em cada seção
4. Marque com ✅ ou ❌ se encontrou trabalhos muito similares
5. Se encontrar trabalhos similares, anote diferenças e contribuições únicas dos nossos frameworks

---

## 🔍 PARTE 1: Pesquisa sobre HPM-KD Framework

### 1.1 Pesquisa Geral do Framework Completo

```
Existe algum framework de knowledge distillation que combine TODOS os seguintes componentes:
1. Seleção automática de configuração via meta-aprendizado (adaptive configuration manager)
2. Destilação progressiva hierárquica com múltiplos teachers em cadeia
3. Multi-teacher ensemble com pesos de atenção aprendidos (attention-weighted)
4. Agendamento adaptativo de temperatura (meta-temperature scheduler)
5. Pipeline de processamento paralelo com cache inteligente
6. Memória compartilhada de otimização entre experimentos

Procuro especificamente trabalhos que integrem pelo menos 4-5 desses componentes simultaneamente, não apenas um ou dois isoladamente.
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos parcialmente similares:
  -
  -
  -

---

### 1.2 Adaptive Configuration Manager

```
Existem trabalhos sobre knowledge distillation que usam meta-aprendizado para selecionar automaticamente:
- Tipo de modelo student
- Temperatura de distillation
- Valores de alpha (peso entre cross-entropy e KD loss)
- Arquitetura do student

Especificamente procuro por "adaptive configuration", "automated hyperparameter selection for distillation", "meta-learning for knowledge distillation configuration", ou "AutoML for knowledge distillation".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 1.3 Progressive Multi-Teacher Distillation

```
Existem frameworks de knowledge distillation que implementam:
1. Cadeia progressiva de destilação (progressive chain) onde:
   - Teacher 1 → Student 1 (intermediate)
   - Student 1 → Student 2 (smaller)
   - Student 2 → Student 3 (final compact model)
2. Com rastreamento de melhoria mínima (minimal improvement tracking)
3. Refinamento incremental hierárquico

Termos de busca: "progressive knowledge distillation", "hierarchical multi-teacher distillation", "cascaded distillation", "incremental knowledge transfer", "multi-step distillation chain".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 1.4 Attention-Weighted Multi-Teacher Ensemble

```
Existem trabalhos sobre multi-teacher knowledge distillation que:
1. Usam múltiplos teachers simultaneamente (ensemble)
2. Aprendem pesos de atenção para cada teacher (não pesos fixos/uniformes)
3. Esses pesos de atenção são aprendidos durante o treinamento (learned attention)
4. Combinam soft targets ponderados por importância de cada teacher

Procuro por "attention-weighted multi-teacher", "learned teacher weighting", "adaptive teacher ensemble", "dynamic teacher selection in KD".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 1.5 Meta-Temperature Scheduler

```
Existem trabalhos que propõem agendamento adaptativo de temperatura (temperature scheduling) em knowledge distillation onde:
1. A temperatura não é fixa durante todo o treinamento
2. A temperatura é ajustada dinamicamente baseada em:
   - Performance no validation set
   - Divergência entre teacher e student
   - Fase do treinamento (early/mid/late)
3. Usa meta-aprendizado para determinar o schedule ótimo

Termos: "adaptive temperature scheduling", "dynamic temperature in knowledge distillation", "meta-learning temperature", "temperature annealing in KD".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 1.6 Parallel Processing e Shared Optimization Memory

```
Existem frameworks de knowledge distillation que implementam:
1. Processamento paralelo de múltiplas configurações de distillation
2. Cache inteligente de:
   - Soft targets do teacher (para evitar recomputação)
   - Embeddings intermediários
   - Predictions
3. Memória compartilhada de otimização entre experimentos:
   - Reutiliza aprendizado de experimentos anteriores
   - Warm-start de configurações similares

Procuro por "parallel knowledge distillation", "cached distillation", "shared memory optimization", "transfer learning across distillation experiments".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 1.7 Nome "HPM-KD" ou Similar

```
Existe algum framework chamado "HPM-KD", "Hierarchical Progressive Multi-Teacher", ou siglas muito similares como:
- HPKD (Hierarchical Progressive Knowledge Distillation)
- MPKD (Multi-Progressive Knowledge Distillation)
- HTPD (Hierarchical Teacher-Progressive Distillation)

Ou trabalhos que usem exatamente a combinação "Hierarchical + Progressive + Multi-Teacher" no título?
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos com nomes similares:
  -
  -

---

## 🔍 PARTE 2: Pesquisa sobre DiXtill Framework

### 2.1 Pesquisa Geral do Framework DiXtill

```
Existe algum framework de knowledge distillation que adiciona um termo de alinhamento de explicabilidade (explanation alignment) na função de perda, especificamente:

L = (1-α)L_CE + α(L_KD + β·L_XAI)

Onde L_XAI alinha explicações (SHAP values, attention weights, ou gradientes) entre teacher e student DURANTE o treinamento (não post-hoc)?

Procuro por trabalhos que transfiram não apenas predictions, mas o PROCESSO DE RACIOCÍNIO (reasoning) do teacher para o student.
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 2.2 SHAP Alignment em Knowledge Distillation

```
Existem trabalhos que usam SHAP values (Shapley Additive Explanations) em knowledge distillation para:
1. Calcular SHAP values do teacher e student
2. Minimizar a distância entre esses SHAP values: ||SHAP_teacher - SHAP_student||²
3. Garantir que feature importances sejam preservadas após distillation

Termos: "SHAP alignment", "Shapley values in distillation", "feature attribution transfer", "explanation-aware distillation".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 2.3 Attention Alignment em Transformers

```
Existem trabalhos sobre knowledge distillation de transformers que:
1. Alinham attention weights entre teacher e student: ||A_teacher - A_student||²
2. Fazem esse alinhamento layer-by-layer
3. Tratam casos onde teacher e student têm diferentes números de layers/heads
4. Usam estratégias de mapeamento (uniform, last-N, skip)

Procuro por "attention transfer", "attention distillation", "attention alignment", especificamente para BERT/transformers.
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 2.4 Gradient Alignment / Saliency Map Alignment

```
Existem frameworks de distillation que alinham gradientes de entrada (input gradients) ou saliency maps entre teacher e student:

L_XAI = ||∇_x log p_teacher - ∇_x log p_student||²

Onde os gradientes indicam quais pixels/features são importantes para a decisão?

Termos: "gradient matching", "saliency alignment", "input gradient distillation", "gradient-based knowledge transfer".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 2.5 XAI-Driven ou Explainability-Driven Distillation

```
Existem trabalhos sobre "explainability-driven distillation", "XAI-guided knowledge distillation", ou "interpretable knowledge distillation" que:
1. Usam técnicas de XAI (SHAP, LIME, CAM, attention, gradients) DURANTE o treinamento
2. Garantem que student seja interpretável-by-design (não apenas post-hoc)
3. Preservam consistência de explicações entre teacher e student

Procuro por trabalhos que combinem XAI + KD de forma integrada.
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 2.6 Reasoning Transfer (Transferência de Raciocínio)

```
Existem trabalhos sobre knowledge distillation que focam em transferir o RACIOCÍNIO (reasoning process) do teacher, não apenas as predictions?

Especificamente trabalhos que argumentam:
- KD tradicional transfere "o que prever"
- Nosso método transfere "por que prever"
- Student aprende o processo de decisão, não só o resultado final

Termos: "reasoning transfer", "decision process transfer", "rationale distillation", "interpretable student models".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

### 2.7 Nome "DiXtill" ou Similar

```
Existe algum framework chamado "DiXtill", "XAI-Driven Distillation", ou nomes muito similares como:
- XAI-KD (XAI Knowledge Distillation)
- Explainable KD
- Interpretable Distillation Framework
- SHAP-Distill

Ou trabalhos que combinem explicitamente "XAI" + "Distillation" no título?
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos com nomes similares:
  -
  -

---

### 2.8 DiXtill Específico para Ambientes Regulados

```
Existem trabalhos sobre knowledge distillation focados em ambientes regulados (finanças, saúde, contratação) onde:
1. Explicabilidade é mandatória por regulação (GDPR, ECOA, EEOC)
2. Student precisa ser interpretável para compliance
3. Explicações do student são consistentes com o teacher (auditabilidade)

Termos: "regulatory-compliant distillation", "explainable distillation for finance", "interpretable student models for healthcare", "GDPR-compliant model compression".
```

**Resultados**:
- [ ] Nenhum trabalho encontrado
- [ ] Trabalhos relacionados:
  -
  -

---

## 🔍 PARTE 3: Trabalhos Relacionados Gerais

### 3.1 Survey Papers sobre Knowledge Distillation (últimos 3 anos)

```
Quais são os survey papers mais recentes (2022-2025) sobre knowledge distillation?
Procuro por:
- "A Survey on Knowledge Distillation"
- "Recent Advances in Knowledge Distillation"
- Reviews abrangentes que cubram multi-teacher, progressive, attention-based methods

Liste os 5-10 surveys mais citados e recentes.
```

**Resultados**:
-
-
-

---

### 3.2 State-of-the-Art em Knowledge Distillation (2024-2025)

```
Quais são os métodos estado-da-arte em knowledge distillation publicados em 2024-2025 nas conferências:
- NeurIPS 2024
- ICML 2024
- ICLR 2024/2025
- AAAI 2024/2025

Liste os papers aceitos nessas conferências relacionados a KD.
```

**Resultados**:
-
-
-

---

### 3.3 Multi-Teacher Distillation (Estado da Arte)

```
Quais são os trabalhos mais citados sobre multi-teacher knowledge distillation?
Especificamente procuro por:
- Deep Mutual Learning (Zhang et al., 2018)
- TAKD - Teacher Assistant Knowledge Distillation (Mirzadeh et al., 2020)
- Online Knowledge Distillation
- Collaborative Learning

Liste os 5-10 papers fundamentais nessa área.
```

**Resultados**:
-
-
-

---

### 3.4 Attention Transfer e Feature-Based Distillation

```
Quais são os trabalhos clássicos sobre:
1. Attention Transfer (Zagoruyko & Komodakis, 2017)
2. FitNets (Romero et al., 2015)
3. Feature-based Knowledge Distillation
4. Intermediate layer matching

Liste os papers fundamentais que fazem matching de representações internas (não apenas soft targets).
```

**Resultados**:
-
-
-

---

### 3.5 XAI Methods (SHAP, LIME, Integrated Gradients)

```
Quais são os papers fundamentais sobre técnicas de XAI que usamos:
1. SHAP (Lundberg & Lee, 2017)
2. LIME (Ribeiro et al., 2016)
3. Integrated Gradients (Sundararajan et al., 2017)
4. Attention Mechanisms for Interpretability
5. Saliency Maps / CAM (Class Activation Mapping)

Liste as referências principais que devemos citar.
```

**Resultados**:
-
-
-

---

## 🔍 PARTE 4: Análise de Novidade e Diferenciação

### 4.1 Análise Comparativa HPM-KD

```
Dado que HPM-KD combina:
- Adaptive configuration (meta-learning)
- Progressive multi-teacher chain
- Attention-weighted ensemble
- Meta-temperature scheduling
- Parallel processing + shared memory

Qual trabalho existente é o MAIS SIMILAR ao HPM-KD?
Liste diferenças específicas e contribuições únicas do HPM-KD.
```

**Análise**:
- Trabalho mais similar:
- Diferenças principais:
  1.
  2.
  3.
- Contribuições únicas do HPM-KD:
  1.
  2.
  3.

---

### 4.2 Análise Comparativa DiXtill

```
Dado que DiXtill:
- Adiciona L_XAI (explanation alignment) na loss function
- Transfere reasoning, não apenas predictions
- Suporta SHAP, Attention, Gradient alignment
- Foca em interpretabilidade-by-design (não post-hoc)

Qual trabalho existente é o MAIS SIMILAR ao DiXtill?
Liste diferenças específicas e contribuições únicas do DiXtill.
```

**Análise**:
- Trabalho mais similar:
- Diferenças principais:
  1.
  2.
  3.
- Contribuições únicas do DiXtill:
  1.
  2.
  3.

---

## 📚 PARTE 5: Bases de Dados para Pesquisa

### Ferramentas de IA Recomendadas

1. **Perplexity AI** (https://www.perplexity.ai/)
   - Melhor para pesquisas acadêmicas com citações
   - Use modo "Academic"

2. **ChatGPT** (https://chat.openai.com/)
   - Use GPT-4 com web browsing ativado
   - Peça para citar fontes específicas

3. **Claude** (https://claude.ai/)
   - Bom para análise comparativa detalhada
   - Peça para comparar trabalhos

4. **Gemini** (https://gemini.google.com/)
   - Integrado com Google Scholar
   - Bom para encontrar papers recentes

5. **Consensus** (https://consensus.app/)
   - Ferramenta especializada em pesquisa acadêmica
   - Busca direta em papers científicos

---

### Bases Acadêmicas para Verificação Manual

1. **Google Scholar** (https://scholar.google.com/)
   - Pesquisa mais abrangente
   - Mostra citações

2. **arXiv** (https://arxiv.org/)
   - Papers em Machine Learning (cs.LG)
   - Pré-prints antes de conferências

3. **Semantic Scholar** (https://www.semanticscholar.org/)
   - IA para encontrar papers relacionados
   - Gráfico de citações

4. **ACM Digital Library** (https://dl.acm.org/)
   - Papers de conferências ACM (FAccT, KDD)

5. **IEEE Xplore** (https://ieeexplore.ieee.org/)
   - Papers de conferências IEEE

6. **Papers With Code** (https://paperswithcode.com/)
   - Papers com código disponível
   - Benchmarks e leaderboards

---

## ✅ Checklist de Validação de Novidade

Antes de submeter os papers, verifique:

### Para HPM-KD:
- [ ] Nenhum trabalho combina TODOS os 6 componentes do HPM-KD
- [ ] Adaptive configuration via meta-learning é original ou tem diferenças claras
- [ ] Progressive multi-teacher chain tem contribuição única
- [ ] Attention-weighted ensemble é diferente de métodos existentes
- [ ] Shared optimization memory não existe em outros frameworks
- [ ] Nome "HPM-KD" não está em uso

### Para DiXtill:
- [ ] L_XAI (explanation alignment loss) é original
- [ ] SHAP alignment em KD não existe ou é muito diferente
- [ ] Transferência de reasoning (não apenas predictions) é contribuição clara
- [ ] Foco em ambientes regulados é único
- [ ] Nome "DiXtill" não está em uso
- [ ] Combinação XAI + KD durante treinamento (não post-hoc) é original

---

## 📝 Notas Finais

- **Data da última pesquisa**: _______
- **Pesquisador responsável**: _______
- **Decisão**:
  - [ ] Prosseguir com submissão (novidade confirmada)
  - [ ] Revisar papers para diferenciação clara
  - [ ] Adiar submissão (trabalho muito similar encontrado)

**Comentários adicionais**:

---

## 🔗 Referências para Citar (em caso de trabalhos relacionados)

### Knowledge Distillation Clássico:
1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

### Multi-Teacher:
2. You, S., Xu, C., Xu, C., & Tao, D. (2017). Learning from multiple teacher networks. KDD.
3. Zhang, Y., Xiang, T., Hospedales, T. M., & Lu, H. (2018). Deep mutual learning. CVPR.

### Attention Transfer:
4. Zagoruyko, S., & Komodakis, N. (2017). Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. ICLR.

### Feature-Based:
5. Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2015). FitNets: Hints for thin deep nets. ICLR.

### XAI (Explainability):
6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
7. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD.
8. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.

---

**Versão**: 1.0
**Última atualização**: 2025-12-07
