# Resumo dos Experimentos - HPM-KD Framework

## Paper: HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation for Efficient Model Compression

**Autores:** Gustavo Coelho Haase, Paulo Dourado
**Instituição:** Universidade Católica de Brasília
**Data:** Novembro 2025

---

## Questões de Pesquisa

O paper foi estruturado para responder 4 questões principais:

- **RQ1 (Eficiência de Compressão)**: HPM-KD consegue alcançar maiores taxas de compressão mantendo acurácia comparado aos métodos estado-da-arte?
- **RQ2 (Contribuição de Componentes)**: Quanto cada um dos 6 componentes do HPM-KD contribui para a performance geral?
- **RQ3 (Generalização)**: HPM-KD generaliza através de domínios diversos (visão, dados tabulares) e escalas de datasets?
- **RQ4 (Eficiência Computacional)**: Qual é o overhead computacional do HPM-KD comparado à destilação tradicional?

---

## Experimentos Realizados

### 1. Experimento Principal: Eficiência de Compressão (RQ1)

**Objetivo:** Avaliar se o HPM-KD consegue alcançar maiores taxas de compressão enquanto mantém a acurácia do modelo, comparado aos métodos estado-da-arte de destilação de conhecimento.

**Datasets:**
- **Visão computacional:** MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- **Dados tabulares:** Adult (Census Income), Credit (German Credit), Wine Quality

**Configuração:**
- Compressão: 3-15× (10.5× para MNIST/Fashion-MNIST, 3.1-10.4× para CIFAR, 10-15× para tabulares)
- Baselines: Direct Training, Traditional KD, FitNets, DML, TAKD
- 5 execuções independentes com seeds diferentes

**Resultados:**
- **MNIST:** 99.15% acurácia (99.87% retenção) vs 99.03% do TAKD - melhor em 0.12pp
- **Fashion-MNIST:** 91.48% acurácia (99.24% retenção) vs 91.15% do TAKD - melhor em 0.33pp
- **CIFAR-10:** 92.34% acurácia (98.74% retenção) vs 91.85% do TAKD - melhor em 0.49pp
- **CIFAR-100:** 70.98% acurácia (96.13% retenção) vs 69.85% do TAKD - melhor em 1.13pp
- **Adult:** 85.24% acurácia (99.44% retenção) vs 84.97% do TAKD - melhor em 0.27pp
- **Credit:** 75.69% acurácia (98.94% retenção) vs 75.28% do TAKD - melhor em 0.41pp
- **Wine Quality:** 60.12% acurácia (97.96% retenção) vs 59.47% do TAKD - melhor em 0.65pp

**Conclusão:** HPM-KD superou todos os baselines em todos os datasets com significância estatística (p<0.01). Maior ganho em CIFAR-100 devido ao alto grau de compressão e grande espaço de saída (100 classes).

---

### 2. Experimento de Generalização Cross-Domain (RQ3)

**Objetivo:** Avaliar se o HPM-KD mantém sua superioridade em datasets diversos com características variadas.

**Datasets:**
- OpenML-CC18: 10 datasets curados com características diversas
  - Tamanho: 150 a 70.000 amostras
  - Features: 4 a 617
  - Classes: 2 a 26

**Configuração:**
- Métricas: Mínimo, Mediana, Máximo e Média de retenção de acurácia

**Resultados:**
| Método | Mínimo | Mediana | Máximo | Média ± Std |
|--------|--------|---------|--------|-------------|
| Traditional KD | 93.4% | 95.8% | 98.1% | 95.9% ± 1.6% |
| FitNets | 94.1% | 96.2% | 98.4% | 96.3% ± 1.4% |
| TAKD | 94.8% | 96.6% | 98.7% | 96.7% ± 1.3% |
| **HPM-KD** | **95.9%** | **97.7%** | **99.2%** | **97.8% ± 1.2%** |

**Conclusão:** HPM-KD demonstrou generalização robusta com 97.8% de retenção média vs 96.7% do TAKD, mantendo melhorias consistentes independentemente de tamanho de dataset, dimensionalidade, número de classes e domínio.

---

### 3. Experimento de Razões de Compressão Variáveis

**Objetivo:** Avaliar como a vantagem do HPM-KD evolui com diferentes taxas de compressão.

**Dataset:** CIFAR-10

**Configuração:**
- Razões de compressão testadas: 2×, 4×, 6×, 8×, 10×, 15×, 20×
- Comparação com Traditional KD e TAKD

**Resultados:**
- **Baixa compressão (2-4×):** Todos os métodos performam similarmente (diferença <0.5pp)
- **Compressão média (6-10×):** HPM-KD começa a mostrar vantagem crescente
- **Alta compressão (10-20×):**
  - HPM-KD: Mantém 95%+ de retenção
  - Baselines: Degradam para 90-93%
  - Vantagem do HPM-KD aumenta de 0.5pp (10×) para 3.2pp (20×)

**Conclusão:** A Progressive Distillation Chain e Adaptive Configuration são particularmente efetivas em cenários de alta compressão onde a lacuna de capacidade é maior.

---

### 4. Análise de Eficiência Computacional (RQ4)

**Objetivo:** Quantificar o overhead computacional do HPM-KD e avaliar o impacto da paralelização.

#### 4.1 Breakdown de Tempo de Treinamento

**Dataset:** CIFAR-10

**Resultados:**
| Método | Busca Config | Treino Teacher | Destilação | Outros | Total |
|--------|--------------|----------------|------------|--------|-------|
| Traditional KD | 1.2h | 2.1h | 0.2h | 0.0h | 3.5h |
| TAKD | 0.8h | 2.1h | 3.4h (2 passos) | 0.1h | 6.4h |
| **HPM-KD** | **0.1h (auto)** | **2.1h** | **2.3h (3 passos)** | **0.2h** | **4.7h** |

**Overhead:** 20-40% comparado ao KD tradicional, mas 26% mais rápido que TAKD.

#### 4.2 Latência de Inferência e Memória

**Resultado crítico:** HPM-KD produz modelos estudantes com **características idênticas** de inferência aos baselines (mesma arquitetura), portanto **zero overhead de inferência**.

| Modelo | CPU Latency | GPU Latency | Parâmetros | Memória |
|--------|-------------|-------------|------------|---------|
| Teacher (ResNet-56) | 8.4ms | 0.9ms | 0.85M | 3.4MB |
| Student (ResNet-20) | 2.7ms | 0.3ms | 0.27M | 1.1MB |

#### 4.3 Speedup com Paralelização

**Objetivo:** Medir ganhos com múltiplos workers na destilação multi-teacher.

**Dataset:** CIFAR-100 (4 teachers)

**Resultados:**
- 1 worker: 12.4 horas (baseline)
- 2 workers: 6.8 horas (1.8× speedup, 90% eficiência)
- 4 workers: 3.9 horas (3.2× speedup, 80% eficiência)
- 8 workers: 2.5 horas (5.0× speedup, 62% eficiência)

**Conclusão:** Speedup quase linear até 4 workers, demonstrando paralelização efetiva.

---

### 5. Estudos de Ablação - Contribuição Individual dos Componentes (RQ2)

**Objetivo:** Isolar a contribuição de cada um dos 6 componentes do HPM-KD para a performance geral.

**Metodologia:**
- Remover cada componente individualmente
- Avaliar impacto na acurácia e tempo de treinamento
- Datasets representativos: CIFAR-10 (visão) e Adult (tabular)

#### Resultados Detalhados

**CIFAR-10:**

| Variante | Acurácia | Retenção | Tempo | Δ Acurácia | Δ Tempo |
|----------|----------|----------|-------|------------|---------|
| **Full HPM-KD** | **92.34%** | **98.74%** | **4.7h** | - | - |
| -AdaptConf | 90.82% | 97.11% | 6.2h | -1.52pp | +1.5h |
| -ProgChain | 89.48% | 95.68% | 3.8h | -2.86pp | -0.9h |
| -MultiTeach | 91.12% | 97.44% | 4.3h | -1.22pp | -0.4h |
| -MetaTemp | 91.56% | 97.91% | 4.8h | -0.78pp | +0.1h |
| -Parallel | 92.34% | 98.74% | 7.1h | 0.00pp | +2.4h |
| -Memory | 92.21% | 98.60% | 4.9h | -0.13pp | +0.2h |

**Adult (Tabular):**

| Variante | Acurácia | Retenção | Δ Acurácia |
|----------|----------|----------|------------|
| **Full HPM-KD** | **85.24%** | **99.44%** | - |
| -AdaptConf | 84.21% | 98.20% | -1.03pp |
| -ProgChain | 83.42% | 97.32% | -1.82pp |
| -MultiTeach | 84.58% | 98.67% | -0.66pp |
| -MetaTemp | 84.87% | 98.98% | -0.37pp |

#### Ranking de Importância dos Componentes

| Posição | Componente | Impacto Médio | Importância |
|---------|------------|---------------|-------------|
| 1 | Progressive Distillation Chain | -2.4pp | **Mais crítico** |
| 2 | Adaptive Configuration Manager | -1.8pp | Alto |
| 3 | Multi-Teacher Attention | -1.2pp | Médio |
| 4 | Meta-Temperature Scheduler | -0.9pp | Médio |
| 5 | Parallel Processing | 0.0pp (tempo apenas) | N/A |
| 6 | Shared Optimization Memory | -0.3pp (primeira rodada) | Baixo (acumula com uso) |

**Conclusões:**
1. **Progressive Chain** é o componente mais crítico (-2.4pp quando removido)
2. **Adaptive Configuration** elimina necessidade de tuning manual (economiza 32-50% de tempo)
3. **Multi-Teacher Attention** oferece ganhos consistentes, especialmente com múltiplos teachers
4. **Meta-Temperature** provê ajuste fino com overhead mínimo
5. **Parallel Processing** reduz tempo em 51% sem afetar acurácia
6. **Shared Memory** benefícios acumulam ao longo de múltiplos experimentos

---

### 6. Análise de Interação entre Componentes

**Objetivo:** Identificar sinergias positivas/negativas entre componentes.

**Metodologia:** Remover pares de componentes e comparar impacto combinado vs soma dos impactos individuais.

**Resultados (CIFAR-10):**

| Componentes Removidos | Impacto Combinado | Soma Individual | Sinergia |
|-----------------------|-------------------|-----------------|----------|
| ProgChain + AdaptConf | -3.90pp | -4.38pp | **Positiva** |
| MultiTeach + MetaTemp | -2.18pp | -2.00pp | **Positiva** |
| AdaptConf + MetaTemp | -2.47pp | -2.30pp | **Positiva** |
| ProgChain + MultiTeach | -4.22pp | -4.08pp | **Positiva** |
| **Todos os 6** | **-6.82pp** | **-6.60pp** | **Positiva** |

**Conclusão:** Todas as combinações mostram sinergia positiva. Remoção de todos os componentes (-6.82pp) excede a soma dos impactos individuais (-6.60pp), validando o design integrado do HPM-KD.

---

### 7. Análise de Sensibilidade a Hiperparâmetros

**Objetivo:** Avaliar robustez do HPM-KD a escolhas de hiperparâmetros comparado ao KD tradicional.

**Dataset:** CIFAR-10

**Configuração:**
- Grid search sobre temperatura T ∈ {2, 4, 6, 8} e peso α ∈ {0.3, 0.5, 0.7, 0.9}
- Medir variância de performance

**Resultados:**
- **Traditional KD:** Alta sensibilidade, região ótima estreita (T=4, α=0.7)
  - Variância de performance: ±2.8pp
- **HPM-KD:** Baixa sensibilidade, performance boa em ampla faixa
  - Variância de performance: ±0.6pp
  - Adaptive Configuration Manager ajusta automaticamente

**Conclusão:** HPM-KD é 4.7× mais robusto a escolhas de hiperparâmetros, graças ao Adaptive Configuration Manager que ajusta parâmetros dinamicamente baseado em dinâmicas de treinamento.

---

### 8. Análise de Comprimento da Cadeia Progressiva

**Objetivo:** Determinar número ideal de passos intermediários na Progressive Distillation Chain.

**Dataset:** CIFAR-10

**Metodologia:** Testar comprimentos de cadeia de 0 (direto) a 5 passos intermediários.

**Resultados:**

| Comprimento | Acurácia Student | Retenção | Tempo | Eficiência (Tempo/Acc) |
|-------------|------------------|----------|-------|------------------------|
| 0 (Direto) | 88.74% | - | 2.1h | - |
| 1 (Single-step KD) | 91.37% | 97.70% | 3.5h | 1.66 |
| 2 passos | 91.92% | 98.29% | 4.1h | 2.17 |
| **3 passos (HPM-KD)** | **92.34%** | **98.74%** | **4.7h** | **2.08** |
| 4 passos | 92.41% | 98.81% | 5.9h | 3.29 |
| 5 passos | 92.43% | 98.83% | 7.2h | 4.44 |

**Análise:**
- Ganhos marginais após 3 passos (<0.1pp)
- Aumento substancial de tempo após 3 passos
- Critério adaptativo do HPM-KD selecionou corretamente 3 passos (melhor custo-benefício)

**Conclusão:** Seleção automática de comprimento funciona bem, identificando o ponto de inflexão onde ganhos marginais não justificam tempo adicional.

---

### 9. Análise de Número de Teachers

**Objetivo:** Avaliar como performance escala com número de modelos teachers na destilação multi-teacher.

**Dataset:** CIFAR-10

**Configuração:**
- Variar número de teachers de 1 a 8
- Comparar HPM-KD com attention vs média uniforme

**Resultados:**

| Número de Teachers | HPM-KD (Attention) | Média Uniforme | Vantagem |
|--------------------|-------------------|----------------|----------|
| 1 | 91.82% | 91.82% | 0.00pp |
| 2 | 92.15% | 91.98% | +0.17pp |
| 3 | 92.34% | 92.08% | +0.26pp |
| 4 | 92.48% | 92.19% | +0.29pp |
| 5 | 92.51% | 92.22% | +0.29pp |
| 6 | 92.52% | 92.21% | +0.31pp |
| 8 | 92.49% | 92.18% | +0.31pp |

**Análise:**
- Benefícios saturam em torno de 4-5 teachers
- Mecanismo de attention começa a ter dificuldade em distinguir expertise dos teachers além desse ponto
- HPM-KD com attention sempre supera média uniforme

**Conclusão:** 4-5 teachers é o ponto ótimo para destilação multi-teacher. Além disso, ganhos adicionais são marginais.

---

### 10. Robustez a Desbalanceamento de Classes

**Objetivo:** Testar robustez do HPM-KD em cenários com distribuição desbalanceada de classes.

**Dataset:** CIFAR-10 (modificado)

**Configuração:**
- Criar versões desbalanceadas subsampling classes minoritárias
- Razões de desbalanceamento: Balanceado, 10:1, 50:1, 100:1

**Resultados (Retenção de Acurácia %):**

| Método | Balanceado | 10:1 | 50:1 | 100:1 |
|--------|------------|------|------|-------|
| Traditional KD | 97.70% | 96.82% | 94.15% | 91.28% |
| TAKD | 98.21% | 97.35% | 95.03% | 92.47% |
| **HPM-KD** | **98.74%** | **97.91%** | **95.86%** | **93.52%** |
| **Δ vs TAKD** | **+0.53pp** | **+0.56pp** | **+0.83pp** | **+1.05pp** |

**Análise:**
- Vantagem do HPM-KD **aumenta** com severidade do desbalanceamento
- De +0.53pp (balanceado) para +1.05pp (100:1 desbalanceado)
- Adaptive Configuration Manager e Progressive Chain são particularmente efetivos para distribuições desafiadoras

**Conclusão:** HPM-KD demonstra robustez superior em cenários de classe desbalanceada, com vantagem crescente conforme desbalanceamento aumenta.

---

### 11. Robustez a Ruído nos Rótulos

**Objetivo:** Avaliar robustez quando rótulos de treinamento contêm erros.

**Dataset:** CIFAR-10

**Configuração:**
- Injetar ruído aleatório flipando p% dos rótulos de treinamento
- Níveis de ruído: 0%, 10%, 20%, 30%

**Resultados (Retenção de Acurácia %):**

| Método | 0% ruído | 10% ruído | 20% ruído | 30% ruído | Degradação Total |
|--------|----------|-----------|-----------|-----------|------------------|
| Traditional KD | 97.70% | 96.18% | 93.82% | 89.64% | -8.06pp |
| TAKD | 98.21% | 96.78% | 94.52% | 90.83% | -7.38pp |
| **HPM-KD** | **98.74%** | **97.42%** | **95.38%** | **92.15%** | **-6.59pp** |

**Análise:**
- HPM-KD mantém menor degradação em todos os níveis de ruído
- A 30% de ruído: HPM-KD degrada 6.59pp vs 7.38pp do TAKD
- Progressive Chain atua como filtro de gradientes ruidosos através de múltiplos estágios

**Conclusão:** HPM-KD demonstra maior robustez a ruído nos rótulos, provavelmente devido ao efeito de "filtragem" da Progressive Chain.

---

### 12. Comparação com Estado-da-Arte Recente

**Objetivo:** Posicionar HPM-KD em relação aos métodos mais recentes de destilação publicados.

**Dataset:** CIFAR-100 (benchmark padrão)

**Configuração:**
- Teacher: ResNet-56 (73.84% acurácia)
- Student: ResNet-20
- Comparação com 8 métodos publicados (2015-2022)

**Resultados:**

| Método | Acurácia Student | Retenção | Ano |
|--------|------------------|----------|-----|
| Traditional KD | 68.92% | 93.34% | 2015 |
| FitNets | 69.47% | 94.08% | 2015 |
| Attention Transfer | 69.28% | 93.82% | 2017 |
| DML | 68.76% | 93.12% | 2018 |
| CRD (Contrastive) | 69.94% | 94.72% | 2020 |
| TAKD | 69.85% | 94.60% | 2020 |
| ReviewKD | 70.12% | 94.97% | 2021 |
| Self-Supervised KD | 70.35% | 95.28% | 2022 |
| **HPM-KD (Ours)** | **70.98%** | **96.13%** | **2025** |

**Melhoria sobre melhor anterior (Self-Supervised KD):** +0.63pp acurácia, +0.85pp retenção

**Conclusão:** HPM-KD supera todos os métodos estado-da-arte recentes, incluindo abordagens especializadas (contrastivas, self-supervised), enquanto oferece framework geral e automatizado.

---

### 13. Visualização de Representações Aprendidas

**Objetivo:** Avaliar qualitativamente a similaridade das representações do student com o teacher.

**Dataset:** CIFAR-10 test set

**Metodologia:**
- Extrair representações da penúltima camada
- Aplicar t-SNE para visualização 2D
- Comparar: Teacher, Direct Training, TAKD, HPM-KD

**Observações Qualitativas:**
- **Teacher:** Separação clara de classes, clusters bem definidos
- **Direct Training:** Separação moderada, alguma sobreposição entre classes similares
- **TAKD:** Melhoria sobre Direct, estrutura mais próxima do teacher
- **HPM-KD:** Separação de classes mais clara, alinhamento mais próximo com estrutura do teacher

**Métricas Quantitativas (Silhouette Score):**
- Teacher: 0.42
- Direct Training: 0.31
- TAKD: 0.37
- **HPM-KD: 0.40** (mais próximo do teacher)

**Conclusão:** HPM-KD aprende representações que melhor replicam a estrutura do espaço de representação do teacher, evidenciando transferência de conhecimento mais efetiva.

---

### 14. Análise de Custo-Benefício

**Objetivo:** Visualizar trade-off entre acurácia e tempo de treinamento.

**Dataset:** CIFAR-10

**Metodologia:**
- Plot scatter: Acurácia vs Tempo de Treinamento
- Identificar fronteira de Pareto

**Resultados:**

| Método | Acurácia | Tempo | Posição |
|--------|----------|-------|---------|
| Direct Training | 88.74% | 2.1h | Baseline |
| Traditional KD | 91.37% | 3.5h | Bom |
| FitNets | 91.68% | 5.2h | Caro |
| DML | 91.42% | 6.8h | Muito caro |
| TAKD | 91.85% | 5.9h | Caro |
| **HPM-KD** | **92.34%** | **4.7h** | **Fronteira Pareto** |

**Análise:**
- HPM-KD alcança melhor acurácia (+0.49pp sobre TAKD) com menor tempo (-1.2h)
- Único método na fronteira de Pareto
- Métodos acima e à direita são dominados

**Conclusão:** HPM-KD oferece o melhor trade-off acurácia-tempo, situando-se na fronteira de Pareto.

---

## Resumo Geral dos Resultados

### Principais Descobertas

1. **Eficiência de Compressão Superior (RQ1)**
   - 95-99% retenção de acurácia com 3-15× compressão
   - Melhoria de 0.3-1.1pp sobre melhores baselines
   - Significância estatística p<0.01 em todos os casos

2. **Generalização Robusta (RQ3)**
   - Consistente em visão e dados tabulares
   - 97.8% retenção média em OpenML-CC18
   - Vantagem aumenta com complexidade do problema

3. **Eficiência Computacional (RQ4)**
   - 20-40% overhead de treino vs KD tradicional
   - 3.2× speedup com paralelização (4 workers)
   - **Zero overhead de inferência**

4. **Contribuições dos Componentes (RQ2)**
   - Progressive Chain: Componente mais crítico (-2.4pp)
   - Todos componentes contribuem significativamente
   - Sinergias positivas entre componentes (+0.22pp além da soma)

5. **Robustez Demonstrada**
   - Menos sensível a hiperparâmetros (4.7× menor variância)
   - Vantagem aumenta com desbalanceamento de classes
   - Menor degradação com ruído nos rótulos (-6.59pp vs -7.38pp a 30% ruído)

6. **Estado-da-Arte**
   - Supera métodos especializados recentes (CRD, ReviewKD, Self-Supervised KD)
   - Melhor trade-off acurácia-tempo (fronteira Pareto)
   - Framework geral aplicável a múltiplos domínios

### Datasets Utilizados

**Total:** 15+ datasets

**Visão Computacional (4):**
- MNIST: 60k train, 10k test, 10 classes
- Fashion-MNIST: 60k train, 10k test, 10 classes
- CIFAR-10: 50k train, 10k test, 10 classes
- CIFAR-100: 50k train, 10k test, 100 classes

**Dados Tabulares (3+):**
- Adult: 32k train, 16k test, 2 classes
- Credit: 700 train, 300 test, 2 classes
- Wine Quality: 4.5k train, 1.9k test, 6 classes
- OpenML-CC18: 10 datasets diversos

### Métricas Avaliadas

1. **Compression Ratio:** Taxa de redução de parâmetros
2. **Accuracy Retention:** % de acurácia do teacher preservada
3. **Relative Improvement:** Melhoria sobre treino direto
4. **Training Time:** Tempo total de destilação
5. **Inference Latency:** Tempo de inferência por amostra
6. **Memory Footprint:** Uso de memória GPU

---

## Limitações e Trabalhos Futuros

**Limitações identificadas:**
1. Benefícios de multi-teacher saturam em 4-5 teachers
2. Shared Memory requer múltiplos experimentos para benefício completo
3. Overhead de 20-40% no tempo de treino (justificado pelos ganhos)

**Direções futuras sugeridas:**
1. Extensão para outras tarefas (detecção de objetos, segmentação)
2. Aplicação a modelos de linguagem (LLMs)
3. Destilação online/contínua
4. Integração com outras técnicas de compressão (quantização, poda)

---

**Implementação:** Framework disponível open-source na biblioteca DeepBridge
**Código e experimentos:** https://github.com/DeepBridge-Validation/DeepBridge

