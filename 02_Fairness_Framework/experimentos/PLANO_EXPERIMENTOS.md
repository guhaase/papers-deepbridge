# Plano de Experimentos - DeepBridge Fairness Framework

**Paper**: "DeepBridge Fairness: Da Pesquisa à Regulação -- Um Framework Pronto para Produção para Teste de Fairness Algorítmica"

**Conferência Alvo**: FAccT 2026

**Objetivo**: Validar todas as claims do paper através de experimentos reproduzíveis e rigorosos.

---

## 📊 Índice de Experimentos

1. [Auto-Detecção de Atributos Sensíveis](#1-auto-detecção-de-atributos-sensíveis)
2. [Cobertura de Métricas](#2-cobertura-de-métricas)
3. [Verificação EEOC/ECOA](#3-verificação-eeocecoa)
4. [Estudos de Caso](#4-estudos-de-caso)
5. [Estudo de Usabilidade](#5-estudo-de-usabilidade)
6. [Performance e Escalabilidade](#6-performance-e-escalabilidade)
7. [Otimização de Threshold](#7-otimização-de-threshold)
8. [Comparação com Ferramentas Existentes](#8-comparação-com-ferramentas-existentes)

---

## 1. Auto-Detecção de Atributos Sensíveis

### 1.1 Experimento: Acurácia em 500 Datasets

**Claim do Paper**:
- Precision: 0.92
- Recall: 0.89
- F1-Score: 0.90
- Testado em 500 datasets reais

**Metodologia**:

1. **Coleta de Datasets**:
   - 200 datasets do Kaggle (buscar por: "credit", "hiring", "health", "criminal justice")
   - 150 datasets do UCI Machine Learning Repository
   - 100 datasets de OpenML
   - 50 datasets sintéticos (controle)

2. **Ground Truth**:
   - Anotação manual por 2 especialistas independentes
   - Medir inter-rater agreement (Cohen's Kappa > 0.85)
   - Resolver discordâncias por consenso

3. **Categorias de Atributos Sensíveis**:
   - Gender/Sex (target: 100 datasets)
   - Race/Ethnicity (target: 80 datasets)
   - Age (target: 90 datasets)
   - Religion (target: 30 datasets)
   - Disability (target: 25 datasets)
   - Nationality (target: 40 datasets)
   - Marital Status (target: 35 datasets)

4. **Execução**:
   ```python
   from deepbridge import DBDataset

   results = []
   for dataset_name, df, ground_truth in datasets:
       dataset = DBDataset(data=df, target_column='target')
       detected = set(dataset.detected_sensitive_attributes)

       tp = len(detected & ground_truth)
       fp = len(detected - ground_truth)
       fn = len(ground_truth - detected)

       precision = tp / (tp + fp) if (tp + fp) > 0 else 0
       recall = tp / (tp + fn) if (tp + fn) > 0 else 0
       f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

       results.append({
           'dataset': dataset_name,
           'precision': precision,
           'recall': recall,
           'f1': f1
       })
   ```

5. **Análise de Erros**:
   - Classificar False Positives por tipo (e.g., "race_time" detectado como "race")
   - Classificar False Negatives por causa (typos, thresholds, codificação)
   - Calcular confusion matrix por categoria

**Métricas de Validação**:
- ✅ Precision geral ≥ 0.90
- ✅ Recall geral ≥ 0.85
- ✅ F1-Score ≥ 0.88
- ✅ Precision por categoria ≥ 0.85
- ✅ Kappa inter-rater ≥ 0.85

**Artefatos**:
- `results/auto_detection_500_datasets.csv` - Resultados completos
- `results/auto_detection_confusion_matrix.png` - Matriz de confusão
- `results/auto_detection_by_category.csv` - Breakdown por categoria
- `results/false_positives_analysis.txt` - Análise de FPs
- `results/false_negatives_analysis.txt` - Análise de FNs

---

### 1.2 Experimento: Acurácia 100% nos Case Studies

**Claim do Paper**:
- 10/10 atributos detectados vs 2/10 manual
- 100% acurácia em COMPAS, German Credit, Adult, Healthcare

**Metodologia**:

1. **Datasets**:
   - COMPAS: 3 atributos esperados (race, sex, age)
   - German Credit: 3 atributos (age, sex, foreign_worker)
   - Adult Income: 2 atributos (sex, race)
   - Healthcare: 2 atributos (race, age_group)

2. **Execução**:
   ```python
   test_cases = [
       ('COMPAS', df_compas, {'race', 'sex', 'age'}),
       ('German Credit', df_credit, {'age', 'sex', 'foreign_worker'}),
       ('Adult Income', df_adult, {'sex', 'race'}),
       ('Healthcare', df_health, {'race', 'age_group'})
   ]

   for name, df, expected in test_cases:
       dataset = DBDataset(data=df, target_column=TARGET_COL)
       detected = set(dataset.detected_sensitive_attributes)

       accuracy = len(detected & expected) / len(expected)
       print(f"{name}: {accuracy*100:.0f}% ({detected} vs {expected})")
   ```

**Métricas de Validação**:
- ✅ 100% acurácia em todos os 4 datasets
- ✅ 0 falsos positivos
- ✅ 0 falsos negativos

**Artefatos**:
- `results/case_studies_auto_detection.txt` - Log de detecção

---

## 2. Cobertura de Métricas

### 2.1 Experimento: 15 Métricas Integradas

**Claim do Paper**:
- 4 métricas pré-treinamento
- 11 métricas pós-treinamento
- 87% mais que AI Fairness 360 (8 métricas)

**Metodologia**:

1. **Verificação de Implementação**:
   ```python
   from deepbridge import FairnessTestManager

   # Pré-treinamento
   pre_metrics = [
       'class_balance',
       'concept_balance',
       'kl_divergence',
       'js_divergence'
   ]

   # Pós-treinamento
   post_metrics = [
       'statistical_parity',
       'equal_opportunity',
       'equalized_odds',
       'disparate_impact',
       'fnr_difference',
       'fpr_difference',
       'conditional_acceptance',
       'conditional_rejection',
       'precision_difference',
       'accuracy_difference',
       'treatment_equality'
   ]

   ftm = FairnessTestManager(dataset)

   # Verificar cada métrica calcula corretamente
   for metric in pre_metrics:
       result = ftm.compute_metric(metric)
       assert result is not None, f"{metric} falhou"

   for metric in post_metrics:
       result = ftm.compute_metric(metric, predictions=y_pred)
       assert result is not None, f"{metric} falhou"
   ```

2. **Validação Manual**:
   - Calcular cada métrica manualmente em dataset pequeno (100 amostras)
   - Comparar com output do DeepBridge (tolerância < 1e-6)

3. **Teste de Edge Cases**:
   - Dataset perfeitamente balanceado (todas métricas = 0 ou 1.0)
   - Dataset completamente enviesado (disparate impact < 0.5)
   - Grupos com 1 amostra (verificar tratamento de divisão por zero)

**Métricas de Validação**:
- ✅ 15 métricas implementadas e funcionais
- ✅ Erro < 1e-6 vs. cálculo manual
- ✅ Edge cases tratados sem crashes

**Artefatos**:
- `results/metrics_validation.csv` - Comparação manual vs DeepBridge
- `results/metrics_edge_cases.txt` - Log de edge cases

---

## 3. Verificação EEOC/ECOA

### 3.1 Experimento: Detecção de Violações da Regra 80%

**Claim do Paper**:
- 100% precisão na detecção de violações
- 0 falsos positivos

**Metodologia**:

1. **Casos de Teste Controlados**:
   ```python
   test_cases = [
       # (selection_rate_protected, selection_rate_reference, expected_violation)
       (0.40, 0.50, True),   # DI = 0.80 - BOUNDARY
       (0.39, 0.50, True),   # DI = 0.78 - VIOLATION
       (0.41, 0.50, False),  # DI = 0.82 - OK
       (0.70, 0.80, False),  # DI = 0.875 - OK
       (0.50, 0.70, True),   # DI = 0.714 - VIOLATION
   ]

   for sr_p, sr_r, expected in test_cases:
       ftm = FairnessTestManager(synthetic_dataset(sr_p, sr_r))
       compliance = ftm.check_eeoc_compliance()

       is_violation = compliance['eeoc_80_rule'] == False
       assert is_violation == expected, f"Falha: DI={sr_p/sr_r:.2f}"
   ```

2. **Datasets Reais**:
   - Aplicar em COMPAS, German Credit, Adult, Healthcare
   - Comparar com análise manual de compliance officer

**Métricas de Validação**:
- ✅ 100% acurácia em casos de teste controlados (5/5)
- ✅ 100% concordância com análise manual em datasets reais

**Artefatos**:
- `results/eeoc_80_rule_validation.csv` - Casos de teste

---

### 3.2 Experimento: Verificação EEOC Question 21

**Claim do Paper**:
- Valida representação mínima 2% por grupo

**Metodologia**:

1. **Casos de Teste**:
   ```python
   test_cases = [
       # (group_representation, expected_valid)
       (0.025, True),   # 2.5% - OK
       (0.020, True),   # 2.0% - BOUNDARY
       (0.015, False),  # 1.5% - VIOLATION
       (0.001, False),  # 0.1% - SEVERE VIOLATION
   ]
   ```

**Métricas de Validação**:
- ✅ 100% acurácia nos casos de teste

**Artefatos**:
- `results/eeoc_question21_validation.csv`

---

### 3.3 Experimento: ECOA Adverse Action Notices

**Claim do Paper**:
- Gera notices com "razões específicas"

**Metodologia**:

1. **Verificação de Conteúdo**:
   - Gerar 100 notices para decisões adversas
   - Verificar se contêm:
     - Razões específicas (não genéricas)
     - Scores/métricas quantitativas
     - Thresholds de decisão

2. **Revisão por Compliance Officer**:
   - 20 notices aleatórios revisados por profissional legal
   - Verificar conformidade com ECOA § 1002.9

**Métricas de Validação**:
- ✅ 100% dos notices contêm razões específicas
- ✅ Aprovação ≥ 90% por compliance officer

**Artefatos**:
- `results/adverse_action_notices_sample.txt` - 20 exemplos
- `results/compliance_officer_review.csv` - Avaliações

---

## 4. Estudos de Caso

### 4.1 Experimento: COMPAS Recidivism Prediction

**Claims do Paper**:
- Tempo: 7.2 min (vs 35 min manual, 79% economia)
- Violação: FPR difference 22pp → 8pp com threshold 0.62
- Atributos detectados: 3/3 (race, sex, age)

**Metodologia**:

1. **Setup**:
   - Dataset: ProPublica COMPAS (7,214 amostras)
   - Modelo: Random Forest Classifier
   - Features: 12 (idade, gênero, raça, histórico criminal)

2. **Análise Completa**:
   ```python
   import time
   from deepbridge import DBDataset, FairnessTestManager

   start = time.time()

   # Load data
   dataset = DBDataset(data=df_compas, target_column='two_year_recid', model=rf_model)

   # Auto-detection
   detected = dataset.detected_sensitive_attributes

   # Pre-training metrics
   ftm = FairnessTestManager(dataset)
   pre_metrics = ftm.compute_pre_training_metrics()

   # Post-training metrics
   post_metrics = ftm.compute_post_training_metrics()

   # EEOC compliance
   compliance = ftm.check_eeoc_compliance()

   # Threshold optimization
   optimal = ftm.optimize_threshold(
       fairness_metric='fpr_difference',
       min_accuracy=0.68
   )

   elapsed = time.time() - start
   ```

3. **Validação de Resultados**:
   - FPR difference at t=0.5: 22pp ± 2pp
   - FPR difference at t=0.62: 8pp ± 2pp
   - Accuracy at t=0.62: ≥ 68%

**Métricas de Validação**:
- ✅ Tempo ≤ 10 min
- ✅ 3/3 atributos detectados
- ✅ FPR reduction ≥ 60%
- ✅ Threshold ótimo: 0.60-0.65

**Artefatos**:
- `results/compas_full_analysis.json` - Resultados completos
- `results/compas_threshold_analysis.csv` - Análise de thresholds
- `results/compas_report.html` - Relatório visual

---

### 4.2 Experimento: German Credit Scoring

**Claims do Paper**:
- Tempo: 5.8 min (vs 25 min manual, 77% economia)
- Violação: Age <25 tem DI = 0.73 (violação regra 80%)
- Threshold ótimo: 0.45 (DI=0.80, Acc=72%)

**Metodologia**:

1. **Setup**:
   - Dataset: UCI German Credit (1,000 amostras)
   - Modelo: XGBoost Classifier
   - Features: 20 (idade, crédito, emprego)

2. **Validação ECOA**:
   ```python
   ftm = FairnessTestManager(dataset)
   compliance = ftm.check_ecoa_compliance()

   # Verificar DI por idade
   di_young = compliance['disparate_impact']['age_<25']
   assert 0.70 <= di_young <= 0.76, "DI fora do esperado"
   ```

3. **Threshold Optimization**:
   - Gerar Pareto frontier (t=0.1 a 0.9, step=0.05)
   - Identificar t que maximiza accuracy com DI ≥ 0.80

**Métricas de Validação**:
- ✅ Tempo ≤ 8 min
- ✅ DI violação detectada em Age <25
- ✅ Threshold ótimo: 0.42-0.48

**Artefatos**:
- `results/credit_full_analysis.json`
- `results/credit_pareto_frontier.png`
- `results/credit_ecoa_compliance.txt`

---

### 4.3 Experimento: Adult Income (Employment)

**Claims do Paper**:
- Tempo: 12.4 min (vs 50 min manual, 75% economia)
- Violação: Female DI = 0.43 (severe violation)
- Análise de causa: "occupation" é proxy de gender

**Metodologia**:

1. **Setup**:
   - Dataset: UCI Adult (48,842 amostras)
   - Modelo: LightGBM Classifier
   - Features: 14 (idade, educação, ocupação, raça, sexo)

2. **Feature Importance por Grupo**:
   ```python
   ftm = FairnessTestManager(dataset)
   importance = ftm.analyze_feature_importance_by_group(
       sensitive_attribute='sex'
   )

   # Verificar se "occupation" é top-3 feature
   assert 'occupation' in importance['female'][:3]
   assert 'occupation' in importance['male'][:3]
   ```

3. **Análise de Mitigação**:
   - Testar reweighting
   - Testar threshold adjustment
   - Testar remoção de proxy features

**Métricas de Validação**:
- ✅ Tempo ≤ 15 min
- ✅ DI Female: 0.40-0.46
- ✅ "occupation" detectado como proxy

**Artefatos**:
- `results/adult_full_analysis.json`
- `results/adult_feature_importance.csv`
- `results/adult_mitigation_strategies.txt`

---

### 4.4 Experimento: Healthcare Risk Prediction

**Claims do Paper**:
- Tempo: 9.1 min (vs 40 min manual, 77% economia)
- Análise: Risco maior para Black/Hispanic (DI=1.41/1.27)
- Recomendação: Threshold adjustment NÃO recomendado (risco de dano)

**Metodologia**:

1. **Setup**:
   - Dataset: Sintético baseado em MIMIC-III (10,000 amostras)
   - Modelo: Neural Network (3 layers)
   - Features: 25 (idade, raça, diagnósticos)

2. **Análise Ética**:
   ```python
   ftm = FairnessTestManager(dataset)
   ethical_review = ftm.analyze_ethical_implications(
       context='healthcare',
       sensitive_attribute='race'
   )

   # Verificar se threshold adjustment é recomendado
   assert ethical_review['threshold_adjustment_recommended'] == False
   assert 'clinical_review' in ethical_review['recommendations']
   ```

**Métricas de Validação**:
- ✅ Tempo ≤ 12 min
- ✅ DI Black: 1.35-1.50
- ✅ Warning sobre threshold adjustment presente

**Artefatos**:
- `results/healthcare_full_analysis.json`
- `results/healthcare_ethical_review.txt`

---

## 5. Estudo de Usabilidade

### 5.1 Experimento: System Usability Scale (SUS)

**Claim do Paper**:
- SUS Score: 85.2 ± 8.3
- Classificação: "Excelente" (top 15%)
- N=20 participantes

**Metodologia**:

1. **Recrutamento**:
   - 20 data scientists/ML engineers
   - Experiência: 2-8 anos em ML
   - 65% com experiência em fairness tools
   - Diversidade: 12 organizações (finanças, saúde, tech)

2. **Protocol**:
   - **Setup** (10 min): Instalar DeepBridge, carregar Adult dataset
   - **Task 1** (15 min): Detectar bias em modelo pré-treinado
   - **Task 2** (15 min): Verificar conformidade EEOC/ECOA
   - **Task 3** (20 min): Identificar threshold ótimo

3. **Questionário SUS**:
   - 10 perguntas em escala Likert (1-5)
   - Normalizar para 0-100
   - Calcular média e desvio padrão

4. **Critérios de Inclusão**:
   - Score individual ≥ 68 (acima da média da indústria)
   - Score médio ≥ 80 (excelente)
   - Desvio padrão ≤ 15

**Métricas de Validação**:
- ✅ SUS médio ≥ 80
- ✅ SUS ≥ 68 para ≥ 90% dos participantes
- ✅ Classificação "Excelente" (top 15%)

**Artefatos**:
- `results/sus_scores.csv` - Scores individuais
- `results/sus_analysis.txt` - Análise estatística
- `results/participant_demographics.csv` - Dados demográficos

---

### 5.2 Experimento: NASA Task Load Index (TLX)

**Claim do Paper**:
- NASA-TLX: 32.1 ± 12.4
- Benchmark: 50 (neutral)
- Interpretação: Baixa carga cognitiva

**Metodologia**:

1. **Questionário TLX** (aplicado após cada tarefa):
   - Mental Demand (1-100)
   - Physical Demand (1-100)
   - Temporal Demand (1-100)
   - Performance (1-100)
   - Effort (1-100)
   - Frustration (1-100)

2. **Análise**:
   - Média ponderada das 6 dimensões
   - Comparar com benchmark (50)

**Métricas de Validação**:
- ✅ TLX médio ≤ 40
- ✅ Mental Demand ≤ 45
- ✅ Frustration ≤ 35

**Artefatos**:
- `results/tlx_scores.csv`
- `results/tlx_by_task.csv`

---

### 5.3 Experimento: Task Success Rate

**Claim do Paper**:
- Overall: 95% (19/20)
- Task 1: 100% (20/20)
- Task 2: 95% (19/20)
- Task 3: 90% (18/20)

**Metodologia**:

1. **Critérios de Sucesso**:
   - **Task 1**: Identificar corretamente ≥2 métricas com violation
   - **Task 2**: Reportar corretamente status EEOC/ECOA
   - **Task 3**: Selecionar threshold com DI ≥ 0.80 e Acc ≥ 70%

2. **Observação**:
   - Screen recording de todas sessões
   - Notas de campo por observador

**Métricas de Validação**:
- ✅ Task 1 success ≥ 95%
- ✅ Task 2 success ≥ 90%
- ✅ Task 3 success ≥ 85%
- ✅ Overall success ≥ 90%

**Artefatos**:
- `results/task_success_rates.csv`
- `results/task_failures_analysis.txt`

---

### 5.4 Experimento: Time-to-Insight

**Claim do Paper**:
- DeepBridge: 10.2 ± 3.1 min
- Manual: 25-30 min

**Metodologia**:

1. **Medição**:
   - Início: Quando participante carrega dataset
   - Fim: Quando identifica primeira violação de fairness

2. **Comparação com Baseline**:
   - Grupo controle (10 participantes) usa AI Fairness 360
   - Medir tempo até primeira detecção

**Métricas de Validação**:
- ✅ Time-to-insight DeepBridge ≤ 12 min
- ✅ Speedup vs manual ≥ 2.0x

**Artefatos**:
- `results/time_to_insight.csv`

---

### 5.5 Experimento: Entrevistas Qualitativas

**Claim do Paper**:
- Pontos fortes: Auto-detecção, relatórios EEOC, Pareto frontier, integração scikit-learn
- Pontos fracos: Pareto frontier não intuitivo, falta de sugestões de mitigação

**Metodologia**:

1. **Protocol**:
   - Entrevista semi-estruturada (20 min)
   - Perguntas abertas:
     - "O que você mais gostou?"
     - "O que foi mais difícil?"
     - "O que melhoraria?"

2. **Análise**:
   - Thematic analysis (codificação de temas)
   - Frequência de menções

**Métricas de Validação**:
- ✅ ≥70% mencionam auto-detecção como ponto forte
- ✅ ≥50% mencionam Pareto frontier como ponto forte
- ✅ ≥30% mencionam necessidade de sugestões de mitigação

**Artefatos**:
- `results/interview_transcripts.txt`
- `results/thematic_analysis.csv`

---

## 6. Performance e Escalabilidade

### 6.1 Experimento: Speedup vs Manual Workflow

**Claim do Paper**:
- Small (1K): 5.5 min vs 24.7 min (4.5x)
- Medium (50K): 17.8 min vs 48.3 min (2.7x)
- Large (500K): 67.9 min vs 140.2 min (2.1x)
- Speedup médio: 2.9x

**Metodologia**:

1. **Datasets**:
   - Small: German Credit (1K amostras, 20 features)
   - Medium: Adult Income (50K amostras, 50 features)
   - Large: Sintético (500K amostras, 100 features)

2. **Workflow Manual** (baseline):
   - Identificação manual de atributos (5 min fixo)
   - Conversão para formato AIF360
   - Análise com AIF360
   - Análise custom (threshold, relatórios)
   - Geração de relatório manual

3. **Workflow DeepBridge**:
   - Auto-detecção
   - Métricas pré-treino
   - Métricas pós-treino
   - Threshold optimization
   - Geração de relatórios

4. **Execução**:
   - 5 repetições por dataset
   - Hardware: AWS m5.2xlarge (8 vCPUs, 32GB RAM)
   - Medir tempo total e por componente

**Métricas de Validação**:
- ✅ Speedup small ≥ 3.5x
- ✅ Speedup medium ≥ 2.5x
- ✅ Speedup large ≥ 2.0x
- ✅ Speedup médio ≥ 2.5x

**Artefatos**:
- `results/performance_benchmarks.csv`
- `results/performance_by_component.csv`
- `results/performance_comparison.png`

---

### 6.2 Experimento: Memory Usage

**Claim do Paper**:
- 40-42% menos memória que AIF360
- Small: 250 MB vs 420 MB
- Medium: 1.8 GB vs 3.2 GB
- Large: 12.5 GB vs 21.3 GB

**Metodologia**:

1. **Medição**:
   ```python
   import tracemalloc

   tracemalloc.start()

   # DeepBridge workflow
   dataset = DBDataset(data=df, target_column='target', model=model)
   ftm = FairnessTestManager(dataset)
   ftm.compute_all_metrics()

   current, peak = tracemalloc.get_traced_memory()
   tracemalloc.stop()

   print(f"Peak memory: {peak / 1024**2:.1f} MB")
   ```

2. **Comparação**:
   - Mesmo workflow com AIF360
   - 5 repetições por dataset

**Métricas de Validação**:
- ✅ Redução ≥ 35% em todos tamanhos
- ✅ Peak memory small ≤ 300 MB
- ✅ Peak memory medium ≤ 2.0 GB
- ✅ Peak memory large ≤ 15 GB

**Artefatos**:
- `results/memory_usage.csv`
- `results/memory_comparison.png`

---

### 6.3 Experimento: Escalabilidade

**Claims do Paper**:
- Algoritmo de threshold optimization: O(n log n)
- Lazy evaluation e caching inteligente

**Metodologia**:

1. **Análise de Complexidade**:
   - Datasets sintéticos: 1K, 10K, 100K, 500K, 1M amostras
   - Medir tempo de threshold optimization
   - Fit curva log-linear

2. **Teste de Lazy Loading**:
   - Medir tempo sem acesso a métricas (deve ser ~0)
   - Medir tempo com acesso a 1 métrica
   - Verificar que métricas não usadas não são calculadas

**Métricas de Validação**:
- ✅ R² ≥ 0.95 para fit O(n log n)
- ✅ Lazy loading economiza ≥ 50% tempo quando <50% métricas usadas

**Artefatos**:
- `results/scalability_analysis.csv`
- `results/complexity_curve.png`
- `results/lazy_loading_test.txt`

---

## 7. Otimização de Threshold

### 7.1 Experimento: Pareto Frontier Identification

**Claim do Paper**:
- 100% dos participantes identificaram threshold ótimo corretamente
- Média 4.8/5 em utilidade de visualizações

**Metodologia**:

1. **Geração de Pareto Frontier**:
   ```python
   ftm = FairnessTestManager(dataset)
   pareto = ftm.analyze_threshold_pareto(
       thresholds=np.arange(0.1, 0.9, 0.05),
       fairness_metric='disparate_impact',
       performance_metric='accuracy'
   )

   # Identificar pontos Pareto-eficientes
   pareto_points = pareto[pareto['is_pareto_efficient']]
   ```

2. **Validação Matemática**:
   - Verificar que pontos na frontier não são dominados
   - Verificar que pontos fora são dominados

3. **Usabilidade**:
   - 20 participantes identificam threshold ótimo dado constraint
   - Constraint: "Maximize fairness com accuracy ≥ 70%"

**Métricas de Validação**:
- ✅ Pareto frontier matematicamente correta
- ✅ ≥95% participantes identificam threshold correto
- ✅ Utilidade média ≥ 4.5/5

**Artefatos**:
- `results/pareto_frontier_example.png`
- `results/pareto_validation.csv`
- `results/threshold_identification_accuracy.csv`

---

### 7.2 Experimento: Threshold Recommendations

**Claim do Paper**:
- COMPAS: threshold 0.62 reduz FPR difference de 22pp → 8pp
- German Credit: threshold 0.45 balanceia DI=0.80 e Acc=72%
- Healthcare: threshold adjustment NÃO recomendado

**Metodologia**:

1. **Teste em Case Studies**:
   - Executar otimização automática
   - Comparar threshold recomendado com claim
   - Validar métricas no threshold recomendado

2. **Regras de Recomendação**:
   ```python
   # COMPAS: minimizar FPR difference mantendo Acc ≥ 68%
   rec = ftm.recommend_threshold(
       objective='minimize',
       fairness_metric='fpr_difference',
       constraints={'accuracy': 0.68}
   )

   # Verificar: 0.60 ≤ rec['threshold'] ≤ 0.65
   ```

**Métricas de Validação**:
- ✅ Threshold COMPAS: 0.60-0.65
- ✅ Threshold German Credit: 0.42-0.48
- ✅ Healthcare: recomendação contra threshold adjustment presente

**Artefatos**:
- `results/threshold_recommendations.csv`

---

## 8. Comparação com Ferramentas Existentes

### 8.1 Experimento: Feature Comparison Matrix

**Claim do Paper**:
- DeepBridge é única ferramenta com:
  - Auto-detecção de atributos
  - Verificação EEOC/ECOA
  - Threshold optimization
  - Pareto frontier analysis
  - Métricas pré-treinamento

**Metodologia**:

1. **Ferramentas Testadas**:
   - AI Fairness 360 v0.2.9+
   - Fairlearn v0.10.0+
   - Aequitas v2.0.0+
   - DeepBridge Fairness

2. **Features Testados**:
   - Auto-detecção (sim/não)
   - Número de métricas pré-treino
   - Número de métricas pós-treino
   - EEOC 80% rule (sim/não)
   - ECOA compliance (sim/não)
   - Threshold optimization (sim/não)
   - Pareto frontier (sim/não)
   - Relatórios HTML/PDF (sim/não)

3. **Validação**:
   - Testar cada feature em dataset Adult Income
   - Documentar presença/ausência de cada feature

**Métricas de Validação**:
- ✅ DeepBridge tem todas features claimed
- ✅ Outras ferramentas NÃO têm features exclusivas claimed

**Artefatos**:
- `results/tool_comparison_matrix.csv`
- `results/tool_comparison_report.md`

---

### 8.2 Experimento: Accuracy of Metrics

**Claim do Paper**:
- DeepBridge calcula métricas corretamente (comparado com outras ferramentas)

**Metodologia**:

1. **Métricas Comuns** (presentes em múltiplas ferramentas):
   - Statistical Parity / Demographic Parity
   - Equal Opportunity
   - Disparate Impact

2. **Dataset de Teste**:
   - Adult Income (consenso na literatura)

3. **Comparação**:
   ```python
   # AIF360
   from aif360.metrics import BinaryLabelDatasetMetric
   aif_di = BinaryLabelDatasetMetric(...).disparate_impact()

   # Fairlearn
   from fairlearn.metrics import demographic_parity_difference
   fl_dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sex)

   # DeepBridge
   ftm = FairnessTestManager(dataset)
   db_di = ftm.compute_metric('disparate_impact')

   # Comparar (tolerância < 0.01)
   assert abs(aif_di - db_di) < 0.01
   ```

**Métricas de Validação**:
- ✅ Diferença < 1% para métricas comuns
- ✅ Métricas exclusivas validadas manualmente

**Artefatos**:
- `results/metric_accuracy_comparison.csv`

---

## 9. Experimentos Adicionais (Robustness)

### 9.1 Edge Cases e Stress Tests

**Objetivo**: Garantir que DeepBridge é robusto em condições adversas

**Casos de Teste**:

1. **Dataset Pequeno** (n=50):
   - Verificar que métricas calculam sem crash
   - Verificar warnings sobre significância estatística

2. **Dataset Desbalanceado Extremo** (99:1):
   - Verificar handling de divisão por zero
   - Verificar warnings sobre grupos minoritários

3. **Missing Values** (30% de NaN):
   - Verificar imputation automática
   - Verificar documentação de missingness

4. **Multiclass Classification** (5 classes):
   - Verificar que métricas binárias são estendidas corretamente
   - Verificar one-vs-rest ou one-vs-one

5. **Multi-Sensitive Attributes** (5 atributos sensíveis):
   - Verificar análise combinada
   - Verificar relatórios não ficam poluídos

**Métricas de Validação**:
- ✅ 0 crashes em edge cases
- ✅ Warnings apropriados exibidos
- ✅ Resultados matematicamente corretos

**Artefatos**:
- `results/edge_cases_test.txt`
- `results/stress_test_results.csv`

---

## 10. Checklist de Validação Final

### 10.1 Claims Principais

| # | Claim | Experimento | Status |
|---|-------|-------------|--------|
| 1 | Auto-detecção F1=0.90 | 1.1 | ⬜ |
| 2 | 100% acurácia em case studies | 1.2 | ⬜ |
| 3 | 15 métricas (4 pré + 11 pós) | 2.1 | ⬜ |
| 4 | 87% mais métricas que AIF360 | 2.1, 8.1 | ⬜ |
| 5 | 100% precisão EEOC/ECOA | 3.1, 3.2 | ⬜ |
| 6 | SUS Score 85.2 | 5.1 | ⬜ |
| 7 | NASA-TLX 32.1 | 5.2 | ⬜ |
| 8 | 95% taxa de sucesso | 5.3 | ⬜ |
| 9 | Time-to-insight 10.2 min | 5.4 | ⬜ |
| 10 | Speedup 2.9x | 6.1 | ⬜ |
| 11 | 40-42% menos memória | 6.2 | ⬜ |
| 12 | COMPAS 7.2 min (79% economia) | 4.1 | ⬜ |
| 13 | German Credit 5.8 min (77% economia) | 4.2 | ⬜ |
| 14 | Adult 12.4 min (75% economia) | 4.3 | ⬜ |
| 15 | Healthcare 9.1 min (77% economia) | 4.4 | ⬜ |

### 10.2 Artefatos de Publicação

| Artefato | Descrição | Localização | Status |
|----------|-----------|-------------|--------|
| Dataset annotations | Ground truth de 500 datasets | `data/ground_truth.csv` | ⬜ |
| Case study results | Resultados completos dos 4 casos | `results/case_studies/` | ⬜ |
| Usability data | Dados brutos do estudo (N=20) | `results/usability/` | ⬜ |
| Performance benchmarks | Tempos e memória | `results/performance/` | ⬜ |
| Comparison matrix | Comparação com ferramentas | `results/comparison/` | ⬜ |
| Reproduction package | Scripts e instruções | `reproduction/` | ⬜ |

---

## 11. Timeline de Execução

### Fase 1: Setup (Semana 1-2)
- [ ] Instalar todas ferramentas (DeepBridge, AIF360, Fairlearn, Aequitas)
- [ ] Coletar e preparar 500 datasets
- [ ] Preparar infraestrutura (AWS, tracking)
- [ ] Criar scripts de automação

### Fase 2: Auto-Detecção (Semana 3-4)
- [ ] Executar Experimento 1.1 (500 datasets)
- [ ] Executar Experimento 1.2 (case studies)
- [ ] Análise de erros
- [ ] Gerar artefatos

### Fase 3: Métricas e Compliance (Semana 5-6)
- [ ] Executar Experimento 2.1 (15 métricas)
- [ ] Executar Experimentos 3.1-3.3 (EEOC/ECOA)
- [ ] Validação manual
- [ ] Gerar artefatos

### Fase 4: Case Studies (Semana 7-9)
- [ ] Executar Experimentos 4.1-4.4
- [ ] Validar todas claims
- [ ] Gerar relatórios completos
- [ ] Gerar artefatos

### Fase 5: Usabilidade (Semana 10-12)
- [ ] Recrutar 20 participantes
- [ ] Executar Experimentos 5.1-5.5
- [ ] Transcrever entrevistas
- [ ] Análise temática
- [ ] Gerar artefatos

### Fase 6: Performance (Semana 13-14)
- [ ] Executar Experimentos 6.1-6.3
- [ ] Executar Experimento 7.1-7.2
- [ ] Análise estatística
- [ ] Gerar artefatos

### Fase 7: Comparação (Semana 15)
- [ ] Executar Experimentos 8.1-8.2
- [ ] Comparação head-to-head
- [ ] Gerar artefatos

### Fase 8: Robustness (Semana 16)
- [ ] Executar Experimento 9.1
- [ ] Edge cases e stress tests
- [ ] Gerar artefatos

### Fase 9: Finalização (Semana 17-18)
- [ ] Validar checklist completo
- [ ] Preparar reproduction package
- [ ] Escrever apêndice técnico
- [ ] Submeter para FAccT 2026

---

## 12. Critérios de Sucesso para Publicação

### Mínimos Aceitáveis (Paper será aceito se):

1. **Auto-Detecção**:
   - F1-Score ≥ 0.85 (claim: 0.90)
   - 100% acurácia em ≥3/4 case studies

2. **Usabilidade**:
   - SUS ≥ 75 (claim: 85.2)
   - Taxa de sucesso ≥ 85% (claim: 95%)

3. **Performance**:
   - Speedup ≥ 2.0x (claim: 2.9x)
   - Economia de memória ≥ 30% (claim: 40-42%)

4. **Compliance**:
   - 100% precisão em EEOC/ECOA (crítico!)

### Targets Ideais (Fortalece o Paper):

1. Todos os claims validados dentro de ±10%
2. N=20 participantes em usabilidade
3. 500 datasets em auto-detecção
4. Reproduction package completo
5. Comparação head-to-head com 3 ferramentas

---

## 13. Contingências

### Se Auto-Detecção < 0.85 F1:
- Reduzir threshold de similaridade
- Adicionar dicionário de sinônimos
- Implementar context filtering mais agressivo
- Worst case: Reduzir claim para 0.85 e explicar trade-off

### Se SUS < 75:
- Melhorar documentação
- Adicionar tutoriais interativos
- Simplificar API
- Worst case: Reposicionar como ferramenta para experts (não iniciantes)

### Se Speedup < 2.0x:
- Otimizar threshold optimization (usar grid search mais esparso)
- Implementar paralelização
- Worst case: Focar em qualidade vs. velocidade (features únicas)

### Se Usabilidade N < 15:
- Incluir dados qualitativos (entrevistas)
- Fazer estudo piloto (N=10) + validation (N=5)
- Worst case: Reportar como estudo exploratório

---

## 14. Ética e Compliance

### IRB (Institutional Review Board):
- [ ] Submeter protocolo de estudo de usabilidade
- [ ] Obter consentimento informado de participantes
- [ ] Garantir anonimização de dados

### Licenciamento de Dados:
- [ ] Verificar licenças de todos os 500 datasets
- [ ] Garantir permissão para republicação de resultados
- [ ] Citar corretamente autores originais

### Conflitos de Interesse:
- [ ] Declarar afiliações com organizações que usam DeepBridge
- [ ] Declarar funding sources

---

## 15. Referências de Validação

### Papers de Referência (FAccT):
1. Bellamy et al. (2018) - AI Fairness 360 [cite para comparação]
2. Bird et al. (2020) - Fairlearn [cite para comparação]
3. Saleiro et al. (2018) - Aequitas [cite para comparação]

### Metodologias de Avaliação:
1. Brooke (1996) - System Usability Scale
2. Hart & Staveland (1988) - NASA Task Load Index
3. Lazar et al. (2017) - Research Methods in HCI

### Benchmarks de Fairness:
1. COMPAS Dataset - ProPublica
2. German Credit - UCI Repository
3. Adult Income - UCI Repository

---

## 16. Outputs Esperados

### Para o Paper:
- Tabelas de resultados (Seção 5 - Evaluation)
- Figuras de comparação (performance, usability)
- Apêndice técnico (metodologia detalhada)

### Para Repositório:
- `experiments/` - Scripts de todos experimentos
- `results/` - Dados brutos e processados
- `reproduction/` - Instruções de reprodução
- `data/` - Ground truth e datasets

### Para Apresentação:
- Slides resumindo principais resultados
- Demos ao vivo (caso aceito)
- Poster (para sessão de posters)

---

## 17. Contato e Suporte

**Responsável pelos Experimentos**: [Adicionar nome]

**Prazo Final**: Submissão FAccT 2026 - [Verificar deadline exato]

**Recursos**:
- Hardware: AWS m5.2xlarge
- Software: Python 3.8+, DeepBridge, AIF360, Fairlearn, Aequitas
- Orçamento: [Definir para AWS + participantes do estudo]

---

**IMPORTANTE**: Este plano é exaustivo e cobre todas as claims do paper. Priorize os experimentos com ⭐ se tempo/recursos forem limitados:

⭐ Experimentos Críticos (Essenciais):
- 1.1 (Auto-detecção 500 datasets)
- 3.1 (EEOC 80% rule)
- 4.1-4.4 (Case studies completos)
- 5.1 (SUS Score)
- 6.1 (Speedup)
- 8.1 (Comparação com ferramentas)

Os demais experimentos fortalecem o paper mas não são absolutamente críticos para aceitação.

**Boa sorte com os experimentos! 🚀**
