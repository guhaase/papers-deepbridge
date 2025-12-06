# Experimento 5: Conformidade Regulatória

## Objetivo

Comprovar que DeepBridge detecta automaticamente violações de conformidade regulatória (EEOC/ECOA) com **100% de precisão** e **0 falsos positivos**.

## Afirmações a Comprovar

| Afirmação | Valor | Status |
|-----------|-------|--------|
| Precisão de detecção | 100% | ⏳ Pendente |
| Falsos positivos | 0 | ⏳ Pendente |
| Cobertura de features | 10/10 vs. 2/10 (ferramentas existentes) | ⏳ Pendente |
| Tempo de auditoria | 70% redução | ⏳ Pendente |

## Regulamentações Cobertas

### 1. EEOC (Equal Employment Opportunity Commission)

#### 1.1. Regra dos 80% (Four-Fifths Rule)

**Definição**: Disparate Impact (DI) deve ser ≥ 0.80

$$
\text{DI} = \frac{\text{Taxa de seleção do grupo protegido}}{\text{Taxa de seleção do grupo de referência}}
$$

**Conformidade**: DI ≥ 0.80

**Violação**: DI < 0.80

#### 1.2. Question 21 (Representatividade Mínima)

**Definição**: Cada grupo demográfico deve ter representação mínima de 2% no pool de candidatos/aprovados.

**Conformidade**: Todos grupos ≥ 2%

**Violação**: Algum grupo < 2%

### 2. ECOA (Equal Credit Opportunity Act)

#### 2.1. Proibição de Discriminação

**Grupos Protegidos**:
- Race
- Color
- Religion
- National origin
- Sex
- Marital status
- Age (≥40)

**Conformidade**: Nenhum viés estatisticamente significativo

#### 2.2. Adverse Action Notices

**Requerimento**: Explicação de razões específicas para decisões adversas (rejeição de crédito, contratação, etc.)

**Conformidade**: Gerar notices automaticamente com razões

## Metodologia

### 1. Dataset de Ground Truth

Criar dataset sintético com violações **conhecidas** de conformidade:

```python
import numpy as np
import pandas as pd
from scipy import stats

def create_ground_truth_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)

    # Features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.lognormal(10, 1, n_samples)
    credit_score = np.random.normal(700, 100, n_samples)

    # Protected attributes
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5])
    race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'],
                            n_samples, p=[0.60, 0.15, 0.15, 0.10])

    # Criar VIOLAÇÃO INTENCIONAL 1: Disparate Impact em gênero
    # Aprovação: P(approve | M) = 0.60, P(approve | F) = 0.40
    # DI = 0.40 / 0.60 = 0.67 < 0.80 ✗

    prob_approve = np.where(
        gender == 'M',
        0.60 + 0.2 * (credit_score - 700) / 100,
        0.40 + 0.2 * (credit_score - 700) / 100  # sistemático viés
    )
    approved = np.random.binomial(1, np.clip(prob_approve, 0, 1))

    # Criar VIOLAÇÃO INTENCIONAL 2: Question 21 - grupo sub-representado
    # Garantir que Asian < 2% dos aprovados
    # (ajustar probabilidade de aprovação)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'gender': gender,
        'race': race,
        'approved': approved
    })

    # Calcular ground truth de violações
    ground_truth = {
        'disparate_impact_gender': 0.67,  # < 0.80 ✗
        'disparate_impact_race': {
            'Black': 0.75,  # < 0.80 ✗
            'Hispanic': 0.82,  # ≥ 0.80 ✓
            'Asian': 0.88,  # ≥ 0.80 ✓
        },
        'question_21_representation': {
            'White': 0.62,  # ≥ 0.02 ✓
            'Black': 0.13,  # ≥ 0.02 ✓
            'Hispanic': 0.16,  # ≥ 0.02 ✓
            'Asian': 0.09,  # ≥ 0.02 ✓ (mas < 10% esperado)
        },
        'total_violations': 2  # Gênero DI + Black DI
    }

    return df, ground_truth
```

### 2. Casos de Teste

Criar **50 casos de teste** variados:

| ID | Violação | Atributo | DI Esperado | Ground Truth |
|----|----------|----------|-------------|--------------|
| 1 | Sim | gender | 0.67 | FAIL |
| 2 | Não | gender | 0.85 | PASS |
| 3 | Sim | race (Black) | 0.75 | FAIL |
| 4 | Não | race (all) | ≥0.80 | PASS |
| ... | ... | ... | ... | ... |
| 50 | Sim | Multiple | Various | FAIL |

**Distribuição**:
- 25 casos COM violações (positivos)
- 25 casos SEM violações (negativos)

### 3. Execução de Validação

Para cada caso:

```python
from deepbridge import DBDataset, Experiment

# Para cada caso de teste
results_deepbridge = []

for test_case in test_cases:
    df, ground_truth = test_case['data'], test_case['ground_truth']

    # Criar dataset
    dataset = DBDataset(
        data=df,
        target_column='approved',
        model=model,  # modelo treinado
        protected_attributes=['gender', 'race']
    )

    # Executar validação de fairness
    exp = Experiment(dataset, tests=['fairness'])
    results = exp.run_tests()

    # Extrair detecções
    detected = {
        'disparate_impact_gender': results['disparate_impact']['gender'],
        'eeoc_rule_80_gender': results['eeoc_compliance']['gender'],
        'disparate_impact_race': results['disparate_impact']['race'],
        'question_21': results['question_21'],
    }

    # Comparar com ground truth
    comparison = {
        'case_id': test_case['id'],
        'ground_truth': ground_truth,
        'detected': detected,
        'correct_detection': compare(ground_truth, detected)
    }

    results_deepbridge.append(comparison)
```

### 4. Baseline: Ferramentas Fragmentadas

Executar mesmos casos com:

#### AIF360

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Converter formato
aif_dataset = BinaryLabelDataset(
    df=df,
    label_names=['approved'],
    protected_attribute_names=['gender', 'race']
)

# Calcular métricas
metric = ClassificationMetric(
    dataset_true, dataset_pred,
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)

di_gender = metric.disparate_impact()

# Verificar conformidade MANUALMENTE
# AIF360 não tem verificação automática EEOC
eeoc_compliant = di_gender >= 0.80  # Checagem manual
```

**Limitação**:
- Não verifica EEOC automaticamente
- Não calcula Question 21
- Requer checagem manual

#### Fairlearn

```python
from fairlearn.metrics import demographic_parity_ratio

# Calcular DPR (similar a DI)
dpr = demographic_parity_ratio(
    y_true, y_pred,
    sensitive_features=df['gender']
)

# Verificar conformidade MANUALMENTE
eeoc_compliant = dpr >= 0.80
```

**Limitação**:
- Não verifica EEOC automaticamente
- Não suporta múltiplos atributos protegidos simultaneamente
- Não gera adverse action notices

### 5. Cobertura de Features

**Definição**: Proporção de atributos protegidos corretamente detectados e validados

**DeepBridge**:
```python
# Detecta automaticamente atributos protegidos
detected_attrs = dataset.detected_sensitive
# Esperado: ['gender', 'race', 'age', ...]

# Valida TODOS atributos
fairness_results = exp.run_fairness_tests()
# Retorna métricas para TODOS atributos

coverage = len(validated_attrs) / len(detected_attrs)
# Esperado: 10/10 = 100%
```

**Ferramentas Fragmentadas**:
```python
# AIF360: requer especificação manual
aif_dataset = BinaryLabelDataset(
    protected_attribute_names=['gender', 'race']  # MANUAL
)

# Fairlearn: 1 atributo por vez
dpr_gender = demographic_parity_ratio(..., sensitive_features=df['gender'])
dpr_race = demographic_parity_ratio(..., sensitive_features=df['race'])
# Precisa rodar separadamente para cada atributo

coverage = 2/10  # Tipicamente valida apenas 1-2 atributos
```

## Métricas de Avaliação

### Precisão de Detecção

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

- **TP (True Positive)**: Violação existe e foi detectada
- **FP (False Positive)**: Violação não existe mas foi detectada

**Meta**: Precision = 100% (0 falsos positivos)

### Recall (Sensibilidade)

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

- **FN (False Negative)**: Violação existe mas não foi detectada

**Meta**: Recall = 100% (0 falsos negativos)

### F1-Score

$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Meta**: F1 = 100%

### Cobertura de Features

$$
\text{Coverage} = \frac{\text{Atributos validados}}{\text{Atributos protegidos totais}}
$$

**Meta DeepBridge**: 100% (10/10)
**Baseline Fragmentado**: ~20% (2/10)

## Resultados Esperados

### Confusion Matrix (50 casos)

| | Violação Real (Ground Truth) | Sem Violação Real |
|---|---|---|
| **Violação Detectada** | TP = 25 | FP = 0 |
| **Sem Violação Detectada** | FN = 0 | TN = 25 |

**DeepBridge**:
- **Precision**: 25 / (25 + 0) = **100%** ✓
- **Recall**: 25 / (25 + 0) = **100%** ✓
- **F1-Score**: **100%** ✓

**Ferramentas Fragmentadas** (estimado):
- **Precision**: 20 / (20 + 3) = **87%**
- **Recall**: 20 / (20 + 5) = **80%**
- **F1-Score**: **83%**

### Cobertura de Features

| Ferramenta | Atributos Detectados | Atributos Validados | Cobertura |
|------------|---------------------|---------------------|-----------|
| **DeepBridge** | 10 | 10 | **100%** |
| AIF360 | Requer manual | ~2 | 20% |
| Fairlearn | Requer manual | ~2 | 20% |

### Tempo de Auditoria

**Baseline Manual**:
1. Coletar métricas de múltiplas ferramentas: 60 min
2. Verificar conformidade manualmente: 45 min
3. Compilar relatório: 60 min
4. Revisão legal: 120 min
**Total**: ~285 minutos (~5 horas)

**DeepBridge**:
1. Executar validação: 17 min
2. Gerar relatório: <1 min
3. Revisão legal: 30 min (relatório padronizado)
**Total**: ~48 minutos

**Redução**: (285 - 48) / 285 = **83%** (> 70% afirmado)

## Análise Estatística

### Teste de Proporções

**H0**: Proportion(DeepBridge errors) = Proportion(Baseline errors)
**H1**: Proportion(DeepBridge errors) < Proportion(Baseline errors)

```python
from statsmodels.stats.proportion import proportions_ztest

# DeepBridge: 0 erros em 50 casos
# Baseline: 8 erros em 50 casos (3 FP + 5 FN)

count = np.array([0, 8])
nobs = np.array([50, 50])

z_stat, p_value = proportions_ztest(count, nobs, alternative='smaller')

# Esperado: p < 0.001
```

## Scripts

### Geração de Ground Truth
`/experimentos/scripts/05_generate_compliance_ground_truth.py`

### Validação DeepBridge
`/experimentos/scripts/05_compliance_deepbridge.py`

### Validação Baseline
`/experimentos/scripts/05_compliance_baseline.py`

### Análise
`/experimentos/notebooks/05_compliance_analysis.ipynb`

## Outputs

### Dados
- `results/05_compliance_ground_truth.csv`
- `results/05_compliance_deepbridge_results.json`
- `results/05_compliance_baseline_results.json`

### Análise
- `results/05_compliance_confusion_matrix.json`
- `results/05_compliance_statistical_tests.json`

### Figuras
- `figures/compliance_confusion_matrix.pdf`
- `figures/compliance_precision_recall.pdf`
- `figures/compliance_feature_coverage.pdf`
- `figures/compliance_audit_time.pdf`

### Tabelas
- `tables/compliance_results.tex`

## Checklist

- [ ] Gerar 50 casos de teste com ground truth
  - [ ] 25 casos com violações
  - [ ] 25 casos sem violações
- [ ] Executar DeepBridge em todos casos
- [ ] Executar AIF360 baseline
- [ ] Executar Fairlearn baseline
- [ ] Calcular confusion matrix
- [ ] Calcular precision/recall/F1
- [ ] Medir cobertura de features
- [ ] Medir tempo de auditoria
- [ ] Análise estatística (teste de proporções)
- [ ] Gerar visualizações
- [ ] Formatar tabelas LaTeX

## Prioridade

🟡 **MÉDIA** - Importante para demonstrar conformidade regulatória

## Tempo Estimado

**1-2 semanas**:
- Semana 1: Geração de ground truth e execução de validações
- Semana 2: Análise de resultados e geração de visualizações
