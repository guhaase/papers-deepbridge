# Experimento 4: HPM-KD Framework

## Objetivo

Comprovar os resultados do framework HPM-KD (Hierarchical Progressive Multi-Teacher Knowledge Distillation) apresentados no paper.

## Afirmações a Comprovar

### Resultados Principais (Seção 5.3, Tabela)

| Método | Acurácia | Compressão | Latência | Status |
|--------|----------|------------|----------|--------|
| Teacher Ensemble | 87.2% | 1.0× | 125ms | ⏳ Pendente |
| Vanilla KD | 82.5% | 10.2× | 12ms | ⏳ Pendente |
| TAKD | 83.8% | 10.1× | 13ms | ⏳ Pendente |
| Auto-KD | 84.4% | 10.3× | 12ms | ⏳ Pendente |
| **HPM-KD** | **85.8%** | **10.3×** | **12ms** | ⏳ Pendente |

### Métricas Derivadas
- **Retenção de acurácia**: 98.4% (85.8% / 87.2%)
- **Compressão de modelo**: 10.3× (2.4GB → 230MB)
- **Speedup de latência**: 10.4× (125ms → 12ms)
- **Redução de custo**: 10× (\$0.05 → \$0.005 por 1K predições)
- **Throughput**: 10× (8 req/s → 83 req/s)

## Metodologia

### 1. Datasets

**Total**: 20 datasets tabulares UCI/OpenML

#### Datasets Sugeridos

**Classificação Binária** (10 datasets):
1. Adult Income
2. Bank Marketing
3. Credit Card Default
4. Credit-g (German Credit)
5. Cylinder Bands
6. Diabetes
7. Heart Disease
8. Ionosphere
9. Mushroom
10. SPECT Heart

**Classificação Multi-Classe** (10 datasets):
1. Car Evaluation
2. Chess (King-Rook vs King-Pawn)
3. Connect-4
4. Letter Recognition
5. Nursery
6. Page Blocks
7. Pen Digits
8. Satimage
9. Vehicle
10. Wine Quality

#### Critérios de Seleção
- Tamanho: 1.000 - 100.000 amostras
- Features: Mix de numéricas e categóricas
- Balance: Incluir datasets balanceados e desbalanceados
- Domínios diversos: finanças, saúde, reconhecimento, jogos

### 2. Modelos Teacher

Para cada dataset, treinar ensemble de 3 modelos:

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Teacher 1: XGBoost
teacher_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)

# Teacher 2: LightGBM
teacher_lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)

# Teacher 3: CatBoost
teacher_catboost = CatBoostClassifier(
    iterations=200,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=False
)

# Treinar todos
teachers = [teacher_xgb, teacher_lgbm, teacher_catboost]
for teacher in teachers:
    teacher.fit(X_train, y_train)

# Ensemble: Soft voting
from sklearn.ensemble import VotingClassifier
teacher_ensemble = VotingClassifier(
    estimators=[
        ('xgb', teacher_xgb),
        ('lgbm', teacher_lgbm),
        ('catboost', teacher_catboost)
    ],
    voting='soft'
)
```

**Métricas do Ensemble**:
- Acurácia média esperada: **87.2%** (agregado sobre 20 datasets)
- Tamanho médio: **2.4GB**
- Latência média: **125ms**

### 3. Modelos Student

**Arquitetura**: Multi-Layer Perceptron (MLP) compacto

```python
from sklearn.neural_network import MLPClassifier

# Student: MLP compacto
student = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 camadas
    activation='relu',
    solver='adam',
    batch_size=256,
    max_iter=100,
    random_state=42
)
```

**Tamanho esperado**: ~230MB (10.3× menor que ensemble)

### 4. Baselines de Destilação

#### Vanilla KD (Hinton et al., 2015)

```python
from sklearn.neural_network import MLPClassifier
import numpy as np

def vanilla_kd(teacher, student, X_train, y_train, temperature=3.0, alpha=0.5):
    # Soft labels do teacher
    soft_labels = teacher.predict_proba(X_train)

    # Hard labels
    hard_labels = y_train

    # Loss combinada (simplificado - na prática usar framework deep learning)
    # L = alpha * L_hard + (1-alpha) * L_soft

    # Treinar student
    student.fit(X_train, y_train)  # simplificado

    return student
```

**Resultado esperado**: 82.5% acurácia, 10.2× compressão

#### TAKD (Teacher Assistant Knowledge Distillation)

```python
def takd(teacher, X_train, y_train, num_stages=2):
    # Stage 1: Teacher -> Assistant
    assistant = MLPClassifier(hidden_layer_sizes=(128, 64))
    assistant = vanilla_kd(teacher, assistant, X_train, y_train)

    # Stage 2: Assistant -> Student
    student = MLPClassifier(hidden_layer_sizes=(64, 32))
    student = vanilla_kd(assistant, student, X_train, y_train)

    return student
```

**Resultado esperado**: 83.8% acurácia, 10.1× compressão

#### Auto-KD (Automated Knowledge Distillation)

```python
from sklearn.model_selection import RandomizedSearchCV

def auto_kd(teacher, student, X_train, y_train):
    # Grid search sobre hiperparâmetros de destilação
    param_dist = {
        'temperature': [1.0, 2.0, 3.0, 5.0, 10.0],
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
        'hidden_layer_sizes': [(64, 32), (128, 64), (64,)]
    }

    # Random search
    # (implementação completa requer framework customizado)

    return best_student
```

**Resultado esperado**: 84.4% acurácia, 10.3× compressão

### 5. HPM-KD Implementation

```python
from deepbridge.compression import HPMKD

# Configurar HPM-KD
hpmkd = HPMKD(
    teachers=[teacher_xgb, teacher_lgbm, teacher_catboost],
    student_architecture=(64, 32),
    num_progressive_stages=3,
    use_attention_weighting=True,
    use_meta_temperature=True,
    use_parallel_processing=True,
    random_state=42
)

# Treinar student via destilação progressiva
hpmkd.fit(X_train, y_train)

# Obter student final
student_hpmkd = hpmkd.student_

# Avaliar
accuracy = student_hpmkd.score(X_test, y_test)
```

### 6. Componentes do HPM-KD a Validar

#### 6.1. Adaptive Configuration Manager

**Afirmação**: Seleciona hiperparâmetros via meta-aprendizado

**Validação**:
```python
# Comparar HPM-KD com config automática vs. manual
configs_to_test = [
    'auto',  # meta-learning
    {'temperature': 3.0, 'alpha': 0.5, 'stages': 2},  # manual 1
    {'temperature': 5.0, 'alpha': 0.7, 'stages': 3},  # manual 2
]

results = {}
for config in configs_to_test:
    hpmkd = HPMKD(config=config)
    hpmkd.fit(X_train, y_train)
    results[config] = hpmkd.score(X_test, y_test)

# Esperado: config 'auto' ≥ configs manuais
```

#### 6.2. Progressive Distillation Chain

**Afirmação**: Refina student incrementalmente através de múltiplos estágios

**Validação**:
```python
# Comparar 1 estágio vs. 2 estágios vs. 3 estágios
num_stages_to_test = [1, 2, 3, 4]

results = {}
for num_stages in num_stages_to_test:
    hpmkd = HPMKD(num_progressive_stages=num_stages)
    hpmkd.fit(X_train, y_train)
    results[num_stages] = hpmkd.score(X_test, y_test)

# Esperado: acurácia cresce com estágios, plateau em 3-4
```

#### 6.3. Attention-Weighted Multi-Teacher

**Afirmação**: Ensemble com pesos de atenção aprendidos

**Validação**:
```python
# Comparar:
# - Uniform weighting (todos teachers peso igual)
# - Fixed weighting (baseado em acurácia)
# - Attention weighting (aprendido)

weighting_schemes = ['uniform', 'fixed', 'attention']

results = {}
for scheme in weighting_schemes:
    hpmkd = HPMKD(weighting_scheme=scheme)
    hpmkd.fit(X_train, y_train)
    results[scheme] = hpmkd.score(X_test, y_test)

# Esperado: attention ≥ fixed ≥ uniform
```

#### 6.4. Meta-Temperature Scheduler

**Afirmação**: Temperatura adaptativa baseada em dificuldade da tarefa

**Validação**:
```python
# Comparar temperatura fixa vs. adaptativa
temperature_modes = [
    'fixed_3.0',
    'fixed_5.0',
    'adaptive_meta_learned'
]

results = {}
for mode in temperature_modes:
    hpmkd = HPMKD(temperature_mode=mode)
    hpmkd.fit(X_train, y_train)
    results[mode] = hpmkd.score(X_test, y_test)

# Esperado: adaptativa ≥ fixas
```

#### 6.5. Parallel Processing Pipeline

**Afirmação**: Speedup via paralelização

**Validação**:
```python
import time

# Medir tempo com diferentes números de workers
num_workers_to_test = [1, 2, 4, 8]

results = {}
for num_workers in num_workers_to_test:
    hpmkd = HPMKD(n_jobs=num_workers)

    start = time.time()
    hpmkd.fit(X_train, y_train)
    elapsed = time.time() - start

    results[num_workers] = {
        'time': elapsed,
        'speedup': results.get(1, {}).get('time', elapsed) / elapsed
    }

# Esperado: speedup quase linear até 4 workers
```

## Métricas de Avaliação

### Por Dataset

```python
metrics = {
    'teacher_accuracy': float,
    'student_accuracy': float,
    'retention_rate': float,  # student / teacher
    'teacher_size_mb': float,
    'student_size_mb': float,
    'compression_ratio': float,  # teacher / student
    'teacher_latency_ms': float,
    'student_latency_ms': float,
    'latency_speedup': float,  # teacher / student
}
```

### Agregadas (20 datasets)

```python
# Média, mediana, std, min, max
aggregated_metrics = {
    'mean_teacher_accuracy': float,
    'mean_student_accuracy': float,
    'mean_retention_rate': float,
    'mean_compression_ratio': float,
    'mean_latency_speedup': float,
}
```

## Resultados Esperados

### Acurácia (média de 20 datasets)

| Método | Média | Std | Min | Max |
|--------|-------|-----|-----|-----|
| Teacher Ensemble | 87.2% | 5.2% | 78.5% | 95.1% |
| Vanilla KD | 82.5% | 5.8% | 72.1% | 90.3% |
| TAKD | 83.8% | 5.5% | 74.2% | 91.5% |
| Auto-KD | 84.4% | 5.3% | 75.1% | 92.0% |
| **HPM-KD** | **85.8%** | **5.1%** | **76.8%** | **93.2%** |

### Retenção de Acurácia

| Método | Média | Std |
|--------|-------|-----|
| Vanilla KD | 94.7% | 2.1% |
| TAKD | 96.1% | 1.8% |
| Auto-KD | 96.8% | 1.5% |
| **HPM-KD** | **98.4%** | **1.2%** |

### Compressão

| Métrica | Teacher | Student | Ratio |
|---------|---------|---------|-------|
| Tamanho (MB) | 2400 | 230 | 10.3× |
| Parâmetros | ~50M | ~5M | 10.0× |

### Latência (1000 predições em batch)

| Modelo | Latência Média | Std |
|--------|----------------|-----|
| Teacher Ensemble | 125ms | 15ms |
| Student HPM-KD | 12ms | 2ms |
| **Speedup** | **10.4×** | - |

### Throughput

| Modelo | Requests/s |
|--------|------------|
| Teacher Ensemble | 8 req/s |
| Student HPM-KD | 83 req/s |
| **Aumento** | **10.4×** |

## Análise Estatística

### Comparação HPM-KD vs. Baselines

**Teste**: Paired t-test (pareado por dataset)

```python
from scipy import stats

# Para cada baseline
baselines = ['vanilla_kd', 'takd', 'auto_kd']

for baseline in baselines:
    # Acurácias em 20 datasets
    acc_hpmkd = [...]  # 20 valores
    acc_baseline = [...]  # 20 valores

    t_stat, p_value = stats.ttest_rel(acc_hpmkd, acc_baseline)

    print(f"HPM-KD vs {baseline}: t={t_stat:.2f}, p={p_value:.4f}")

# Esperado: p < 0.01 para todos
```

### Ablation Study

**Objetivo**: Quantificar contribuição de cada componente

```python
# Versões ablacionadas
ablations = {
    'full': HPMKD(all_features=True),
    'no_progressive': HPMKD(num_stages=1),
    'no_attention': HPMKD(weighting='uniform'),
    'no_meta_temp': HPMKD(temperature='fixed'),
    'no_parallel': HPMKD(n_jobs=1),
}

results = {}
for name, model in ablations.items():
    model.fit(X_train, y_train)
    results[name] = model.score(X_test, y_test)

# Calcular contribuição de cada componente
contributions = {}
full_acc = results['full']
for name, acc in results.items():
    if name != 'full':
        contributions[name] = full_acc - acc

# Esperado:
# - Progressive: ~1.5% contribuição
# - Attention: ~0.8% contribuição
# - Meta-temp: ~0.5% contribuição
# - Parallel: 0% (só afeta tempo)
```

## Ambiente de Execução

### Hardware
- **CPU**: Intel i7-12700K (12 cores) ou similar
- **GPU**: NVIDIA RTX 3080 (para acelerar training do MLP)
- **RAM**: 32GB
- **Storage**: SSD 500GB

### Software
- **Python**: 3.10
- **PyTorch**: 2.0 (para MLP student)
- **XGBoost**: 1.7
- **LightGBM**: 3.3
- **CatBoost**: 1.2
- **Scikit-learn**: 1.3

## Scripts

### Principal
`/experimentos/scripts/04_hpmkd_main.py`

### Baselines
`/experimentos/scripts/04_hpmkd_baselines.py`

### Ablation Study
`/experimentos/scripts/04_hpmkd_ablation.py`

### Análise
`/experimentos/notebooks/04_hpmkd_analysis.ipynb`

## Outputs

### Por Dataset
- `results/04_hpmkd_{dataset_name}_metrics.json`

### Agregados
- `results/04_hpmkd_aggregated_results.csv`
- `results/04_hpmkd_statistical_tests.json`

### Figuras
- `figures/hpmkd_accuracy_comparison.pdf`
- `figures/hpmkd_retention_rates.pdf`
- `figures/hpmkd_compression_latency.pdf`
- `figures/hpmkd_ablation_study.pdf`

### Tabelas
- `tables/hpmkd_results.tex`

## Checklist

- [ ] Obter 20 datasets UCI/OpenML
- [ ] Pré-processar datasets (train/test split, encoding)
- [ ] Treinar teacher ensembles para cada dataset
- [ ] Medir tamanho e latência dos teachers
- [ ] Implementar Vanilla KD baseline
- [ ] Implementar TAKD baseline
- [ ] Implementar Auto-KD baseline
- [ ] Executar HPM-KD em todos datasets
- [ ] Validar componente: Adaptive Config
- [ ] Validar componente: Progressive Distillation
- [ ] Validar componente: Attention Weighting
- [ ] Validar componente: Meta-Temperature
- [ ] Validar componente: Parallel Processing
- [ ] Executar ablation study
- [ ] Calcular métricas agregadas
- [ ] Realizar testes estatísticos
- [ ] Gerar visualizações
- [ ] Formatar tabelas LaTeX

## Prioridade

🔴 **ALTA** - HPM-KD é uma contribuição central do paper

## Tempo Estimado

**3-4 semanas**:
- Semana 1: Preparação de datasets e training de teachers
- Semana 2: Implementação e execução de baselines
- Semana 3: Execução de HPM-KD e ablation studies
- Semana 4: Análise de resultados e geração de visualizações
