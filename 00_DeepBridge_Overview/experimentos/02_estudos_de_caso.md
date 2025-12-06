# Experimento 2: Estudos de Caso em 6 Domínios

## Objetivo

Comprovar os resultados apresentados na **Tabela 3: Resultados dos Estudos de Caso** do paper, demonstrando a aplicação de DeepBridge em cenários reais de produção.

## Afirmações a Comprovar

| Domínio | Amostras | Violações | Tempo | Achado Principal | Status |
|---------|----------|-----------|-------|------------------|--------|
| Crédito | 1.000 | 2 | 17 min | DI=0.74 (gênero) | ⏳ Pendente |
| Contratação | 7.214 | 1 | 12 min | DI=0.59 (raça) | ⏳ Pendente |
| Saúde | 101.766 | 0 | 23 min | Bem calibrado | ⏳ Pendente |
| Hipoteca | 450.000 | 1 | 45 min | Violação ECOA | ⏳ Pendente |
| Seguros | 595.212 | 0 | 38 min | Passa todos testes | ⏳ Pendente |
| Fraude | 284.807 | 0 | 31 min | Alta resiliência | ⏳ Pendente |

### Estatísticas Agregadas
- **Tempo médio**: 27.7 minutos
- **Violações detectadas**: 4/6 casos
- **Precisão de detecção**: 100%
- **Falsos positivos**: 0

---

## Estudo de Caso 1: Credit Scoring

### Contexto (Seção 2.1)
- Instituição financeira, aprovação de crédito pessoal
- Modelo: XGBoost
- Volume: 50.000+ aplicações/mês
- Tempo de validação: **17 minutos**

### Dataset
**Opção 1: Dataset Público**
- **Nome**: German Credit Data (UCI)
- **URL**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- **Tamanho Original**: 1.000 amostras
- **Features**: 20 features (7 numéricas, 13 categóricas)
- **Target**: Binário (good/bad credit)
- **Atributos Protegidos**: age, sex

**Opção 2: Dataset Sintético**
- Gerar 1.000 amostras sintéticas usando Gaussian Copula do DeepBridge
- Manter distribuições realistas

### Modelo
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
```

### Experimento

```python
from deepbridge import DBDataset, Experiment
import time

# Criar dataset
dataset = DBDataset(
    data=df_test,
    target_column='credit_risk',
    model=model,
    protected_attributes=['gender', 'age']
)

# Validação completa
start_time = time.time()
exp = Experiment(dataset, tests='all')
results = exp.run_tests()
validation_time = time.time() - start_time

# Verificar detecções esperadas
fairness_results = results['fairness']
assert fairness_results['disparate_impact']['gender'] < 0.80  # Violação 1
assert fairness_results['eeoc_compliance']['gender'] == 'FAIL'  # Violação 2
```

### Resultados Esperados

**Violações Detectadas**:
1. **Disparate Impact (DI) para gênero**: DI = 0.74 (< 0.80) ✗
2. **Violação regra 80% EEOC** ✗

**Análise de Subgrupos**:
- Subgrupo vulnerável: Mulheres com idade < 25 anos e valor > $5.000
- Acurácia no subgrupo: 0.62
- Acurácia global: 0.85
- Gap: 0.23

**Tempo**:
- Tempo total: ~17 minutos (±1 min)
- Fairness: ~5 min
- Robustez: ~7 min
- Incerteza: ~3 min
- Resiliência: ~2 min

**Relatório**:
- PDF de 12 páginas
- Visualizações: confusion matrix, DI por grupo, calibration plot
- Recomendações: re-ponderação, threshold adjustment

### Script
`/experimentos/scripts/02_case_study_credit.py`

---

## Estudo de Caso 2: Contratação (Hiring)

### Contexto (Seção 2.2)
- Empresa de tecnologia
- Sistema de triagem automatizada de currículos
- Modelo: Random Forest
- Volume: 10.000+ candidatos/ano
- Tempo de validação: **12 minutos**

### Dataset
**Opção 1: Dataset Público**
- **Nome**: Adult Income Dataset adaptado para hiring
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult
- **Tamanho**: 7.214 amostras (subset)
- **Features**: education, occupation, workclass, etc.
- **Target**: Binário (hired/not hired)
- **Atributos Protegidos**: race, sex, age

**Opção 2: Dataset Sintético**
- Gerar dados de candidatos com distribuição realista

### Modelo
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
```

### Experimento

```python
from deepbridge import DBDataset, Experiment

dataset = DBDataset(
    data=df_test,
    target_column='hired',
    model=model,
    protected_attributes=['race', 'sex', 'age']
)

# Validação
exp = Experiment(dataset, tests='all')
results = exp.run_tests()

# Verificações
fairness_results = results['fairness']
assert fairness_results['disparate_impact']['race'] < 0.80  # DI = 0.59
assert fairness_results['eeoc_compliance']['race'] == 'FAIL'
assert fairness_results['question_21']['race'] == 'PASS'  # Rep. ≥ 2%
```

### Resultados Esperados

**Violações Detectadas**:
1. **Disparate Impact para raça**: DI = 0.59 (< 0.80) ✗

**Conformidade**:
- Question 21 EEOC: PASS (todos grupos ≥ 2% representação)
- Regra 80%: FAIL para raça

**Teste de Robustez**:
- Perturbações testadas: typos, formatos alternativos
- Performance mantida: ✓

**Tempo**:
- Tempo total: ~12 minutos (±1 min)

**Relatório**:
- Adverse action notices gerados automaticamente
- Aprovado por equipe jurídica

### Script
`/experimentos/scripts/02_case_study_hiring.py`

---

## Estudo de Caso 3: Saúde (Healthcare)

### Contexto (Seção 2.3)
- Hospital universitário
- Modelo de priorização para triagem de emergência
- Predição: risco de complicações graves em 24h
- Volume: 800+ pacientes/dia
- Tempo de validação: **23 minutos**
- Amostras de validação: **101.766**

### Dataset
**Opção 1: Dataset Público**
- **Nome**: MIMIC-III Clinical Database (subset)
- **URL**: https://physionet.org/content/mimiciii/
- **Tamanho**: 101.766 amostras
- **Features**: sinais vitais, laboratório, demografia
- **Target**: Complicações em 24h (binário)
- **Atributos Protegidos**: ethnicity, gender, age

**Opção 2: Dataset Sintético**
- Gerar dados clínicos sintéticos com distribuição realista

### Modelo
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)
```

### Experimento

```python
from deepbridge import DBDataset, Experiment

dataset = DBDataset(
    data=df_test,
    target_column='complication_24h',
    model=model,
    protected_attributes=['ethnicity', 'gender', 'age_group']
)

# Validação completa
exp = Experiment(dataset, tests='all')
results = exp.run_tests()

# Verificações
fairness_results = results['fairness']
assert all(results['fairness']['equal_opportunity'].values() > 0.80)  # PASS

uncertainty_results = results['uncertainty']
assert uncertainty_results['ece'] < 0.05  # ECE = 0.042

conformal_results = results['conformal_prediction']
assert conformal_results['coverage'] >= 0.95  # 95% cobertura
```

### Resultados Esperados

**Violações Detectadas**: 0 ✓

**Fairness**:
- Equal Opportunity em 4 grupos étnicos: PASS
- Equal Opportunity em 2 gêneros: PASS
- Equal Opportunity em 5 faixas etárias: PASS

**Calibração**:
- ECE (Expected Calibration Error): 0.042 (< 0.05) ✓
- Confiável para decisões médicas

**Predição Conformal**:
- Intervalos com 95% cobertura garantida
- Coverage real: 95.2%

**Robustez**:
- Perturbações em sinais vitais: ±5%
- Performance mantida

**Drift Detection**:
- Monitoramento contínuo configurado
- PSI, KL divergence

**Tempo**:
- Tempo total: ~23 minutos (±2 min)
- (maior devido ao tamanho do dataset: 101.766 amostras)

**Aprovação**:
- Comitê de ética médica: aprovado
- 0 violações detectadas em produção

### Script
`/experimentos/scripts/02_case_study_healthcare.py`

---

## Estudo de Caso 4: Hipoteca (Mortgage)

### Contexto
- Instituição financeira de grande porte
- Aprovação de empréstimos hipotecários
- Modelo: Gradient Boosting
- Tempo de validação: **45 minutos**
- Amostras: **450.000**

### Dataset
**Opção 1: Dataset Público**
- **Nome**: Home Mortgage Disclosure Act (HMDA) Data
- **URL**: https://www.consumerfinance.gov/data-research/hmda/
- **Tamanho**: 450.000 amostras (subset)
- **Features**: loan amount, income, property type, etc.
- **Target**: Aprovação (approved/denied)
- **Atributos Protegidos**: race, ethnicity, gender

**Opção 2: Dataset Sintético**
- Gerar dados de hipoteca sintéticos

### Resultados Esperados

**Violações Detectadas**: 1
- Violação ECOA (detalhes a definir)

**Tempo**:
- Tempo total: ~45 minutos (±3 min)
- (maior devido ao tamanho: 450.000 amostras)

### Script
`/experimentos/scripts/02_case_study_mortgage.py`

---

## Estudo de Caso 5: Seguros (Insurance)

### Contexto
- Companhia de seguros
- Precificação e subscrição
- Modelo: XGBoost
- Tempo de validação: **38 minutos**
- Amostras: **595.212**

### Dataset
**Opção 1: Dataset Público**
- **Nome**: Porto Seguro Safe Driver Prediction
- **URL**: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
- **Tamanho**: 595.212 amostras
- **Features**: características do segurado e veículo
- **Target**: Sinistro (claim/no claim)

### Resultados Esperados

**Violações Detectadas**: 0 ✓
- Passa todos os testes

**Tempo**:
- Tempo total: ~38 minutos (±3 min)

### Script
`/experimentos/scripts/02_case_study_insurance.py`

---

## Estudo de Caso 6: Detecção de Fraude

### Contexto
- Instituição financeira
- Detecção de transações fraudulentas
- Modelo: LightGBM
- Tempo de validação: **31 minutos**
- Amostras: **284.807**

### Dataset
**Opção 1: Dataset Público**
- **Nome**: Credit Card Fraud Detection
- **URL**: https://www.kaggle.com/mlg-ulb/creditcardfraud
- **Tamanho**: 284.807 amostras
- **Features**: PCA features (anonimizadas)
- **Target**: Fraude (fraud/legitimate)

### Resultados Esperados

**Violações Detectadas**: 0 ✓
- Alta resiliência a drift
- Bem calibrado

**Tempo**:
- Tempo total: ~31 minutos (±2 min)

### Script
`/experimentos/scripts/02_case_study_fraud.py`

---

## Análise Agregada

### Estatísticas de Tempo

```python
import numpy as np

times = [17, 12, 23, 45, 38, 31]  # minutos
print(f"Média: {np.mean(times):.1f} min")  # 27.7 min
print(f"Std: {np.std(times):.1f} min")
print(f"Min: {np.min(times)} min")
print(f"Max: {np.max(times)} min")
```

**Esperado**: Média = 27.7 minutos

### Precisão de Detecção

**Violações Reais**: 4 casos (Crédito tem 2 violações, Contratação 1, Hipoteca 1)
**Violações Detectadas**: 4 ✓
**Falsos Positivos**: 0 ✓
**Precisão**: 100%
**Recall**: 100%

### Aprovação de Relatórios

- **Relatórios gerados**: 6
- **Aprovados sem modificações**: 6
- **Taxa de aprovação**: 100%

## Outputs

### Por Caso de Uso
1. **Resultados Brutos**: `results/02_case_study_{domain}_results.json`
2. **Relatórios PDF**: `results/02_case_study_{domain}_report.pdf`
3. **Métricas de Tempo**: `results/02_case_study_{domain}_times.csv`

### Agregados
1. **Tabela LaTeX**: `tables/case_studies_summary.tex`
2. **Análise Estatística**: `results/02_case_studies_analysis.json`
3. **Visualizações**:
   - `figures/case_studies_times.pdf`
   - `figures/case_studies_violations.pdf`

## Checklist

- [ ] Obter/gerar dataset para Credit Scoring
- [ ] Obter/gerar dataset para Hiring
- [ ] Obter/gerar dataset para Healthcare
- [ ] Obter/gerar dataset para Mortgage
- [ ] Obter/gerar dataset para Insurance
- [ ] Obter/gerar dataset para Fraud Detection
- [ ] Treinar modelo para cada domínio
- [ ] Executar validação DeepBridge para cada caso
- [ ] Medir tempos precisos
- [ ] Validar violações detectadas
- [ ] Gerar relatórios PDF
- [ ] Calcular estatísticas agregadas
- [ ] Formatar tabela em LaTeX
- [ ] Gerar visualizações

## Prioridade

🔴 **ALTA** - Estes são os principais resultados práticos do paper

## Tempo Estimado

**4-6 semanas**:
- Semana 1-2: Obtenção/geração de datasets e treinamento de modelos
- Semana 3-4: Execução de validações e coleta de métricas
- Semana 5-6: Análise de resultados e geração de relatórios
