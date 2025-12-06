# Experimento 2: Estudos de Caso em 6 Domínios

## Objetivo

Comprovar os resultados apresentados na **Tabela 3: Resultados dos Estudos de Caso** do paper, demonstrando a aplicação de DeepBridge em cenários reais de produção em 6 domínios diferentes.

## Estrutura do Experimento

### Casos de Uso

| # | Domínio | Amostras | Violações | Tempo (min) | Script |
|---|---------|----------|-----------|-------------|--------|
| 1 | **Crédito** | 1.000 | 2 | 17 | `case_study_credit.py` |
| 2 | **Contratação** | 7.214 | 1 | 12 | `case_study_hiring.py` |
| 3 | **Saúde** | 101.766 | 0 | 23 | `case_study_healthcare.py` |
| 4 | **Hipoteca** | 450.000 | 1 | 45 | `case_study_mortgage.py` |
| 5 | **Seguros** | 595.212 | 0 | 38 | `case_study_insurance.py` |
| 6 | **Fraude** | 284.807 | 0 | 31 | `case_study_fraud.py` |

### Estatísticas Esperadas

- **Tempo médio**: 27.7 minutos
- **Violações detectadas**: 4/6 casos
- **Precisão de detecção**: 100%
- **Falsos positivos**: 0

## Estrutura de Diretórios

```
02_estudos_de_caso/
├── config/          # Configurações dos experimentos
├── scripts/         # Scripts de execução
│   ├── case_study_credit.py
│   ├── case_study_hiring.py
│   ├── case_study_healthcare.py
│   ├── case_study_mortgage.py
│   ├── case_study_insurance.py
│   ├── case_study_fraud.py
│   ├── aggregate_analysis.py
│   └── utils.py
├── results/         # Resultados em JSON/CSV
├── figures/         # Gráficos e visualizações
├── tables/          # Tabelas em LaTeX
├── logs/            # Logs de execução
└── README.md
```

## Como Executar

### 1. Instalação

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/02_estudos_de_caso
pip install -r requirements.txt
```

### 2. Executar Casos Individuais

```bash
# Caso 1: Crédito
python scripts/case_study_credit.py

# Caso 2: Contratação
python scripts/case_study_hiring.py

# Caso 3: Saúde
python scripts/case_study_healthcare.py

# Caso 4: Hipoteca
python scripts/case_study_mortgage.py

# Caso 5: Seguros
python scripts/case_study_insurance.py

# Caso 6: Fraude
python scripts/case_study_fraud.py
```

### 3. Executar Todos os Casos

```bash
python scripts/run_all_cases.py
```

### 4. Gerar Análise Agregada

```bash
python scripts/aggregate_analysis.py
```

## Outputs

### Por Caso de Uso

Cada caso gera:
- `results/case_study_{domain}_results.json` - Resultados brutos
- `results/case_study_{domain}_report.pdf` - Relatório completo
- `results/case_study_{domain}_times.csv` - Métricas de tempo
- `logs/case_study_{domain}_{timestamp}.log` - Log de execução

### Agregados

- `tables/case_studies_summary.tex` - Tabela LaTeX para o paper
- `results/case_studies_analysis.json` - Análise estatística agregada
- `figures/case_studies_times.pdf` - Gráfico de tempos
- `figures/case_studies_violations.pdf` - Gráfico de violações

## Detalhes dos Casos de Uso

### 1. Credit Scoring (Crédito)
- Dataset: German Credit Data (UCI)
- Modelo: XGBoost
- Violações Esperadas: DI=0.74 (gênero), violação EEOC

### 2. Hiring (Contratação)
- Dataset: Adult Income Dataset (adaptado)
- Modelo: Random Forest
- Violações Esperadas: DI=0.59 (raça)

### 3. Healthcare (Saúde)
- Dataset: MIMIC-III (subset) ou sintético
- Modelo: XGBoost
- Violações Esperadas: Nenhuma (bem calibrado, ECE=0.042)

### 4. Mortgage (Hipoteca)
- Dataset: HMDA Data
- Modelo: Gradient Boosting
- Violações Esperadas: Violação ECOA

### 5. Insurance (Seguros)
- Dataset: Porto Seguro Safe Driver
- Modelo: XGBoost
- Violações Esperadas: Nenhuma

### 6. Fraud Detection (Fraude)
- Dataset: Credit Card Fraud Detection
- Modelo: LightGBM
- Violações Esperadas: Nenhuma

## Validações

Cada caso realiza:
- ✅ Teste de Fairness (Disparate Impact, Equal Opportunity, EEOC)
- ✅ Teste de Robustez (perturbações, drift)
- ✅ Teste de Incerteza (calibração, ECE)
- ✅ Teste de Resiliência (adversarial)
- ✅ Medição precisa de tempo
- ✅ Geração de relatório PDF

## Próximos Passos

Ver arquivo `02_estudos_de_caso.md` para detalhes completos de implementação de cada caso.

## Referências

- German Credit: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- Adult Income: https://archive.ics.uci.edu/ml/datasets/adult
- HMDA Data: https://www.consumerfinance.gov/data-research/hmda/
- Porto Seguro: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
- Credit Card Fraud: https://www.kaggle.com/mlg-ulb/creditcardfraud
