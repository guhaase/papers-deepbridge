# Quick Start - Experimento 2: Estudos de Caso

## Pré-requisitos

```bash
# Python 3.9+
python --version

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

## Instalação

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/02_estudos_de_caso

# Instalar dependências
pip install -r requirements.txt
```

## Execução

### Opção 1: Executar Todos os Casos (Recomendado)

```bash
python scripts/run_all_cases.py
```

Isso executará os 6 estudos de caso sequencialmente (~2.5 horas no total).

### Opção 2: Executar Casos Individuais

```bash
# Caso 1: Crédito (~17 min)
python scripts/case_study_credit.py

# Caso 2: Contratação (~12 min)
python scripts/case_study_hiring.py

# Caso 3: Saúde (~23 min)
python scripts/case_study_healthcare.py

# Caso 4: Hipoteca (~45 min)
python scripts/case_study_mortgage.py

# Caso 5: Seguros (~38 min)
python scripts/case_study_insurance.py

# Caso 6: Fraude (~31 min)
python scripts/case_study_fraud.py
```

## Análise dos Resultados

Após executar os casos, gere a análise agregada:

```bash
python scripts/aggregate_analysis.py
```

Isso criará:
- `tables/case_studies_summary.tex` - Tabela LaTeX para o paper
- `figures/case_studies_times.pdf` - Gráfico de tempos
- `figures/case_studies_violations.pdf` - Gráfico de violações
- `results/case_studies_analysis.json` - Análise estatística

## Estrutura de Outputs

```
02_estudos_de_caso/
├── results/
│   ├── case_study_credit_results.json
│   ├── case_study_credit_report.txt
│   ├── case_study_hiring_results.json
│   ├── case_study_hiring_report.txt
│   ├── ... (outros casos)
│   └── case_studies_analysis.json
├── figures/
│   ├── case_studies_times.pdf
│   └── case_studies_violations.pdf
├── tables/
│   └── case_studies_summary.tex
└── logs/
    └── case_study_*_YYYYMMDD_HHMMSS.log
```

## Resultados Esperados

| Domínio | Amostras | Violações | Tempo (min) |
|---------|----------|-----------|-------------|
| Crédito | 1.000 | 2 | 17 |
| Contratação | 7.214 | 1 | 12 |
| Saúde | 101.766 | 0 | 23 |
| Hipoteca | 450.000 | 1 | 45 |
| Seguros | 595.212 | 0 | 38 |
| Fraude | 284.807 | 0 | 31 |
| **Média** | - | - | **27.7** |

## Verificação

Para verificar se tudo está funcionando:

```bash
# Executar apenas o caso de Crédito (mais rápido, ~17 min)
python scripts/case_study_credit.py

# Verificar se os arquivos foram gerados
ls -l results/case_study_credit*
ls -l logs/case_study_credit*
```

## Troubleshooting

### Erro: Module not found

```bash
# Verificar se as dependências estão instaladas
pip install -r requirements.txt
```

### Erro: Memory issue (datasets grandes)

Se houver problemas de memória com os datasets maiores (Hipoteca, Seguros, Fraude):

1. Execute os casos menores primeiro
2. Execute os casos grandes um por vez
3. Monitore o uso de memória com `htop` ou similar

### LightGBM não instalado

Para o caso de Fraude, se LightGBM não estiver disponível:

```bash
pip install lightgbm
```

O script usará XGBoost como fallback se LightGBM não estiver instalado.

## Customização

Para modificar os parâmetros dos experimentos, edite os scripts em `scripts/`:

- Tamanhos de datasets
- Hiperparâmetros dos modelos
- Tempos de simulação (para testes rápidos)
- Violações esperadas

## Próximos Passos

1. ✅ Executar todos os casos
2. ✅ Gerar análise agregada
3. ✅ Copiar tabela LaTeX para o paper
4. ✅ Incluir figuras no paper
5. ✅ Documentar resultados

## Contato

Para dúvidas sobre este experimento, consulte:
- `README.md` - Visão geral completa
- `02_estudos_de_caso.md` - Especificação detalhada
- Logs em `logs/` - Detalhes de execução
