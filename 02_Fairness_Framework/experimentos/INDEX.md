# 📑 Índice Completo - Experimentos DeepBridge Fairness

Índice de todos os arquivos criados e sua finalidade.

---

## 📄 Documentação Principal

### 1. [RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md) ⭐ **LEIA PRIMEIRO**
- Visão geral de 15 claims a validar
- Timeline de 18 semanas
- Recursos necessários (~$1,300)
- Riscos e mitigações
- Dashboard de progresso

### 2. [PLANO_EXPERIMENTOS.md](PLANO_EXPERIMENTOS.md) 📋 **DOCUMENTO MASTER**
- **17 seções detalhadas** cobrindo:
  - 8 grupos de experimentos principais
  - Metodologias step-by-step
  - Métricas de validação
  - Critérios de sucesso
  - Timeline detalhado
  - Contingências e mitigações

### 3. [GUIA_EXECUCAO.md](GUIA_EXECUCAO.md) 🚀 **PASSO A PASSO**
- Setup do ambiente (Python, deps)
- Execução fase por fase
- Comandos exatos
- Troubleshooting comum
- Checklist final

### 4. [CHECKLIST_RAPIDO.md](CHECKLIST_RAPIDO.md) ✅ **TRACKING DIÁRIO**
- 6 experimentos críticos
- Tabela de validação de claims
- Red flags e ações
- Timeline resumido (8-18 semanas)

### 5. [README.md](README.md) 📖 **OVERVIEW**
- Estrutura de arquivos
- Quick start
- Claims principais
- Critérios mínimos para publicação

---

## 🐍 Scripts Python

### Experimentos Principais

#### 1. [scripts/exp1_auto_detection.py](scripts/exp1_auto_detection.py)
**Experimento 1: Auto-Detecção de Atributos Sensíveis**
- Testa em 500 datasets
- Valida F1-Score ≥ 0.90
- Análise de erros (FP/FN)
- Uso: `python exp1_auto_detection.py --quick`

#### 2. [scripts/exp3_eeoc_validation.py](scripts/exp3_eeoc_validation.py)
**Experimento 3: Verificação EEOC/ECOA**
- Testa Regra 80% (10 casos controlados)
- Testa Question 21 (7 casos)
- Valida Adverse Action Notices
- **CRÍTICO**: 100% precisão obrigatória
- Uso: `python exp3_eeoc_validation.py`

#### 3. [scripts/exp4_case_studies.py](scripts/exp4_case_studies.py)
**Experimento 4: Case Studies**
- COMPAS (recidivism)
- German Credit (credit scoring)
- Adult Income (employment)
- Healthcare (readmission)
- Valida tempo de análise
- Uso: `python exp4_case_studies.py --dataset compas`

### Scripts Auxiliares

#### 4. [scripts/utils.py](scripts/utils.py)
**Utilidades Comuns**
- `timer()`: Context manager para medir tempo
- `save_json()`, `save_csv()`: Helpers de I/O
- `create_synthetic_dataset()`: Geração de dados sintéticos
- `validate_claim()`: Validação de claims
- `ExperimentLogger`: Logger estruturado
- `check_dependencies()`: Verifica instalações

#### 5. [scripts/calculate_inter_rater_agreement.py](scripts/calculate_inter_rater_agreement.py)
**Análise de Concordância entre Anotadores**
- Calcula Cohen's Kappa
- Valida Kappa ≥ 0.85
- Identifica discordâncias
- Uso: `python calculate_inter_rater_agreement.py --reviewer1 r1.csv --reviewer2 r2.csv`

---

## ⚙️ Configuração e Setup

### 1. [requirements.txt](requirements.txt)
Dependências Python:
- Core: `deepbridge`, `pandas`, `numpy`, `scipy`
- ML: `scikit-learn`, `xgboost`, `lightgbm`
- Fairness: `aif360`, `fairlearn`, `aequitas`
- Viz: `matplotlib`, `seaborn`, `plotly`
- Testes: `pytest`

Instalar: `pip install -r requirements.txt`

### 2. [setup.sh](setup.sh)
Script automatizado de setup:
- Cria venv
- Instala dependências
- Cria diretórios
- Testa instalação
- Uso: `chmod +x setup.sh && ./setup.sh`

---

## 📊 Dados e Templates

### 1. [data/ground_truth_template.csv](data/ground_truth_template.csv)
Template para anotação manual:
- Colunas: dataset_name, source, n_samples, sensitive_attributes
- Exemplos: COMPAS, German Credit, Adult, Healthcare
- Use como base para anotar 500 datasets

### Estrutura de Diretórios

```
data/
├── ground_truth.csv              # Anotações finalizadas (500 datasets)
├── annotations_reviewer1.csv     # Anotações do revisor 1
├── annotations_reviewer2.csv     # Anotações do revisor 2
├── case_studies/                 # Datasets dos case studies
│   ├── compas.csv
│   ├── german_credit.csv
│   ├── adult.csv
│   └── healthcare.csv
└── synthetic/                    # Datasets sintéticos para testes
```

---

## 📈 Resultados

Estrutura onde resultados são salvos:

```
results/
├── auto_detection/
│   ├── auto_detection_results.csv
│   ├── summary.json
│   ├── confusion_matrix.png
│   └── false_positives_analysis.txt
├── eeoc_validation/
│   ├── eeoc_80_rule_validation.csv
│   ├── eeoc_question_21_validation.csv
│   ├── adverse_action_notices_sample.json
│   └── summary.json
├── case_studies/
│   ├── compas_result.json
│   ├── german_credit_result.json
│   ├── adult_income_result.json
│   ├── healthcare_result.json
│   └── case_studies_summary.csv
├── usability/
│   ├── sus_scores.csv
│   ├── tlx_scores.csv
│   └── P01/, P02/, ... P20/     # Por participante
├── performance/
│   └── performance_benchmarks.csv
└── comparison/
    └── tool_comparison_matrix.csv
```

---

## 📊 Relatórios

```
reports/
├── experiment_summary.pdf        # Resumo consolidado
├── reproduction_guide.md         # Como reproduzir
└── figures/                      # Figuras para o paper
    ├── auto_detection_f1.png
    ├── sus_scores.png
    ├── performance_speedup.png
    └── ...
```

---

## 🎯 Fluxo de Trabalho Recomendado

### Fase 1: Setup Inicial (Dia 1)
```bash
# 1. Ler documentação
cat RESUMO_EXECUTIVO.md
cat GUIA_EXECUCAO.md

# 2. Setup ambiente
./setup.sh

# 3. Teste rápido
cd scripts/
python exp1_auto_detection.py --quick
python exp3_eeoc_validation.py
```

### Fase 2: Coleta de Dados (Semanas 1-2)
```bash
# 1. Coletar 500 datasets (Kaggle, UCI, OpenML)
# 2. Anotar ground truth (2 revisores)
# 3. Calcular concordância
python scripts/calculate_inter_rater_agreement.py \
    --reviewer1 data/annotations_reviewer1.csv \
    --reviewer2 data/annotations_reviewer2.csv
```

### Fase 3: Experimentos Core (Semanas 3-9)
```bash
# Experimento 1: Auto-detecção
python scripts/exp1_auto_detection.py --n-datasets 500

# Experimento 3: EEOC/ECOA
python scripts/exp3_eeoc_validation.py

# Experimento 4: Case Studies
python scripts/exp4_case_studies.py --dataset all
```

### Fase 4: Usabilidade (Semanas 10-13)
```bash
# Recrutamento → Execução → Análise
# (Scripts para exp5_usability.py ainda não criados)
```

### Fase 5: Validação (Semanas 14-16)
```bash
# Performance, Comparação, Robustness
# (Scripts ainda não criados)
```

### Fase 6: Finalização (Semanas 17-18)
```bash
# Gerar relatórios
python scripts/generate_reports.py --experiments all

# Criar reproduction package
python scripts/create_reproduction_package.py
```

---

## ✅ Checklist de Arquivos Criados

### Documentação (6 arquivos)
- [x] RESUMO_EXECUTIVO.md
- [x] PLANO_EXPERIMENTOS.md
- [x] GUIA_EXECUCAO.md
- [x] CHECKLIST_RAPIDO.md
- [x] README.md
- [x] INDEX.md (este arquivo)

### Scripts Principais (3 arquivos)
- [x] exp1_auto_detection.py
- [x] exp3_eeoc_validation.py
- [x] exp4_case_studies.py

### Scripts Auxiliares (2 arquivos)
- [x] utils.py
- [x] calculate_inter_rater_agreement.py

### Configuração (3 arquivos)
- [x] requirements.txt
- [x] setup.sh
- [x] data/ground_truth_template.csv

### Estrutura de Diretórios
- [x] scripts/
- [x] data/ (case_studies/, synthetic/)
- [x] results/ (6 subdiretórios)
- [x] reports/ (figures/)

**Total: 14 arquivos + estrutura de diretórios** ✅

---

## 🚧 Próximas Implementações (TODO)

### Scripts Faltantes

1. **exp2_metrics_coverage.py**
   - Validar 15 métricas (4 pré + 11 pós)
   - Comparar com cálculo manual
   - Edge cases

2. **exp5_usability.py**
   - Protocol para participantes
   - Coleta de SUS/TLX
   - Análise de tarefas

3. **exp6_performance.py**
   - Benchmarks (Small/Medium/Large)
   - Speedup vs manual
   - Memory profiling

4. **exp7_threshold_optimization.py**
   - Pareto frontier validation
   - Threshold recommendations

5. **exp8_comparison.py**
   - Feature matrix (AIF360, Fairlearn, Aequitas)
   - Metric accuracy comparison

6. **exp9_edge_cases.py**
   - Dataset pequeno (n=50)
   - Desbalanceado extremo (99:1)
   - Missing values
   - Multiclass

7. **generate_reports.py**
   - Consolidar todos resultados
   - Gerar figuras para paper
   - LaTeX tables

8. **create_reproduction_package.py**
   - Zip com scripts + dados + README
   - Para submission

---

## 📞 Suporte e Troubleshooting

### Problemas Comuns

**1. Import Error: No module named 'deepbridge'**
```bash
pip install deepbridge
# ou
pip install -r requirements.txt
```

**2. Experimento falhou: Dataset não encontrado**
- Verifique se o arquivo existe em `data/case_studies/`
- Use `--quick` para gerar dados sintéticos

**3. Kappa < 0.85**
- Revisar guidelines de anotação
- Re-anotar datasets com discordância
- Consultar `GUIA_EXECUCAO.md` seção Troubleshooting

**4. Tempo excedeu target**
- Normal em primeira execução (overhead)
- Rode múltiplas vezes e calcule média
- Otimize código se necessário

### Onde Encontrar Ajuda

- **Metodologia**: `PLANO_EXPERIMENTOS.md` seção específica
- **Execução**: `GUIA_EXECUCAO.md`
- **Tracking**: `CHECKLIST_RAPIDO.md`
- **Código**: Comentários inline nos scripts

---

## 📊 Status do Projeto

**Última Atualização**: 2025-12-06

**Fase Atual**: Planejamento completo ✅

**Próximos Passos**:
1. Setup ambiente (`./setup.sh`)
2. Teste rápido (`python exp1_auto_detection.py --quick`)
3. Coletar 500 datasets
4. Iniciar Experimento 1

**Progresso Geral**: 0% (planejamento 100%)

---

## 🎓 Para Citação

Se você usar este framework de experimentos, por favor cite:

```bibtex
@misc{deepbridge_fairness_experiments2025,
  title={Experimental Framework for DeepBridge Fairness Validation},
  author={[Seu Nome]},
  year={2025},
  note={Framework para validação de claims do paper DeepBridge Fairness}
}
```

---

**Boa sorte com os experimentos! 🚀**

**Questões**: Consulte `README.md` ou abra issue no repositório.
