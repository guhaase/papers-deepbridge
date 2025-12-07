# 🚀 Guia de Execução Passo a Passo

Guia prático para executar todos os experimentos necessários para validar o paper DeepBridge Fairness.

---

## 📋 Pré-requisitos

### 1. Ambiente Python

```bash
# Verificar versão Python (necessário ≥ 3.8)
python --version

# Criar ambiente virtual
cd /home/guhaase/projetos/DeepBridge/papers/02_Fairness_Framework/experimentos
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install --upgrade pip
pip install deepbridge
pip install aif360 fairlearn aequitas
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn plotly
pip install jupyter notebook  # Para análises interativas
pip install pytest pytest-cov  # Para testes
```

### 2. Verificar Instalação

```bash
python -c "from deepbridge import DBDataset; print('✅ DeepBridge OK')"
python -c "from aif360.datasets import BinaryLabelDataset; print('✅ AIF360 OK')"
python -c "import fairlearn; print('✅ Fairlearn OK')"
```

### 3. Estrutura de Diretórios

```bash
# Já foi criada automaticamente, mas verifique:
tree -L 2 experimentos/

# Deve mostrar:
# experimentos/
# ├── scripts/
# ├── data/
# ├── results/
# └── reports/
```

---

## 🎯 Fase 1: Teste Rápido (1 dia)

**Objetivo**: Validar que tudo está funcionando antes de executar experimentos completos.

### Passo 1.1: Executar Auto-Detecção em Modo Rápido

```bash
cd scripts/
python exp1_auto_detection.py --quick
```

**Saída Esperada**:
```
🔬 EXPERIMENTO 1: AUTO-DETECÇÃO DE ATRIBUTOS SENSÍVEIS
========================================================
🚀 Iniciando experimento de auto-detecção
📊 Total de datasets: 5

[1/5] Processando: compas_synthetic
   Atributos esperados: ['age', 'race', 'sex']
   ✅ Detectado: ['age', 'race', 'sex']
   📈 Precision: 1.000 | Recall: 1.000 | F1: 1.000
...

📊 RESULTADOS AGREGADOS
========================================================
📈 Métricas Gerais (N=5):
   Precision: 0.XXX ± 0.XXX
   Recall:    0.XXX ± 0.XXX
   F1-Score:  0.XXX ± 0.XXX

✅ Validação de Claims:
   Precision ≥ 0.92: ✅ PASS
   Recall ≥ 0.89:    ✅ PASS
   F1-Score ≥ 0.90:  ✅ PASS
```

**✅ Se passou**: Continue para próximo passo
**❌ Se falhou**: Verifique instalação do DeepBridge ou abra issue

---

## 📊 Fase 2: Coleta de Dados (2-3 semanas)

**Objetivo**: Coletar 500 datasets com ground truth anotado.

### Passo 2.1: Coletar Datasets

**Fontes**:
1. **Kaggle** (200 datasets):
   ```bash
   # Instalar Kaggle CLI
   pip install kaggle

   # Configurar API key (https://www.kaggle.com/docs/api)
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   # Buscar datasets relevantes
   kaggle datasets list -s "credit scoring"
   kaggle datasets list -s "hiring"
   kaggle datasets list -s "healthcare"
   kaggle datasets list -s "criminal justice"
   ```

2. **UCI Repository** (150 datasets):
   - Acesse: https://archive.ics.uci.edu/ml/datasets.php
   - Filtros: Classification, Tabular, >1000 samples
   - Baixe datasets relevantes para fairness

3. **OpenML** (100 datasets):
   ```python
   from sklearn.datasets import fetch_openml

   # Exemplo
   data = fetch_openml(name='adult', version=1, as_frame=True)
   ```

4. **Sintéticos** (50 datasets):
   - Use scripts de geração sintética para controle

### Passo 2.2: Anotar Ground Truth

**Criar arquivo**: `data/ground_truth.csv`

**Formato**:
```csv
dataset_name,source,target_column,sensitive_attributes,n_samples,n_features
compas,kaggle,two_year_recid,"race,sex,age",7214,12
german_credit,uci,credit_risk,"age,sex,foreign_worker",1000,20
adult,uci,income,">50K,sex,race",48842,14
...
```

**Processo de Anotação**:

1. **Revisor 1** anota todos 500 datasets
2. **Revisor 2** anota todos 500 datasets (independentemente)
3. Calcular Cohen's Kappa (deve ser > 0.85)
4. Resolver discordâncias por consenso

**Script auxiliar**:
```bash
# Criar template para anotação
python scripts/create_annotation_template.py --n-datasets 500

# Calcular inter-rater agreement
python scripts/calculate_kappa.py \
    --reviewer1 data/annotations_reviewer1.csv \
    --reviewer2 data/annotations_reviewer2.csv
```

---

## 🔬 Fase 3: Experimentos Principais (4-6 semanas)

### Semana 1-2: Auto-Detecção

```bash
# Executar experimento completo (500 datasets)
python scripts/exp1_auto_detection.py --n-datasets 500

# Analisar resultados
jupyter notebook analysis/exp1_analysis.ipynb

# Verificar se passou nos critérios
grep "PASS" results/auto_detection/summary.json
```

**Critérios de Sucesso**:
- [ ] F1-Score ≥ 0.85 (target: 0.90)
- [ ] Precision ≥ 0.90
- [ ] Recall ≥ 0.85
- [ ] Kappa inter-rater ≥ 0.85

### Semana 3: Verificação EEOC/ECOA

```bash
# Executar testes de conformidade
python scripts/exp3_eeoc_validation.py

# Verificar 100% precisão
cat results/eeoc_validation/summary.txt
```

**Critérios de Sucesso**:
- [ ] 100% precisão em regra 80% (0 erros)
- [ ] 100% precisão em Question 21
- [ ] 0 falsos positivos

### Semana 4-6: Case Studies

```bash
# COMPAS
python scripts/exp4_case_studies.py --dataset compas

# German Credit
python scripts/exp4_case_studies.py --dataset german_credit

# Adult Income
python scripts/exp4_case_studies.py --dataset adult

# Healthcare
python scripts/exp4_case_studies.py --dataset healthcare

# Ou executar todos de uma vez
python scripts/exp4_case_studies.py --all
```

**Critérios de Sucesso** (cada dataset):
- [ ] Tempo de análise documentado
- [ ] Atributos detectados corretamente
- [ ] Violações identificadas
- [ ] Threshold ótimo calculado
- [ ] Relatório gerado

---

## 👥 Fase 4: Estudo de Usabilidade (3-4 semanas)

### Semana 1: Recrutamento

**Perfil dos Participantes**:
- Data Scientists ou ML Engineers
- 2-8 anos de experiência em ML
- Pelo menos 65% com experiência em fairness tools
- N = 20 participantes

**Canais de Recrutamento**:
- LinkedIn (grupos de ML)
- Twitter (#MachineLearning #ResponsibleAI)
- Conferências (NeurIPS, ICML, FAccT)
- Empresas parceiras

**Incentivos**:
- $50 Amazon gift card
- Co-autoria em acknowledgments
- Early access to tool

### Semana 2-3: Execução

**Protocol** (60 minutos por participante):

1. **Briefing** (5 min):
   - Explicar objetivo do estudo
   - Obter consentimento informado
   - Configurar screen recording

2. **Setup** (10 min):
   - Instalar DeepBridge
   - Carregar dataset Adult Income
   - Verificar ambiente funcional

3. **Tarefas** (35 min):
   - **Task 1** (15 min): Detectar bias em modelo
   - **Task 2** (10 min): Verificar EEOC compliance
   - **Task 3** (10 min): Encontrar threshold ótimo

4. **Questionários** (10 min):
   - System Usability Scale (SUS)
   - NASA Task Load Index (TLX)
   - Perguntas demográficas

5. **Entrevista** (10 min):
   - "O que você mais gostou?"
   - "O que foi mais difícil?"
   - "O que você mudaria?"

**Executar**:
```bash
# Para cada participante
python scripts/exp5_usability.py --participant-id P01

# Isso irá:
# 1. Gerar instruções personalizadas
# 2. Cronometrar tarefas
# 3. Coletar métricas
# 4. Salvar resultados em results/usability/P01/
```

### Semana 4: Análise

```bash
# Calcular SUS scores
python scripts/analyze_sus.py --input results/usability/

# Calcular TLX
python scripts/analyze_tlx.py --input results/usability/

# Análise qualitativa (entrevistas)
python scripts/thematic_analysis.py --transcripts results/usability/*/interview.txt
```

**Critérios de Sucesso**:
- [ ] N ≥ 15 participantes completaram
- [ ] SUS médio ≥ 75 (target: 85.2)
- [ ] Taxa de sucesso ≥ 85% (target: 95%)
- [ ] NASA-TLX ≤ 40 (target: 32.1)

---

## ⚡ Fase 5: Performance (1-2 semanas)

### Passo 5.1: Configurar Hardware

**Opção 1: AWS** (recomendado para reprodutibilidade)
```bash
# Lançar instância m5.2xlarge
aws ec2 run-instances \
    --image-id ami-xxxxx \
    --instance-type m5.2xlarge \
    --key-name my-key

# SSH na instância
ssh -i my-key.pem ubuntu@<instance-ip>

# Setup
git clone <repo>
cd experimentos
source setup_aws.sh
```

**Opção 2: Local** (se tiver hardware equivalente)
- 8 CPUs
- 32GB RAM
- SSD storage

### Passo 5.2: Executar Benchmarks

```bash
# Todos os tamanhos (Small, Medium, Large)
python scripts/exp6_performance.py --all-sizes

# Isso irá executar 5 repetições de cada:
# - Small: 1K amostras, 20 features
# - Medium: 50K amostras, 50 features
# - Large: 500K amostras, 100 features
```

**Duração Estimada**: 8-12 horas

**Critérios de Sucesso**:
- [ ] Speedup Small ≥ 3.5x
- [ ] Speedup Medium ≥ 2.5x
- [ ] Speedup Large ≥ 2.0x
- [ ] Redução memória ≥ 35%

---

## 🔄 Fase 6: Comparação com Ferramentas (1 semana)

### Passo 6.1: Instalar Ferramentas

```bash
pip install aif360==0.2.9
pip install fairlearn==0.10.0
pip install aequitas==2.0.0
```

### Passo 6.2: Executar Comparação

```bash
# Feature comparison
python scripts/exp8_comparison.py --tools all --test-features

# Metric accuracy comparison
python scripts/exp8_comparison.py --tools all --test-accuracy
```

**Critérios de Sucesso**:
- [ ] DeepBridge tem todas features claimed
- [ ] Outras ferramentas NÃO têm features exclusivas claimed
- [ ] Diferença de métricas < 1%

---

## 📊 Fase 7: Análise e Relatórios (1-2 semanas)

### Passo 7.1: Gerar Relatórios

```bash
# Relatório consolidado
python scripts/generate_reports.py \
    --experiments all \
    --output reports/experiment_summary.pdf

# Figuras para o paper
python scripts/generate_figures.py \
    --output reports/figures/ \
    --format pdf,png

# Tabelas LaTeX
python scripts/generate_tables.py \
    --output reports/tables/ \
    --format latex
```

### Passo 7.2: Validar Checklist

```bash
# Validar todas as claims
python scripts/validate_claims.py --checklist CHECKLIST_RAPIDO.md

# Saída esperada:
# ✅ Claim 1: Auto-detecção F1=0.90 - VALIDATED
# ✅ Claim 2: 100% acurácia case studies - VALIDATED
# ...
```

### Passo 7.3: Preparar Reproduction Package

```bash
# Criar package para submission
python scripts/create_reproduction_package.py \
    --include scripts,data,results \
    --output reproduction_package.zip

# Conteúdo:
# - README.md com instruções
# - Scripts completos
# - Dados (se permitido por licença)
# - Resultados agregados
# - Requirements.txt
```

---

## ⚠️ Troubleshooting

### Problema: Auto-detecção F1 < 0.85

**Diagnóstico**:
```bash
python scripts/debug_auto_detection.py --analyze-errors
```

**Soluções**:
1. Ajustar threshold de similaridade
2. Expandir dicionário de sinônimos
3. Melhorar context filtering
4. Revisar ground truth (possíveis erros de anotação)

### Problema: SUS < 75

**Diagnóstico**:
```bash
python scripts/analyze_usability_issues.py --detailed
```

**Soluções**:
1. Melhorar documentação
2. Adicionar tutoriais
3. Simplificar API
4. Mais exemplos práticos

### Problema: Speedup < 2.0x

**Diagnóstico**:
```bash
python scripts/profile_performance.py --component threshold_opt
```

**Soluções**:
1. Otimizar threshold optimization (grid search esparso)
2. Paralelizar cálculos
3. Melhorar caching
4. Usar numba/cython para hot paths

### Problema: Recrutamento < 15 participantes

**Ações**:
1. Aumentar incentivos ($75)
2. Estender prazo de recrutamento
3. Recrutar em mais canais
4. Fazer estudo piloto (N=10) + validation (N=5)

---

## 📞 Suporte

### Dúvidas Técnicas:
- Consulte `PLANO_EXPERIMENTOS.md` seção específica
- Abra issue no repositório
- Email: [seu-email]

### Problemas com Scripts:
```bash
# Ativar modo debug
export DEBUG=1
python scripts/exp1_auto_detection.py --quick --verbose
```

### Issues Conhecidos:
- Ver `issues.md` para lista atualizada

---

## ✅ Checklist Final

Antes de submeter o paper, verifique:

### Experimentos:
- [ ] Auto-detecção: 500 datasets, F1 ≥ 0.85
- [ ] EEOC/ECOA: 100% precisão
- [ ] Case Studies: 4/4 completos
- [ ] Usabilidade: N ≥ 15, SUS ≥ 75
- [ ] Performance: Speedup ≥ 2.0x
- [ ] Comparação: 3 ferramentas testadas

### Artefatos:
- [ ] Todos resultados em `results/`
- [ ] Figuras em `reports/figures/`
- [ ] Tabelas em `reports/tables/`
- [ ] Reproduction package criado
- [ ] README atualizado

### Documentação:
- [ ] Metodologia documentada
- [ ] Resultados documentados
- [ ] Limitações documentadas
- [ ] IRB approval (se necessário)
- [ ] Licenças de dados verificadas

### Paper:
- [ ] Seção 5 (Evaluation) atualizada com resultados
- [ ] Figuras inseridas
- [ ] Tabelas inseridas
- [ ] Claims validadas
- [ ] Apêndice técnico incluído

---

**Boa sorte! 🚀**

**Estimativa Total de Tempo**: 12-18 semanas (3-4.5 meses)

**Próximos Passos Imediatos**:
1. ✅ Executar teste rápido (`python exp1_auto_detection.py --quick`)
2. 📊 Iniciar coleta de datasets
3. 👥 Começar recrutamento para usabilidade
4. 📅 Revisar timeline e ajustar se necessário
