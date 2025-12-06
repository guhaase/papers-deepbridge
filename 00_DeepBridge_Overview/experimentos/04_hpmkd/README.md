# Experimento 4: HPM-KD Framework

## Objetivo

Comprovar os resultados do framework **HPM-KD (Hierarchical Progressive Multi-Teacher Knowledge Distillation)** apresentados no paper, demonstrando:

- **Compressão**: 10.3× redução de tamanho
- **Acurácia**: 98.4% de retenção (85.8% vs 87.2% teacher)
- **Latência**: 10.4× speedup (12ms vs 125ms)
- **Superioridade** sobre baselines (Vanilla KD, TAKD, Auto-KD)

## Resultados Esperados

### Tabela Principal

| Método | Acurácia | Compressão | Latência | Retenção |
|--------|----------|------------|----------|----------|
| Teacher Ensemble | 87.2% | 1.0× | 125ms | 100% |
| Vanilla KD | 82.5% | 10.2× | 12ms | 94.7% |
| TAKD | 83.8% | 10.1× | 13ms | 96.1% |
| Auto-KD | 84.4% | 10.3× | 12ms | 96.8% |
| **HPM-KD** | **85.8%** | **10.3×** | **12ms** | **98.4%** |

### Métricas Derivadas

- **Redução de custo**: 10× ($0.05 → $0.005 por 1K predições)
- **Throughput**: 10.4× (8 req/s → 83 req/s)
- **Tamanho**: 2.4GB → 230MB

## Estrutura do Experimento

### Datasets

**20 datasets tabulares** UCI/OpenML:
- 10 classificação binária (Adult, Bank, Credit, etc.)
- 10 classificação multi-classe (Car, Chess, Letter, etc.)

### Modelos

**Teachers (Ensemble de 3)**:
- XGBoost (200 estimators)
- LightGBM (200 estimators)
- CatBoost (200 iterations)

**Student**:
- MLP compacto (64, 32) - PyTorch

### Baselines

1. **Vanilla KD**: Destilação simples com temperatura
2. **TAKD**: Teacher-Assistant KD (2 estágios)
3. **Auto-KD**: Busca automática de hiperparâmetros

### Componentes HPM-KD

1. **Adaptive Configuration Manager**: Meta-learning para configuração
2. **Progressive Distillation Chain**: Múltiplos estágios de refinamento
3. **Attention-Weighted Multi-Teacher**: Ensemble com atenção aprendida
4. **Meta-Temperature Scheduler**: Temperatura adaptativa
5. **Parallel Processing Pipeline**: Paralelização para speedup

## Estrutura de Diretórios

```
04_hpmkd/
├── config/          # Configurações YAML
├── data/            # Dados processados
├── datasets/        # Datasets UCI/OpenML (baixados)
├── models/          # Modelos treinados (teachers, students)
├── scripts/         # Scripts Python
│   ├── utils.py
│   ├── datasets_loader.py
│   ├── train_teachers.py
│   ├── baselines.py
│   ├── hpmkd_model.py
│   ├── run_experiment.py
│   ├── ablation_study.py
│   └── analyze_results.py
├── notebooks/       # Análise exploratória
├── results/         # Resultados JSON/CSV
├── figures/         # Visualizações
├── tables/          # Tabelas LaTeX
└── logs/            # Logs de execução
```

## Como Executar

### 1. Instalação

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd
pip install -r requirements.txt
```

### 2. Execução Completa

```bash
# Pipeline completo (todos os 20 datasets)
python scripts/run_experiment.py --mode full

# Ou por etapas:
python scripts/datasets_loader.py      # 1. Carregar datasets
python scripts/train_teachers.py       # 2. Treinar teachers
python scripts/baselines.py            # 3. Executar baselines
python scripts/hpmkd_model.py          # 4. Executar HPM-KD
python scripts/ablation_study.py       # 5. Ablation studies
python scripts/analyze_results.py      # 6. Análise e visualizações
```

### 3. Execução Rápida (Mock/Demo)

```bash
# Teste rápido com dados sintéticos
python scripts/run_experiment.py --mode demo --n-datasets 3
```

## Outputs

### Por Dataset

Cada dataset gera:
- `results/hpmkd_{dataset_name}_metrics.json` - Métricas detalhadas
- `models/teachers/{dataset_name}_ensemble.pkl` - Teacher ensemble
- `models/students/{dataset_name}_hpmkd.pkl` - Student HPM-KD

### Agregados

- `results/hpmkd_aggregated_results.csv` - Resultados consolidados
- `results/hpmkd_statistical_tests.json` - Testes estatísticos
- `results/hpmkd_ablation_results.json` - Estudos de ablação

### Visualizações

- `figures/hpmkd_accuracy_comparison.pdf` - Comparação de acurácia
- `figures/hpmkd_retention_rates.pdf` - Taxas de retenção
- `figures/hpmkd_compression_latency.pdf` - Trade-off compressão/latência
- `figures/hpmkd_ablation_study.pdf` - Contribuição de componentes

### Tabelas

- `tables/hpmkd_results.tex` - Tabela LaTeX para o paper

## Análise Estatística

### Testes Realizados

1. **Paired t-test**: HPM-KD vs. cada baseline (20 pares)
   - H0: Não há diferença de acurácia
   - H1: HPM-KD > baseline
   - Esperado: p < 0.01 para todos

2. **Effect Size**: Cohen's d
   - HPM-KD vs Vanilla KD: d > 0.8 (large effect)
   - HPM-KD vs TAKD: d > 0.5 (medium effect)
   - HPM-KD vs Auto-KD: d > 0.3 (small-medium effect)

3. **Ablation Study**: Quantificar contribuição de cada componente
   - Progressive Distillation: ~1.5%
   - Attention Weighting: ~0.8%
   - Meta-Temperature: ~0.5%
   - Parallel Processing: 0% (apenas tempo)

## Componentes do HPM-KD

### 1. Adaptive Configuration Manager

Seleciona automaticamente:
- Temperatura inicial
- Taxa de aprendizado
- Número de estágios
- Pesos dos teachers

**Validação**: Comparar config automática vs. manual

### 2. Progressive Distillation Chain

Refina student em múltiplos estágios:
- Stage 1: Teacher → Assistant (128, 64)
- Stage 2: Assistant → Student (64, 32)
- Stage 3: Fine-tuning

**Validação**: Comparar 1, 2, 3, 4 estágios

### 3. Attention-Weighted Multi-Teacher

Aprende pesos de atenção para cada teacher:
- Pesos baseados em performance local
- Adaptativos por amostra
- Treinados end-to-end

**Validação**: Comparar uniform, fixed, attention weighting

### 4. Meta-Temperature Scheduler

Ajusta temperatura baseado em:
- Dificuldade da tarefa (entropia)
- Progresso do treinamento
- Performance em validação

**Validação**: Comparar temperatura fixa vs. adaptativa

### 5. Parallel Processing Pipeline

Paraleliza:
- Training de múltiplos teachers
- Inferência em batches
- Cross-validation

**Validação**: Medir speedup com 1, 2, 4, 8 workers

## Limitações

### Implementação Atual (Mock)

Os scripts atuais usam **simulação simplificada** porque:
1. HPM-KD completo requer implementação profunda em PyTorch
2. Alguns datasets requerem download/pré-processamento
3. Training de 20 ensembles × 3 modelos = 60 modelos é demorado

**Características Mock**:
- ✅ Estrutura completa do pipeline
- ✅ Métricas simuladas realistas
- ✅ Análise estatística real
- ✅ Visualizações reais
- ⚠️ Modelos não são treinados de verdade (valores simulados)

### Implementação Real (Futuro)

Para execução real:
1. Implementar HPM-KD completo em PyTorch
2. Baixar 20 datasets reais
3. Treinar 60 teachers (20 datasets × 3 modelos)
4. Executar destilação real
5. Validar métricas empiricamente

**Tempo estimado**: 3-4 semanas
**Hardware recomendado**: GPU NVIDIA RTX 3080+, 32GB RAM

## Referências

### Knowledge Distillation

- Hinton et al. (2015). Distilling the Knowledge in a Neural Network
- Mirzadeh et al. (2020). Improved Knowledge Distillation via Teacher Assistant (TAKD)

### Datasets

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
- OpenML: https://www.openml.org/

### Frameworks

- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- CatBoost: https://catboost.ai/
- PyTorch: https://pytorch.org/

## Próximos Passos

1. ⏳ Implementar HPM-KD real em PyTorch
2. ⏳ Baixar e pré-processar 20 datasets
3. ⏳ Treinar teachers para todos datasets
4. ⏳ Executar baselines
5. ⏳ Executar HPM-KD
6. ⏳ Realizar ablation studies
7. ⏳ Gerar resultados finais
8. ⏳ Integrar no paper

## Tempo Estimado

**Mock (teste)**: ~5 minutos
**Real (completo)**: ~3-4 semanas

---

**Status**: ✅ Estrutura completa, ⏳ Implementação mock, ⏳ Aguarda implementação real
