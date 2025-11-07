# Guia de Experimentos no Google Colab - HPM-KD Framework

**Repositório:** https://github.com/guhaase/papers-deepbridge
**Paper:** HPM-KD: Hierarchical Progressive Multi-Teacher Knowledge Distillation
**Última atualização:** 07 Novembro 2025

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura de Notebooks](#estrutura-de-notebooks)
3. [Configuração Inicial](#configuração-inicial)
4. [Experimentos Principais](#experimentos-principais)
5. [Resultados e Análises](#resultados-e-análises)
6. [Troubleshooting](#troubleshooting)

---

## 📊 Visão Geral

Este guia documenta **14 experimentos** implementados para validar o paper HPM-KD, organizados em **4 notebooks principais** (um por Research Question).

### Research Questions (RQs)

- **RQ1:** Eficiência de Compressão - HPM-KD supera baselines em compression ratio + accuracy?
- **RQ2:** Contribuição de Componentes - Quanto cada componente contribui (ablation studies)?
- **RQ3:** Generalização - HPM-KD generaliza cross-domain e diferentes escalas?
- **RQ4:** Eficiência Computacional - Qual overhead computacional vs traditional KD?

### Experimentos Mapeados

| RQ | Experimentos | Notebook |
|----|--------------|----------|
| **RQ1** | 1, 2, 3, 12 | `01_compression_efficiency.ipynb` |
| **RQ2** | 5, 6, 7, 8, 9 | `02_ablation_studies.ipynb` |
| **RQ3** | 2, 10, 11, 13 | `03_generalization.ipynb` |
| **RQ4** | 4, 14 | `04_computational_efficiency.ipynb` |

---

## 📁 Estrutura de Notebooks

### Ordem de Execução

```
00_setup_colab_UPDATED.ipynb        # SEMPRE EXECUTAR PRIMEIRO
├── 01_compression_efficiency.ipynb # RQ1 - Principal
├── 02_ablation_studies.ipynb       # RQ2 - Componentes
├── 03_generalization.ipynb         # RQ3 - Cross-domain
└── 04_computational_efficiency.ipynb # RQ4 - Performance
```

### Tempos Estimados (GPU T4)

| Notebook | Tempo Estimado | GPU Recomendada |
|----------|----------------|-----------------|
| 00_setup | 5-10 min | Qualquer |
| 01_compression (quick) | 30-45 min | T4/V100 |
| 01_compression (full) | 2-4 horas | V100/A100 |
| 02_ablation | 1-2 horas | T4/V100 |
| 03_generalization | 2-3 horas | V100/A100 |
| 04_efficiency | 30-60 min | T4/V100 |

---

## ⚙️ Configuração Inicial

### 1. Setup do Ambiente

```python
# Executar SEMPRE antes dos experimentos
# Notebook: 00_setup_colab_UPDATED.ipynb

# Este notebook:
# ✅ Verifica GPU
# ✅ Clona repositório papers-deepbridge
# ✅ Instala DeepBridge e dependências
# ✅ Monta Google Drive
# ✅ Cria estrutura de diretórios
# ✅ Salva configuração (colab_config.json)
```

### 2. Estrutura de Diretórios (Google Drive)

```
MyDrive/
└── papers-deepbridge-results/
    └── HPM-KD/
        └── 20251107/                    # Data da execução
            ├── experiments/             # Resultados de experimentos
            │   ├── exp01_compression/
            │   ├── exp02_ablation/
            │   ├── exp03_generalization/
            │   └── exp04_efficiency/
            ├── models/                  # Modelos treinados (.pth)
            ├── figures/                 # Figuras geradas
            ├── logs/                    # Logs de treinamento
            └── colab_config.json        # Configuração do setup
```

### 3. Carregar Configuração (Nos Experimentos)

```python
import json

# Carregar config salva pelo setup
config_path = '/content/drive/MyDrive/papers-deepbridge-results/latest_config.json'
with open(config_path) as f:
    config = json.load(f)

# Usar nas configurações
experiments_dir = config['experiments_dir']
results_dir = config['results_dir']
gpu_name = config['gpu_name']
```

---

## 🧪 Experimentos Principais

### Notebook 1: Compression Efficiency (RQ1)

**Arquivo:** `01_compression_efficiency.ipynb`

**Experimentos incluídos:**
1. **Experimento 1:** Comparison com baselines (7 datasets)
2. **Experimento 2:** Cross-domain generalization (OpenML-CC18)
3. **Experimento 3:** Compression ratio scaling (2-20×)
4. **Experimento 12:** SOTA comparison (CIFAR-100)

**Datasets:**
- **Visão:** MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- **Tabular:** Adult, Credit, Wine Quality
- **OpenML-CC18:** 10 datasets curados

**Outputs:**
- Tabelas de acurácia comparativa
- Gráficos Accuracy vs Compression Ratio
- Statistical significance tests (t-tests)
- Fronteira de Pareto

**Modo Quick (30-45 min):**
```python
# Usar subsets menores
QUICK_MODE = True
MNIST_SAMPLES = 10000      # vs 60000 full
CIFAR_SAMPLES = 10000      # vs 50000 full
EPOCHS_TEACHER = 10        # vs 50 full
EPOCHS_STUDENT = 5         # vs 30 full
```

**Modo Full (2-4 horas):**
```python
QUICK_MODE = False
# Usa datasets completos
# Recomendado: GPU V100 ou A100
```

---

### Notebook 2: Ablation Studies (RQ2)

**Arquivo:** `02_ablation_studies.ipynb`

**Experimentos incluídos:**
5. **Experimento 5:** Ablation per component (6 components)
6. **Experimento 6:** Component interactions (pairwise)
7. **Experimento 7:** Hyperparameter sensitivity
8. **Experimento 8:** Progressive chain length (0-5 steps)
9. **Experimento 9:** Number of teachers (1-8)

**Componentes avaliados:**
1. Progressive Distillation Chain
2. Adaptive Configuration Manager
3. Multi-Teacher Attention
4. Meta-Temperature Scheduler
5. Parallel Processing
6. Shared Optimization Memory

**Outputs:**
- Tabelas de contribuição individual
- Heatmap de interações
- Gráficos de sensibilidade
- Ranking de importância

**Configuração Recomendada:**
```python
DATASET = 'CIFAR-10'  # Representativo e rápido
COMPRESSION_RATIO = 3.1  # Padrão ResNet-56 → ResNet-20
N_RUNS = 5  # Para statistical significance
```

---

### Notebook 3: Generalization (RQ3)

**Arquivo:** `03_generalization.ipynb`

**Experimentos incluídos:**
2. **Experimento 2:** Cross-domain (repetido com foco em análise)
10. **Experimento 10:** Class imbalance robustness (10:1, 50:1, 100:1)
11. **Experimento 11:** Label noise robustness (10%, 20%, 30%)
13. **Experimento 13:** Representation visualization (t-SNE)

**Análises:**
- Performance em diferentes domínios
- Robustez a distribuições desbalanceadas
- Robustez a ruído nos labels
- Similaridade de representações (Silhouette Score)

**Outputs:**
- Boxplots de retenção cross-domain
- Gráficos de degradação com imbalance/noise
- Visualizações t-SNE
- Métricas de qualidade de representação

---

### Notebook 4: Computational Efficiency (RQ4)

**Arquivo:** `04_computational_efficiency.ipynb`

**Experimentos incluídos:**
4. **Experimento 4:** Training time breakdown (profiling)
   - 4.1: Time breakdown per component
   - 4.2: Inference latency + memory
   - 4.3: Speedup with parallelization (1-8 workers)
14. **Experimento 14:** Cost-benefit analysis (Pareto frontier)

**Métricas avaliadas:**
- Training time (total e per-component)
- Inference latency (CPU e GPU)
- Memory footprint
- Speedup efficiency
- Accuracy vs Time trade-off

**Outputs:**
- Stacked bar chart de tempo breakdown
- Speedup curves (workers vs time)
- Pareto frontier (accuracy vs time)
- Cost-benefit tables

---

## 📊 Resultados e Análises

### Estrutura de Resultados

Cada notebook gera automaticamente:

1. **Relatório Markdown** (`{experiment}_report.md`)
   - Configuração do experimento
   - Resultados principais
   - Conclusões

2. **Figuras** (`.png`)
   - Gráficos comparativos
   - Visualizações
   - Tabelas formatadas

3. **Dados Brutos** (`.csv`, `.json`)
   - Métricas detalhadas
   - Logs de treinamento
   - Configurações

4. **Modelos Treinados** (`.pth`)
   - Teachers
   - Students
   - Intermediate models (TAs)

### Consolidação de Resultados

Após executar todos os notebooks, rode:

```python
# No último notebook ou célula separada
from scripts.report_generator import consolidate_all_reports

# Gera relatório consolidado Paper 1
consolidate_all_reports(
    results_dir=results_dir,
    output_file='PAPER1_CONSOLIDATED_REPORT.md'
)
```

Este relatório final incluirá:
- Resumo de todos os 14 experimentos
- Tabelas consolidadas
- Todas as figuras principais
- Conclusões por RQ
- Formato pronto para paper

---

## 🔧 Troubleshooting

### Problemas Comuns

#### 1. GPU Out of Memory

```python
# Reduzir batch size
BATCH_SIZE = 64  # vs 128 ou 256

# Reduzir número de teachers
N_TEACHERS = 3  # vs 4 ou 5

# Usar gradient checkpointing
USE_CHECKPOINT = True

# Limpar cache regularmente
import torch
torch.cuda.empty_cache()
```

#### 2. Timeout no Colab (>12 horas)

```python
# Salvar checkpoints regularmente
SAVE_CHECKPOINT_EVERY = 10  # epochs

# Usar modo Quick para testes
QUICK_MODE = True

# Dividir experimentos em sessões separadas
# Sessão 1: MNIST + Fashion-MNIST
# Sessão 2: CIFAR-10
# Sessão 3: CIFAR-100
```

#### 3. Resultados não salvam no Drive

```python
# Verificar montagem
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)

# Verificar permissões
!ls -la /content/drive/MyDrive/papers-deepbridge-results/
```

#### 4. Imports falham

```python
# Reinstalar DeepBridge
%cd /content/DeepBridge-lib
!pip install -e . --upgrade --force-reinstall

# Verificar sys.path
import sys
print(sys.path)

# Adicionar manualmente se necessário
sys.path.insert(0, '/content/papers-deepbridge')
```

---

## 📝 Checklist de Execução

### Antes de Começar

- [ ] Configurar runtime para GPU (T4 mínimo, V100/A100 recomendado)
- [ ] Executar `00_setup_colab_UPDATED.ipynb` com sucesso
- [ ] Verificar `colab_config.json` foi criado
- [ ] Google Drive montado e diretórios criados
- [ ] GPU disponível e funcionando

### Durante os Experimentos

- [ ] Salvar checkpoints regularmente (a cada 10 epochs)
- [ ] Monitorar uso de memória GPU
- [ ] Verificar que resultados estão sendo salvos no Drive
- [ ] Manter sessão ativa (movimento de mouse/touch)

### Depois dos Experimentos

- [ ] Todos os 4 notebooks executados com sucesso
- [ ] Relatórios `.md` gerados para cada experimento
- [ ] Figuras salvas em `/figures`
- [ ] Modelos salvos em `/models`
- [ ] Consolidar relatório final
- [ ] Backup do Google Drive (download local)

---

## 🚀 Workflow Recomendado

### Dia 1: Setup + Experimento 1 (Quick)

1. Executar `00_setup_colab_UPDATED.ipynb` (10 min)
2. Executar `01_compression_efficiency.ipynb` (QUICK_MODE=True) (45 min)
3. Verificar resultados parciais
4. Ajustar configurações se necessário

### Dia 2: Experimento 1 (Full) + Experimento 2

1. Reexecutar setup (recarregar config)
2. Executar `01_compression_efficiency.ipynb` (QUICK_MODE=False) (3 horas)
3. Executar `02_ablation_studies.ipynb` (2 horas)

### Dia 3: Experimentos 3 + 4

1. Executar `03_generalization.ipynb` (2.5 horas)
2. Executar `04_computational_efficiency.ipynb` (1 hora)
3. Consolidar relatório final
4. Verificar todas as figuras e tabelas

### Total: ~10-12 horas de GPU

**Custo estimado (Colab Pro):** $0 (free tier) a $10 (Colab Pro com A100)

---

## 📚 Referências Rápidas

### Datasets Disponíveis

```python
DATASETS_VISION = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
DATASETS_TABULAR = ['Adult', 'Credit', 'WineQuality']
DATASETS_OPENML = [...]  # 10 datasets OpenML-CC18
```

### Compression Ratios

```python
# MNIST / Fashion-MNIST
TEACHER = 'LeNet5-300-100'   # ~90K params
STUDENT = 'LeNet5-Small'     # ~8.5K params
COMPRESSION = 10.5x

# CIFAR-10 / CIFAR-100
TEACHER = 'ResNet-56'        # 0.85M params
STUDENT = 'ResNet-20'        # 0.27M params
COMPRESSION = 3.1x

# Tabular (Custom)
COMPRESSION = 10-15x (configurável)
```

### Baselines Implementados

```python
BASELINES = [
    'Direct Training',        # Student trained from scratch
    'Traditional KD',         # Hinton et al. 2015
    'FitNets',               # Romero et al. 2015
    'DML',                   # Zhang et al. 2018
    'TAKD',                  # Mirzadeh et al. 2020
    'HPM-KD',                # Ours
]
```

---

## 💡 Dicas de Otimização

### Para GPU T4 (Colab Grátis)

- Use QUICK_MODE para testes iniciais
- Prefira MNIST e Fashion-MNIST (mais rápidos)
- Batch size máximo: 128
- Evite CIFAR-100 com 5+ teachers (OOM)

### Para GPU V100/A100 (Colab Pro)

- Use datasets completos (QUICK_MODE=False)
- Batch size: 256-512
- Todos os experimentos com 5 runs para statistical significance
- Paralelização com 4-8 workers

### Para Economizar Tempo

- Reutilizar teachers treinados (salvar/carregar .pth)
- Pular experimentos redundantes (ex: experimento 2 aparece em RQ1 e RQ3)
- Usar early stopping (monitor validation loss)

---

## ✅ Validação de Resultados

### Métricas Esperadas (Ballpark)

```python
# MNIST
HPMKD_ACCURACY = 99.10 - 99.20%
RETENTION_RATE = 99.80 - 99.90%

# CIFAR-10
HPMKD_ACCURACY = 92.00 - 92.50%
RETENTION_RATE = 98.50 - 99.00%

# CIFAR-100
HPMKD_ACCURACY = 70.50 - 71.50%
RETENTION_RATE = 95.50 - 96.50%

# Ablation (CIFAR-10)
PROGRESSIVE_CHAIN_IMPACT = -2.0 to -3.0pp
ADAPTIVE_CONFIG_IMPACT = -1.5 to -2.0pp
```

Se seus resultados estiverem **significativamente** diferentes (>2pp), verifique:
- Configuração de hiperparâmetros
- Seeds aleatórios (deve usar múltiplas)
- GPU/precisão numérica
- Versões de bibliotecas

---

**FIM DO GUIA**

Para questões ou bugs, abra uma issue em:
https://github.com/guhaase/papers-deepbridge/issues

Boa sorte com os experimentos! 🚀
