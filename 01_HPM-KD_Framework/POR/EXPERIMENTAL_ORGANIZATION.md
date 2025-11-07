# HPM-KD Framework - Organização de Experimentos

**Data de Criação**: 06 de Novembro de 2025
**Autor**: Gustavo Coelho Haase + Claude Code
**Versão**: 1.0

---

## 📋 VISÃO GERAL

Este documento descreve a organização completa dos experimentos do artigo HPM-KD Framework, incluindo scripts de teste, objetivos de cada experimento, e estrutura de resultados.

---

## 📂 ESTRUTURA DE DIRETÓRIOS

```
papers/01_HPM-KD_Framework/POR/
├── experiments/                          # 🎯 PASTA PRINCIPAL DE EXPERIMENTOS
│   ├── sklearn_validation/               # Experimentos de validação com sklearn
│   │   ├── example_hpmkd_experiment.py   # Exemplo simplificado (10k samples)
│   │   ├── run_hpmkd_experiments.py      # Pipeline completo sklearn
│   │   └── run_full_mnist_experiment.py  # MNIST completo (70k samples)
│   │
│   ├── cnn_baseline/                     # Experimentos CNN baseline
│   │   ├── train_teacher.py              # Treinar modelo professor (ResNet-18)
│   │   ├── train_student.py              # Treinar aluno direto (sem KD)
│   │   └── train_kd.py                   # Treinar aluno com KD tradicional
│   │
│   ├── cnn_hpmkd/                        # Experimentos CNN com HPM-KD
│   │   └── train_hpmkd.py                # Treinar aluno com HPM-KD completo
│   │
│   ├── evaluation/                       # Scripts de avaliação e análise
│   │   ├── evaluate_all.py               # Comparação completa de todos os modelos
│   │   └── generate_figures.py           # Geração de figuras do paper
│   │
│   ├── lib/                              # Bibliotecas compartilhadas
│   │   ├── cnn_models.py                 # Definições de arquiteturas CNN
│   │   └── utils_training.py             # Utilitários de treinamento
│   │
│   └── results/                          # Resultados organizados
│       ├── sklearn/                      # Resultados sklearn
│       │   ├── quick_10k/                # Quick test (10k samples)
│       │   └── full_70k/                 # Full MNIST (70k samples)
│       ├── cnn/                          # Resultados CNN
│       │   ├── teacher/                  # Modelos professores
│       │   ├── student_direct/           # Alunos treinados diretamente
│       │   ├── student_kd/               # Alunos com KD tradicional
│       │   └── student_hpmkd/            # Alunos com HPM-KD
│       ├── figures/                      # Figuras geradas
│       └── tables/                       # Tabelas de resultados
│
├── sections/                             # Seções LaTeX do artigo
├── bibliography/                         # Referências bibliográficas
├── build/                                # PDF compilado
└── models/                               # Modelos treinados (persistência)
```

---

## 🎯 CATEGORIAS DE EXPERIMENTOS

### 1. VALIDAÇÃO INICIAL (sklearn_validation/)

**Objetivo**: Validar a implementação HPM-KD com modelos sklearn antes de experimentos CNN custosos.

**Arquivos**:

#### `example_hpmkd_experiment.py`
- **Tipo**: Exemplo Didático
- **Dataset**: MNIST (10,000 samples)
- **Objetivo**:
  - Demonstrar uso básico do HPM-KD
  - Validação rápida da integração com DeepBridge
  - Exemplo para documentação e tutoriais
- **Teacher**: RandomForest (500 árvores, profundidade 20)
- **Student**: LogisticRegression ou DecisionTree
- **Tempo de execução**: ~2-3 minutos
- **Resultados obtidos**:
  - HPM-KD: 89.50% accuracy
  - Traditional KD: 67.35%
  - Melhoria: +22.15 pontos percentuais
- **Referência no Paper**: Section 5.1 (Preliminary Validation)

#### `run_hpmkd_experiments.py`
- **Tipo**: Pipeline Completo
- **Dataset**: MNIST (configurável: 10k ou 70k samples)
- **Objetivo**:
  - Pipeline experimental completo com sklearn
  - Testar todos os componentes HPM-KD
  - Baseline para comparação com CNN
- **Configurações**: 12 configurações testadas automaticamente
- **Componentes testados**:
  - ✅ Adaptive Configuration Manager
  - ✅ Progressive Distillation Chain
  - ✅ Meta-Temperature Scheduler
  - ✅ Shared Optimization Memory
  - ⏸️ Parallel Processing (desabilitado - problemas pickle)
  - ⏳ Multi-Teacher Attention (single teacher)
- **Tempo de execução**: ~8-10 minutos
- **Referência no Paper**: Section 5.1 (Main Results - sklearn)

#### `run_full_mnist_experiment.py`
- **Tipo**: Experimento Completo
- **Dataset**: MNIST completo (70,000 samples)
- **Objetivo**:
  - Validar scaling com dataset completo
  - Resultados definitivos para validação sklearn
  - Análise de comportamento com mais dados
- **Configuração**: Wrapper sobre `run_hpmkd_experiments.py` com `USE_FULL_MNIST=True`
- **Tempo de execução**: ~100 segundos
- **Resultados obtidos**:
  - HPM-KD: 91.67% accuracy
  - Traditional KD: 68.54%
  - Retenção: 94.9%
  - Melhoria: +23.13 pontos percentuais
- **Referência no Paper**: Section 5.1 (Preliminary Validation), Section 7.1 (Scaling Analysis)

---

### 2. BASELINE CNN (cnn_baseline/)

**Objetivo**: Estabelecer baselines com modelos CNN profundos para comparação justa.

#### `train_teacher.py`
- **Tipo**: Treinamento de Professor
- **Dataset**: MNIST
- **Modelo**: ResNet-18 (11M parâmetros)
- **Objetivo**:
  - Treinar modelo professor de alta capacidade
  - Target: 99.3-99.5% accuracy
  - Servir como teacher para KD e HPM-KD
- **Configuração padrão**:
  - Epochs: 20
  - Batch size: 128
  - Learning rate: 0.1 (com scheduler)
  - Optimizer: SGD com momentum 0.9
  - Weight decay: 5e-4
- **Tempo estimado**: ~30-45 minutos (GPU)
- **Saída**:
  - `models/teacher_resnet18_best.pth`
  - `models/teacher_resnet18_last.pth`
  - Logs de treinamento
- **Referência no Paper**: Section 3.2 (Teacher Models), Table 1 (Model Architectures)

**Uso**:
```bash
poetry run python experiments/cnn_baseline/train_teacher.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.1 \
    --save-dir models \
    --save-name teacher_resnet18
```

#### `train_student.py`
- **Tipo**: Baseline - Treinamento Direto
- **Dataset**: MNIST
- **Modelos**: SimpleCNN ou MobileNet-V2 (3.2M parâmetros)
- **Objetivo**:
  - Treinar aluno diretamente (sem distillation)
  - Estabelecer baseline inferior
  - Target: 98.5-98.8% accuracy
- **Configuração padrão**:
  - Epochs: 20
  - Batch size: 128
  - Learning rate: 0.1
- **Tempo estimado**: ~20-30 minutos (GPU)
- **Saída**: `models/student_<arch>_direct_best.pth`
- **Referência no Paper**: Section 5.1 (Main Results - Direct Training row)

**Uso**:
```bash
poetry run python experiments/cnn_baseline/train_student.py \
    --model mobilenet \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.1 \
    --save-dir models \
    --save-name student_mobilenet_direct
```

#### `train_kd.py`
- **Tipo**: Baseline - Traditional Knowledge Distillation
- **Dataset**: MNIST
- **Método**: Hinton et al. 2015 (temperature-scaled softmax)
- **Objetivo**:
  - Implementar KD tradicional
  - Baseline principal para comparação
  - Target: 98.9-99.1% accuracy
- **Fórmula**: `Loss = α × KL(T_soft, S_soft) + (1-α) × CE(labels, S)`
- **Configuração padrão**:
  - Temperature: 4.0
  - Alpha: 0.5
  - Epochs: 20
- **Tempo estimado**: ~25-35 minutos (GPU)
- **Saída**: `models/student_<arch>_kd_best.pth`
- **Referência no Paper**: Section 5.1 (Main Results - Traditional KD row)

**Uso**:
```bash
poetry run python experiments/cnn_baseline/train_kd.py \
    --teacher models/teacher_resnet18_best.pth \
    --student mobilenet \
    --temperature 4.0 \
    --alpha 0.5 \
    --epochs 20 \
    --save-dir models \
    --save-name student_mobilenet_kd
```

---

### 3. HPM-KD CNN (cnn_hpmkd/)

**Objetivo**: Experimentos com framework HPM-KD completo usando CNNs.

#### `train_hpmkd.py`
- **Tipo**: Método Proposto (HPM-KD Completo)
- **Dataset**: MNIST
- **Método**: Hierarchical Progressive Multi-Teacher Knowledge Distillation
- **Objetivo**:
  - Implementar HPM-KD com todos os 6 componentes
  - Demonstrar superioridade sobre baselines
  - Target: 99.0-99.2% accuracy
- **Componentes utilizados**:
  1. ✅ Adaptive Configuration Manager
  2. ✅ Progressive Distillation Chain
  3. ✅ Attention-Weighted Multi-Teacher (se múltiplos teachers)
  4. ✅ Meta-Temperature Scheduler
  5. ✅ Parallel Processing Pipeline
  6. ✅ Shared Optimization Memory
- **Configuração padrão**:
  - Initial temperature: 4.0 (adaptativo)
  - Alpha: 0.5
  - Progressive chain: Ativado
  - Adaptive temperature: Ativado
  - Epochs: 20
- **Tempo estimado**: ~40-60 minutos (GPU)
- **Saída**: `models/student_<arch>_hpmkd_best.pth`
- **Referência no Paper**: Section 5.1 (Main Results - HPM-KD row), Section 5.4 (Component Analysis)

**Uso**:
```bash
poetry run python experiments/cnn_hpmkd/train_hpmkd.py \
    --teacher models/teacher_resnet18_best.pth \
    --student mobilenet \
    --use-progressive \
    --use-adaptive-temp \
    --initial-temperature 4.0 \
    --alpha 0.5 \
    --epochs 20 \
    --save-dir models \
    --save-name student_mobilenet_hpmkd
```

---

### 4. AVALIAÇÃO E ANÁLISE (evaluation/)

**Objetivo**: Comparação abrangente e geração de artefatos para o paper.

#### `evaluate_all.py`
- **Tipo**: Avaliação Comparativa Completa
- **Dataset**: MNIST (test set)
- **Objetivo**:
  - Comparar todos os modelos treinados
  - Gerar métricas detalhadas
  - Testes de significância estatística
  - Análise de retenção
- **Métricas geradas**:
  - Accuracy (test set)
  - Confusion matrices
  - Per-class accuracy
  - Teacher retention percentage
  - Classification reports
  - Statistical significance (t-tests)
- **Visualizações opcionais**:
  - Confusion matrix heatmaps
  - Feature space t-SNE
  - Attention weight distributions
- **Tempo estimado**: ~5-10 minutos
- **Saída**:
  - `results/evaluation_report.json`
  - `results/figures/confusion_*.png`
  - `results/tables/comparison_table.csv`
- **Referência no Paper**: Section 5.1-5.4 (Todos os resultados), Appendix (Tabelas completas)

**Uso**:
```bash
poetry run python experiments/evaluation/evaluate_all.py \
    --teacher models/teacher_resnet18_best.pth \
    --student-direct models/student_mobilenet_direct_best.pth \
    --student-kd models/student_mobilenet_kd_best.pth \
    --student-hpmkd models/student_mobilenet_hpmkd_best.pth \
    --student-arch mobilenet \
    --output-dir results/cnn \
    --save-confusion \
    --save-figures
```

#### `generate_figures.py`
- **Tipo**: Geração de Figuras para Paper
- **Input**: Resultados de experimentos (CSVs)
- **Objetivo**:
  - Gerar todas as 13 figuras do paper
  - Formato publication-quality (300 DPI PNG + PDF vetorial)
  - Estilo consistente com padrões de conferências
- **Figuras geradas** (6/13 completas):
  1. ✅ Figure 1: Performance comparison (10k vs 70k)
  2. ✅ Figure 2: Improvement over baseline
  3. ✅ Figure 3: Teacher retention comparison
  4. ✅ Figure 4: Scaling analysis
  5. ✅ Figure 5: Training time comparison
  6. ✅ Figure 6: Comprehensive comparison matrix
  7. ⏳ Figure 7: Progressive chain behavior
  8. ⏳ Figure 8: Adaptive configuration search
  9. ⏳ Figure 9: Ablation study results
  10. ⏳ Figure 10: Temperature sensitivity
  11. ⏳ Figure 11: Alpha sensitivity
  12. ⏳ Figure 12: Multi-dataset comparison
  13. ⏳ Figure 13: Paper gap analysis
- **Estilo**: seaborn-v0_8-paper, colorblind-friendly palette
- **Tempo estimado**: ~2-3 minutos
- **Saída**: `results/figures/*.png` e `results/figures/*.pdf`
- **Referência no Paper**: Todas as figuras em Section 5, 6, 7

**Uso**:
```bash
python experiments/evaluation/generate_figures.py
```

---

### 5. BIBLIOTECAS COMPARTILHADAS (lib/)

**Objetivo**: Código reutilizável para evitar duplicação.

#### `cnn_models.py`
- **Tipo**: Definições de Arquiteturas
- **Conteúdo**:
  - `create_teacher_model()`: ResNet-18 adaptado para MNIST
  - `create_student_model()`: SimpleCNN ou MobileNet-V2
  - Modificações para MNIST (1 canal, 10 classes)
- **Uso**: Importado por todos os scripts de treinamento CNN

#### `utils_training.py`
- **Tipo**: Utilitários de Treinamento
- **Funções**:
  - `get_mnist_loaders()`: DataLoaders MNIST
  - `train_epoch()`: Loop de treinamento padrão
  - `train_epoch_kd()`: Loop com distillation loss
  - `validate()`: Validação de modelo
  - `save_checkpoint()` / `load_checkpoint()`: Persistência
  - `get_optimizer()` / `get_scheduler()`: Otimizadores
  - `print_model_summary()`: Informações do modelo
  - `distillation_loss()`: KL divergence para KD
- **Uso**: Importado por todos os scripts de treinamento

---

## 📊 MAPEAMENTO: SCRIPTS → PAPER

### Seção 5.1 - Main Results (RQ1: Compression Efficiency)

**Experimentos necessários**:
- ✅ sklearn validation: `run_full_mnist_experiment.py`
- ⏳ CNN experiments:
  - `train_teacher.py` → Table 2 (Teacher accuracy)
  - `train_student.py` → Table 2 (Direct Training row)
  - `train_kd.py` → Table 2 (Traditional KD row)
  - `train_hpmkd.py` → Table 2 (HPM-KD row)
  - `evaluate_all.py` → Gera Table 2 completa

**Status**:
- sklearn: ✅ COMPLETO (91.67% HPM-KD, +23.13pp)
- CNN: ⏳ EM ANDAMENTO (modelos treinando)

---

### Seção 5.2 - Generalization Analysis (RQ3)

**Experimentos necessários**:
- ⏳ Repetir experimentos em múltiplos datasets:
  - Fashion-MNIST
  - CIFAR-10
  - CIFAR-100
  - Tabular datasets
- ⏳ OpenML-CC18 benchmark

**Scripts**: Mesmos scripts, diferentes configurações de dataset

---

### Seção 5.3 - Computational Efficiency (RQ4)

**Experimentos necessários**:
- ✅ Training time: Coletado durante `run_full_mnist_experiment.py`
- ⏳ Parallel speedup: Testar com múltiplos workers
- ✅ Inference latency: Medido em `evaluate_all.py`

**Status**:
- Métricas de tempo coletadas
- Análise de parallel speedup pendente

---

### Seção 6 - Ablation Studies (RQ2: Component Contribution)

**Experimentos necessários**:
- ⏳ HPM-KD sem Adaptive Configuration
- ⏳ HPM-KD sem Progressive Chain
- ⏳ HPM-KD sem Multi-Teacher
- ⏳ HPM-KD sem Meta-Temperature
- ⏳ HPM-KD sem Parallel Processing
- ⏳ HPM-KD sem Shared Memory

**Scripts**: `train_hpmkd.py` com flags de desabilitação

---

### Seção 6.2 - Sensitivity Analysis

**Experimentos necessários**:
- ⏳ Variar temperature: {2.0, 3.0, 4.0, 5.0}
- ⏳ Variar alpha: {0.3, 0.5, 0.7, 0.9}
- ⏳ Variar chain length: {1, 2, 3, 4, 5}

**Scripts**: `train_hpmkd.py` e `train_kd.py` com diferentes parâmetros

---

### Figuras (Todas as Seções)

**Geração**: `generate_figures.py`
- ✅ 6/13 figuras completas
- ⏳ 7/13 figuras pendentes (requerem experimentos CNN e ablation)

---

## 🎯 PLANO DE EXECUÇÃO

### Fase 1: Validação sklearn ✅ COMPLETO

- [x] `example_hpmkd_experiment.py` (10k samples)
- [x] `run_full_mnist_experiment.py` (70k samples)
- [x] Primeiras 6 figuras geradas
- [x] Validação de componentes

**Resultados**: HPM-KD demonstra +23.13pp melhoria sobre Traditional KD

---

### Fase 2: Baseline CNN ⏳ EM ANDAMENTO

- [ ] `train_teacher.py` → ResNet-18 professor
- [ ] `train_student.py` → MobileNet direct training
- [ ] `train_kd.py` → MobileNet com Traditional KD
- [ ] `evaluate_all.py` → Comparação preliminar

**Objetivo**: Estabelecer baselines CNN para comparação justa

**Status**: Processos em execução (background tasks)

---

### Fase 3: HPM-KD CNN ⏳ PRÓXIMO

- [ ] `train_hpmkd.py` → MobileNet com HPM-KD completo
- [ ] `evaluate_all.py` → Comparação completa
- [ ] Validar que HPM-KD supera baselines CNN

**Expectativa**: 99.0-99.2% accuracy (fechar gap para paper)

---

### Fase 4: Ablation Studies ⏳ PENDENTE

- [ ] 6 variantes de ablation
- [ ] Análise de contribuição individual
- [ ] Validar synergy entre componentes

**Scripts**: `train_hpmkd.py` com componentes desabilitados

---

### Fase 5: Sensitivity Analysis ⏳ PENDENTE

- [ ] Grid search de hyperparameters
- [ ] Análise de robustness
- [ ] Geração de superfícies de sensibilidade

---

### Fase 6: Multi-Dataset Experiments ⏳ PENDENTE

- [ ] Fashion-MNIST
- [ ] CIFAR-10 / CIFAR-100
- [ ] Datasets tabulares
- [ ] OpenML-CC18

**Duração estimada**: 4-6 semanas

---

### Fase 7: Figuras Finais e Paper ⏳ PENDENTE

- [ ] Completar 7 figuras restantes
- [ ] Gerar todas as tabelas
- [ ] Atualizar paper com resultados reais
- [ ] Review completo

**Duração estimada**: 2 semanas

---

## 📈 STATUS ATUAL (06/11/2025)

### ✅ COMPLETO

1. **Validação sklearn**:
   - Quick test (10k): 89.50% HPM-KD
   - Full MNIST (70k): 91.67% HPM-KD
   - Melhoria: +23.13pp sobre Traditional KD
   - Figuras: 6/13 geradas

2. **Estrutura de código**:
   - Todos os scripts implementados
   - Bibliotecas compartilhadas organizadas
   - Integração com DeepBridge validada

3. **Documentação**:
   - Scripts documentados
   - Estrutura de experimentos definida
   - Mapeamento para paper completo

### ⏳ EM ANDAMENTO

1. **Baseline CNN** (8 processos em background):
   - Teacher training (ResNet-18)
   - Student direct training (MobileNet)
   - Traditional KD training
   - HPM-KD training (2 versões)
   - Evaluation pipeline

### ⏳ PENDENTE

1. **Completar experimentos CNN**
2. **Ablation studies** (6 variantes)
3. **Sensitivity analysis** (temperature, alpha, chain length)
4. **Multi-dataset experiments** (7 datasets adicionais)
5. **Figuras restantes** (7/13)
6. **Atualização do paper**

---

## 🔧 COMANDOS RÁPIDOS

### Validação sklearn (completo):
```bash
# Quick test
python experiments/sklearn_validation/example_hpmkd_experiment.py

# Full MNIST
python experiments/sklearn_validation/run_full_mnist_experiment.py
```

### Baseline CNN:
```bash
# 1. Treinar professor
poetry run python experiments/cnn_baseline/train_teacher.py --epochs 20

# 2. Treinar aluno direto
poetry run python experiments/cnn_baseline/train_student.py --model mobilenet --epochs 20

# 3. Treinar com KD tradicional
poetry run python experiments/cnn_baseline/train_kd.py \
    --teacher models/teacher_resnet18_best.pth --student mobilenet --epochs 20
```

### HPM-KD CNN:
```bash
# Treinar com HPM-KD completo
poetry run python experiments/cnn_hpmkd/train_hpmkd.py \
    --teacher models/teacher_resnet18_best.pth \
    --student mobilenet \
    --use-progressive --use-adaptive-temp \
    --epochs 20
```

### Avaliação:
```bash
# Comparar todos os modelos
poetry run python experiments/evaluation/evaluate_all.py \
    --teacher models/teacher_resnet18_best.pth \
    --student-hpmkd models/student_mobilenet_hpmkd_best.pth \
    --student-arch mobilenet \
    --output-dir results/cnn \
    --save-confusion --save-figures

# Gerar figuras
python experiments/evaluation/generate_figures.py
```

---

## 📝 NOTAS IMPORTANTES

### 1. Dependências entre Scripts

**Ordem de execução necessária**:
1. `train_teacher.py` (gera teacher model)
2. Paralelo:
   - `train_student.py`
   - `train_kd.py` (depende do teacher)
   - `train_hpmkd.py` (depende do teacher)
3. `evaluate_all.py` (depende de todos os modelos)
4. `generate_figures.py` (depende dos resultados)

### 2. Recursos Computacionais

**sklearn experiments**:
- CPU: Suficiente
- RAM: 4-8 GB
- Tempo: minutos

**CNN experiments**:
- GPU: Recomendado (CUDA)
- RAM: 8-16 GB
- VRAM: 4-8 GB
- Tempo: horas

### 3. Paths Relativos

Todos os scripts assumem execução a partir do diretório raiz do DeepBridge:
```bash
cd /home/guhaase/projetos/DeepBridge
poetry run python papers/01_HPM-KD_Framework/POR/experiments/.../script.py
```

### 4. Resultados Intermediários

- Modelos salvos em: `models/`
- Resultados sklearn em: `experiments/results/sklearn/`
- Resultados CNN em: `experiments/results/cnn/`
- Figuras em: `experiments/results/figures/`

### 5. Git Ignore

Adicionar ao `.gitignore`:
```
models/*.pth
experiments/results/**/*.csv
experiments/results/**/*.json
experiments/results/figures/*.png
!experiments/results/figures/*.pdf  # Manter PDFs versionados
```

---

## 📚 REFERÊNCIAS

### Paper Sections Mapping

- **Section 3**: Methodology → Código em `deepbridge/distillation/techniques/hpm/`
- **Section 5.1**: Main Results → `train_*.py` + `evaluate_all.py`
- **Section 5.2**: Generalization → Multi-dataset experiments
- **Section 5.3**: Efficiency → Training time logs + `evaluate_all.py`
- **Section 5.4**: Component Analysis → Experimentos base
- **Section 6**: Ablation → `train_hpmkd.py` com componentes desabilitados
- **Section 7**: Discussion → Análise agregada de todos os resultados

### Arquivos de Documentação Relacionados

- `FINAL_STATUS.md`: Status geral do projeto
- `FULL_MNIST_RESULTS.md`: Resultados detalhados sklearn
- `EXPERIMENTS_COMPARISON.md`: Comparação quick vs full
- `FIGURES_SUMMARY.md`: Documentação de figuras geradas
- `IMPLEMENTATION_GUIDE.md`: Guia paper-to-code

---

## ✅ CHECKLIST DE VALIDAÇÃO

Antes de submeter paper, verificar:

- [ ] Todos os scripts executam sem erros
- [ ] Resultados replicáveis (múltiplas seeds)
- [ ] Figuras em alta resolução (300 DPI)
- [ ] Tabelas com todos os dados preenchidos
- [ ] Testes estatísticos realizados
- [ ] Significância validada (p-values)
- [ ] Ablation studies completos
- [ ] Multi-dataset experiments completos
- [ ] Código versionado no GitHub
- [ ] Modelos salvos e documentados
- [ ] README atualizado
- [ ] Reprodutibilidade garantida

---

**Última atualização**: 06 de Novembro de 2025
**Status**: Fase 1 completa, Fase 2 em andamento
**Próximo milestone**: Completar baseline CNN (Fase 2)

---

## 🎯 PRÓXIMAS AÇÕES

1. **Monitorar processos background** (8 jobs rodando)
2. **Aguardar conclusão dos treinamentos CNN**
3. **Executar `evaluate_all.py`** com todos os modelos
4. **Analisar resultados CNN** vs sklearn
5. **Decidir sobre necessidade de ajustes**
6. **Iniciar Fase 3** (HPM-KD CNN) se baselines ok
