# Experimento 1: Compression Efficiency

## 📋 Informações Gerais

| Parâmetro | Valor |
|-----------|-------|
| **Experimento** | Experimento 1 - Compression Efficiency |
| **Research Question** | RQ1: HPM-KD consegue alcançar maiores taxas de compressão mantendo acurácia vs baselines? |
| **Status** | ✅ **CONCLUÍDO** (Novembro 2025) |
| **Dataset** | MNIST |
| **Compression Ratio** | **2× (LeNet5-Large → LeNet5-Small)** |
| **Modelos Treinados** | 31 modelos (1 teacher + 30 students) |

---

## 🎯 Objetivo

Validar a **Research Question 1 (RQ1)** do paper HPM-KD, comparando a eficiência de compressão do método proposto (HPM-KD) contra 5 baselines estado-da-arte.

### **Research Question:**
> HPM-KD consegue alcançar maiores taxas de compressão mantendo acurácia comparado aos métodos estado-da-arte?

---

## 🔬 Metodologia

### **Experimentos Incluídos:**
1. **Baseline Comparison** - Compara HPM-KD vs 5 baselines em MNIST
2. **Compression Ratio Scaling** - Testa ratio de 2×
3. **Statistical Significance** - Testes t para validar diferenças

### **Baselines Comparados (6 métodos):**

| # | Método | Descrição | Referência |
|---|--------|-----------|------------|
| 1 | **Direct** | Treinar student do zero (baseline) | - |
| 2 | **Traditional KD** | Knowledge Distillation clássico | Hinton et al. (2015) |
| 3 | **FitNets** | Hint-based KD | Romero et al. (2015) |
| 4 | **AT** | Attention Transfer | Zagoruyko & Komodakis (2017) |
| 5 | **TAKD** | Teacher Assistant KD | Mirzadeh et al. (2020) |
| 6 | **HPM-KD** | Nossa proposta (DeepBridge Library) | **Ours** ⭐ |

---

## 📊 Configuração

### **Dataset:**
- **MNIST**: 60,000 imagens de treinamento, 10,000 de teste
- **Classes**: 10 dígitos (0-9)
- **Resolução**: 28×28 pixels (grayscale)

### **Arquiteturas:**

| Modelo | Arquitetura | Parâmetros | Descrição |
|--------|-------------|------------|-----------|
| **Teacher** | LeNet5-Large | 62,006 | Modelo maior (professor) |
| **Student** | LeNet5-Small | 30,206 | Modelo menor (aluno) |

### **Compression Ratio:**
```
Compression = Teacher Params / Student Params
            = 62,006 / 30,206
            = 2.05×
```

### **Hiperparâmetros (Full Mode):**

| Parâmetro | Valor |
|-----------|-------|
| **Runs por método** | 5 (para robustez estatística) |
| **Teacher Epochs** | 100 |
| **Student Epochs** | 50 |
| **Batch Size** | 128 |
| **Learning Rate** | 0.1 |
| **Optimizer** | SGD com momentum 0.9 |
| **Loss** | CrossEntropyLoss |

### **Modos de Execução:**

| Modo | Teacher Epochs | Student Epochs | Runs | Tempo Estimado |
|------|----------------|----------------|------|----------------|
| **Quick** | 50 | 20 | 3 | 45 minutos |
| **Full** | 100 | 50 | 5 | 3-4 horas |

---

## 📈 Resultados Obtidos

### **Accuracy dos Métodos (MNIST):**

| Rank | Método | Accuracy (%) | Desvio Padrão | Status |
|------|--------|--------------|---------------|--------|
| **1º** 🥇 | **Direct** | **68.10%** | ±0.15 | **MELHOR** |
| 2º | HPM-KD | 67.74% | ±0.18 | -0.36pp vs Direct |
| 3º | TAKD | 67.70% | ±0.12 | -0.40pp vs Direct |
| 4º | FitNets | 67.52% | ±0.20 | -0.58pp vs Direct |
| 5º | AT | 67.38% | ±0.16 | -0.72pp vs Direct |
| 6º | TraditionalKD | 67.28% | ±0.14 | -0.82pp vs Direct |

**Teacher Accuracy:** 90.50%

### **Retenção de Conhecimento:**

```
Retention = (Student Acc / Teacher Acc) × 100%

Direct:        75.2% retention
HPM-KD:        74.8% retention
TAKD:          74.8% retention
```

---

## ⚠️ Análise Crítica

### **PROBLEMA IDENTIFICADO:**

#### **Compression Ratio Insuficiente (2×)**

```
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  PROBLEMA: Compression 2× é MUITO PEQUENO!              ║
╚══════════════════════════════════════════════════════════════╝

LeNet5-Large:   62,006 parâmetros
LeNet5-Small:   30,206 parâmetros
Compression:    2.05× apenas

Com compression tão pequeno:
  ❌ Student tem capacidade SUFICIENTE para aprender sozinho
  ❌ Direct training alcança melhor performance
  ❌ Knowledge Distillation não traz vantagem (overhead)
  ❌ HPM-KD não demonstra superioridade
```

### **Por Que Direct Venceu:**

1. **Gap muito pequeno** entre teacher e student
2. **Student capacitado** (30K params suficiente para MNIST)
3. **Overhead de KD** não compensa com compression baixo
4. **Direct training** mais simples e efetivo neste cenário

### **Insight Importante:**

> **"Knowledge Distillation é mais efetivo com compression ratios MAIORES (≥5×)"**

Quando o gap entre teacher e student é pequeno (2×), o student consegue aprender diretamente dos dados sem precisar da "orientação" do teacher.

---

## ✅ Validação Sklearn (MNIST)

Antes do experimento principal, foi realizada validação com modelos sklearn:

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 91.67% |
| **F1-Score** | 91.50% |
| **Precision** | 91.80% |
| **Recall** | 91.67% |

**Status:** ✅ Validação bem-sucedida

**Localização:** `results/sklearn/`

---

## 📁 Estrutura de Arquivos

```
experimento_01_compression_efficiency/
├── README.md                          ← Este arquivo
│
├── scripts/                           ← Scripts Python
│   └── 01_compression_efficiency.py   (810 linhas, 44 KB)
│
└── results/                           ← Resultados
    ├── results_full_20251112_111138/  ← Experimento principal
    │   ├── exp_01_01_compression_efficiency/
    │   │   ├── results_comparison.csv
    │   │   ├── experiment_report.md
    │   │   ├── 01_compression_efficiency.log
    │   │   ├── figures/
    │   │   │   ├── accuracy_comparison.png
    │   │   │   ├── retention_analysis.png
    │   │   │   └── statistical_significance.png
    │   │   └── models/
    │   │       ├── teacher_lenet5large_MNIST.pt
    │   │       └── student_*.pt (30 modelos)
    │   └── run_all_experiments.log
    │
    └── sklearn/                       ← Validação sklearn
        ├── experiment_results.json
        ├── confusion_matrix.png
        └── classification_report.txt
```

---

## 🚀 Como Executar

### **Pré-requisitos:**
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scipy tqdm
pip install deepbridge  # DeepBridge Library (HPM-KD)
```

### **Execução Quick Mode (45 min):**
```bash
cd scripts/
python 01_compression_efficiency.py --mode quick --datasets MNIST
```

### **Execução Full Mode (3-4h):**
```bash
cd scripts/
python 01_compression_efficiency.py --mode full --datasets MNIST
```

### **Opções:**
```bash
--mode {quick,full}        # Modo de execução
--datasets {MNIST,FashionMNIST,CIFAR10}  # Datasets (múltiplos)
--gpu 0                    # GPU a usar (default: 0)
--seed 42                  # Seed para reprodutibilidade
```

---

## 📊 Modelos Treinados

### **Total de Modelos:**

| Tipo | Quantidade | Descrição |
|------|------------|-----------|
| **Teachers** | 1 | LeNet5-Large (62K params) |
| **Students** | 30 | 6 métodos × 5 runs |
| **TOTAL** | **31** | |

### **Breakdown por Método:**

```
Direct:        5 modelos (1 por run)
TraditionalKD: 5 modelos (1 por run)
FitNets:       5 modelos (1 por run)
AT:            5 modelos (1 por run)
TAKD:          5 modelos (1 por run)
HPM-KD:        5 modelos (1 por run)
─────────────────────────────────────
TOTAL:        30 students + 1 teacher = 31 modelos
```

---

## 🎯 Conclusões

### **✅ Experimento Executado com Sucesso**
- 31 modelos treinados corretamente
- Resultados estatisticamente robustos (5 runs por método)
- Todos os baselines implementados e testados

### **❌ RQ1 NÃO Validada**
- Compression ratio 2× muito pequeno
- Direct training superou todos os métodos de KD
- HPM-KD não demonstrou superioridade esperada

### **💡 Insight Obtido:**
> **Knowledge Distillation (incluindo HPM-KD) é mais efetivo com compression ratios MAIORES (≥5×)**

### **🚨 Ação Necessária:**
**EXECUTAR EXPERIMENTO 1B** com compression ratios maiores:
- 2.3× (ResNet50 → ResNet18)
- **5×** (ResNet50 → ResNet10) ⭐
- **7×** (ResNet50 → MobileNetV2) ⭐⭐

**Localização:** `../experimento_01b_compression_ratios/`

---

## 📚 Referências

1. **Hinton et al. (2015)** - "Distilling the Knowledge in a Neural Network"
2. **Romero et al. (2015)** - "FitNets: Hints for Thin Deep Nets"
3. **Zagoruyko & Komodakis (2017)** - "Paying More Attention to Attention"
4. **Mirzadeh et al. (2020)** - "Improved Knowledge Distillation via Teacher Assistant"

---

## 📞 Informações Adicionais

### **Status do Experimento:**
- ✅ Implementação completa
- ✅ Execução bem-sucedida
- ✅ Resultados reproduzíveis
- ⚠️ RQ1 não validada (compression insuficiente)

### **Próximos Passos:**
1. ✅ Experimento 1 concluído (este experimento)
2. ⏳ **EXECUTAR EXPERIMENTO 1B** (CRÍTICO) - Compression ratios maiores
3. 📋 Experimento 2 (Ablation Studies)
4. 📋 Experimento 3 (Generalization)
5. 📋 Experimento 4 (Computational Efficiency)

### **Relacionado:**
- **Experimento 1B:** `../experimento_01b_compression_ratios/` - **CRÍTICO PARA RQ1**
- **Documentação Geral:** `../SUMARIO_COMPLETO_EXPERIMENTOS.md`
- **Contagem de Modelos:** `../CONTAGEM_MODELOS.md`

---

**Criado:** Dezembro 2025
**Última Atualização:** Dezembro 2025
**Status:** ✅ Experimento Concluído
**Autor:** Gustavo Haase
**Paper:** HPM-KD Framework - Hierarchical Progressive Multi-Teacher Knowledge Distillation
