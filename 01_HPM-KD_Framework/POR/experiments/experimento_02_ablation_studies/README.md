# Experimento 2: Ablation Studies

## 📋 Informações Gerais

| Parâmetro | Valor |
|-----------|-------|
| **Experimento** | Experimento 2 - Ablation Studies |
| **Research Question** | RQ2: Qual a contribuição individual de cada componente do HPM-KD? |
| **Status** | 📋 **PENDENTE** (Script criado, não executado) |
| **Dataset** | CIFAR100 (Full Mode) / MNIST (Quick Mode) |
| **Modelos a Treinar** | ~280 modelos (Full Mode) |
| **Tempo Estimado** | ~2h (Full Mode) |

---

## 🎯 Objetivo

Validar a **Research Question 2 (RQ2)** do paper HPM-KD analisando a **contribuição individual** de cada componente do método proposto e como eles **interagem** entre si.

### **Research Question:**
> Qual a contribuição individual de cada componente do HPM-KD e como eles interagem?

---

## 🔬 Componentes HPM-KD (DeepBridge Library)

| # | Componente | Descrição |
|---|------------|-----------|
| 1 | **ProgChain** | Progressive chaining de modelos intermediários |
| 2 | **AdaptConf** | Adaptive confidence weighting |
| 3 | **MultiTeach** | Multi-teacher ensemble |
| 4 | **MetaTemp** | Meta-learned temperature |
| 5 | **Parallel** | Parallel distillation paths |
| 6 | **Memory** | Memory-augmented distillation |

---

## 📊 Sub-Experimentos

### **2.1. Component Ablation (Exp 5)**
**Objetivo:** Testar cada componente isolado vs HPM-KD completo

**Configurações (7 configs):**
1. Baseline (nenhum componente)
2. ProgChain apenas
3. AdaptConf apenas
4. MultiTeach apenas
5. MetaTemp apenas
6. Parallel apenas
7. HPM-KD Full (todos componentes)

**Modelos:** 7 configs × 5 runs = **35 modelos**

---

### **2.2. Component Interactions (Exp 6)**
**Objetivo:** Identificar sinergias entre componentes

**Configurações (~15 combinações):**
- Pares: ProgChain+AdaptConf, ProgChain+MultiTeach, etc.
- Trios: ProgChain+AdaptConf+MultiTeach, etc.

**Modelos:** ~15 configs × 5 runs = **~75 modelos**

---

### **2.3. Hyperparameter Sensitivity (Exp 7)**
**Objetivo:** Sensibilidade a temperatura (T) e alpha (α)

**Grid Search:**
- **T:** [1, 2, 4, 6, 8, 10] (6 valores)
- **α:** [0.1, 0.3, 0.5, 0.7, 0.9] (5 valores)
- **Total:** 6 × 5 = 30 combinações

**Modelos:** 30 configs × 3 runs = **90 modelos**

---

### **2.4. Progressive Chain Length (Exp 8)**
**Objetivo:** Número ótimo de modelos intermediários

**Configurações:**
- Chain lengths: [1, 2, 3, 4, 5, 6]

**Modelos:** 6 configs × 5 runs = **30 modelos**

---

### **2.5. Number of Teachers (Exp 9)**
**Objetivo:** Quantos teachers são necessários (saturação)

**Configurações:**
- Number of teachers: [1, 2, 3, 4, 5, 6, 8, 10]

**Modelos:** 8 configs × 5 runs = **40 modelos**

---

## ⚙️ Configuração

### **Modos de Execução:**

| Modo | Dataset | Runs | Tempo |
|------|---------|------|-------|
| **Quick** | MNIST | 3 | ~2-3h |
| **Full** | CIFAR100 | 5 | ~10-15h |

### **Total de Modelos (Full Mode):**
```
Component Ablation:        35 modelos
Component Interactions:    75 modelos
Hyperparameter Sensitivity: 90 modelos
Progressive Chain Length:   30 modelos
Number of Teachers:         40 modelos
────────────────────────────────────
TOTAL:                    ~280 modelos
```

---

## 🚀 Como Executar

### **Quick Mode:**
```bash
cd scripts/
python 02_ablation_studies.py --mode quick --dataset MNIST
```

### **Full Mode:**
```bash
cd scripts/
python 02_ablation_studies.py --mode full --dataset CIFAR100 --gpu 0
```

---

## 📁 Estrutura de Arquivos

```
experimento_02_ablation_studies/
├── README.md                          ← Este arquivo
├── scripts/
│   └── 02_ablation_studies.py        (script principal)
└── results/
    └── (resultados após execução)
```

---

## 📊 Resultados Esperados

### **Hipóteses:**

1. **HPM-KD Full > componentes individuais**
2. **Sinergias positivas** entre componentes (ProgChain + AdaptConf > soma individual)
3. **T ∈ [4, 6], α ∈ [0.5, 0.7]** são ótimos
4. **Chain length = 3-4** é ótimo (tradeoff accuracy vs overhead)
5. **Saturação em ~4-6 teachers** (mais não ajuda significativamente)

### **Se Confirmado:**
- ✅ RQ2 validada
- ✅ Justifica complexidade do HPM-KD
- ✅ Identifica componentes essenciais
- ✅ Guia para configuração ótima

---

## 📚 Relacionado

- **Experimento 1:** `../experimento_01_compression_efficiency/`
- **Experimento 1B:** `../experimento_01b_compression_ratios/` ⭐ **Execute primeiro!**
- **Documentação Geral:** `../SUMARIO_COMPLETO_EXPERIMENTOS.md`

---

**Criado:** Dezembro 2025
**Status:** 📋 Pendente
**Prioridade:** 2 (executar após Experimento 1B)
**Autor:** Gustavo Haase
