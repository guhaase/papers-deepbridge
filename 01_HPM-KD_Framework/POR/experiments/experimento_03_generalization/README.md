# Experimento 3: Generalization

## 📋 Informações Gerais

| Parâmetro | Valor |
|-----------|-------|
| **Experimento** | Experimento 3 - Generalization |
| **Research Question** | RQ3: HPM-KD generaliza melhor que baselines em condições adversas? |
| **Status** | 📋 **PENDENTE** (Script criado, não executado) |
| **Dataset** | CIFAR10 |
| **Modelos a Treinar** | ~83 modelos (Full Mode) |
| **Tempo Estimado** | ~3h (Full Mode) |

---

## 🎯 Objetivo

Validar a **Research Question 3 (RQ3)** do paper HPM-KD testando a **robustez** do método proposto em condições adversas: desbalanceamento de classes e ruído nos rótulos.

### **Research Question:**
> HPM-KD generaliza melhor que baselines em condições adversas (desbalanceamento, ruído)?

---

## 🔬 Sub-Experimentos

### **3.1. Class Imbalance (Exp 10)**
**Objetivo:** Robustez a desbalanceamento de classes

**Cenários (4 cenários):**
1. **Balanced** (baseline)
2. **Imbalance 10:1**
3. **Imbalance 50:1**
4. **Imbalance 100:1**

**Métodos:** HPM-KD, TAKD (baseline)

**Modelos:** 4 cenários × 2 métodos × 5 runs = **40 modelos**

---

### **3.2. Label Noise (Exp 11)**
**Objetivo:** Robustez a ruído nos rótulos

**Cenários (4 cenários):**
1. **No noise** (baseline)
2. **10% noise**
3. **20% noise**
4. **30% noise**

**Métodos:** HPM-KD, TAKD (baseline)

**Modelos:** 4 cenários × 2 métodos × 5 runs = **40 modelos**

---

### **3.3. Representation Visualization (Exp 13)**
**Objetivo:** Qualidade das representações aprendidas

**Técnicas:**
- t-SNE visualization
- Silhouette Score

**Métodos:** Direct, TAKD, HPM-KD

**Modelos:** 3 métodos × 1 run = **3 modelos** (análise qualitativa)

---

## ⚙️ Configuração

### **Modos de Execução:**

| Modo | Runs | Tempo |
|------|------|-------|
| **Quick** | 3 | ~1.5h |
| **Full** | 5 | ~3h |

### **Total de Modelos (Full Mode):**
```
Class Imbalance:           40 modelos
Label Noise:               40 modelos
Representation Viz:         3 modelos
────────────────────────────────────
TOTAL:                     83 modelos
```

---

## 🚀 Como Executar

### **Quick Mode:**
```bash
cd scripts/
python 03_generalization.py --mode quick --dataset CIFAR10
```

### **Full Mode:**
```bash
cd scripts/
python 03_generalization.py --mode full --dataset CIFAR10 --gpu 0
```

---

## 📁 Estrutura de Arquivos

```
experimento_03_generalization/
├── README.md                          ← Este arquivo
├── scripts/
│   └── 03_generalization.py          (script principal)
└── results/
    └── (resultados após execução)
```

---

## 📊 Resultados Esperados

### **Hipóteses:**

1. **Class Imbalance:** HPM-KD mais robusto que TAKD em ratios altos (≥50:1)
2. **Label Noise:** HPM-KD menos sensível a ruído que TAKD (≥20%)
3. **Representations:** HPM-KD aprende features mais separáveis (Silhouette Score maior)

### **Métricas:**

- **Accuracy** em cada cenário
- **F1-Score** (importante para imbalance)
- **Degradation**: queda de accuracy vs baseline
- **Silhouette Score**: qualidade de clusters (t-SNE)

### **Se Confirmado:**
- ✅ RQ3 validada
- ✅ HPM-KD é robusto a condições adversas
- ✅ Generalization superior aos baselines

---

## 📚 Relacionado

- **Experimento 1B:** `../experimento_01b_compression_ratios/` ⭐ **Execute primeiro!**
- **Experimento 2:** `../experimento_02_ablation_studies/`
- **Documentação Geral:** `../SUMARIO_COMPLETO_EXPERIMENTOS.md`

---

**Criado:** Dezembro 2025
**Status:** 📋 Pendente
**Prioridade:** 3 (executar após Experimentos 1B e 2)
**Autor:** Gustavo Haase
