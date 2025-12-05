# Experimento 4: Computational Efficiency

## 📋 Informações Gerais

| Parâmetro | Valor |
|-----------|-------|
| **Experimento** | Experimento 4 - Computational Efficiency |
| **Research Question** | RQ4: Qual o overhead computacional do HPM-KD comparado aos baselines? |
| **Status** | 📋 **PENDENTE** (Script criado, não executado) |
| **Dataset** | CIFAR10 (Full Mode) / MNIST (Quick Mode) |
| **Modelos a Treinar** | ~8 modelos (maioria é benchmarking) |
| **Tempo Estimado** | ~1h (Full Mode) |

---

## 🎯 Objetivo

Validar a **Research Question 4 (RQ4)** do paper HPM-KD medindo o **overhead computacional** do método proposto comparado aos baselines.

### **Research Question:**
> Qual o overhead computacional do HPM-KD comparado aos baselines?

---

## 🔬 Sub-Experimentos

### **4.1. Time Breakdown**
**Objetivo:** Tempo de cada componente do HPM-KD

**Medições:**
- Total training time
- Per epoch time
- Per component time (ProgChain, AdaptConf, etc.)

**Modelos:** 1 método × 5 runs = **5 modelos** (medição de tempo)

---

### **4.2. Inference Latency**
**Objetivo:** Latência de inferência CPU/GPU com diferentes batch sizes

**Batch Sizes:**
- Batch=1 (latência mínima)
- Batch=32 (médio)
- Batch=128 (throughput máximo)

**Plataformas:**
- CPU
- GPU

**Modelos:** 3 métodos × 1 run = **3 modelos** (benchmarking)

---

### **4.3. Speedup Parallelization**
**Objetivo:** Ganhos com paralelização (multiple workers)

**Workers:** [1, 2, 4, 8, 16, 32]

**Modelos:** Reutiliza modelos existentes (sem treino adicional)

---

### **4.4. Cost-Benefit Analysis (Exp 14)**
**Objetivo:** Pareto frontier: accuracy vs time

**Análise:**
- Plotar accuracy vs training time
- Identificar sweet spot (melhor tradeoff)

**Modelos:** Reutiliza resultados de Experimento 1B

---

## ⚙️ Configuração

### **Modos de Execução:**

| Modo | Dataset | Runs | Tempo |
|------|---------|------|-------|
| **Quick** | MNIST | 3 | ~30min |
| **Full** | CIFAR10 | 5 | ~1h |

### **Total de Modelos (Full Mode):**
```
Time Breakdown:             5 modelos
Inference Latency:          3 modelos
Speedup Parallelization:    0 (reutiliza)
Cost-Benefit Analysis:      0 (reutiliza)
────────────────────────────────────
TOTAL:                      8 modelos
```

---

## 🚀 Como Executar

### **Quick Mode:**
```bash
cd scripts/
python 04_computational_efficiency.py --mode quick --dataset MNIST
```

### **Full Mode:**
```bash
cd scripts/
python 04_computational_efficiency.py --mode full --dataset CIFAR10 --gpu 0
```

---

## 📁 Estrutura de Arquivos

```
experimento_04_computational_efficiency/
├── README.md                          ← Este arquivo
├── scripts/
│   └── 04_computational_efficiency.py (script principal)
└── results/
    └── (resultados após execução)
```

---

## 📊 Resultados Esperados

### **Hipóteses:**

1. **Training Time:** HPM-KD overhead ~20-30% vs Traditional KD
2. **Inference:** Sem overhead (mesmo modelo student final)
3. **Parallelization:** Speedup linear até ~4 workers
4. **Cost-Benefit:** HPM-KD oferece melhor accuracy/time ratio em compression ≥5×

### **Métricas:**

- **Training Time** (total, per epoch)
- **Inference Latency** (ms/sample)
- **Throughput** (samples/sec)
- **Memory Consumption** (GPU/CPU)
- **Speedup** (parallel workers)
- **Efficiency** (speedup/workers)

### **Se Confirmado:**
- ✅ RQ4 validada
- ✅ Overhead computacional aceitável
- ✅ Justifica uso de HPM-KD (benefício > custo)

---

## 📚 Relacionado

- **Experimento 1B:** `../experimento_01b_compression_ratios/` ⭐ **Execute primeiro!**
- **Experimento 2:** `../experimento_02_ablation_studies/`
- **Experimento 3:** `../experimento_03_generalization/`
- **Documentação Geral:** `../SUMARIO_COMPLETO_EXPERIMENTOS.md`

---

**Criado:** Dezembro 2025
**Status:** 📋 Pendente
**Prioridade:** 4 (pode executar em paralelo com Experimento 3)
**Autor:** Gustavo Haase
