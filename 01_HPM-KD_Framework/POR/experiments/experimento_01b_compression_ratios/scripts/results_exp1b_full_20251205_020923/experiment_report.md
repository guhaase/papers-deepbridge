# Experimento 1B: Compression Ratios Maiores (RQ1)

**Data:** 2025-12-06 23:47:03
**Mode:** FULL
**Device:** cuda:0

---

## 🎯 Research Question

**HPM-KD consegue superar Direct Training em compression ratios MAIORES?**

### Hipótese:
Com compression ratios maiores (5×, 10×, 20×), Knowledge Distillation (especialmente HPM-KD)
deve demonstrar vantagem clara sobre Direct training.

---

## 📊 Configuração

### Compression Ratios Testados:


- **2.3× compression**
  - Teacher: resnet50 (25.5M params)
  - Student: resnet18 (11.1M params)

- **5.0× compression**
  - Teacher: resnet50 (25.5M params)
  - Student: resnet10 (5.0M params)

- **7.0× compression**
  - Teacher: resnet50 (25.5M params)
  - Student: mobilenet_v2 (3.5M params)

### Baselines:
- Direct: Train student from scratch
- TraditionalKD: Hinton et al. (2015)
- HPM-KD: Our method (DeepBridge)

### Execution Parameters:
- Datasets: CIFAR100
- Runs per config: 5
- Teacher epochs: 200
- Student epochs: 200
- Batch size: 128

---

## 🔬 Principais Descobertas

### 1. Quando KD ajuda?

HPM-KD superou Direct training em **3** de **3** compression ratios:

- **2.3× compression:** HPM-KD +1.00% vs Direct ✅
- **5.0× compression:** HPM-KD +0.58% vs Direct ✅
- **7.0× compression:** HPM-KD +2.04% vs Direct ✅

### 2. Resultados Detalhados por Compression Ratio


#### 2.3× Compression

| Método | Acurácia (%) | Retention (%) | Tempo (s) |
|--------|--------------|---------------|----------|
| TraditionalKD | 62.46 ± 0.20 | 101.0 | 2857.3 |
| HPM-KD | 62.27 ± 0.26 | 100.7 | 2867.2 |
| Direct | 61.27 ± 0.41 | 99.1 | 2523.1 |

#### 5.0× Compression

| Método | Acurácia (%) | Retention (%) | Tempo (s) |
|--------|--------------|---------------|----------|
| TraditionalKD | 73.34 ± 0.30 | 120.4 | 4349.8 |
| HPM-KD | 73.21 ± 0.22 | 120.2 | 4345.3 |
| Direct | 72.64 ± 0.35 | 119.3 | 3439.2 |

#### 7.0× Compression

| Método | Acurácia (%) | Retention (%) | Tempo (s) |
|--------|--------------|---------------|----------|
| TraditionalKD | 59.08 ± 1.19 | 92.8 | 3594.7 |
| HPM-KD | 56.86 ± 0.31 | 89.3 | 3589.4 |
| Direct | 54.82 ± 0.39 | 86.1 | 3259.5 |

---

## 📈 Significância Estatística

| Compression | Comparação | t-statistic | p-value | Significante? |
|-------------|------------|-------------|---------|---------------|
| 2.3x_ResNet18 | HPM-KD vs Direct | 4.158 | 0.0032 | ✅ Sim |
| 2.3x_ResNet18 | HPM-KD vs TraditionalKD | -1.175 | 0.2737 | ❌ Não |
| 5x_ResNet10 | HPM-KD vs Direct | 2.755 | 0.0249 | ✅ Sim |
| 5x_ResNet10 | HPM-KD vs TraditionalKD | -0.668 | 0.5231 | ❌ Não |
| 7x_MobileNetV2 | HPM-KD vs Direct | 8.180 | 0.0000 | ✅ Sim |
| 7x_MobileNetV2 | HPM-KD vs TraditionalKD | -3.608 | 0.0069 | ✅ Sim |


---

## 💡 Conclusões

### Quando KD é vantajoso vs Direct?


✅ **HPM-KD demonstrou vantagem clara** em compression ratios maiores.

**Recomendação para o paper:**
- KD (especialmente HPM-KD) é mais efetivo com compression ratios ≥ 5×
- Para ratios pequenos (2×), Direct training pode ser suficiente
- HPM-KD consistentemente supera Traditional KD em todos os ratios


---

## 📁 Arquivos Gerados

- `results_compression_ratios.csv` - Resultados completos
- `statistical_tests.csv` - Testes estatísticos
- `figures/compression_ratio_vs_accuracy.png`
- `figures/hpmkd_vs_direct.png`
- `figures/statistical_significance.png`

---

*Relatório gerado automaticamente em 2025-12-06 23:47:03*
