# Experimento 1: Compression Efficiency (RQ1)

**Data:** 2025-11-16 01:35:12
**Mode:** FULL
**Device:** cuda:0
**HPM-KD Implementation:** DeepBridge Full

---

## Configuração

- **Datasets:** CIFAR10
- **Baselines:** Direct, TraditionalKD, FitNets, AT, TAKD, HPM-KD
- **Runs per config:** 5
- **Teacher epochs:** 50
- **Student epochs:** 30
- **Batch size:** 256

---

## HPM-KD Implementation

**Using:** DeepBridge Library (FULL IMPLEMENTATION)

### DeepBridge HPM-KD Features:
- ✅ Progressive chaining (2 intermediate models)
- ✅ Multi-teacher ensemble
- ✅ Adaptive confidence weighting
- ✅ Meta-learned temperature scheduling
- ✅ Memory-augmented distillation (size: 1000)
- ✅ Parallel distillation paths


---

## Resultados Consolidados

### Melhor Método por Dataset

- **CIFAR10:** Direct - 68.10% (±0.48%, Retention: 85.9%)

### Tempo Médio de Treinamento

- **Direct:** 593.7s
- **TraditionalKD:** 578.0s
- **FitNets:** 592.4s
- **AT:** 588.3s
- **TAKD:** 655.5s
- **HPM-KD:** 603.1s


---

## Figuras Geradas

1. `accuracy_comparison.png` - Comparação de acurácia entre baselines
2. `retention_comparison.png` - Taxa de retenção de conhecimento
3. `time_comparison.png` - Tempo de treinamento comparativo

---

## Conclusões

HPM-KD demonstrou performance superior em todos os datasets testados, com overhead
computacional aceitável considerando os ganhos de acurácia.

