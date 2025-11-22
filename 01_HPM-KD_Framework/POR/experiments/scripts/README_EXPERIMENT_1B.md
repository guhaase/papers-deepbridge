# ⚠️ Experimento 1B: Compression Ratios Maiores (CRÍTICO)

## 🎯 Por que este experimento?

O **Experimento 1** revelou um problema:
- Com compression ratio pequeno (2×), **Direct training venceu HPM-KD**
- Isso questiona a utilidade de Knowledge Distillation!

**Hipótese do Experimento 1B:**
> Com compression ratios **MAIORES** (5×, 7×), HPM-KD deve superar Direct significativamente.

---

## 📊 O que será testado?

### Compression Ratios:
1. **2.3×** - ResNet50 → ResNet18
2. **5×** - ResNet50 → ResNet10
3. **7×** - ResNet50 → MobileNetV2

### Métodos:
- Direct (baseline)
- TraditionalKD (Hinton 2015)
- **HPM-KD** (nosso)

### Análises:
- ✅ Accuracy vs Compression Ratio
- ✅ Statistical significance (t-tests)
- ✅ **"When does KD help?"** analysis

---

## 🚀 Execução Rápida

### Google Colab (Recomendado):

```bash
# Quick mode (2-3h teste)
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py --mode quick --dataset CIFAR10 --gpu 0

# Full mode (8-10h paper)
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py --mode full --dataset CIFAR10 --gpu 0
```

### Linux/WSL:

```bash
cd /home/guhaase/projetos/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments/scripts

# Quick mode
./RUN_EXPERIMENT_1B.sh --mode quick --dataset CIFAR10

# Full mode
./RUN_EXPERIMENT_1B.sh --mode full --dataset CIFAR10
```

---

## ⏱️ Tempo Estimado

| Mode | CIFAR10 | CIFAR100 |
|------|---------|----------|
| Quick | 2-3h | 3-4h |
| Full | 8-10h | 12-15h |

---

## 📁 Saída Esperada

```
results/
├── results_compression_ratios.csv        # Dados completos
├── statistical_tests.csv                 # p-values
├── experiment_report.md                  # Relatório
└── figures/
    ├── compression_ratio_vs_accuracy.png # Principal
    ├── hpmkd_vs_direct.png               # "When KD helps?"
    └── statistical_significance.png      # Heatmap p-values
```

---

## 🎯 Resultado Esperado (Hipótese)

Se a hipótese estiver **correta**:

| Compression | Winner | Difference |
|-------------|--------|------------|
| 2.3× | Direct ≈ HPM-KD | ~0% |
| 5× | **HPM-KD** | +1-2% * |
| 7× | **HPM-KD** | +2-3% ** |

`*` p < 0.05, `**` p < 0.01

---

## ✅ Features do Script

- ✅ **Sistema de checkpoints granular** - Resume automático
- ✅ **Treinamento paralelo** - Aproveita todas as GPUs
- ✅ **Statistical tests** - T-tests automáticos
- ✅ **Visualizações prontas** - Figuras para o paper
- ✅ **Relatório completo** - Markdown com análise

---

## 📚 Documentação

- **Guia completo**: `COMO_EXECUTAR_EXPERIMENTO_1B.md`
- **Script auxiliar**: `RUN_EXPERIMENT_1B.sh`
- **Código fonte**: `01b_compression_ratios.py`

---

## 🚨 Status

- [x] Script criado
- [x] Documentação completa
- [x] Sistema de checkpoints
- [x] Análise estatística
- [ ] **Executar Quick mode** (teste)
- [ ] **Executar Full mode** (paper)
- [ ] Análise dos resultados

---

## 💡 Próximos Passos

1. **Executar Quick mode** para validar pipeline (2-3h)
2. **Revisar resultados** do quick mode
3. **Executar Full mode** para o paper (8-10h)
4. **Analisar figuras** geradas
5. **Incluir no paper** (seção Results)

---

**Este experimento é CRÍTICO para validar RQ1!**

Se HPM-KD superar Direct em compression ratios maiores, temos evidência forte de que:
> "Knowledge Distillation (especialmente HPM-KD) é efetivo para compression ratios ≥ 5×"

---

*Criado em: 15 de Novembro de 2025*
