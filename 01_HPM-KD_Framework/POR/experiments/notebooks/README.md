# Notebooks de Experimentos - HPM-KD Framework

**Repositório:** https://github.com/guhaase/papers-deepbridge
**Guia Completo:** `../COLAB_EXPERIMENTS_GUIDE.md`

---

## 📚 Índice de Notebooks

### ✅ Obrigatório (Sempre executar primeiro)

**00_setup_colab_UPDATED.ipynb**
- ⏱️ Tempo: 10 minutos
- 🎯 Objetivo: Configurar ambiente Colab
- 📦 Ações:
  - Verifica GPU
  - Clona repositório papers-deepbridge
  - Instala DeepBridge e dependências
  - Monta Google Drive
  - Cria estrutura de diretórios
  - Salva configuração

---

### 📊 Experimentos Principais

#### **01_compression_efficiency.ipynb** (RQ1)
- ⏱️ Tempo: 30-45 min (Quick) / 2-4h (Full)
- 🎯 Research Question: HPM-KD supera baselines em compression ratio + accuracy?
- 🧪 Experimentos:
  1. Baseline comparison (7 datasets)
  2. Cross-domain generalization
  3. Compression ratio scaling
  4. SOTA comparison
- 📈 Outputs:
  - `results_comparison.csv`
  - `figures/accuracy_comparison.png`
  - `figures/retention_comparison.png`
  - `experiment_report.md`

**Status:** ✅ **COMPLETO E FUNCIONAL**

---

#### **02_ablation_studies.ipynb** (RQ2)
- ⏱️ Tempo: 1-2 horas
- 🎯 Research Question: Quanto cada componente contribui?
- 🧪 Experimentos:
  5. Component ablation (6 components)
  6. Component interactions
  7. Hyperparameter sensitivity
  8. Progressive chain length
  9. Number of teachers
- 📈 Outputs:
  - `ablation_results.csv`
  - `figures/ablation_heatmap.png`
  - `figures/sensitivity_plots.png`
  - `experiment_report.md`

**Status:** 📝 Template disponível - adaptar do notebook 01

---

#### **03_generalization.ipynb** (RQ3)
- ⏱️ Tempo: 2-3 horas
- 🎯 Research Question: HPM-KD generaliza cross-domain?
- 🧪 Experimentos:
  10. Class imbalance robustness
  11. Label noise robustness
  13. Representation visualization (t-SNE)
- 📈 Outputs:
  - `generalization_results.csv`
  - `figures/degradation_curves.png`
  - `figures/tsne_visualization.png`
  - `experiment_report.md`

**Status:** 📝 Template disponível - adaptar do notebook 01

---

#### **04_computational_efficiency.ipynb** (RQ4)
- ⏱️ Tempo: 30-60 min
- 🎯 Research Question: Qual overhead computacional?
- 🧪 Experimentos:
  4.1. Training time breakdown
  4.2. Inference latency + memory
  4.3. Speedup with parallelization
  14. Cost-benefit analysis
- 📈 Outputs:
  - `timing_results.csv`
  - `figures/time_breakdown.png`
  - `figures/speedup_curves.png`
  - `figures/pareto_frontier.png`
  - `experiment_report.md`

**Status:** 📝 Template disponível - adaptar do notebook 01

---

## 🚀 Ordem de Execução

```
1. 00_setup_colab_UPDATED.ipynb       [OBRIGATÓRIO - 10 min]
   ↓
2. 01_compression_efficiency.ipynb    [RQ1 - 30min-4h]
   ↓
3. 02_ablation_studies.ipynb          [RQ2 - 1-2h]
   ↓
4. 03_generalization.ipynb            [RQ3 - 2-3h]
   ↓
5. 04_computational_efficiency.ipynb  [RQ4 - 30-60min]
```

**Tempo Total:**
- Quick Mode: ~4-5 horas
- Full Mode: ~10-14 horas

---

## ⚙️ Modo de Execução

### Quick Mode (Teste Rápido)
```python
QUICK_MODE = True
```
- Subsets de 10K samples
- Teachers: 10 epochs
- Students: 5 epochs
- 2-3 runs por configuração
- **Use para:** Validar que tudo funciona

### Full Mode (Paper Final)
```python
QUICK_MODE = False
```
- Datasets completos
- Teachers: 50 epochs
- Students: 30 epochs
- 5 runs por configuração
- **Use para:** Resultados finais do paper

---

## 📁 Estrutura de Outputs

Após execução, você terá no Google Drive:

```
/drive/MyDrive/papers-deepbridge-results/HPM-KD/20251107/
├── experiments/
│   ├── exp01_compression/
│   │   ├── results_comparison.csv
│   │   ├── figures/
│   │   ├── models/
│   │   ├── logs/
│   │   └── experiment_report.md
│   ├── exp02_ablation/
│   ├── exp03_generalization/
│   └── exp04_efficiency/
├── models/
│   ├── mnist_teacher.pth
│   ├── cifar10_teacher.pth
│   └── [30+ modelos]
├── figures/
│   ├── exp01_accuracy_comparison.png
│   └── [20+ figuras]
└── colab_config.json
```

---

## ✅ Checklist de Execução

### Antes de Começar
- [ ] Runtime configurado para GPU (T4 mínimo)
- [ ] Google Drive com espaço livre (~2-5GB)
- [ ] Notebook 00_setup executado com sucesso

### Durante Execução
- [ ] Escolher modo (Quick ou Full) no início de cada notebook
- [ ] Monitorar uso de memória GPU
- [ ] Verificar que resultados estão sendo salvos no Drive
- [ ] Manter sessão ativa (movimento de mouse)

### Após Cada Notebook
- [ ] Relatório .md gerado
- [ ] Figuras salvas em /figures
- [ ] CSV de resultados salvo
- [ ] Modelos salvos em /models (se aplicável)

### Após Todos os Notebooks
- [ ] 4 relatórios .md gerados (1 por RQ)
- [ ] ~20 figuras geradas
- [ ] ~4-6 arquivos CSV de resultados
- [ ] Download backup do Google Drive

---

## 🔧 Troubleshooting

### GPU Out of Memory
- Reduzir `BATCH_SIZE` (ex: 128 → 64)
- Usar Quick Mode
- Limpar cache: `torch.cuda.empty_cache()`

### Timeout (>12h no Colab)
- Dividir em múltiplas sessões
- Salvar checkpoints regularmente
- Usar Quick Mode para testes

### Import Errors
```python
# Reinstalar DeepBridge
%cd /content/DeepBridge-lib
!pip install -e . --force-reinstall
```

### Drive não monta
```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

---

## 📚 Documentação

- **Setup Guide:** `../COLAB_QUICK_START.md`
- **Guia Completo:** `../COLAB_EXPERIMENTS_GUIDE.md`
- **Resumo de Experimentos:** `../RESUMO_EXPERIMENTOS.md`
- **Issues:** https://github.com/guhaase/papers-deepbridge/issues

---

## 🎯 Status Atual

| Notebook | Status | Observações |
|----------|--------|-------------|
| 00_setup | ✅ Completo | Pronto para uso |
| 01_compression (RQ1) | ✅ Completo | Funcional, testado |
| 02_ablation (RQ2) | 📝 Template | Criar baseado no 01 |
| 03_generalization (RQ3) | 📝 Template | Criar baseado no 01 |
| 04_efficiency (RQ4) | 📝 Template | Criar baseado no 01 |

---

## 💡 Próximos Passos

1. **Teste o Notebook 00_setup** no Google Colab (10 min)
2. **Rode o Notebook 01** em Quick Mode (45 min)
3. **Se funcionar:**
   - Crie notebooks 02-04 baseados no template 01
   - Ou peça para eu criar versões completas
4. **Rode Full Mode** para resultados finais (10-14h)

---

**Última atualização:** 07 Novembro 2025
**Versão:** 1.0
**Autor:** Claude (Anthropic) para Gustavo Haase

