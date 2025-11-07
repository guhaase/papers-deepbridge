# 🚀 Quick Start - Experimentos HPM-KD no Google Colab

**Para:** Gustavo Haase
**Repositório:** https://github.com/guhaase/papers-deepbridge
**Objetivo:** Executar experimentos do Paper 1 (HPM-KD) no Google Colab

---

## ⚠️ ATENÇÃO - Mudanças nas Importações

**Data:** 2025-11-07
**DeepBridge versão:** 0.1.54+

As importações do DeepBridge foram atualizadas. Se você vir erros como:
- `No module named 'deepbridge.data'`
- `No module named 'deepbridge.core.knowledge_distillation'`

**Solução:** Consulte o **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** para as importações corretas.

**Notebooks atualizados:**
- ✅ `00_setup_colab_UPDATED.ipynb` (use este)
- ⚠️ Outros notebooks precisam ser migrados (veja MIGRATION_GUIDE.md)

---

## ⚡ Início Rápido (5 minutos)

### 1. Abrir Google Colab

Acesse: https://colab.research.google.com/

### 2. Upload do Notebook de Setup

**Opção A: Upload Manual**
1. Download: `notebooks/00_setup_colab_UPDATED.ipynb`
2. No Colab: File → Upload notebook
3. Selecione o arquivo

**Opção B: Abrir do GitHub (quando push for feito)**
```
File → Open notebook → GitHub
URL: https://github.com/guhaase/papers-deepbridge
Path: 01_HPM-KD_Framework/POR/experiments/notebooks/00_setup_colab_UPDATED.ipynb
```

### 3. Configurar GPU

```
Runtime → Change runtime type → Hardware accelerator: GPU
```

**Recomendações:**
- **Teste rápido:** GPU T4 (Colab gratuito)
- **Experimentos completos:** GPU V100 ou A100 (Colab Pro - $10/mês)

### 4. Executar Setup

```python
# No notebook 00_setup_colab_UPDATED.ipynb
# Clique em: Runtime → Run all

# Ou execute célula por célula (Shift+Enter)
```

**Tempo:** ~10 minutos

**Resultado esperado:**
```
✅ GPU pronta para uso!
✅ Repositório clonado!
✅ DeepBridge importado com sucesso!
✅ Google Drive montado!
✅ SETUP CONCLUÍDO COM SUCESSO!
```

### 5. Verificar Estrutura

Após setup, você terá:

```
/content/
├── papers-deepbridge/           # Repositório clonado
│   └── 01_HPM-KD_Framework/
│       └── POR/
│           └── experiments/
│               └── notebooks/
│                   ├── 00_setup_colab_UPDATED.ipynb  ← VOCÊ ESTÁ AQUI
│                   ├── 01_compression_efficiency.ipynb  ← PRÓXIMO
│                   ├── 02_ablation_studies.ipynb
│                   ├── 03_generalization.ipynb
│                   └── 04_computational_efficiency.ipynb
│
└── drive/MyDrive/papers-deepbridge-results/
    └── HPM-KD/
        └── 20251107/            # Data de hoje
            ├── experiments/
            ├── models/
            ├── figures/
            ├── logs/
            └── colab_config.json  ← CONFIG SALVA
```

---

## 🧪 Executar Experimentos

### Ordem Recomendada

```
✅ 00_setup_colab_UPDATED.ipynb      [10 min]   ← OBRIGATÓRIO PRIMEIRO
↓
📊 01_compression_efficiency.ipynb    [30-45 min QUICK / 2-4h FULL]
↓
🔬 02_ablation_studies.ipynb          [1-2 horas]
↓
🌍 03_generalization.ipynb            [2-3 horas]
↓
⚡ 04_computational_efficiency.ipynb  [30-60 min]
```

### Modo Quick vs Full

**Modo Quick (Teste Rápido):**
```python
QUICK_MODE = True
# Usa subsets de 10K samples
# Teachers: 10 epochs
# Students: 5 epochs
# Total: ~2 horas para todos os 4 notebooks
```

**Modo Full (Paper):**
```python
QUICK_MODE = False
# Usa datasets completos
# Teachers: 50 epochs
# Students: 30 epochs
# Total: ~10-12 horas para todos os 4 notebooks
```

**Recomendação:**
1. **Dia 1:** Rode QUICK_MODE para testar tudo (2 horas)
2. **Dia 2-3:** Rode FULL_MODE com GPU V100/A100 (10-12 horas)

---

## 📊 Resultados Esperados

Após executar todos os notebooks, você terá:

### 1. Relatórios Markdown

```
/drive/MyDrive/papers-deepbridge-results/HPM-KD/20251107/experiments/
├── exp01_compression_efficiency_report.md
├── exp02_ablation_studies_report.md
├── exp03_generalization_report.md
└── exp04_efficiency_report.md
```

### 2. Figuras

```
/drive/MyDrive/papers-deepbridge-results/HPM-KD/20251107/figures/
├── exp01_accuracy_comparison.png
├── exp01_pareto_frontier.png
├── exp02_ablation_heatmap.png
├── exp03_tsne_visualization.png
├── exp04_speedup_curves.png
└── [20+ figuras no total]
```

### 3. Modelos Treinados

```
/drive/MyDrive/papers-deepbridge-results/HPM-KD/20251107/models/
├── mnist_teacher_resnet56.pth
├── mnist_student_hpmkd_resnet20.pth
├── cifar10_teacher_resnet56.pth
├── cifar10_student_hpmkd_resnet20.pth
└── [30+ modelos salvos]
```

### 4. Dados Brutos

```
/drive/MyDrive/papers-deepbridge-results/HPM-KD/20251107/experiments/
├── exp01_results.csv
├── exp02_ablation_data.json
├── exp03_generalization_metrics.csv
└── exp04_timing_breakdown.json
```

---

## 🎯 Experimentos por Research Question

### RQ1: Eficiência de Compressão

**Notebook:** `01_compression_efficiency.ipynb`

**Experimentos:**
- Comparação com 5 baselines em 7 datasets
- Cross-domain generalization (OpenML-CC18)
- Compression ratio scaling (2-20×)
- Comparação com SOTA

**Resultados-chave:**
- Tabela: HPM-KD vs Baselines (accuracy, retention)
- Gráfico: Accuracy vs Compression Ratio
- Fronteira de Pareto
- Statistical significance tests

**Tempo:** 30-45 min (Quick) / 2-4 horas (Full)

---

### RQ2: Contribuição de Componentes

**Notebook:** `02_ablation_studies.ipynb`

**Experimentos:**
- Ablation de 6 componentes individuais
- Análise de interações (pairwise)
- Sensibilidade a hiperparâmetros
- Comprimento ideal da cadeia progressiva
- Número ótimo de teachers

**Resultados-chave:**
- Tabela: Contribuição de cada componente
- Heatmap: Interações entre componentes
- Ranking de importância
- Gráficos de sensibilidade

**Tempo:** 1-2 horas

---

### RQ3: Generalização

**Notebook:** `03_generalization.ipynb`

**Experimentos:**
- Cross-domain performance (10 datasets OpenML)
- Robustez a class imbalance (10:1, 50:1, 100:1)
- Robustez a label noise (10%, 20%, 30%)
- Visualização de representações (t-SNE)

**Resultados-chave:**
- Boxplot: Retenção cross-domain
- Curvas de degradação (imbalance, noise)
- Visualizações t-SNE
- Silhouette scores

**Tempo:** 2-3 horas

---

### RQ4: Eficiência Computacional

**Notebook:** `04_computational_efficiency.ipynb`

**Experimentos:**
- Training time breakdown (profiling)
- Inference latency (CPU vs GPU)
- Memory footprint analysis
- Speedup com paralelização (1-8 workers)
- Cost-benefit analysis

**Resultados-chave:**
- Stacked bar: Time breakdown por componente
- Gráfico: Speedup vs Workers
- Fronteira de Pareto (accuracy vs time)
- Tabela de trade-offs

**Tempo:** 30-60 min

---

## 🔧 Troubleshooting

### GPU Out of Memory

```python
# Reduzir batch size
BATCH_SIZE = 64  # em vez de 128

# Reduzir número de teachers
N_TEACHERS = 3  # em vez de 4-5

# Limpar cache
import torch
torch.cuda.empty_cache()
```

### Timeout (>12 horas)

```python
# Usar QUICK_MODE
QUICK_MODE = True

# Salvar checkpoints regularmente
SAVE_CHECKPOINT_EVERY = 10

# Dividir em múltiplas sessões
# Sessão 1: MNIST + Fashion-MNIST
# Sessão 2: CIFAR-10
# Sessão 3: CIFAR-100
```

### Drive não monta

```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

### Import falha

```python
# Reinstalar
!pip install deepbridge --upgrade --force-reinstall

# Ou instalar do source
%cd /content/DeepBridge-lib
!pip install -e .
```

---

## 📝 Checklist de Validação

Antes de considerar os experimentos completos, verifique:

### Experimento 1 (Compression)
- [ ] MNIST accuracy: ~99.15% (HPM-KD)
- [ ] CIFAR-10 accuracy: ~92.34% (HPM-KD)
- [ ] HPM-KD supera todos os baselines
- [ ] Diferença estatisticamente significativa (p<0.01)

### Experimento 2 (Ablation)
- [ ] Progressive Chain impacto: -2.0 a -3.0pp
- [ ] Adaptive Config impacto: -1.5 a -2.0pp
- [ ] Todos os componentes contribuem positivamente
- [ ] Sinergias positivas detectadas (+0.2pp)

### Experimento 3 (Generalization)
- [ ] OpenML média: ~97.8% retenção
- [ ] Robustez a imbalance demonstrada
- [ ] Robustez a noise demonstrada
- [ ] Silhouette score HPM-KD > TAKD

### Experimento 4 (Efficiency)
- [ ] Overhead de treino: 20-40% vs Traditional KD
- [ ] Speedup 4 workers: ~3.2×
- [ ] Zero overhead de inferência
- [ ] HPM-KD na fronteira de Pareto

---

## 💰 Estimativa de Custos

### Colab Gratuito (GPU T4)
- **Quick Mode:** $0 (dentro do limite gratuito)
- **Full Mode:** Pode ultrapassar limite (não recomendado)

### Colab Pro ($10/mês)
- **Quick Mode:** $0-1
- **Full Mode:** $2-5
- **GPU V100/A100 ilimitada**

### Colab Pro+ ($50/mês)
- **Full Mode:** $0-2
- **GPU A100 prioritária**
- **Background execution**

**Recomendação:** Colab Pro por 1 mês ($10) é suficiente para todos os experimentos.

---

## 📚 Documentação Adicional

- **Guia Completo:** `COLAB_EXPERIMENTS_GUIDE.md` (50+ páginas)
- **Resumo de Experimentos:** `RESUMO_EXPERIMENTOS.md`
- **Paper Structure:** `../ESTRUTURA_PAPER1_TECNICO.md`

---

## 🎯 Próximos Passos

### Depois dos Experimentos

1. **Consolidar Resultados:**
```python
# Execute no último notebook
from scripts.report_generator import consolidate_all_reports

consolidate_all_reports(
    results_dir=results_dir,
    output_file='PAPER1_FINAL_REPORT.md'
)
```

2. **Download Backup:**
- Drive → MyDrive → papers-deepbridge-results → Download
- Backup local: ~500MB-2GB (dependendo de QUICK vs FULL)

3. **Preparar Tabelas para Paper:**
- Abrir relatórios `.md` gerados
- Copiar tabelas para LaTeX
- Inserir figuras `.png` no paper

4. **Push para GitHub:**
```bash
cd /local/papers-deepbridge
git add 01_HPM-KD_Framework/POR/experiments/results/
git commit -m "Add experiment results from Colab"
git push origin main
```

---

## ✅ Workflow Completo

```
Dia 1 (2 horas):
├── Setup Colab (10 min)
├── Experimento 1 Quick (45 min)
├── Experimento 2 Quick (30 min)
├── Experimento 3 Quick (30 min)
└── Experimento 4 Quick (15 min)
   → Verificar que tudo funciona ✅

Dia 2-3 (10-12 horas):
├── Experimento 1 Full (3 horas)
├── Experimento 2 Full (2 horas)
├── Experimento 3 Full (3 horas)
├── Experimento 4 Full (1 hora)
└── Consolidação (1 hora)
   → Resultados finais para paper ✅

Dia 4 (2 horas):
├── Revisar relatórios
├── Ajustar figuras
├── Preparar tabelas LaTeX
└── Push para GitHub
   → Paper 1 pronto! 🎉
```

---

**BOA SORTE COM OS EXPERIMENTOS! 🚀**

Para dúvidas:
- Abra issue: https://github.com/guhaase/papers-deepbridge/issues
- Consulte: `COLAB_EXPERIMENTS_GUIDE.md`
- Revise: `RESUMO_EXPERIMENTOS.md`
