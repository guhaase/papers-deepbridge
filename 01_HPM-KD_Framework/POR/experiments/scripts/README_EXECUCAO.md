# 🚀 Guia de Execução - Experimentos HPM-KD

## 📋 Visão Geral

Este diretório contém **4 experimentos** que validam o framework HPM-KD (Hierarchical Progressive Multi-teacher Knowledge Distillation) usando a **biblioteca DeepBridge**.

## 📂 Scripts Disponíveis

### Scripts Individuais:
1. **01_compression_efficiency.py** - Compara HPM-KD vs 5 baselines (RQ1)
2. **02_ablation_studies.py** - Analisa cada componente do HPM-KD (RQ2)
3. **03_generalization.py** - Testa robustez em condições adversas (RQ3)
4. **04_computational_efficiency.py** - Mede overhead computacional (RQ4)

### Scripts de Execução:
- **run_all_experiments.py** - Executa todos os experimentos automaticamente
- **RUN_COLAB.py** - Versão simplificada para Google Colab

---

## 🎯 Execução no Google Colab (RECOMENDADO)

### Opção 1: Execução Simplificada (Mais Fácil) ⭐

```python
# No Google Colab, execute:

# 1. Clone ou faça upload dos scripts
!git clone <seu-repositorio>
%cd papers/01_HPM-KD_Framework/POR/experiments/scripts

# 2. Instale dependências
!pip install deepbridge torch torchvision scikit-learn seaborn tqdm

# 3. Execute TODOS os experimentos (modo rápido)
!python RUN_COLAB.py

# OU modo completo (mais demorado)
!python RUN_COLAB.py --full

# OU customizar dataset
!python RUN_COLAB.py --dataset CIFAR10
```

**Tempo estimado:**
- Modo `quick`: 3-4 horas
- Modo `full`: 8-10 horas

### Opção 2: Execução Avançada

```python
# Mais controle sobre os parâmetros
!python run_all_experiments.py --mode quick --datasets MNIST --gpu 0

# Executar apenas experimentos específicos
!python run_all_experiments.py --mode quick --only 1 2 --dataset MNIST

# Pular experimentos
!python run_all_experiments.py --mode quick --skip 4 --dataset MNIST

# Múltiplos datasets (apenas Exp 1 suporta)
!python run_all_experiments.py --mode full --datasets MNIST CIFAR10 --gpu 0
```

---

## 💻 Execução Local

### Pré-requisitos:

```bash
pip install deepbridge torch torchvision scikit-learn seaborn tqdm matplotlib pandas numpy
```

### Execução:

```bash
# Modo rápido (recomendado para testes)
python run_all_experiments.py --mode quick --dataset MNIST --gpu 0

# Modo completo (para resultados finais do paper)
python run_all_experiments.py --mode full --dataset CIFAR10 --gpu 0

# CPU (sem GPU)
python run_all_experiments.py --mode quick --dataset MNIST
```

---

## 📊 Estrutura de Resultados

Após a execução, você terá:

```
results_quick_20250307_123456/
├── exp_01_compression_efficiency/
│   ├── results/
│   │   ├── baseline_comparison.csv
│   │   └── statistical_tests.csv
│   ├── figures/
│   │   ├── accuracy_comparison.png
│   │   ├── retention_rates.png
│   │   └── training_time.png
│   ├── models/
│   │   ├── teacher.pth
│   │   ├── hpmkd_student.pth
│   │   └── ...
│   └── report.md
│
├── exp_02_ablation_studies/
│   ├── results/
│   ├── figures/
│   └── report.md
│
├── exp_03_generalization/
│   ├── results/
│   ├── figures/
│   └── report.md
│
├── exp_04_computational_efficiency/
│   ├── results/
│   ├── figures/
│   └── report.md
│
├── run_all_experiments.log
├── results.json
└── RELATORIO_FINAL.md  ⭐ RELATÓRIO CONSOLIDADO
```

---

## 📈 Experimentos Detalhados

### Exp 1: Compression Efficiency (RQ1)
**Objetivo:** HPM-KD alcança maiores taxas de compressão mantendo acurácia?

**Baselines comparados:**
- Direct (train from scratch)
- Traditional KD (Hinton et al. 2015)
- FitNets (Romero et al. 2015)
- Attention Transfer (Zagoruyko & Komodakis 2017)
- TAKD (Mirzadeh et al. 2020)
- **HPM-KD (DeepBridge)** ⭐

**Tempo:** Quick: 45min | Full: 4h

---

### Exp 2: Ablation Studies (RQ2)
**Objetivo:** Qual a contribuição individual de cada componente?

**Componentes testados:**
- ProgChain (Progressive chaining)
- AdaptConf (Adaptive confidence)
- MultiTeach (Multi-teacher ensemble)
- MetaTemp (Meta-learned temperature)
- Parallel (Parallel paths)
- Memory (Memory-augmented)

**Experimentos:**
- Component ablation (5)
- Component interactions (6)
- Hyperparameter sensitivity (7)
- Chain length (8)
- Number of teachers (9)

**Tempo:** Quick: 1h | Full: 2h

---

### Exp 3: Generalization (RQ3)
**Objetivo:** HPM-KD generaliza melhor em condições adversas?

**Cenários:**
- Class Imbalance (ratios 10:1, 50:1, 100:1)
- Label Noise (10%, 20%, 30%)
- t-SNE Visualization + Silhouette Score

**Tempo:** Quick: 1.5h | Full: 3h

---

### Exp 4: Computational Efficiency (RQ4)
**Objetivo:** Qual o overhead computacional do HPM-KD?

**Métricas:**
- Training time breakdown
- Inference latency (CPU/GPU, batch 1-128)
- Memory consumption
- Throughput (samples/sec)
- Speedup com paralelização
- Cost-benefit analysis (Pareto frontier)

**Tempo:** Quick: 30min | Full: 1h

---

## 🔧 Parâmetros Disponíveis

### run_all_experiments.py

```bash
--mode {quick,full}           # Modo de execução (padrão: quick)
--datasets [MNIST ...]        # Datasets a usar (padrão: MNIST)
--dataset MNIST               # Dataset único (alias)
--gpu GPU_ID                  # ID da GPU (None = CPU)
--output OUTPUT_DIR           # Diretório de saída
--skip [1 2 ...]              # Pular experimentos específicos
--only [1 2 ...]              # Executar apenas experimentos específicos
```

### Scripts individuais

```bash
--mode {quick,full}           # Modo de execução
--dataset MNIST               # Dataset (scripts 2, 3, 4)
--datasets MNIST CIFAR10      # Múltiplos datasets (script 1)
--gpu GPU_ID                  # ID da GPU
--output OUTPUT_DIR           # Diretório de saída
```

---

## ⚠️ Requerimentos Importantes

1. **DeepBridge Library** - OBRIGATÓRIA
   ```bash
   pip install deepbridge
   ```
   ❌ Scripts **FALHAM** se DeepBridge não estiver instalado (sem fallback)

2. **GPU Recomendada**
   - Modo quick: funciona em CPU (lento)
   - Modo full: GPU altamente recomendada

3. **Espaço em disco**
   - ~2-5 GB para resultados completos (modelos, figuras, logs)

4. **RAM**
   - Mínimo: 8 GB
   - Recomendado: 16 GB+

---

## 📊 Datasets Suportados

| Dataset | Classes | Samples | Tamanho | Tempo (quick) |
|---------|---------|---------|---------|---------------|
| MNIST | 10 | 60k | 28x28 | Rápido (~30min) |
| FashionMNIST | 10 | 60k | 28x28 | Rápido (~30min) |
| CIFAR10 | 10 | 50k | 32x32x3 | Médio (~1h) |
| CIFAR100 | 100 | 50k | 32x32x3 | Lento (~2h) |

**Recomendação:**
- **Testes rápidos:** MNIST (quick)
- **Resultados paper:** CIFAR10 (full)

---

## 🎓 Citação

Se você usar estes experimentos, por favor cite:

```bibtex
@article{hpmkd2025,
  title={HPM-KD: Hierarchical Progressive Multi-teacher Knowledge Distillation},
  author={Seu Nome et al.},
  journal={Conferência},
  year={2025}
}
```

---

## 🐛 Troubleshooting

### Erro: "DeepBridge library não está disponível"
**Solução:**
```bash
pip install deepbridge
```

### Erro: "CUDA out of memory"
**Soluções:**
1. Use dataset menor (MNIST)
2. Use modo `quick`
3. Reduza batch size (edite configs nos scripts)
4. Use CPU (remove `--gpu`)

### Scripts muito lentos
**Soluções:**
1. Use `--mode quick`
2. Use GPU: `--gpu 0`
3. Use dataset pequeno: `--dataset MNIST`
4. Execute apenas alguns experimentos: `--only 1 2`

### No Google Colab: Session timeout
**Soluções:**
1. Use Google Colab Pro (mais tempo de sessão)
2. Execute experimentos individualmente
3. Use `--only` para executar poucos de cada vez

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `run_all_experiments.log`
2. Leia os relatórios individuais em cada `exp_XX/report.md`
3. Consulte a documentação do DeepBridge

---

## ✅ Checklist de Execução

Antes de executar:
- [ ] DeepBridge instalado
- [ ] PyTorch + torchvision instalados
- [ ] GPU disponível (opcional mas recomendado)
- [ ] Espaço em disco suficiente (~5 GB)
- [ ] Tempo disponível (3-10 horas)

Para executar no Colab:
```python
!python RUN_COLAB.py
```

Para executar localmente:
```bash
python run_all_experiments.py --mode quick --dataset MNIST --gpu 0
```

**Boa sorte! 🚀**
