# Experimento 1B: Compression Ratios Maiores (CRÍTICO) ⭐⭐⭐

## 📋 Informações Gerais

| Parâmetro | Valor |
|-----------|-------|
| **Experimento** | Experimento 1B - Compression Ratios Maiores |
| **Research Question** | RQ1: HPM-KD supera Direct Training com compression ratios ≥ 5×? |
| **Status** | ⏳ **PRONTO PARA EXECUTAR** (Dezembro 2025) |
| **Importância** | ⭐⭐⭐ **CRÍTICO** - Valida RQ1 efetivamente |
| **Dataset** | CIFAR10 |
| **Compression Ratios** | **2.3×, 5×, 7×** |
| **Modelos a Treinar** | 46 modelos (1 teacher + 45 students) |
| **Plataforma** | **RunPod.io** (GPU contratada por hora) |

---

## 🎯 Objetivo

**Validar efetivamente a Research Question 1 (RQ1)** do paper HPM-KD testando compression ratios **REALISTAS** (5×, 7×) ao invés do compression insuficiente (2×) do Experimento 1.

### **Por Que Este Experimento É CRÍTICO:**

```
╔═══════════════════════════════════════════════════════════════════╗
║  ⭐⭐⭐ EXPERIMENTO MAIS IMPORTANTE DO PAPER                      ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ✅ Experimento 1 FALHOU em validar RQ1 (compression 2× baixo)  ║
║  ✅ Experimento 1B CORRIGE com compression ratios ≥5×           ║
║  ✅ Testa hipótese: "KD é efetivo com gaps MAIORES"            ║
║  ✅ Arquiteturas MODERNAS (ResNet50 → ResNet18/10/MobileNetV2) ║
║  ✅ Dataset DESAFIADOR (CIFAR10 com 10 classes)                ║
║  ✅ 100% PRONTO para executar com GPU dedicada (RunPod)        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### **Research Question:**
> Com compression ratios maiores (5×, 7×), HPM-KD consegue superar Direct training e validar RQ1?

---

## 🔬 Metodologia

### **Hipótese Central:**

> **"Knowledge Distillation (HPM-KD) é mais efetivo que Direct Training quando o gap entre teacher e student é GRANDE (compression ≥5×)"**

### **Diferenças vs Experimento 1:**

| Aspecto | Experimento 1 ❌ | Experimento 1B ✅ |
|---------|------------------|-------------------|
| **Compression** | 2× (insuficiente) | **2.3×, 5×, 7×** (realista) |
| **Teacher** | LeNet5-Large (62K) | **ResNet50 (25M)** |
| **Student** | LeNet5-Small (30K) | **ResNet18/10/MobileNetV2** |
| **Dataset** | MNIST (simples) | **CIFAR10 (desafiador)** |
| **Gap** | Pequeno | **GRANDE** |
| **Resultado** | Direct venceu | **HPM-KD deve vencer** |
| **RQ1** | ❌ Não validada | ✅ **Deve validar** |

---

## 📊 Configuração

### **Dataset: CIFAR10**
- **Treinamento**: 50,000 imagens
- **Teste**: 10,000 imagens
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Resolução**: 32×32 pixels (RGB)
- **Dificuldade**: Média-alta

### **Compression Ratios Testados (3 ratios):**

| Compression | Teacher | Student | Params Teacher | Params Student | Ratio Real | Importância |
|-------------|---------|---------|----------------|----------------|------------|-------------|
| **2.3×** | ResNet50 | ResNet18 | 25.6M | 11.2M | **2.3×** | Baseline |
| **5×** ⭐ | ResNet50 | ResNet10 | 25.6M | 5.0M | **5.0×** | **Crítico** |
| **7×** ⭐⭐ | ResNet50 | MobileNetV2 | 25.6M | 3.5M | **7.3×** | **Mais crítico** |

### **Métodos Comparados (3 métodos):**

| # | Método | Descrição | Hiperparâmetros | Referência |
|---|--------|-----------|-----------------|------------|
| 1 | **Direct** | Treinar student do zero | - | Baseline |
| 2 | **Traditional KD** | KD clássico | T=4.0, α=0.5 | Hinton et al. (2015) |
| 3 | **HPM-KD** | Nossa proposta | T=6.0, α=0.7 | **Ours** ⭐ |

---

## ⚙️ Hiperparâmetros

### **Full Mode (Recomendado):**

| Parâmetro | Valor |
|-----------|-------|
| **Teacher Epochs** | 100 |
| **Student Epochs** | 50 |
| **Runs por método** | **5** (maior robustez estatística) |
| **Batch Size** | 128 |
| **Learning Rate** | 0.1 |
| **Optimizer** | SGD (momentum=0.9, weight_decay=5e-4) |
| **Scheduler** | MultiStepLR [60, 120, 160] |
| **Data Augmentation** | RandomCrop, RandomHorizontalFlip |

**Modelos:** 1 teacher + (3 compression × 3 métodos × 5 runs) = **46 modelos**

**Tempo Estimado:**
- **GPU RTX 4090:** ~5-7h
- **GPU A100:** ~3-5h
- **GPU V100:** ~7-10h

---

## 🚀 Como Executar (RunPod)

### **Passo 1: Setup RunPod**

1. Acesse https://www.runpod.io/
2. Selecione template **PyTorch** ou **CUDA**
3. GPU recomendada: **RTX 4090**, **A100**, ou **V100**
4. Storage: mínimo 50GB

### **Passo 2: Preparar Ambiente**

```bash
# Instalar dependências
pip install torch torchvision numpy pandas matplotlib seaborn scipy tqdm
pip install deepbridge  # DeepBridge Library (HPM-KD)

# Clonar repositório (se necessário)
git clone <seu-repo>
cd papers/01_HPM-KD_Framework/POR/experiments/experimento_01b_compression_ratios/scripts/

# Verificar GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
```

### **Passo 3: Executar Experimento**

```bash
# Full Mode (recomendado para paper)
python 01b_compression_ratios.py --mode full --dataset CIFAR10 --gpu 0

# Quick Mode (teste rápido - 3 runs)
python 01b_compression_ratios.py --mode quick --dataset CIFAR10 --gpu 0

# Compression específico (apenas 5×)
python 01b_compression_ratios.py --mode full --dataset CIFAR10 --compression 5x --gpu 0
```

### **Opções Disponíveis:**
```bash
--mode {quick,full}              # Modo de execução
--dataset {CIFAR10,CIFAR100}     # Dataset
--compression {all,2.3x,5x,7x}   # Compression ratio específico
--gpu 0                          # GPU ID
--seed 42                        # Seed para reprodutibilidade
```

---

## 📊 Outputs Gerados

### **Estrutura de Saída:**
```
results/exp1b_full_YYYYMMDD_HHMMSS/
├── results.csv                       📊 Dados numéricos (CSV)
├── experiment_report.md              📄 Relatório completo (Markdown)
├── experiment.log                    📋 Log de execução detalhado
│
├── figures/                          📈 Visualizações (PNG 300 DPI)
│   ├── accuracy_vs_compression.png  ⭐⭐⭐ FIGURA PRINCIPAL (paper)
│   ├── hpmkd_vs_direct.png          ⭐⭐ "When does KD help?"
│   └── retention_analysis.png       📊 Retenção de conhecimento
│
└── models/                           💾 Modelos treinados
    ├── teacher_resnet50_CIFAR10.pt  2.6 MB (reutilizado!)
    └── student_*.pt                 45 modelos (227 KB cada)
```

**Tamanho Total:** ~2 GB (Full Mode)

### **Figuras Geradas (PNG 300 DPI):**

1. **`accuracy_vs_compression.png`** ⭐⭐⭐ **PRINCIPAL**
   - Accuracy vs Compression Ratio
   - 3 métodos (Direct, TraditionalKD, HPM-KD)
   - Error bars (desvio padrão)
   - **USO:** Section 5 (Results) do paper

2. **`hpmkd_vs_direct.png`** ⭐⭐
   - Delta (HPM-KD - Direct) vs Compression
   - Mostra onde KD ajuda
   - **USO:** Analysis "When does KD help?"

3. **`retention_analysis.png`** 📊
   - Knowledge retention (%)
   - Por método e compression

---

## 📈 Resultados Esperados

### **Hipótese:**

| Compression | Direct | Traditional KD | HPM-KD | Δ (HPM-KD - Direct) | Conclusão |
|-------------|--------|----------------|--------|---------------------|-----------|
| **2.3×** | ~88.5% | ~88.6% | ~88.7% | **+0.2pp** | ≈ Empate (gap pequeno) |
| **5×** ⭐ | ~85.0% | ~86.5% | **~87.5%** | **+2.5pp** ✅ | **HPM-KD vence** |
| **7×** ⭐⭐ | ~82.0% | ~84.5% | **~86.0%** | **+4.0pp** ✅✅ | **HPM-KD vence forte** |

**Teacher Accuracy esperado:** ~92-93% (ResNet50 em CIFAR10)

### **Análise "When Does KD Help?":**

```
Compression 2.3×:  Gap pequeno  → Direct ≈ HPM-KD (empate)
Compression 5×:    Gap médio   → HPM-KD > Direct (+2.5pp) ✅
Compression 7×:    Gap grande  → HPM-KD >> Direct (+4.0pp) ✅✅

CONCLUSÃO: KD (HPM-KD) é efetivo quando gap ≥ 5×
```

### **Se Confirmado:**
- ✅ **RQ1 VALIDADA**: HPM-KD supera baselines com compression ≥5×
- ✅ **Paper fortalecido**: Identificamos quando KD é vantajoso
- ✅ **Figuras prontas**: accuracy_vs_compression.png para Section 5
- ✅ **Publicação viável**: Resultados robustos e significativos

---

## 📁 Estrutura de Arquivos

```
experimento_01b_compression_ratios/
├── README.md                          ← Este arquivo
│
├── scripts/                           ← Scripts Python
│   └── 01b_compression_ratios.py     Script principal (822 linhas)
│
└── results/                           ← Resultados (vazio - aguardando execução)
    └── (outputs serão salvos aqui após execução)
```

---

## ⏱️ Estimativa de Tempo (RunPod)

### **Por GPU:**

| GPU | Full Mode | Quick Mode | Custo Estimado (Full) |
|-----|-----------|------------|----------------------|
| **RTX 4090** | 5-7h | 2-3h | $5-7 USD |
| **A100** | 3-5h | 1-2h | $10-15 USD |
| **V100** | 7-10h | 3-4h | $7-10 USD |
| **RTX 3090** | 8-12h | 3-5h | $6-9 USD |

**Recomendação:** RTX 4090 (melhor custo-benefício)

---

## 🎯 Análise de Resultados (Pós-Execução)

### **Métricas Principais:**

1. **Accuracy (%)**: Acurácia no test set
2. **Retention (%)**: `(Student Acc / Teacher Acc) × 100%`
3. **Δ (pp)**: Diferença HPM-KD - Direct
4. **Statistical Significance**: t-test (p < 0.05)

### **Critérios de Sucesso (RQ1):**

```
✅ RQ1 VALIDADA se:
  1. HPM-KD > Direct em compression 5× (Δ > +1.5pp, p < 0.05)
  2. HPM-KD > Direct em compression 7× (Δ > +2.5pp, p < 0.05)
  3. Figura accuracy_vs_compression mostra tendência clara

❌ RQ1 NÃO VALIDADA se:
  1. Direct ≥ HPM-KD em todos os compression ratios
  2. Diferenças não são estatisticamente significativas (p > 0.05)
```

---

## 📚 Documentação Relacionada

- **Experimento 1:** `../experimento_01_compression_efficiency/README.md` (concluído)
- **Sumário Completo:** `../SUMARIO_COMPLETO_EXPERIMENTOS.md`
- **Biblioteca lib/:** `../lib/` (utils compartilhados)

---

## ✅ Checklist de Execução

### **Antes de Executar:**
- [ ] RunPod configurado com GPU adequada
- [ ] Dependências instaladas (PyTorch, DeepBridge, etc.)
- [ ] GPU verificada (`nvidia-smi`)
- [ ] Disco com ≥50GB disponível
- [ ] Script `01b_compression_ratios.py` disponível

### **Durante Execução:**
- [ ] GPU sendo utilizada (verificar `nvidia-smi`)
- [ ] Dataset CIFAR10 baixando
- [ ] Teacher ResNet50 treinando
- [ ] Progress bars aparecendo
- [ ] Logs sendo gerados

### **Após Execução:**
- [ ] Ver `results.csv` (dados numéricos)
- [ ] Ler `experiment_report.md` (relatório)
- [ ] Analisar `accuracy_vs_compression.png` ⭐⭐⭐
- [ ] Download resultados localmente
- [ ] Backup em cloud storage
- [ ] Incluir figuras no paper (Section 5)
- [ ] Atualizar paper com resultados
- [ ] Validar RQ1 ✅ ou ❌

---

## 🎉 Conclusão

### **Por Que Este Experimento É Essencial:**

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║  Este é o EXPERIMENTO MAIS IMPORTANTE do paper HPM-KD!   ║
║                                                            ║
║  ✅ Corrige o problema do Experimento 1 (compression 2×) ║
║  ✅ Testa compression ratios REALISTAS (5×, 7×)          ║
║  ✅ Valida efetivamente RQ1 do paper                     ║
║  ✅ 100% pronto para executar com GPU dedicada           ║
║  ✅ Resultados em 3-10h (dependendo da GPU)              ║
║  ✅ Figuras prontas para publicação                      ║
║                                                            ║
║  SEM ESTE EXPERIMENTO, O PAPER NÃO TEM VALIDAÇÃO DE RQ1! ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### **Próximo Passo:**
**EXECUTAR AGORA NO RUNPOD!** 🚀

```bash
# Setup RunPod
cd scripts/
python 01b_compression_ratios.py --mode full --dataset CIFAR10 --gpu 0

# Aguarde 3-10h (dependendo da GPU)
# Valide RQ1 com os resultados
```

---

**Criado:** Dezembro 2025
**Última Atualização:** Dezembro 2025
**Status:** ⏳ Pronto para Executar
**Importância:** ⭐⭐⭐ **CRÍTICO**
**Autor:** Gustavo Haase
**Paper:** HPM-KD Framework
**Plataforma:** RunPod.io (GPU dedicada)
