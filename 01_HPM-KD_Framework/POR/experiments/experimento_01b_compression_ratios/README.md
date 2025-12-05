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
| **Plataforma** | **Kaggle** (9-12h sessões vs 90min Colab) |

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
║  ✅ 100% PRONTO para executar no Kaggle                        ║
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

**Nota:** Reduzido de 6 para 3 métodos para focar nos mais importantes e economizar tempo.

---

## ⚙️ Hiperparâmetros

### **Quick Mode (Teste - 2-3h):**

| Parâmetro | Valor |
|-----------|-------|
| **Teacher Epochs** | 50 |
| **Student Epochs** | 20 |
| **Runs por método** | 3 |
| **Batch Size** | 128 |
| **Learning Rate** | 0.1 |
| **Optimizer** | SGD (momentum=0.9, weight_decay=5e-4) |
| **Scheduler** | MultiStepLR [60, 120, 160] |
| **Data Augmentation** | RandomCrop, RandomHorizontalFlip |

**Modelos:** 1 teacher + (3 compression × 3 métodos × 3 runs) = **28 modelos**

**Tempo:** 2-3h (Kaggle GPU T4)

### **Full Mode (Paper - 8-10h):**

| Parâmetro | Valor |
|-----------|-------|
| **Teacher Epochs** | 100 |
| **Student Epochs** | 50 |
| **Runs por método** | **5** (maior robustez) |
| **Batch Size** | 128 |
| **Learning Rate** | 0.1 |
| **Optimizer** | SGD (momentum=0.9, weight_decay=5e-4) |
| **Scheduler** | MultiStepLR [60, 120, 160] |
| **Data Augmentation** | RandomCrop, RandomHorizontalFlip |

**Modelos:** 1 teacher + (3 compression × 3 métodos × 5 runs) = **46 modelos**

**Tempo:** 8-10h (Kaggle GPU T4) ou 5-7h (Kaggle GPU P100)

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

## 🚀 Como Executar (Kaggle)

### **Por Que Kaggle e Não Colab:**

| Aspecto | Google Colab | Kaggle ✅ |
|---------|--------------|-----------|
| **Sessão** | 90 minutos | **9-12 horas** |
| **GPU/semana** | ~12h | **30h** |
| **Desconexões** | Frequentes | Raras |
| **Experimento 1B** | ❌ Não completa | ✅ **Completa** |

**Decisão:** Migrado para Kaggle devido a sessões longas (experimento leva 8-10h).

### **Passo 1: Setup Kaggle (2 minutos)**

1. Acesse: https://www.kaggle.com/code
2. Clique em **New Notebook**
3. **Settings** → **Accelerator** → **GPU T4 x2** → **Save**
4. **Settings** → **Internet** → **ON** → **Save**

### **Passo 2: Upload Script (1 minuto)**

1. Baixe: `scripts/run_exp1b_kaggle.py`
2. Kaggle → **Add Data** → **Upload**
3. Execute em célula:
```python
!cp /kaggle/input/*/run_exp1b_kaggle.py /kaggle/working/
```

### **Passo 3: Executar (2-10 horas)**

#### **Quick Mode (2-3h) - Testar Pipeline:**
```python
!python /kaggle/working/run_exp1b_kaggle.py --mode quick --dataset CIFAR10
```

#### **Full Mode (8-10h) - Resultados para Paper:**
```python
!python /kaggle/working/run_exp1b_kaggle.py --mode full --dataset CIFAR10
```

#### **Compression Específico (1h) - Apenas 5×:**
```python
!python /kaggle/working/run_exp1b_kaggle.py --mode quick --compression 5x
```

#### **Retomar se Desconectar (raro):**
```python
!python /kaggle/working/run_exp1b_kaggle.py --mode full --resume
```

### **Monitoramento:**
```python
# Ver progresso
!tail -50 /kaggle/working/experiment.log

# GPU usage
!nvidia-smi

# Checkpoints salvos
!ls -lh /kaggle/working/exp1b_*/checkpoints/
```

---

## 💾 Sistema de Checkpoints

### **Features:**

- ✅ **Teacher reutilizado**: Treinado UMA VEZ e usado para todos os students (economia 30min-1h!)
- ✅ **Granular**: Checkpoint por experimento/método/run
- ✅ **Resume automático**: `--resume` flag retoma de onde parou
- ✅ **Robusto**: Salva estado completo (pickle) após cada run

### **Estrutura:**
```python
checkpoints/
├── experiment_state.pkl              # Estado completo (resume)
├── teacher_resnet50_CIFAR10.pt      # 2.6 MB (reutilizado!)
└── student_*.pt                      # 27 (quick) ou 45 (full) modelos
```

### **Se Kaggle Desconectar (raro):**
```python
!python run_exp1b_kaggle.py --mode full --resume
# Retoma de onde parou! Teacher já treinado não é retreinado.
```

---

## 📊 Outputs Gerados

### **Estrutura de Saída:**
```
/kaggle/working/exp1b_full_YYYYMMDD_HHMMSS/
├── results.csv                       📊 Dados numéricos (CSV)
├── experiment_report.md              📄 Relatório completo (Markdown)
├── experiment.log                    📋 Log de execução detalhado
│
├── figures/                          📈 Visualizações (PNG 300 DPI)
│   ├── accuracy_vs_compression.png  ⭐⭐⭐ FIGURA PRINCIPAL (paper)
│   ├── hpmkd_vs_direct.png          ⭐⭐ "When does KD help?"
│   └── retention_analysis.png       📊 Retenção de conhecimento
│
├── checkpoints/                      💾 Para retomar se desconectar
│   ├── experiment_state.pkl         Estado completo
│   ├── teacher_resnet50_CIFAR10.pt  2.6 MB (reutilizado!)
│   └── student_*.pt                 27 ou 45 modelos (227 KB cada)
│
└── data/                             📦 CIFAR10 (auto-download)
    └── cifar-10-batches-py/
```

**Tamanho Total:**
- Quick Mode: ~500 MB
- Full Mode: ~2 GB

### **Download:**
```
Output tab (canto superior direito) → Download All (ZIP)
```

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

## 📁 Estrutura de Arquivos

```
experimento_01b_compression_ratios/
├── README.md                          ← Este arquivo
│
├── scripts/                           ← Scripts Python
│   ├── 01b_compression_ratios.py     Script original (822 linhas)
│   └── run_exp1b_kaggle.py           ⭐ Script Kaggle (810 linhas) - USAR ESTE
│
└── results/                           ← Resultados (vazio - aguardando execução)
    └── (outputs serão salvos aqui após execução)
```

---

## ⏱️ Estimativa de Tempo

### **Kaggle GPU T4:**

| Modo | Tempo | Breakdown |
|------|-------|-----------|
| **Quick** | 2-3h | Teacher: 30min, Students: 1.5-2.5h |
| **Full** | 8-10h | Teacher: 1h, Students: 7-9h |
| **5× only** | 45-60min | Teacher: 30min, Students 5×: 15-30min |

### **Kaggle GPU P100 (40% mais rápido):**

| Modo | Tempo | Breakdown |
|------|-------|-----------|
| **Quick** | 1.5-2h | Teacher: 20min, Students: 1-1.5h |
| **Full** | 5-7h | Teacher: 40min, Students: 4.5-6.5h |
| **5× only** | 30-45min | Teacher: 20min, Students 5×: 10-25min |

**Limite Kaggle:** 9-12h por sessão → **Suficiente para Full Mode!**

**Quota Kaggle:** 30h GPU/semana grátis

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

### **Possíveis Cenários:**

| Cenário | Resultado | Ação |
|---------|-----------|-------|
| **A** | HPM-KD > Direct (5× e 7×) | ✅ **RQ1 validada! Incluir no paper** |
| **B** | HPM-KD > Direct (apenas 7×) | ⚠️ Validado parcialmente, discutir no paper |
| **C** | Direct ≥ HPM-KD (todos) | ❌ RQ1 falhou, rever método ou hipótese |

---

## 📚 Documentação Relacionada

### **Guias Kaggle:**
- **Quick Start:** `../../kaggle/QUICK_START_KAGGLE.md` (3 passos)
- **Guia Completo:** `../../kaggle/README_KAGGLE.md` (516 linhas)
- **Índice:** `../../kaggle/INDEX.md`

### **Comparação de Plataformas:**
- **Kaggle vs Colab:** `../../README_PLATAFORMAS.md`

### **Documentação Geral:**
- **Sumário Completo:** `../../SUMARIO_COMPLETO_EXPERIMENTOS.md`
- **Contagem de Modelos:** `../../CONTAGEM_MODELOS.md`

---

## ✅ Checklist de Execução

### **Antes de Executar:**
- [ ] Conta Kaggle criada
- [ ] Telefone verificado (libera GPU)
- [ ] Lido `../../kaggle/QUICK_START_KAGGLE.md`
- [ ] Script `run_exp1b_kaggle.py` baixado
- [ ] Notebook Kaggle criado
- [ ] GPU ativada (Settings → GPU T4)
- [ ] Internet ON (Settings → Internet)

### **Durante Execução:**
- [ ] GPU P100 ou T4 detectada
- [ ] Dataset CIFAR10 baixando
- [ ] Teacher ResNet50 treinando
- [ ] Progress bars aparecendo
- [ ] Checkpoints salvando automaticamente
- [ ] Não fechar aba do navegador

### **Após Execução:**
- [ ] Ver `results.csv` (dados numéricos)
- [ ] Ler `experiment_report.md` (relatório)
- [ ] Analisar `accuracy_vs_compression.png` ⭐⭐⭐
- [ ] Download All (Output tab)
- [ ] Save Version (guardar outputs permanentemente)
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
║  ✅ 100% pronto para executar no Kaggle                  ║
║  ✅ Resultados em 8-10h (Full Mode)                      ║
║  ✅ Figuras prontas para publicação                      ║
║                                                            ║
║  SEM ESTE EXPERIMENTO, O PAPER NÃO TEM VALIDAÇÃO DE RQ1! ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

### **Próximo Passo:**
**EXECUTAR AGORA NO KAGGLE!** 🚀

1. Leia `../../kaggle/QUICK_START_KAGGLE.md` (5 minutos)
2. Upload `scripts/run_exp1b_kaggle.py` no Kaggle
3. Execute Quick Mode (2-3h) para testar
4. Execute Full Mode (8-10h) para o paper
5. Aguarde resultados e valide RQ1

---

**Criado:** Dezembro 2025
**Última Atualização:** Dezembro 2025
**Status:** ⏳ Pronto para Executar
**Importância:** ⭐⭐⭐ **CRÍTICO**
**Autor:** Gustavo Haase
**Paper:** HPM-KD Framework
