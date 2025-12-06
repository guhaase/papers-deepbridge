# GPU Requirements - Experimentos DeepBridge

**Data:** 2025-12-06

---

## 📊 Resumo Executivo

**Resposta Rápida**: Apenas **1 de 6 experimentos** requer GPU (opcional):

| Experimento | GPU Necessária? | Tipo de Computação | Pode Rodar em CPU? |
|-------------|-----------------|--------------------|--------------------|
| 01 - Benchmarks | ❌ NÃO | Dados tabulares (XGBoost, Adult Income) | ✅ Sim |
| 02 - Casos de Uso | ❌ NÃO | Dados tabulares (6 domínios) | ✅ Sim |
| 03 - Usabilidade | ❌ NÃO | Análise estatística (mock data) | ✅ Sim |
| **04 - HPM-KD** | ⚠️ **OPCIONAL** | **PyTorch MLP** (student) | ⚠️ Sim, mas lento |
| 05 - Conformidade | ❌ NÃO | Compliance tests | ✅ Sim |
| 06 - Ablation Studies | ❌ NÃO | Análise de componentes | ✅ Sim |

**Conclusão**: Você pode executar **TODOS os experimentos em CPU**. GPU só acelera o Experimento 04.

---

## 🔍 Análise Detalhada por Experimento

### ✅ Experimento 01: Benchmarks de Tempo

**GPU Necessária?** ❌ **NÃO**

**Por quê?**
- Dataset: Adult Income (dados tabulares)
- Modelos: XGBoost, LightGBM (tree-based, otimizados para CPU)
- DeepBridge: Opera sobre modelos já treinados
- Workflow fragmentado: Usa bibliotecas tradicionais (AIF360, Fairlearn)

**Hardware Recomendado:**
- CPU: 4+ cores
- RAM: 8GB
- Tempo: ~30 minutos (10 runs)

**Status**: ✅ Já executado com sucesso em CPU

---

### ✅ Experimento 02: Estudos de Caso

**GPU Necessária?** ❌ **NÃO**

**Por quê?**
- 6 domínios: Crédito, Contratação, Saúde, Hipoteca, Seguros, Fraude
- Todos usam dados tabulares
- Modelos: XGBoost, Random Forest, LightGBM, Gradient Boosting
- 1.4M amostras processadas (mas em tree models, não neural networks)

**Hardware Recomendado:**
- CPU: 8+ cores (para paralelizar 6 casos)
- RAM: 16GB
- Tempo: ~15 minutos (com dados sintéticos)

**Status**: ✅ Já executado com sucesso em CPU

---

### ✅ Experimento 03: Usabilidade

**GPU Necessária?** ❌ **NÃO**

**Por quê?**
- Análise estatística pura
- Cálculo de SUS, NASA TLX scores
- Testes de normalidade, correlações
- Geração de visualizações (matplotlib)
- Nenhum treinamento de modelo

**Hardware Recomendado:**
- CPU: 2+ cores
- RAM: 4GB
- Tempo: ~3 minutos

**Status**: ✅ Já executado com sucesso em CPU

---

### ⚠️ **Experimento 04: HPM-KD** (ÚNICO QUE USA DEEP LEARNING)

**GPU Necessária?** ⚠️ **OPCIONAL** (recomendada para versão real)

**Por quê?**

#### Teachers (NÃO precisam de GPU):
- XGBoost (200 estimators)
- LightGBM (200 estimators)
- CatBoost (200 iterations)
- **Total**: ~2.4GB
- **Treinamento**: CPU suficiente (tree-based models)

#### **Student (PODE se beneficiar de GPU):**
- **MLP compacto PyTorch** (64, 32 hidden layers)
- **Framework**: PyTorch
- **Total**: ~230MB
- **Treinamento**: Knowledge Distillation

**Análise:**

| Cenário | Hardware | Tempo Estimado (20 datasets) | Viável? |
|---------|----------|------------------------------|---------|
| **CPU Only** | 8+ cores, 16GB RAM | ~5-7 dias | ⚠️ Lento mas viável |
| **GPU (RTX 3080)** | CUDA, 32GB RAM | ~3-4 semanas | ✅ Recomendado |

**Dependências PyTorch:**
```python
torch>=2.0.0
torchvision>=0.15.0
```

**Observações:**
- MLP é uma rede **pequena** (64, 32 neurons)
- Com CPU, treinar 1 student pode levar ~1-2 horas
- Com GPU, treinar 1 student leva ~5-10 minutos
- **Total de students**: 60 (20 datasets × 3 métodos)

**Recomendação:**
- **Mock/Demo**: CPU suficiente (poucos datasets, teste rápido)
- **Versão Real (20 datasets)**: GPU altamente recomendada

**Status**: ⏳ Mock implementation (CPU viável)

---

### ✅ Experimento 05: Conformidade

**GPU Necessária?** ❌ **NÃO**

**Por quê?**
- Testes de compliance com regulações (GDPR, EEOC, ECOA, etc.)
- Validação de métricas de fairness
- Análise de documentação e relatórios
- Nenhum treinamento de modelo pesado

**Hardware Recomendado:**
- CPU: 4+ cores
- RAM: 8GB

**Status**: 📋 Planejado (não requer GPU)

---

### ✅ Experimento 06: Ablation Studies

**GPU Necessária?** ❌ **NÃO**

**Por quê?**
- Análise de componentes do DeepBridge
- Remoção incremental de features
- Medição de impacto na performance
- Usa modelos já treinados (análise, não treinamento)

**Hardware Recomendado:**
- CPU: 4+ cores
- RAM: 8GB

**Status**: 📋 Planejado (não requer GPU)

---

## 🎯 Recomendações Práticas

### Para Você (Agora)

**Sua Situação**: Tem acesso a GPU (servidor RunPod/Kaggle)

**Recomendação**:
1. ✅ **Experimentos 01, 02, 03, 05, 06**: Execute em **CPU local**
   - Rápidos, leves, não justificam custo de GPU
   - Total: ~1 hora de execução combinada

2. ⚠️ **Experimento 04 (HPM-KD)**:
   - **Mock/Demo**: CPU local (teste rápido)
   - **Versão Real (20 datasets)**: GPU no RunPod/Kaggle
   - Economiza ~5 dias de CPU vs ~3-4 semanas em GPU

### Comparação de Custo-Benefício

#### CPU Local (Todos os Experimentos):
```
Exp 01:  ~30 min   ✅ Viável
Exp 02:  ~15 min   ✅ Viável
Exp 03:  ~3 min    ✅ Viável
Exp 04:  ~5-7 dias ⚠️ Lento (versão real)
Exp 05:  ~20 min   ✅ Viável
Exp 06:  ~30 min   ✅ Viável
─────────────────────────────────────
TOTAL:   ~1.5h + 5-7 dias (Exp04 real)
```

#### GPU RunPod (Apenas Exp 04):
```
Exp 04 (GPU RTX 3080):  ~3-4 semanas
Custo estimado:         ~$50-100 USD
─────────────────────────────────────
Speedup vs CPU:         ~10-20× mais rápido
```

---

## 📋 Checklist de Decisão

### Quando usar CPU?

- [x] Você quer testar/validar a infraestrutura
- [x] Você está trabalhando com mock/demo data
- [x] Experimentos 01, 02, 03, 05, 06
- [x] Experimento 04 com poucos datasets (≤3)
- [x] Budget limitado

### Quando usar GPU?

- [ ] Experimento 04 com **20 datasets reais**
- [ ] Você quer resultados em **semanas** ao invés de **meses**
- [ ] Treinamento de múltiplos students (60 modelos)
- [ ] Budget disponível (~$50-100 para RunPod)

---

## 💡 Dicas de Otimização

### Se Usar CPU para Experimento 04:

1. **Paralelizar por Dataset** (não por método):
   ```python
   # Processar datasets em paralelo (8 cores)
   from joblib import Parallel, delayed
   results = Parallel(n_jobs=8)(
       delayed(train_student)(dataset) for dataset in datasets
   )
   ```

2. **Reduzir Epochs**:
   ```python
   # Em vez de 200 epochs, usar 50-100
   epochs = 50  # Para teste
   ```

3. **Começar com Subset**:
   ```python
   # Testar com 3 datasets primeiro
   datasets = ['Adult', 'Bank', 'Credit']  # Ao invés de 20
   ```

### Se Usar GPU para Experimento 04:

1. **Batch Size Maior**:
   ```python
   batch_size = 512  # Aproveitar VRAM
   ```

2. **Mixed Precision Training**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

3. **DataLoader com pin_memory**:
   ```python
   DataLoader(..., pin_memory=True, num_workers=4)
   ```

---

## 📊 Tabela Resumo Final

| Experimento | GPU? | Justificativa | Tempo CPU | Tempo GPU | Prioridade GPU |
|-------------|------|---------------|-----------|-----------|----------------|
| **01 - Benchmarks** | ❌ | Tree models | ~30 min | - | Nenhuma |
| **02 - Casos de Uso** | ❌ | Tree models | ~15 min | - | Nenhuma |
| **03 - Usabilidade** | ❌ | Estatística | ~3 min | - | Nenhuma |
| **04 - HPM-KD** | ⚠️ | PyTorch MLP | ~5-7 dias* | ~3-4 sem | **ALTA*** |
| **05 - Conformidade** | ❌ | Análise | ~20 min | - | Nenhuma |
| **06 - Ablation** | ❌ | Análise | ~30 min | - | Nenhuma |

\* Para versão real com 20 datasets

---

## 🚀 Próximos Passos Recomendados

### Curto Prazo (Esta Semana):

1. ✅ Executar **Exp 01, 02, 03** em CPU local
   - Já estão prontos e documentados
   - Total: ~1 hora

2. ✅ Testar **Exp 04 (mock)** em CPU local
   - 1-3 datasets apenas
   - Validar infraestrutura

### Médio Prazo (Próximas 2-3 Semanas):

3. 🚀 Executar **Exp 04 (real)** em GPU RunPod/Kaggle
   - 20 datasets completos
   - Investimento justificado (~$50-100)

4. ✅ Executar **Exp 05, 06** em CPU local
   - Após ter resultados do Exp 04

---

## ✅ Conclusão

**Resposta Direta**: Apenas o **Experimento 04 (HPM-KD)** usa PyTorch e pode se beneficiar de GPU, mas **TODOS podem rodar em CPU**.

**Recomendação Prática**:
- Execute **5 de 6 experimentos em CPU** (rápidos, ~1-2 horas total)
- Reserve **GPU apenas para Exp 04 versão real** (quando necessário)
- Para testes/demos, **CPU é suficiente**

**Seu Caso (RunPod ativo agora)**:
- Se está rodando **Exp 1B do HPM-KD Framework** (CIFAR100): ✅ Ótimo uso de GPU!
- Para **Exp 04 do DeepBridge Overview** (dados tabulares): ⚠️ GPU não é crítico (MLP pequeno)

---

**Documento criado em**: 2025-12-06
**Status**: ✅ Análise Completa
**Próxima Ação**: Decidir se mantém GPU para Exp 04 ou libera para outros experimentos
