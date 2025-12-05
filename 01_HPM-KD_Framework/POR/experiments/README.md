# Experimentos - HPM-KD Framework

## 📊 Visão Geral

Repositório organizado com todos os experimentos do paper **HPM-KD Framework** (Hierarchical Progressive Multi-Teacher Knowledge Distillation).

---

## 📂 Estrutura dos Experimentos

```
experiments/
│
├── experimento_01_compression_efficiency/       ✅ CONCLUÍDO
├── experimento_01b_compression_ratios/          ⏳ PRONTO ⭐⭐⭐ CRÍTICO
├── experimento_02_ablation_studies/             📋 PENDENTE
├── experimento_03_generalization/               📋 PENDENTE
├── experimento_04_computational_efficiency/     📋 PENDENTE
│
├── lib/                                         Bibliotecas compartilhadas
└── SUMARIO_COMPLETO_EXPERIMENTOS.md             Documentação completa
```

---

## 🎯 Experimentos

| # | Experimento | RQ | Status | Modelos | Prioridade |
|---|-------------|-------|--------|---------|------------|
| **1** | Compression Efficiency | RQ1 | ✅ Concluído | 31 | - |
| **1B** | Compression Ratios ⭐ | RQ1 | ⏳ **PRONTO** | 46 | **1** |
| **2** | Ablation Studies | RQ2 | 📋 Pendente | ~280 | 2 |
| **3** | Generalization | RQ3 | 📋 Pendente | ~83 | 3 |
| **4** | Computational Efficiency | RQ4 | 📋 Pendente | ~8 | 4 |

**TOTAL:** ~448 modelos

---

## 🚀 Início Rápido

### **Executar Experimento 1B (CRÍTICO)** ⭐⭐⭐

```bash
# Navegue para o experimento
cd experimento_01b_compression_ratios/

# Leia a documentação
cat README.md

# Execute no RunPod
cd scripts/
python 01b_compression_ratios.py --mode full --dataset CIFAR10 --gpu 0
```

---

## 📚 Documentação

### **Por Experimento:**
- Cada pasta `experimento_XX_nome/` tem seu próprio `README.md` completo
- Acesse a pasta e leia o README para instruções detalhadas

### **Geral:**
- `SUMARIO_COMPLETO_EXPERIMENTOS.md` - Documentação detalhada de todos os experimentos
- `lib/` - Bibliotecas Python compartilhadas (cnn_models.py, utils_training.py)

---

## 🎯 Research Questions (RQs)

| RQ | Pergunta | Experimento |
|----|----------|-------------|
| **RQ1** | HPM-KD consegue maiores compression ratios mantendo acurácia? | Exp 1 + **1B** ⭐ |
| **RQ2** | Qual a contribuição de cada componente do HPM-KD? | Exp 2 |
| **RQ3** | HPM-KD generaliza melhor em condições adversas? | Exp 3 |
| **RQ4** | Qual o overhead computacional do HPM-KD? | Exp 4 |

---

## ⚙️ Plataforma de Execução

**Atual:** RunPod.io (GPU dedicada)
- RTX 4090, A100, V100
- Storage: 50GB+
- Custo: $5-15 USD por experimento (Full Mode)

---

## 📝 Status Atual

```
✅ Experimento 1:   CONCLUÍDO (Novembro 2025)
                    - Resultado: Direct venceu (compression 2× insuficiente)
                    - Ação: Executar Experimento 1B

⏳ Experimento 1B:  PRONTO PARA EXECUTAR (CRÍTICO) ⭐⭐⭐
                    - Compression: 2.3×, 5×, 7×
                    - Plataforma: RunPod.io
                    - Tempo: 3-10h (dependendo da GPU)
                    - Objetivo: Validar RQ1

📋 Experimentos 2, 3, 4: PENDENTES
                    - Executar após validação de RQ1
```

---

## 🔗 Links Úteis

- **Paper:** `../../ENG/` (versão em inglês)
- **Biblioteca DeepBridge:** `pip install deepbridge`
- **RunPod:** https://www.runpod.io/

---

## 👤 Autor

**Gustavo Haase**
**Data:** Dezembro 2025
**Status:** ✅ Estrutura Organizada e Pronta

---

## 🚀 Próximo Passo

**EXECUTAR EXPERIMENTO 1B NO RUNPOD** para validar RQ1!

```bash
cd experimento_01b_compression_ratios/
cat README.md  # Leia as instruções
```
