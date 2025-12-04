# 🚀 Experimento 1B - Escolha Sua Plataforma

## 📊 Comparação de Plataformas

| Característica | Google Colab | **Kaggle** ⭐ |
|----------------|--------------|--------------|
| **Tempo de sessão** | 90 minutos | **9-12 horas** |
| **GPU grátis/semana** | ~12h | **30h** |
| **Desconexões** | Frequentes | Raras |
| **GPU** | T4 (16GB) | **P100 (16GB) ou T4** |
| **RAM** | 12GB | **16GB** |
| **Outputs persistem** | Não | **Sim, automaticamente** |
| **Checkpoints** | Manual | **Automático** |
| **Melhor para** | Testes rápidos (<1h) | **Experimentos longos (2-10h)** |

---

## ✅ Recomendação

### **Para Este Experimento: Use KAGGLE! 🎯**

**Motivo:** Experimento 1B leva 2-10 horas (dependendo do modo).

- ❌ **Colab:** Desconecta após 90 minutos → Você perde progresso!
- ✅ **Kaggle:** Sessões de 9-12h → Experimento completa sem interrupção!

---

## 📁 Arquivos Disponíveis

### **🔵 Kaggle (RECOMENDADO)**

📂 **Localização:** `experiments/kaggle/`

**Arquivos:**
- `run_exp1b_kaggle.py` - Script principal (810 linhas)
- `INDEX.md` - Visão geral
- `README_KAGGLE.md` - Guia completo
- `QUICK_START_KAGGLE.md` - Guia rápido

**Como usar:**
```bash
cd experiments/kaggle/
# Leia QUICK_START_KAGGLE.md
```

---

### **🟡 Google Colab (NÃO RECOMENDADO)**

📂 **Localização:** `experiments/scripts/`

**Arquivos:**
- `run_exp1b_colab.py` - Script Colab (822 linhas)

**⚠️ Limitação:** Sessões de 90 minutos → Experimento não completa!

**Use apenas para:** Testes muito rápidos (<1h)

---

## 🚀 Quick Start (Kaggle)

### **Passo 1: Criar Notebook**
1. https://www.kaggle.com/code → New Notebook
2. Settings → Accelerator → **GPU T4 x2**
3. Settings → Internet → **ON**

### **Passo 2: Upload Script**
1. Baixe `experiments/kaggle/run_exp1b_kaggle.py`
2. Kaggle → Add Data → Upload
3. Execute:
```bash
!cp /kaggle/input/*/run_exp1b_kaggle.py /kaggle/working/
```

### **Passo 3: Executar**
```bash
# Quick Mode (2-3h) - TESTE
!python /kaggle/working/run_exp1b_kaggle.py --mode quick

# Full Mode (8-10h) - PAPER
!python /kaggle/working/run_exp1b_kaggle.py --mode full
```

---

## ⏱️ Tempo de Execução

### **Kaggle:**
| Modo | GPU P100 | GPU T4 |
|------|----------|--------|
| Quick | 1.5-2h ✅ | 2-3h ✅ |
| Full | 5-7h ✅ | 8-10h ✅ |

### **Colab:**
| Modo | GPU T4 | Status |
|------|--------|--------|
| Quick | 2-3h | ❌ Desconecta (90min) |
| Full | 8-10h | ❌ Desconecta (90min) |

**Conclusão:** Apenas Kaggle suporta este experimento!

---

## 📊 Resultados Gerados

```
/kaggle/working/exp1b_full_YYYYMMDD/
├── results.csv                      ⭐ Dados
├── experiment_report.md             ⭐ Relatório
├── figures/
│   ├── accuracy_vs_compression.png ⭐⭐⭐ PRINCIPAL
│   └── hpmkd_vs_direct.png         ⭐⭐
└── checkpoints/                     💾 Resume automático
```

**Download:** Output tab → Download All

---

## 💾 Sistema de Checkpoints (Kaggle)

**Se desconectar (raro):**
```bash
!python run_exp1b_kaggle.py --mode full --resume
```

Retoma de onde parou! Teacher já treinado é reutilizado.

---

## 🎯 Resultado Esperado

| Compression | Direct | HPM-KD | Δ | Conclusão |
|-------------|--------|--------|---|-----------|
| 2.3× | ~88.5% | ~88.7% | +0.2pp | Empate |
| 5× | ~85.0% | ~87.5% | **+2.5pp** ✅ | **HPM-KD vence** |
| 7× | ~82.0% | ~86.0% | **+4.0pp** ✅✅ | **HPM-KD vence** |

**Se confirmado:** ✅ Valida RQ1 do paper!

---

## 📚 Documentação

### **Kaggle (Recomendado):**
1. **Quick Start:** `kaggle/QUICK_START_KAGGLE.md` (3 passos)
2. **Guia Completo:** `kaggle/README_KAGGLE.md` (516 linhas)
3. **Índice:** `kaggle/INDEX.md`
4. **Resumo:** `COMO_USAR_KAGGLE.txt`

### **Colab (Não Recomendado):**
1. ~~`scripts/run_exp1b_colab.py`~~ (limitação de 90min)

---

## ✅ Checklist

### **Antes de Executar:**
- [ ] Conta Kaggle criada
- [ ] Telefone verificado (para GPU)
- [ ] Leu `kaggle/QUICK_START_KAGGLE.md`
- [ ] Baixou `run_exp1b_kaggle.py`

### **Durante Execução:**
- [ ] GPU ativada (P100 ou T4)
- [ ] Internet ON
- [ ] Monitora log: `!tail -f experiment.log`

### **Após Execução:**
- [ ] Download resultados (Output tab)
- [ ] Revisar `experiment_report.md`
- [ ] Incluir figuras no paper

---

## 💡 Dicas Pro

1. **Use Kaggle** - Sessões longas (9-12h)
2. **GPU P100** - 40% mais rápido que T4 (quando disponível)
3. **Quick Mode** - Primeiro para testar (2-3h)
4. **Full Mode** - Depois para o paper (8-10h)
5. **Checkpoints** - Resume automático se desconectar
6. **Save Version** - Após execução para guardar outputs

---

## 📞 Suporte

**Kaggle:**
- Documentação: `kaggle/README_KAGGLE.md`
- Quick Start: `kaggle/QUICK_START_KAGGLE.md`
- Community: https://www.kaggle.com/discussions

**Colab:**
- ❌ Não recomendado para este experimento (90min timeout)

---

## 🎉 Resumo

✅ **Use Kaggle** para Experimento 1B
✅ Leia `kaggle/QUICK_START_KAGGLE.md`
✅ Upload `run_exp1b_kaggle.py`
✅ Execute `--mode quick` (teste) ou `--mode full` (paper)
✅ Aguarde 2-10 horas (dependendo do modo)
✅ Download resultados e incluir no paper

**Boa sorte! 🚀**

---

**Criado:** Dezembro 2025
**Status:** ✅ Pronto para uso
**Plataforma recomendada:** Kaggle
