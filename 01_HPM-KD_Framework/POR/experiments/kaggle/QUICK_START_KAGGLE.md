# ⚡ Quick Start - Experimento 1B no Kaggle

## 🎯 3 Passos Rápidos

### **1. Setup GPU (1 minuto)**
```
Kaggle Notebook → Settings (⚙️) → Accelerator → GPU T4 x2 → Save
```

### **2. Upload Script (1 minuto)**
- Baixe: `run_exp1b_kaggle.py`
- Sidebar → ➕ Add Data → Upload
- OU cole código com `%%writefile`

### **3. Executar (2-10 horas)**
```bash
# Quick Mode (2-3h) - TESTE
!python run_exp1b_kaggle.py --mode quick --dataset CIFAR10

# Full Mode (8-10h) - PAPER
!python run_exp1b_kaggle.py --mode full --dataset CIFAR10
```

---

## 📋 Comandos Essenciais

### **Executar:**
```bash
# Quick mode
!python run_exp1b_kaggle.py --mode quick

# Full mode
!python run_exp1b_kaggle.py --mode full

# Apenas compression 5× (mais crítico)
!python run_exp1b_kaggle.py --mode quick --compression 5x

# Retomar se desconectou
!python run_exp1b_kaggle.py --mode full --resume
```

### **Monitorar:**
```bash
# Ver progresso
!tail -50 /kaggle/working/experiment.log

# GPU usage
!nvidia-smi

# Modelos salvos
!ls -lh /kaggle/working/exp1b_*/checkpoints/*.pt
```

### **Ver Resultados:**
```python
import pandas as pd
from IPython.display import Markdown, Image, display

# Dados
df = pd.read_csv('/kaggle/working/exp1b_*/results.csv')
print(df)

# Relatório
with open('/kaggle/working/exp1b_*/experiment_report.md') as f:
    display(Markdown(f.read()))

# Figura principal
display(Image(filename='/kaggle/working/exp1b_*/figures/accuracy_vs_compression.png'))
```

### **Download:**
```
Output tab (canto superior direito) → Download All
```

---

## ⏱️ Tempo Esperado

| Modo | GPU P100 | GPU T4 |
|------|----------|--------|
| Quick | 1.5-2h | 2-3h |
| Full | 5-7h | 8-10h |

---

## 🔥 Copy-Paste Completo (Kaggle Notebook)

```python
# ===== CÉLULA 1: Verificar GPU =====
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - ATIVAR GPU!'}")

# ===== CÉLULA 2: Upload Script =====
# Use: Sidebar → Add Data → Upload run_exp1b_kaggle.py
# Depois execute:
!cp /kaggle/input/*/run_exp1b_kaggle.py /kaggle/working/

# ===== CÉLULA 3: Executar Quick Mode =====
!python /kaggle/working/run_exp1b_kaggle.py --mode quick --dataset CIFAR10

# ===== CÉLULA 4: Ver Resultados =====
import pandas as pd
from IPython.display import Markdown, Image, display

# Resultados
df = pd.read_csv('/kaggle/working/exp1b_*/results.csv')
print("\n📊 RESULTADOS:")
print(df.to_string())

# Relatório
with open('/kaggle/working/exp1b_*/experiment_report.md') as f:
    display(Markdown(f.read()))

# Figuras
display(Image(filename='/kaggle/working/exp1b_*/figures/accuracy_vs_compression.png', width=800))
display(Image(filename='/kaggle/working/exp1b_*/figures/hpmkd_vs_direct.png', width=800))

# ===== CÉLULA 5: Download =====
# Clique em "Output" (canto superior direito) → Download All
```

---

## ⚠️ Se Desconectar

**Não se preocupe!** Kaggle salva checkpoints automaticamente.

Apenas adicione `--resume`:
```bash
!python run_exp1b_kaggle.py --mode full --dataset CIFAR10 --resume
```

Ele retoma de onde parou (teacher já treinado é reutilizado)!

---

## ✅ Outputs Gerados

```
/kaggle/working/exp1b_full_YYYYMMDD_HHMMSS/
├── results.csv                      ⭐ Dados numéricos
├── experiment_report.md             ⭐ Relatório final
├── figures/
│   ├── accuracy_vs_compression.png ⭐⭐⭐ PRINCIPAL
│   └── hpmkd_vs_direct.png         ⭐⭐ "When KD helps?"
└── checkpoints/                     💾 Para retomar
```

**Download:** Output tab → Download All (ZIP ~500MB-2GB)

---

## 🎯 Resultado Esperado

| Compression | Direct | HPM-KD | Δ | Conclusão |
|-------------|--------|--------|---|-----------|
| 2.3× | ~88.5% | ~88.7% | +0.2pp | Empate |
| 5× | ~85.0% | ~87.5% | **+2.5pp** ✅ | HPM-KD vence |
| 7× | ~82.0% | ~86.0% | **+4.0pp** ✅✅ | HPM-KD vence |

**Conclusão:** HPM-KD é superior com compression ≥ 5× → **Valida RQ1!**

---

## 💡 Pro Tip

**Sessões curtas?** Execute 1 compression por vez:

```bash
# Sessão 1 (1h)
!python run_exp1b_kaggle.py --mode quick --compression 5x

# Sessão 2 (1h)
!python run_exp1b_kaggle.py --mode quick --compression 2.3x

# Sessão 3 (1h)
!python run_exp1b_kaggle.py --mode quick --compression 7x
```

Depois junte os CSVs manualmente!

---

**Pronto! 🚀 Só copiar e colar no Kaggle!**
