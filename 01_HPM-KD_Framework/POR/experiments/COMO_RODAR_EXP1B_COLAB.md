# 🚀 Como Executar Experimento 1B no Google Colab

**Script Standalone:** `run_exp1b_colab.py`

---

## ⚡ Quick Start (3 Passos)

### **1. Configurar GPU no Colab**
```
Runtime → Change runtime type → Hardware accelerator → GPU
```

### **2. Fazer Upload do Script**
Faça upload do arquivo `run_exp1b_colab.py` para o Colab, OU cole o código diretamente.

### **3. Executar**

```bash
# Quick Mode (2-3h) - TESTE RÁPIDO
!python run_exp1b_colab.py --mode quick --dataset CIFAR10

# Full Mode (8-10h) - PARA O PAPER
!python run_exp1b_colab.py --mode full --dataset CIFAR10

# Apenas compression 5× (mais crítico)
!python run_exp1b_colab.py --mode quick --compression 5x
```

---

## 📋 Comandos Completos

### **Opção 1: Quick Mode (Recomendado para Teste)**
```python
# Cole isto em uma célula do Colab:
!python run_exp1b_colab.py \
    --mode quick \
    --dataset CIFAR10 \
    --gpu 0
```

**Tempo:** 2-3 horas
**Runs:** 3 por método
**Epochs:** Teacher=50, Student=20

---

### **Opção 2: Full Mode (Para o Paper)**
```python
# Cole isto em uma célula do Colab:
!python run_exp1b_colab.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0
```

**Tempo:** 8-10 horas
**Runs:** 5 por método
**Epochs:** Teacher=100, Student=50

---

### **Opção 3: Testar Apenas Compression Específico**

```python
# Apenas 2.3× (ResNet18)
!python run_exp1b_colab.py --mode quick --compression 2.3x

# Apenas 5× (ResNet10) - MAIS CRÍTICO
!python run_exp1b_colab.py --mode quick --compression 5x

# Apenas 7× (MobileNetV2)
!python run_exp1b_colab.py --mode quick --compression 7x
```

---

### **Opção 4: Salvar no Google Drive**

```python
# 1. Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Executar salvando no Drive
!python run_exp1b_colab.py \
    --mode quick \
    --dataset CIFAR10 \
    --output /content/drive/MyDrive/HPM-KD-Results/exp1b_$(date +%Y%m%d)
```

---

## 📊 O Que Será Gerado

```
exp1b_results_quick_YYYYMMDD_HHMMSS/
├── experiment_report.md              ⭐ Relatório completo em Markdown
├── results_compression_ratios.csv    ⭐ Dados numéricos
├── models/                           💾 Modelos treinados
│   ├── teacher_resnet50_CIFAR10.pt
│   ├── student_2.3x_ResNet18_Direct_run1.pt
│   ├── student_2.3x_ResNet18_TradKD_run1.pt
│   ├── student_2.3x_ResNet18_HPMKD_run1.pt
│   ├── student_5x_ResNet10_*.pt
│   └── student_7x_MobileNetV2_*.pt
└── figures/                          📊 Figuras PNG
    ├── accuracy_vs_compression.png   ⭐⭐⭐ PRINCIPAL
    ├── hpmkd_vs_direct.png           ⭐⭐ "When KD helps?"
    └── retention_analysis.png
```

---

## 📈 Ver Resultados Durante Execução

### **Monitorar Progresso:**
```python
# Ver últimas 50 linhas do log
!tail -50 experiment.log

# Ver progresso em tempo real (Ctrl+C para parar)
!tail -f experiment.log
```

### **Ver Resultados Parciais:**
```python
import pandas as pd

# Carregar CSV
df = pd.read_csv('exp1b_results_quick_*/results_compression_ratios.csv')
print(df.to_string())
```

### **Ver Relatório:**
```python
from IPython.display import Markdown, display

with open('exp1b_results_quick_*/experiment_report.md', 'r') as f:
    display(Markdown(f.read()))
```

### **Ver Figuras:**
```python
from IPython.display import Image, display
import glob

# Figura principal
fig = glob.glob('exp1b_results_*/figures/accuracy_vs_compression.png')[0]
display(Image(filename=fig, width=800))

# HPM-KD vs Direct
fig = glob.glob('exp1b_results_*/figures/hpmkd_vs_direct.png')[0]
display(Image(filename=fig, width=800))
```

---

## 🎯 Resultados Esperados

### **Hipótese:**
> HPM-KD deve superar Direct em compression ratios ≥ 5×

### **Resultados Esperados:**

| Compression | Direct | HPM-KD | Δ | Status |
|-------------|--------|--------|---|--------|
| **2.3×** (ResNet18) | ~88.5% | ~88.7% | +0.2pp | ≈ Empate |
| **5×** (ResNet10) | ~85.0% | ~87.5% | **+2.5pp** ✅ | HPM-KD vence |
| **7×** (MobileNetV2) | ~82.0% | ~86.0% | **+4.0pp** ✅✅ | HPM-KD vence forte |

### **Conclusão Esperada:**
```
✅ HPM-KD é mais efetivo com compression ratios MAIORES (≥5×)
✅ Com gaps pequenos (2-3×), Direct e KD têm performance similar
✅ Para compression alta (≥7×), HPM-KD oferece ganhos significativos
```

---

## 🔧 Troubleshooting

### **Problema 1: Out of Memory**
```python
# Editar batch_size no script:
# Linha ~280: batch_size=128 → batch_size=64
```

### **Problema 2: Colab Desconecta**
```python
# O script salva checkpoints automaticamente
# Pode retomar mais tarde (teacher já estará treinado)

# Para sessões longas, use Colab Pro (24h ao invés de 12h)
```

### **Problema 3: GPU Não Detectada**
```python
import torch
print(torch.cuda.is_available())  # Deve ser True

# Se False: Runtime → Change runtime type → GPU
```

### **Problema 4: Download Lento do Dataset**
```python
# CIFAR10 é pequeno (~170MB), deve baixar rápido
# Se falhar, tente novamente ou use espelho:
# datasets.CIFAR10(root='./data', train=True, download=True)
```

---

## 📦 Dependências (Auto-instaladas)

O script verifica e usa:
- ✅ PyTorch (com CUDA)
- ✅ torchvision
- ✅ numpy, pandas
- ✅ matplotlib, seaborn
- ✅ tqdm (progress bars)
- ✅ scipy (para estatísticas futuras)

**Não precisa instalar nada manualmente!**

---

## ⏱️ Estimativas de Tempo

### **Quick Mode:**
| Compression | Teacher | Direct | TradKD | HPM-KD | Total |
|-------------|---------|--------|--------|--------|-------|
| 2.3× | 30 min | 20 min | 20 min | 20 min | ~1.5h |
| 5× | (reusa) | 15 min | 15 min | 15 min | ~45 min |
| 7× | (reusa) | 12 min | 12 min | 12 min | ~35 min |
| **TOTAL** | **30 min** | **47 min** | **47 min** | **47 min** | **~2.5h** |

### **Full Mode:**
| Compression | Teacher | Direct | TradKD | HPM-KD | Total |
|-------------|---------|--------|--------|--------|-------|
| 2.3× | 1h | 40 min | 40 min | 40 min | ~3h |
| 5× | (reusa) | 30 min | 30 min | 30 min | ~1.5h |
| 7× | (reusa) | 25 min | 25 min | 25 min | ~1.3h |
| **TOTAL** | **1h** | **1.5h** | **1.5h** | **1.5h** | **~5.5h** |

*Tempos para GPU Tesla T4. V100/A100 serão ~40% mais rápidos.*

---

## 💾 Download de Resultados

```python
# Compactar resultados
!zip -r exp1b_results.zip exp1b_results_*/

# Download (se não estiver usando Drive)
from google.colab import files
files.download('exp1b_results.zip')
```

---

## 📚 Argumentos Disponíveis

```bash
--mode {quick,full}           # Modo de execução (default: quick)
--dataset {CIFAR10,CIFAR100}  # Dataset (default: CIFAR10)
--compression {all,2.3x,5x,7x} # Compression específico (default: all)
--output PATH                 # Diretório de saída (default: auto)
--gpu {0,1,...}               # GPU ID (default: 0)
```

### **Exemplos:**
```bash
# Todos os compressions, quick mode
!python run_exp1b_colab.py --mode quick

# Apenas 5×, full mode
!python run_exp1b_colab.py --mode full --compression 5x

# CIFAR100, quick mode
!python run_exp1b_colab.py --mode quick --dataset CIFAR100

# Salvar em local específico
!python run_exp1b_colab.py --mode quick --output /content/drive/MyDrive/results
```

---

## 🎯 Checklist

### **Antes de Executar:**
- [ ] GPU configurada (Runtime → GPU)
- [ ] Script upload ou código colado
- [ ] Tem 2-3h disponíveis (quick) ou 8-10h (full)
- [ ] (Opcional) Google Drive montado

### **Durante Execução:**
- [ ] Monitor progresso: `!tail -f experiment.log`
- [ ] Verificar GPU: `!nvidia-smi`
- [ ] Não fechar aba do Colab

### **Após Execução:**
- [ ] Ver relatório: `experiment_report.md`
- [ ] Analisar figura: `accuracy_vs_compression.png`
- [ ] Verificar CSV: `results_compression_ratios.csv`
- [ ] Download ou copiar para Drive

---

## ✅ Script Pronto!

**O script é autocontido e não precisa de arquivos externos!**

Basta fazer upload e executar:
```bash
!python run_exp1b_colab.py --mode quick --dataset CIFAR10
```

**Boa sorte! 🚀**

---

## 📞 Suporte

Se encontrar problemas:
1. Verificar GPU: `!nvidia-smi`
2. Verificar PyTorch CUDA: `import torch; print(torch.cuda.is_available())`
3. Ver log de erros: `!tail -100 experiment.log`
4. Reduzir batch_size se OOM

---

**Criado em:** Dezembro 2025
**Versão:** 1.0
**Status:** ✅ Pronto para uso
