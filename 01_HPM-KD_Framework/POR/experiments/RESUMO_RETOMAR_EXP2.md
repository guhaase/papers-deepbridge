# 🚀 RESUMO: Como Retomar o Experimento 2

## ⚡ Método Rápido (Copy-Paste no Google Colab)

### 1️⃣ Montar Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2️⃣ Clonar Repositório (se ainda não clonou)
```bash
!git clone https://github.com/seu-usuario/papers-deepbridge.git /content/papers-deepbridge
```

### 3️⃣ Instalar Dependências
```bash
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q matplotlib seaborn pandas numpy scipy tqdm
```

### 4️⃣ Retomar Experimento (COMANDO PRINCIPAL)
```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full"
```

**Pronto!** O script detecta automaticamente o checkpoint `teacher_CIFAR10.pt` e continua de onde parou.

---

## 📊 Verificar Status Antes de Rodar

```bash
# Ver o que você já tem
!ls -lh /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/models/
!ls -lh /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/*.csv
```

**Se você vir:**
- ✅ `teacher_CIFAR10.pt` → Não precisa retreinar o teacher (~45 min economizados!)
- ✅ `exp05_*.csv` → Experimento 5 já completado
- ✅ `exp06_*.csv` → Experimento 6 já completado
- etc.

**O script pula automaticamente o que já foi feito!**

---

## 🔍 Monitorar Progresso (Executar em Paralelo)

Em outra célula do Colab:

```python
import time
from IPython.display import clear_output

while True:
    clear_output(wait=True)
    print("📊 PROGRESSO DO EXPERIMENTO 2")
    print("=" * 60)

    # CSVs gerados
    !ls -1 /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/*.csv 2>/dev/null || echo "Nenhum CSV ainda"

    # Figuras
    !ls -1 /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/figures/*.png 2>/dev/null || echo "Nenhuma figura ainda"

    # GPU
    !nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

    time.sleep(30)
```

---

## ⏱️ Tempo Estimado

| Situação | Tempo |
|----------|-------|
| Sem checkpoint do teacher | ~2 horas |
| **Com checkpoint do teacher** ✅ | **~1.5 horas** |
| Quick mode | ~45 minutos |

---

## 📁 Arquivos Criados

Para saber mais detalhes, consulte:

1. **Guia Completo:** `COMO_RETOMAR_EXPERIMENTO_2_DO_CHECKPOINT.md`
2. **Notebook Interativo:** `RETOMAR_EXPERIMENTO_2_COLAB.ipynb` (abra no Colab)
3. **Script de Verificação:** `scripts/check_and_resume_exp2.py`

---

## ❓ FAQ Rápido

**P: O script vai sobrescrever meus resultados?**
R: Não! Ele detecta o que já foi feito e pula.

**P: E se o Colab desconectar?**
R: Sem problemas! O checkpoint está salvo no Drive. Basta remontar e executar novamente.

**P: Como sei se deu certo?**
R: Você verá 5 arquivos CSV no final:
- `exp05_component_ablation.csv`
- `exp06_component_interactions.csv`
- `exp07_hyperparameter_sensitivity.csv`
- `exp08_progressive_chain.csv`
- `exp09_num_teachers.csv`

**P: Posso usar outro dataset?**
R: Sim! Troque `--dataset CIFAR10` por `MNIST`, `FashionMNIST` ou `CIFAR100`.

---

## ✅ Checklist Antes de Rodar

- [ ] Google Drive montado
- [ ] GPU ativada no Colab (Runtime > Change runtime type > GPU)
- [ ] Repositório clonado
- [ ] PyTorch instalado
- [ ] Checkpoint `teacher_CIFAR10.pt` existe (ou aceita retreinar)

---

**Criado em:** 17 de Novembro de 2025
