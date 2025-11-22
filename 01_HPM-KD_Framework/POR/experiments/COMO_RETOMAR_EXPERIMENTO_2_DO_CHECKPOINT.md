# 🔄 Como Retomar o Experimento 2 do Checkpoint no Google Colab

## 📋 Situação Atual

Você tem o modelo do professor (teacher) treinado e salvo no Google Drive:
```
/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/models/teacher_CIFAR10.pt
```

O script de Ablation Studies tem **suporte completo a checkpoints**, então você pode continuar de onde parou sem precisar retreinar o modelo do professor!

---

## 🚀 Passos para Retomar no Google Colab

### Passo 1: Verificar Estrutura no Drive

Primeiro, execute no Colab para ver o que você tem:

```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navegar para a pasta de resultados
import os
os.chdir('/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full')

# Verificar estrutura
!ls -lh
!ls -lh models/
!ls -lh logs/ 2>/dev/null || echo "Pasta logs não existe"
!ls -lh figures/ 2>/dev/null || echo "Pasta figures não existe"
```

**O que esperar:**
- `models/teacher_CIFAR10.pt` ✅ (você já tem)
- `logs/` - pode conter logs parciais
- `figures/` - pode estar vazio se não completou nenhum experimento
- Arquivos CSV dos experimentos (se algum foi concluído):
  - `exp05_component_ablation.csv`
  - `exp06_component_interactions.csv`
  - `exp07_hyperparameter_sensitivity.csv`
  - `exp08_progressive_chain.csv`
  - `exp09_num_teachers.csv`

---

### Passo 2: Clonar/Atualizar Repositório

```bash
# Se ainda não clonou, clone o repositório
!git clone https://github.com/seu-usuario/papers-deepbridge.git /content/papers-deepbridge

# OU, se já clonou, atualize:
!cd /content/papers-deepbridge && git pull
```

---

### Passo 3: Instalar Dependências

```bash
# Instalar PyTorch com CUDA
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar bibliotecas necessárias
!pip install matplotlib seaborn pandas numpy scipy tqdm
```

**Nota:** Você **NÃO precisa** instalar DeepBridge para este experimento (ele usa PyTorch puro).

---

### Passo 4: Continuar Experimento (Modo Automático)

O script detecta automaticamente o checkpoint do teacher e **não retreina**:

```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full"
```

**O que acontece:**
1. ✅ Script detecta `teacher_CIFAR10.pt` e **carrega** (não retreina!)
2. ✅ Continua executando os experimentos 5, 6, 7, 8, 9
3. ✅ Salva resultados incrementalmente no Drive

---

## 🔍 Como Monitorar o Progresso

### Opção A: Ver Logs em Tempo Real

Em uma **nova célula** do Colab (enquanto o experimento roda):

```python
import time

# Loop para monitorar progresso
while True:
    !clear
    print("=" * 80)
    print("PROGRESSO DO EXPERIMENTO 2")
    print("=" * 80)

    # Ver últimos arquivos modificados
    print("\n📁 ARQUIVOS RECENTES:")
    !ls -lth /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/ | head -10

    print("\n📊 ARQUIVOS CSV GERADOS:")
    !ls -1 /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/*.csv 2>/dev/null || echo "  Nenhum CSV gerado ainda"

    print("\n🖼️ FIGURAS GERADAS:")
    !ls -1 /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/figures/*.png 2>/dev/null || echo "  Nenhuma figura gerada ainda"

    print("\n🔥 USO DA GPU:")
    !nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

    time.sleep(30)  # Atualiza a cada 30 segundos
```

### Opção B: Verificar Logs Salvos

```python
# Ver log completo (se existir)
!tail -50 /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/logs/*.log 2>/dev/null || echo "Nenhum log encontrado"
```

---

## 📊 Estrutura de Saída Esperada (Completa)

Ao final, você terá:

```
exp02_ablation_full/
├── models/
│   ├── teacher_CIFAR10.pt                      ✅ (você já tem)
│   └── student_CIFAR10_*.pt                     (modelos dos experimentos)
│
├── exp05_component_ablation.csv                 (Experimento 5)
├── exp06_component_interactions.csv             (Experimento 6)
├── exp07_hyperparameter_sensitivity.csv         (Experimento 7)
├── exp08_progressive_chain.csv                  (Experimento 8)
├── exp09_num_teachers.csv                       (Experimento 9)
│
├── figures/
│   ├── component_ablation.png
│   ├── component_interactions_heatmap.png
│   ├── hyperparameter_sensitivity.png
│   ├── chain_length_analysis.png
│   └── num_teachers_saturation.png
│
└── logs/
    └── experiment_2_YYYYMMDD_HHMMSS.log
```

---

## ⏱️ Tempo Estimado Restante

Como você já tem o **teacher treinado**, o tempo restante é:

| Modo | Tempo Original | Tempo Restante (sem teacher) |
|------|----------------|------------------------------|
| Quick | ~1 hora | ~45 minutos |
| Full (CIFAR10) | ~2 horas | **~1.5 horas** |

**Razão:** O treinamento do teacher demora ~30-45 minutos, e você já o tem!

---

## ✅ Checklist Antes de Rodar

- [ ] Google Drive montado
- [ ] Checkpoint `teacher_CIFAR10.pt` existe e é válido
- [ ] GPU disponível no Colab (`!nvidia-smi`)
- [ ] PyTorch instalado (`!python3 -c "import torch; print(torch.__version__)"`)
- [ ] Repositório clonado em `/content/papers-deepbridge`
- [ ] Caminho de output correto: `/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full`

---

## 🛠️ Troubleshooting

### Problema 1: "Checkpoint corrupted"

Se o script relatar que o checkpoint está corrompido:

```python
# Verificar checkpoint manualmente
import torch

checkpoint_path = '/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/models/teacher_CIFAR10.pt'

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("✅ Checkpoint válido!")
    print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"   Train time: {checkpoint['train_time']:.2f}s")
    print(f"   Timestamp: {checkpoint['timestamp']}")
except Exception as e:
    print(f"❌ Erro ao carregar checkpoint: {e}")
```

**Solução:** Se corrompido, delete o arquivo e retreine:
```bash
!rm /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/models/teacher_CIFAR10.pt
```

### Problema 2: Google Drive Desconectou Durante Execução

```python
# Forçar reconexão
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problema 3: "CUDA out of memory"

Se a GPU ficar sem memória:

```python
# Limpar cache da GPU
import torch
torch.cuda.empty_cache()
```

Ou reduza o batch size editando o script:
```python
# No arquivo 02_ablation_studies.py, linha ~1085
'batch_size': 128,  # em vez de 256
```

### Problema 4: Colab Desconectou (Sessão Expirou)

**Não tem problema!** O checkpoint do teacher está salvo no Drive.

Basta:
1. Remontar o Drive
2. Reexecutar o comando do **Passo 4**
3. O script detecta o checkpoint e continua

---

## 🎯 Comando Final Copy-Paste

```bash
# COMANDO COMPLETO PARA RETOMAR
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full"
```

---

## 📈 Próximos Passos Após Conclusão

1. **Verificar Resultados:**
   ```bash
   !ls -lh /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/*.csv
   ```

2. **Ver Figuras Geradas:**
   ```python
   from IPython.display import Image, display
   import os

   figures_dir = '/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/figures'
   for img in os.listdir(figures_dir):
       if img.endswith('.png'):
           print(f"\n{'='*60}\n{img}\n{'='*60}")
           display(Image(filename=os.path.join(figures_dir, img)))
   ```

3. **Analisar Resultados CSV:**
   ```python
   import pandas as pd

   # Experimento 5: Component Ablation
   df5 = pd.read_csv('/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/exp05_component_ablation.csv')
   print("EXPERIMENTO 5: Component Ablation")
   print(df5)

   # Experimento 6: Interactions
   df6 = pd.read_csv('/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/exp06_component_interactions.csv')
   print("\nEXPERIMENTO 6: Component Interactions")
   print(df6)
   ```

---

## 🎉 Você Está Pronto!

Execute o comando do **Passo 4** e acompanhe o progresso. O script é inteligente e detecta automaticamente o que já foi feito.

**Boa sorte!** 🚀

---

*Criado em: 17 de Novembro de 2025*
*Última atualização: 17 de Novembro de 2025*
