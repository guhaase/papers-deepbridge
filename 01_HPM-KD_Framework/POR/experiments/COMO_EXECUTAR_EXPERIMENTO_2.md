# 🚀 Como Executar o Experimento 2: Ablation Studies (RQ2)

## 📋 Sobre o Experimento 2

**Research Question (RQ2):** Qual a contribuição individual de cada componente do HPM-KD e como eles interagem?

### Experimentos Incluídos:

1. **Component Ablation (Exp 5)** - Impacto individual de cada componente
2. **Component Interactions (Exp 6)** - Sinergias entre componentes
3. **Hyperparameter Sensitivity (Exp 7)** - Sensibilidade a T e α
4. **Progressive Chain Length (Exp 8)** - Número ótimo de passos intermediários
5. **Number of Teachers (Exp 9)** - Saturação com múltiplos teachers

### Componentes HPM-KD Testados:

- ✅ **ProgChain**: Progressive chaining de modelos intermediários
- ✅ **AdaptConf**: Adaptive confidence weighting
- ✅ **MultiTeach**: Multi-teacher ensemble
- ✅ **MetaTemp**: Meta-learned temperature
- ✅ **Parallel**: Parallel distillation paths
- ✅ **Memory**: Memory-augmented distillation

---

## ⏱️ Tempo Estimado

- **Quick Mode:** ~1 hora
- **Full Mode:** ~2 horas

---

## 🖥️ Opções de Execução

### Opção 1: Execução Local (WSL/Linux) ⚠️ Requer GPU

Se você tem GPU local NVIDIA:

```bash
cd /home/guhaase/projetos/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments/scripts

# Quick mode (teste rápido)
python3 02_ablation_studies.py \
    --mode quick \
    --dataset MNIST \
    --gpu 0 \
    --output ../results/exp02_ablation_quick

# Full mode (completo)
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output ../results/exp02_ablation_full
```

### Opção 2: Google Colab (Recomendado) ✅

Execute no Google Colab com GPU gratuita:

#### Passo 1: Montar Drive e Clonar Repo

```python
from google.colab import drive
import os

# Montar Google Drive (se ainda não montou)
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# Clonar repositório (se ainda não clonou)
if not os.path.exists('/content/papers-deepbridge'):
    !git clone https://github.com/seu-usuario/papers-deepbridge.git /content/papers-deepbridge
```

#### Passo 2: Instalar Dependências

```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install deepbridge
!pip install matplotlib seaborn pandas numpy scipy tqdm
```

#### Passo 3: Executar Experimento

**Quick Mode (Teste):**
```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 02_ablation_studies.py \
    --mode quick \
    --dataset MNIST \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_quick"
```

**Full Mode (Completo para o Paper):**
```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full"
```

---

## 📊 Parâmetros Disponíveis

| Parâmetro | Opções | Padrão | Descrição |
|-----------|--------|--------|-----------|
| `--mode` | `quick`, `full` | `quick` | Modo de execução |
| `--dataset` | `MNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100` | `MNIST` | Dataset a usar |
| `--gpu` | `0`, `1`, etc. | `0` | ID da GPU |
| `--output` | caminho | `./results/exp02_ablation` | Diretório de saída |

### Exemplos de Uso:

**Teste rápido com MNIST:**
```bash
python3 02_ablation_studies.py --mode quick --dataset MNIST --gpu 0
```

**Experimento completo com CIFAR10:**
```bash
python3 02_ablation_studies.py --mode full --dataset CIFAR10 --gpu 0 \
    --output "/caminho/para/resultados"
```

**Experimento completo com CIFAR100 (mais complexo):**
```bash
python3 02_ablation_studies.py --mode full --dataset CIFAR100 --gpu 0 \
    --output "/caminho/para/resultados"
```

---

## 🎯 Diferenças entre Quick e Full Mode

### Quick Mode (Teste - ~1h):
- Menos épocas de treinamento
- Menos repetições (runs)
- Subset menor do dataset
- Útil para **testar o pipeline**

### Full Mode (Paper - ~2h):
- Épocas completas (30-50)
- 5 repetições por configuração
- Dataset completo
- Resultados **publicáveis**

---

## 📁 Estrutura de Saída Esperada

```
exp02_ablation_full/
├── ablation_results.csv              # Resultados tabulares
├── experiment_report.md              # Relatório completo
├── figures/
│   ├── component_ablation.png        # Impacto individual
│   ├── component_interactions.png    # Sinergias
│   ├── hyperparameter_sensitivity.png
│   ├── chain_length_analysis.png
│   └── num_teachers_saturation.png
├── results/
│   ├── exp5_component_ablation.json
│   ├── exp6_interactions.json
│   ├── exp7_hyperparams.json
│   ├── exp8_chain_length.json
│   └── exp9_num_teachers.json
└── models/
    └── checkpoints/                  # Modelos salvos
```

---

## 🔍 Como Monitorar a Execução

### Opção A: Ver progresso em tempo real (Colab)

Em uma **nova célula** do Colab:

```python
# Ver logs em tempo real
!tail -f /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/*.log

# Ver últimos arquivos criados
!ls -lth /content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full/ | head -10

# Ver uso da GPU
!nvidia-smi
```

### Opção B: Script de monitoramento

```bash
#!/bin/bash
# monitor_exp2.sh

watch -n 10 '
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

echo -e "\n=== Latest Files ==="
ls -lth /caminho/resultados/ | head -5
'
```

---

## ⚠️ Troubleshooting

### Problema: "No module named 'torch'"

**Solução:** Instalar PyTorch
```bash
pip install torch torchvision
```

### Problema: "CUDA out of memory"

**Solução:** Reduzir batch size no script ou usar dataset menor
```bash
# Use MNIST em vez de CIFAR100
--dataset MNIST
```

### Problema: Script trava durante execução

**Solução:** Verificar se há múltiplos processos rodando
```bash
# Matar processos duplicados
killall -9 python3

# Reiniciar
python3 02_ablation_studies.py ...
```

### Problema: Google Drive desconectou

**Solução:** Remontar Drive
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## 🎯 Recomendação para o Paper

### Para publicação, execute com:

```bash
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_CIFAR10_full"
```

**Justificativa:**
- CIFAR10 é padrão na literatura
- Full mode garante resultados estatisticamente significativos
- Consistente com Experimento 1

---

## 📊 O que Esperar nos Resultados

### Experimento 5: Component Ablation

**Tabela esperada:**

| Config | Components | Accuracy | Improvement |
|--------|-----------|----------|-------------|
| Full | All 6 | 67.74% | baseline |
| -ProgChain | 5 components | 67.2% | -0.54% |
| -AdaptConf | 5 components | 67.5% | -0.24% |
| ... | ... | ... | ... |

**Insight:** Identifica qual componente é MAIS importante.

### Experimento 6: Component Interactions

**Gráfico esperado:** Heatmap de sinergias entre componentes.

### Experimento 7: Hyperparameter Sensitivity

**Gráfico esperado:** Curvas de acurácia vs Temperature (T) e Alpha (α).

### Experimento 8: Chain Length

**Gráfico esperado:** Acurácia vs Número de Intermediate Models (0, 1, 2, 3).

### Experimento 9: Number of Teachers

**Gráfico esperado:** Acurácia vs Número de Teachers (1-5) - mostra saturação.

---

## ✅ Checklist Antes de Executar

- [ ] GPU disponível (`nvidia-smi`)
- [ ] PyTorch instalado (`python3 -c "import torch; print(torch.__version__)"`)
- [ ] DeepBridge instalado (`python3 -c "import deepbridge"`)
- [ ] Google Drive montado (se usando Colab)
- [ ] Espaço em disco suficiente (~2GB para modelos + resultados)
- [ ] Definido modo (`quick` para teste, `full` para paper)
- [ ] Definido dataset (CIFAR10 recomendado)

---

## 🚀 Comando Final (Copy-Paste)

**Google Colab - Full Mode - CIFAR10:**

```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 02_ablation_studies.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full_$(date +%Y%m%d_%H%M%S)"
```

O sufixo `$(date ...)` cria uma pasta única com timestamp para evitar sobrescrever resultados.

---

**Boa execução!** 🎉

Após concluir, execute a análise dos resultados com:
```bash
python3 analyze_experiment_2.py --input /caminho/para/resultados
```

---

*Criado em: 15 de Novembro de 2025*
