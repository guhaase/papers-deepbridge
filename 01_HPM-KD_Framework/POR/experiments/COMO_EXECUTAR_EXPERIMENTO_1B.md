# 🚀 Como Executar o Experimento 1B: Compression Ratios Maiores

## ⚠️ EXPERIMENTO CRÍTICO

Este é o experimento **MAIS IMPORTANTE** para validar a RQ1 adequadamente.

### 🎯 Por que este experimento é crítico?

O Experimento 1 mostrou que com compression ratio pequeno (2×), **Direct training superou todos os métodos de KD**, incluindo HPM-KD.

**Hipótese deste experimento:**
Com compression ratios maiores (5×, 7×), HPM-KD deve demonstrar vantagem clara sobre Direct.

---

## 📊 O que será testado?

### Compression Ratios:

| Ratio | Teacher | Student | Params Teacher | Params Student |
|-------|---------|---------|----------------|----------------|
| **2.3×** | ResNet50 | ResNet18 | 25.5M | 11.1M |
| **5×** | ResNet50 | ResNet10 | 25.5M | 5.0M |
| **7×** | ResNet50 | MobileNetV2 | 25.5M | 3.5M |

### Baselines:
- ✅ **Direct**: Train student from scratch
- ✅ **TraditionalKD**: Hinton et al. (2015)
- ✅ **HPM-KD**: Our method (DeepBridge)

### Análises Incluídas:
1. **Compression Ratio Scaling** - Accuracy vs Compression
2. **Statistical Significance** - T-tests (HPM-KD vs Direct)
3. **"When does KD help?"** - Identificar threshold onde KD vence Direct

---

## ⏱️ Tempo Estimado

| Mode | CIFAR10 | CIFAR100 |
|------|---------|----------|
| **Quick** | 2-3 horas | 3-4 horas |
| **Full** | 8-10 horas | 12-15 horas |

**Recomendação:** Comece com Quick mode para testar, depois execute Full mode para o paper.

---

## 🖥️ Opções de Execução

### Opção 1: Google Colab (Recomendado) ✅

#### Passo 1: Setup Inicial

```python
from google.colab import drive
import os

# Montar Google Drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# Clonar repositório
if not os.path.exists('/content/papers-deepbridge'):
    !git clone https://github.com/seu-usuario/papers-deepbridge.git /content/papers-deepbridge

# Instalar dependências
!pip install torch torchvision
!pip install deepbridge matplotlib seaborn pandas scipy tqdm
```

#### Passo 2: Executar Experimento

**Quick Mode (Teste - 2-3h):**
```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py \
    --mode quick \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp1b_quick"
```

**Full Mode (Paper - 8-10h):**
```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp1b_full"
```

**Full Mode com CIFAR100 (Mais complexo - 12-15h):**
```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py \
    --mode full \
    --dataset CIFAR100 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp1b_cifar100_full"
```

---

### Opção 2: Execução Local (WSL/Linux)

```bash
cd /home/guhaase/projetos/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments/scripts

# Quick mode
python3 01b_compression_ratios.py \
    --mode quick \
    --dataset CIFAR10 \
    --gpu 0 \
    --output ../results/exp1b_quick

# Full mode
python3 01b_compression_ratios.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output ../results/exp1b_full
```

---

## 📋 Parâmetros Disponíveis

| Parâmetro | Opções | Padrão | Descrição |
|-----------|--------|--------|-----------|
| `--mode` | `quick`, `full` | `quick` | Modo de execução |
| `--dataset` | `CIFAR10`, `CIFAR100` | `CIFAR10` | Dataset principal |
| `--datasets` | lista | `['CIFAR10']` | Múltiplos datasets |
| `--gpu` | `0`, `1`, etc. | `0` | ID da GPU |
| `--output` | caminho | auto | Diretório de saída |
| `--compressions` | lista | all | Quais compression ratios testar |

### Exemplos Avançados:

**Testar apenas compression ratio específico:**
```bash
python3 01b_compression_ratios.py \
    --mode quick \
    --dataset CIFAR10 \
    --compressions "5x_ResNet10" \
    --gpu 0
```

**Testar múltiplos datasets:**
```bash
python3 01b_compression_ratios.py \
    --mode full \
    --datasets CIFAR10 CIFAR100 \
    --gpu 0
```

---

## 🔄 Sistema de Checkpoints (Resume-Friendly)

O experimento **salva checkpoints granulares** para cada modelo treinado. Se a execução for interrompida, basta executar novamente com os mesmos parâmetros:

```bash
# Execução original
python3 01b_compression_ratios.py --mode full --dataset CIFAR10 --output /meu/path

# Se interrompido, retomar com o MESMO comando
python3 01b_compression_ratios.py --mode full --dataset CIFAR10 --output /meu/path
```

O script automaticamente:
- ✅ Detecta modelos já treinados
- ✅ Carrega checkpoints existentes
- ✅ Continua apenas o que falta

### Estrutura de Checkpoints:

```
output_dir/
└── models/
    ├── 2.3x_ResNet18/
    │   ├── teacher_CIFAR10.pt
    │   ├── student_CIFAR10_Direct_run1.pt
    │   ├── student_CIFAR10_Direct_run2.pt
    │   ├── student_CIFAR10_TraditionalKD_run1.pt
    │   └── student_CIFAR10_HPM-KD_run1.pt
    ├── 5x_ResNet10/
    │   └── ...
    └── 7x_MobileNetV2/
        └── ...
```

---

## 📊 Saída Esperada

### Arquivos Gerados:

```
exp1b_full/
├── results_compression_ratios.csv     # Dados completos
├── statistical_tests.csv              # T-tests e p-values
├── experiment_report.md               # Relatório detalhado
├── figures/
│   ├── compression_ratio_vs_accuracy.png
│   ├── hpmkd_vs_direct.png
│   └── statistical_significance.png
└── models/
    └── [checkpoints organizados por compression ratio]
```

### Visualizações:

1. **compression_ratio_vs_accuracy.png**
   - Mostra como accuracy varia com compression ratio
   - Compara Direct, TraditionalKD, HPM-KD

2. **hpmkd_vs_direct.png**
   - Mostra quando HPM-KD supera Direct
   - Barra verde = HPM-KD vence
   - Barra vermelha = Direct vence

3. **statistical_significance.png**
   - Heatmap com p-values dos t-tests
   - Verde = diferença significativa
   - Vermelho = não significativa

---

## 🎯 Resultados Esperados

### Hipótese de Sucesso:

Se a hipótese estiver correta, você deve ver:

1. **Com 2.3× compression:**
   - Direct ≈ HPM-KD (diferença pequena ou Direct vence)

2. **Com 5× compression:**
   - HPM-KD > Direct (+1-2%)
   - p-value < 0.05 (estatisticamente significativo)

3. **Com 7× compression:**
   - HPM-KD >> Direct (+2-3% ou mais)
   - p-value < 0.01 (muito significativo)

### Para o Paper:

```latex
Our experiments show that HPM-KD demonstrates clear advantages
over direct training when the compression ratio exceeds 5×.
Specifically, with 7× compression (ResNet50 → MobileNetV2),
HPM-KD achieved X.XX% higher accuracy than direct training
(p < 0.01), demonstrating the effectiveness of knowledge
distillation in scenarios with large capacity gaps.
```

---

## 🔍 Monitoramento

### Ver progresso em tempo real (Colab):

```python
# Em uma nova célula
!tail -f /content/drive/MyDrive/HPM-KD_Results/exp1b_full/*.log

# Ver GPU usage
!nvidia-smi

# Ver últimos arquivos criados
!ls -lth /content/drive/MyDrive/HPM-KD_Results/exp1b_full/models/**/*.pt | head -10
```

### Script de monitoramento (Linux):

```bash
#!/bin/bash
watch -n 30 '
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo -e "\n=== Progress ==="
ls -lh results/exp1b_full/models/*/*.pt | wc -l
echo "models trained"
'
```

---

## ⚠️ Troubleshooting

### Problema: "CUDA out of memory"

**Solução 1:** Reduzir batch size

Edite o script e mude:
```python
'batch_size': 128,  # Mudar para 64 ou 32
```

**Solução 2:** Usar dataset menor temporariamente
```bash
--mode quick  # Usa subset menor
```

### Problema: Treinamento muito lento

**Possível causa:** CPU mode (sem GPU)

**Verificar:**
```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Solução:** Usar Google Colab com GPU ativado:
- Runtime → Change runtime type → GPU → T4

### Problema: Script trava sem mensagem

**Diagnóstico:**
```bash
# Ver se o processo está rodando
ps aux | grep 01b_compression

# Ver logs
tail -f output_dir/*.log
```

**Solução:** Pode estar treinando (normal levar tempo). Aguarde ou verifique GPU usage.

---

## ✅ Checklist Antes de Executar

- [ ] GPU disponível e funcionando (`nvidia-smi`)
- [ ] PyTorch instalado com CUDA (`python3 -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Espaço em disco suficiente (~10GB para modelos + resultados)
- [ ] Google Drive montado (se usando Colab)
- [ ] Definido modo adequado (`quick` para teste, `full` para paper)
- [ ] Tempo disponível (8-10h para full mode)

---

## 🚀 Comando Recomendado para o Paper

**Google Colab - Full Mode - CIFAR10:**

```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py \
    --mode full \
    --dataset CIFAR10 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp1b_full_$(date +%Y%m%d_%H%M%S)"
```

**Após conclusão, execute também com CIFAR100** para ter resultados em 2 datasets:

```bash
!cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \
python3 01b_compression_ratios.py \
    --mode full \
    --dataset CIFAR100 \
    --gpu 0 \
    --output "/content/drive/MyDrive/HPM-KD_Results/exp1b_cifar100_full_$(date +%Y%m%d_%H%M%S)"
```

---

## 📝 Após a Execução

1. **Revisar relatório:**
   ```bash
   cat exp1b_full/experiment_report.md
   ```

2. **Verificar significância estatística:**
   ```bash
   cat exp1b_full/statistical_tests.csv
   ```

3. **Visualizar gráficos:**
   - Abrir arquivos PNG em `exp1b_full/figures/`

4. **Incluir no paper:**
   - Usar as figuras geradas
   - Citar estatísticas do relatório
   - Adicionar discussão sobre "when does KD help?"

---

**Boa execução!** 🎉

Este experimento é **CRÍTICO** para validar a RQ1. Os resultados vão determinar se HPM-KD realmente supera Direct training em cenários reais de compressão.

---

*Criado em: 15 de Novembro de 2025*
