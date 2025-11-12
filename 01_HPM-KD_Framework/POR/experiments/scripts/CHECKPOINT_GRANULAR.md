# 🔄 Sistema de Checkpoint Granular

## Problema Resolvido

Antes, se o Colab desconectasse **durante** um experimento, você perdia TODO o progresso daquele experimento, mesmo que já tivesse treinado vários modelos.

## Solução Implementada

Agora o sistema salva **cada modelo individualmente** assim que termina de treinar, permitindo retomar **exatamente de onde parou**, mesmo dentro de um experimento.

---

## Como Funciona

### Estrutura de Checkpoints

Cada modelo é salvo com metadata completa:

```
models/
├── teacher_CIFAR10.pt              # Teacher (salvo após treinar)
├── student_CIFAR10_Direct_run1.pt   # Direct, run 1
├── student_CIFAR10_Direct_run2.pt   # Direct, run 2
├── student_CIFAR10_Direct_run3.pt   # Direct, run 3
├── student_CIFAR10_TraditionalKD_run1.pt
├── student_CIFAR10_TraditionalKD_run2.pt
├── student_CIFAR10_FitNets_run1.pt
...
└── student_CIFAR10_HPM-KD_run5.pt  # HPM-KD, run 5
```

### Conteúdo do Checkpoint

Cada arquivo `.pt` contém:
- `model_state_dict`: Pesos do modelo
- `accuracy`: Acurácia alcançada
- `train_time`: Tempo de treinamento
- `timestamp`: Quando foi treinado
- `metadata`: Dataset, baseline, run, epochs, etc.

### Fluxo de Execução

```python
# Para cada modelo:
1. Verifica se checkpoint existe
   └─ SIM: Carrega modelo salvo (⏭️ pula treinamento)
   └─ NÃO: Treina modelo → Salva checkpoint

2. Continua para próximo modelo
```

---

## Exemplo Prático

### Cenário: Experimento 1 com CIFAR10 (Mode Full)

```
Estrutura do experimento:
├─ Teacher (1 modelo, 50 epochs)
└─ 6 Baselines × 5 runs = 30 modelos students (30 epochs cada)
   ├─ Direct (5 runs)
   ├─ TraditionalKD (5 runs)
   ├─ FitNets (5 runs)
   ├─ AT (5 runs)
   ├─ TAKD (5 runs)
   └─ HPM-KD (5 runs)

Tempo estimado: ~4 horas
```

### Execução Original (Primeira Vez)

```bash
# SESSÃO 1: Início
!python RUN_COLAB.py --full --dataset CIFAR10

# Output:
✅ Granular checkpointing enabled (resume-friendly)

Dataset: CIFAR10
Training Teacher...
  Teacher: 79.37% in 1013.7s
💾 Checkpoint saved: teacher_CIFAR10.pt (acc=79.37%)

Testing Direct...
  Run 1/5...
    69.12% in 611.6s
  💾 Checkpoint saved: student_CIFAR10_Direct_run1.pt (acc=69.12%)

  Run 2/5...
    68.48% in 610.7s
  💾 Checkpoint saved: student_CIFAR10_Direct_run2.pt (acc=68.48%)

  Run 3/5...
    # ... Colab desconecta aqui! ❌
```

**Progresso antes da desconexão:**
- ✅ Teacher CIFAR10: treinado e salvo
- ✅ Direct Run 1: treinado e salvo
- ✅ Direct Run 2: treinado e salvo
- ❌ Direct Run 3: perdido (estava treinando)

### Retomando Após Desconexão

```bash
# SESSÃO 2: Reconectar e retomar
!python RUN_COLAB.py --resume

# Output:
✅ Granular checkpointing enabled (resume-friendly)

Dataset: CIFAR10
⏭️ Teacher checkpoint found - loading...
✅ Loaded checkpoint: teacher_CIFAR10.pt (acc=79.37%)
  Teacher: 79.37% in 1013.7s  # ← Carregado instantaneamente!

Testing Direct...
  Run 1/5...
  ⏭️ Checkpoint found - loading...
  ✅ Loaded checkpoint: student_CIFAR10_Direct_run1.pt (acc=69.12%)
    69.12% in 611.6s  # ← Carregado instantaneamente!

  Run 2/5...
  ⏭️ Checkpoint found - loading...
  ✅ Loaded checkpoint: student_CIFAR10_Direct_run2.pt (acc=68.48%)
    68.48% in 610.7s  # ← Carregado instantaneamente!

  Run 3/5...
    # Nenhum checkpoint → Treina do zero
    67.92% in 615.3s
  💾 Checkpoint saved: student_CIFAR10_Direct_run3.pt (acc=67.92%)

  Run 4/5...
    68.75% in 608.9s
  💾 Checkpoint saved: student_CIFAR10_Direct_run4.pt (acc=68.75%)

  Run 5/5...
    69.23% in 612.1s
  💾 Checkpoint saved: student_CIFAR10_Direct_run5.pt (acc=69.23%)

Testing TraditionalKD...
  # ... continua normalmente ...
```

**Economia de tempo:**
- Teacher: ~17 minutos economizados
- Direct Run 1 e 2: ~20 minutos economizados
- **Total economizado: ~37 minutos!**

---

## Vantagens do Checkpoint Granular

### ✅ **Zero Perda de Progresso**
- Cada modelo salvo individualmente
- Desconexões não perdem trabalho já concluído

### ✅ **Retomada Inteligente**
- Detecta automaticamente modelos já treinados
- Pula treinamentos já concluídos
- Continua exatamente de onde parou

### ✅ **Economia de Tempo Massiva**
- Não retreina modelos que já existem
- Carregamento instantâneo (~1s vs ~10-20min de treinamento)

### ✅ **Debugging Facilitado**
- Pode inspecionar cada modelo salvo
- Fácil identificar onde algo deu errado

### ✅ **Reprodutibilidade**
- Todos os modelos salvos com metadata completa
- Pode recriar experimentos exatos

---

## Implementação Técnica

### Funções Helper (Código)

```python
# 1. Gerar caminho do checkpoint
get_model_checkpoint_path(output_dir, dataset, model_type, baseline, run)
# → models/teacher_CIFAR10.pt
# → models/student_CIFAR10_HPM-KD_run3.pt

# 2. Verificar se checkpoint existe (e é válido)
model_checkpoint_exists(checkpoint_path)
# → True/False

# 3. Salvar checkpoint
save_model_checkpoint(model, checkpoint_path, accuracy, train_time, metadata)
# → Salva atomicamente (evita corrupção)

# 4. Carregar checkpoint
model, accuracy, train_time = load_model_checkpoint(model, checkpoint_path)
# → Carrega modelo + metadata
```

### Fluxo no Código

```python
# Para Teacher
teacher_checkpoint_path = get_model_checkpoint_path(output_dir, dataset, 'teacher')

if model_checkpoint_exists(teacher_checkpoint_path):
    # Carrega do checkpoint
    teacher, acc, time = load_model_checkpoint(teacher, teacher_checkpoint_path)
else:
    # Treina do zero
    teacher, acc, time = train_teacher(...)
    # Salva checkpoint
    save_model_checkpoint(teacher, teacher_checkpoint_path, acc, time)

# Para cada Student
for run in range(n_runs):
    student_checkpoint_path = get_model_checkpoint_path(
        output_dir, dataset, 'student', baseline, run+1
    )

    if model_checkpoint_exists(student_checkpoint_path):
        # Carrega do checkpoint
        student, acc, time = load_model_checkpoint(student, student_checkpoint_path)
    else:
        # Treina do zero
        student, acc, time = train_baseline(...)
        # Salva checkpoint
        save_model_checkpoint(student, student_checkpoint_path, acc, time)
```

---

## Status de Implementação

### ✅ Totalmente Implementado

- [x] **01_compression_efficiency.py** - Checkpoint granular completo
  - Teacher checkpointing
  - Student checkpointing (6 baselines × 5 runs = 30 checkpoints)
  - Detecção automática e skip de modelos já treinados
  - **Mais crítico** - treina 30+ modelos!

- [x] **02_ablation_studies.py** - Checkpoint granular
  - Teacher checkpointing
  - Student checkpointing para Experimento 5 (Component Ablation)
  - Funções helper disponíveis para Experimentos 6-9
  - Estrutura pronta para expandir checkpointing

- [x] **03_generalization.py** - Checkpoint granular básico
  - Funções helper de checkpointing disponíveis
  - Teacher checkpointing implementado
  - Pronto para adicionar student checkpointing conforme necessário

- [x] **04_computational_efficiency.py** - Checkpoint granular básico
  - Funções helper de checkpointing disponíveis
  - Teacher checkpointing implementado
  - Pronto para adicionar student checkpointing conforme necessário

**Todos os 4 experimentos** agora têm suporte básico para checkpoint granular!

---

## Testando Localmente

```bash
# 1. Começar experimento
cd /path/to/scripts
python 01_compression_efficiency.py --mode quick --datasets MNIST --output /tmp/test

# 2. Cancelar no meio (Ctrl+C)
# ... cancele após alguns modelos serem salvos ...

# 3. Verificar checkpoints
ls -lh /tmp/test/models/
# → teacher_MNIST.pt
# → student_MNIST_Direct_run1.pt
# → student_MNIST_Direct_run2.pt

# 4. Retomar
python 01_compression_efficiency.py --mode quick --datasets MNIST --output /tmp/test

# Output deve mostrar:
# ⏭️ Teacher checkpoint found - loading...
# ⏭️ Checkpoint found - loading...
# (pula modelos já treinados e continua os pendentes)
```

---

## Monitoramento

### Ver Checkpoints Salvos

```bash
# No Colab
!ls -lh /content/drive/MyDrive/HPM-KD_Results/results_full_*/models/

# Output:
# teacher_CIFAR10.pt              45.2 MB
# student_CIFAR10_Direct_run1.pt   11.3 MB
# student_CIFAR10_Direct_run2.pt   11.3 MB
# ...
```

### Inspecionar Checkpoint

```python
import torch

# Carregar checkpoint
ckpt = torch.load('/path/to/student_CIFAR10_Direct_run1.pt')

print(f"Accuracy: {ckpt['accuracy']:.2f}%")
print(f"Train time: {ckpt['train_time']:.1f}s")
print(f"Timestamp: {ckpt['timestamp']}")
print(f"Metadata: {ckpt['metadata']}")

# Output:
# Accuracy: 69.12%
# Train time: 611.6s
# Timestamp: 2025-01-12T00:23:04
# Metadata: {'dataset': 'CIFAR10', 'baseline': 'Direct', 'run': 1, 'epochs': 30}
```

---

## FAQ

### P: E se eu quiser retreinar um modelo específico?

**R:** Apenas delete o checkpoint desse modelo:

```bash
# Retreinar Direct run 3
!rm /content/drive/MyDrive/.../models/student_CIFAR10_Direct_run3.pt

# Retomar experimento - vai retreinar apenas esse modelo
!python RUN_COLAB.py --resume
```

### P: E se o checkpoint estiver corrompido?

**R:** O sistema detecta automaticamente e retreina:

```python
def model_checkpoint_exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        return 'model_state_dict' in checkpoint
    except:
        logger.warning("Checkpoint corrupted - will retrain")
        return False
```

### P: Quanto espaço os checkpoints ocupam?

**R:** Varia por modelo:
- Teacher (LeNet5-based): ~45 MB
- Student (smaller): ~11 MB cada

Para Experimento 1 Full (1 teacher + 30 students):
- Total: ~45 MB + (30 × 11 MB) = **~375 MB**

### P: Posso mover os checkpoints para outro lugar?

**R:** Sim, mas você precisa especificar o caminho com `--output`:

```bash
# Mover checkpoints
!mv /content/drive/.../results_full_20251111/ /content/drive/Backup/

# Retomar apontando para novo local
!python RUN_COLAB.py --resume --output /content/drive/Backup/results_full_20251111/
```

---

## Conclusão

O sistema de checkpoint granular transforma experimentos longos e frágeis em processos **robustos e resilientes**. Agora você pode:

- ✅ Rodar experimentos de 4+ horas sem medo de perder progresso
- ✅ Desconectar/reconectar o Colab quantas vezes quiser
- ✅ Economizar horas de reprocessamento
- ✅ Debugar problemas mais facilmente

**Apenas use `--resume` e o sistema cuida do resto!** 🎉

---

**Última atualização:** 2025-01-12
