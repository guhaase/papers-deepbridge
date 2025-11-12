# 🔄 Guia: Sistema de Checkpoint para Google Colab

## Problema
O Google Colab desconecta após algumas horas de execução, interrompendo experimentos longos e perdendo todo o progresso.

## Solução em Dois Níveis

Implementamos um **sistema duplo de checkpointing**:

### 🔷 Nível 1: Checkpoint de Experimentos
- ✅ Salva o progresso após **cada experimento concluído** (Exp 1, 2, 3, 4)
- ✅ Permite **retomar de onde parou** entre experimentos
- ✅ Salva tudo no **Google Drive** (persistente)

### 🔷 Nível 2: Checkpoint Granular de Modelos (NOVO!)
- ✅ Salva **cada modelo** assim que termina de treinar
- ✅ Retoma **dentro** de um experimento, não perde progresso parcial
- ✅ Exemplo: Se treinou 15 de 30 modelos, retoma do 16º
- ✅ Ver detalhes em: `CHECKPOINT_GRANULAR.md`

---

## 🚀 Uso Básico

### 1️⃣ Primeira Execução

```python
# No Google Colab
!python RUN_COLAB.py
```

Isso vai:
- Montar o Google Drive automaticamente
- Criar um diretório de resultados: `/content/drive/MyDrive/HPM-KD_Results/results_quick_YYYYMMDD_HHMMSS/`
- Executar os experimentos em sequência
- **Salvar checkpoint após cada experimento**

### 2️⃣ Se o Colab Desconectar

**✨ SUPER SIMPLES: Apenas use `--resume`**

```python
# Restaura TUDO automaticamente: modo (quick/full), datasets, GPU, progresso!
!python RUN_COLAB.py --resume
```

**🎯 Não precisa repetir `--full`, `--dataset`, ou qualquer outro parâmetro!**
- O sistema detecta automaticamente o checkpoint mais recente
- Restaura o modo (quick/full) que você estava usando
- Restaura os datasets que você estava processando
- Continua exatamente de onde parou

**OPÇÃO B: Especificar Diretório Manualmente** (se tiver múltiplos checkpoints)

```python
# Use o caminho exato do diretório anterior
!python RUN_COLAB.py --resume --output /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022
```

**OPÇÃO C: Começar de um Experimento Específico**

```python
# Começar do experimento 3 em diante
!python RUN_COLAB.py --start-from 3 --output /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022
```

---

## 📋 Cenários Comuns

### Cenário 1: Colab desconectou após completar 2 experimentos

```python
# Quando você reconectar, monte o Drive novamente
from google.colab import drive
drive.mount('/content/drive')

# Retome automaticamente - vai continuar do experimento 3
!python RUN_COLAB.py --resume
```

### Cenário 2: Um experimento falhou, mas quer continuar os outros

```python
# Pule o experimento que falhou (ex: experimento 2)
!python RUN_COLAB.py --skip 2 --output /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022
```

### Cenário 3: Executar apenas experimentos específicos

```python
# Executar apenas experimentos 3 e 4
!python RUN_COLAB.py --only 3 4
```

### Cenário 4: Modo Full (8-10 horas)

```python
# Primeira vez
!python RUN_COLAB.py --full

# Se desconectar, retomar
!python RUN_COLAB.py --resume --full
```

---

## 📁 Estrutura de Arquivos

Após iniciar os experimentos, a estrutura no Google Drive será:

```
/content/drive/MyDrive/HPM-KD_Results/
└── results_quick_20250111_143022/
    ├── checkpoint.json              ← ARQUIVO DE CHECKPOINT (principal)
    ├── run_all_experiments.log      ← Log de execução
    ├── results.json                  ← Resultados finais
    ├── RELATORIO_FINAL.md           ← Relatório consolidado
    ├── exp_01_compression_efficiency/
    │   ├── results/
    │   ├── figures/
    │   ├── models/
    │   └── report.md
    ├── exp_02_ablation_studies/
    │   └── ...
    ├── exp_03_generalization/
    │   └── ...
    └── exp_04_computational_efficiency/
        └── ...
```

### Arquivo `checkpoint.json`

O checkpoint contém:
```json
{
  "timestamp": "2025-01-11T14:35:22",
  "last_completed_experiment": 2,
  "completed_experiments": [1, 2],
  "failed_experiments": [],
  "results": [...]
}
```

---

## 🔍 Verificar Progresso

### Ver quais experimentos foram concluídos

```python
import json

# Carregar checkpoint
with open('/content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022/checkpoint.json', 'r') as f:
    checkpoint = json.load(f)

print("Experimentos concluídos:", checkpoint['completed_experiments'])
print("Último experimento:", checkpoint['last_completed_experiment'])
```

### Ver logs em tempo real

```python
!tail -f /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022/run_all_experiments.log
```

---

## ⚠️ Dicas Importantes

1. **SEMPRE monte o Google Drive primeiro**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Anote o diretório de resultados**
   - O script mostra o caminho no início: `/content/drive/MyDrive/HPM-KD_Results/results_quick_YYYYMMDD_HHMMSS`
   - Copie e salve esse caminho!

3. **Checkpoint automático**
   - O checkpoint é salvo **automaticamente** após cada experimento
   - Não precisa fazer nada manualmente

4. **Se o script falhar ao retomar**
   - Tente especificar o diretório manualmente com `--output`
   - Verifique se o arquivo `checkpoint.json` existe

5. **Múltiplas sessões do Colab**
   - EVITE executar o mesmo experimento em múltiplas sessões ao mesmo tempo
   - Isso pode causar conflitos nos arquivos

---

## 🎯 Resumo de Comandos

```bash
# ===== COMEÇAR NOVA EXECUÇÃO =====
# Modo Quick (rápido, padrão)
!python RUN_COLAB.py

# Modo Full (completo)
!python RUN_COLAB.py --full

# Com dataset específico
!python RUN_COLAB.py --dataset CIFAR10


# ===== RETOMAR EXECUÇÃO =====
# ✨ SIMPLES: Apenas --resume (restaura tudo automaticamente!)
!python RUN_COLAB.py --resume

# NÃO precisa: !python RUN_COLAB.py --full --resume
# O --resume já restaura o modo full automaticamente!


# ===== OPÇÕES AVANÇADAS =====
# Retomar de diretório específico
!python RUN_COLAB.py --resume --output /caminho/completo

# Começar de experimento específico
!python RUN_COLAB.py --start-from 3 --output /caminho/completo

# Executar apenas alguns experimentos
!python RUN_COLAB.py --only 2 3 4

# Pular experimentos
!python RUN_COLAB.py --skip 1 --output /caminho/completo
```

---

## 🆘 Troubleshooting

### Problema: "Checkpoint não encontrado"

**Solução:**
```python
# Liste os diretórios disponíveis
!ls -lah /content/drive/MyDrive/HPM-KD_Results/

# Especifique o diretório manualmente
!python RUN_COLAB.py --resume --output /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022
```

### Problema: "Drive não está montado"

**Solução:**
```python
# Monte manualmente
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Problema: "Experimento continua rodando do início"

**Solução:**
- Certifique-se de usar o flag `--resume`
- Verifique se está usando o mesmo diretório de output (`--output`)
- Verifique se o arquivo `checkpoint.json` existe no diretório

### Problema: "Erro ao salvar checkpoint"

**Solução:**
- Verifique permissões do Google Drive
- Tente remontar o Drive com `force_remount=True`
- Certifique-se de que há espaço suficiente no Drive

---

## 📊 Exemplo Completo

```python
# ========================================
# SESSÃO 1: Início
# ========================================

from google.colab import drive
drive.mount('/content/drive')

# Começar experimentos
!python RUN_COLAB.py --mode quick --dataset MNIST

# Output mostra:
# 💾 Resultados serão salvos NO GOOGLE DRIVE:
#    /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022
#
# ... executa experimento 1 ...
# ✅ Checkpoint salvo em: checkpoint.json
#
# ... executa experimento 2 ...
# ✅ Checkpoint salvo em: checkpoint.json
#
# ... Colab desconecta! ...

# ========================================
# SESSÃO 2: Retomando (depois de horas)
# ========================================

from google.colab import drive
drive.mount('/content/drive')

# Retomar automaticamente
!python RUN_COLAB.py --resume

# Output mostra:
# ♻️  RETOMANDO EXECUÇÃO ANTERIOR:
#    /content/drive/MyDrive/HPM-KD_Results/results_quick_20250111_143022
#
# ✅ Checkpoint encontrado!
#    Última execução: 2025-01-11T15:42:18
#    Experimentos concluídos: [1, 2]
#
# ♻️  Retomando execução - 2 experimentos restantes
#
# ... executa experimento 3 ...
# ✅ Checkpoint salvo em: checkpoint.json
#
# ... executa experimento 4 ...
# ✅ Checkpoint salvo em: checkpoint.json
#
# 🎉 Todos os experimentos concluídos com sucesso!
```

### 📊 Exemplo com Modo Full

```python
# ========================================
# SESSÃO 1: Começar em Modo FULL
# ========================================

from google.colab import drive
drive.mount('/content/drive')

# Começar experimentos em modo FULL
!python RUN_COLAB.py --full

# Output mostra:
# 💾 Resultados serão salvos NO GOOGLE DRIVE:
#    /content/drive/MyDrive/HPM-KD_Results/results_full_20250111_143022
# Modo: FULL
#
# ... executa experimento 1 ...
# ✅ Checkpoint salvo (mode: full, datasets: ['MNIST'])
#
# ... Colab desconecta após 3 horas! ...

# ========================================
# SESSÃO 2: Retomando (NÃO precisa --full!)
# ========================================

from google.colab import drive
drive.mount('/content/drive')

# ✨ APENAS --resume! Não precisa repetir --full
!python RUN_COLAB.py --resume

# Output mostra:
# ♻️  Modo restaurado do checkpoint: FULL
# ♻️  RETOMANDO EXECUÇÃO ANTERIOR:
#    /content/drive/MyDrive/HPM-KD_Results/results_full_20250111_143022
#
# Modo: FULL  ← Restaurado automaticamente!
# Retomando: SIM ♻️
# Datasets: MNIST  ← Também restaurado!
# Experimentos concluídos: [1, 2]
#
# ... continua experimento 3 ...
# 🎉 Tudo restaurado automaticamente!
```

---

## 🎉 Benefícios

- ✅ **Zero perda de progresso** - nunca mais perder horas de trabalho
- ✅ **Retomada automática** - apenas um comando para continuar
- ✅ **Persistência no Drive** - resultados salvos permanentemente
- ✅ **Flexibilidade** - pule, reexecute ou continue experimentos específicos
- ✅ **Segurança** - checkpoints salvos atomicamente (sem corrupção)
- ✅ **Checkpoint Granular** - salva cada modelo individualmente (ver `CHECKPOINT_GRANULAR.md`)

---

## 📚 Documentação Adicional

### Checkpoint Granular de Modelos

Para entender como o sistema salva **cada modelo individualmente** durante os experimentos, consulte:

📄 **`CHECKPOINT_GRANULAR.md`** - Documentação completa do checkpoint granular

**Resumo rápido:**
- Cada modelo (teacher/student) é salvo assim que termina de treinar
- Se desconectar no meio do Experimento 1, não perde modelos já treinados
- Exemplo: treinou 15 de 30 modelos → retoma do 16º, não do 1º!

**Implementado em:**
- ✅ **Experimento 1** (Compression Efficiency) - Checkpoint granular completo (30+ modelos)
- ✅ **Experimento 2** (Ablation Studies) - Checkpoint granular implementado
- ✅ **Experimento 3** (Generalization) - Checkpoint básico (teacher + estrutura)
- ✅ **Experimento 4** (Computational Efficiency) - Checkpoint básico (teacher + estrutura)

**Todos os 4 experimentos** estão protegidos contra desconexões!

---

**Última atualização:** 2025-01-12
