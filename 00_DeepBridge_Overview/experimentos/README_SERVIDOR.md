# README - Execução no Servidor GPU

**Status**: ✅ **PRONTO PARA EXECUTAR**
**Última atualização**: 2025-12-08

---

## 🚀 Quick Start (3 comandos)

```bash
# 1. Ir para o diretório
cd /workspace/papers-deepbridge/00_DeepBridge_Overview/experimentos

# 2. Dar permissão ao script
chmod +x deploy_e_executar.sh

# 3. Executar
./deploy_e_executar.sh
```

O script vai:
- ✅ Verificar GPU
- ✅ Criar ambiente virtual
- ✅ Instalar PyTorch com CUDA
- ✅ Instalar todas as dependências
- ✅ Testar configuração
- ✅ Executar os experimentos

---

## 📊 Experimentos Disponíveis

### Experimento 4: HPM-KD Framework
- **Descrição**: Knowledge Distillation real (XGBoost+LightGBM → PyTorch)
- **Tempo**: ~1 hora (3 datasets)
- **GPU**: Sim (recomendado)
- **Output**: `04_hpmkd/results/hpmkd_results_REAL.json`

### Experimento 6: Ablation Studies
- **Descrição**: Comparação DeepBridge vs Baseline fragmentado
- **Tempo**: ~10 minutos (10 runs)
- **GPU**: Parcial (DeepBridge usa)
- **Output**: `06_ablation_studies/results/ablation_study_REAL.json`

---

## 🔧 Opções de Execução

### Opção 1: Script Automatizado (RECOMENDADO)

```bash
./deploy_e_executar.sh
```

Você será perguntado qual experimento executar:
1. Apenas HPM-KD (~1h)
2. Apenas Ablation (~10min)
3. Ambos (~1h 10min)

### Opção 2: Manual

#### Experimento 4 (HPM-KD)

```bash
cd /workspace/papers-deepbridge/00_DeepBridge_Overview/experimentos
source venv_gpu/bin/activate
cd 04_hpmkd
python scripts/run_hpmkd_REAL.py
```

#### Experimento 6 (Ablation)

```bash
cd /workspace/papers-deepbridge/00_DeepBridge_Overview/experimentos
source venv_gpu/bin/activate
cd 06_ablation_studies
python scripts/run_ablation_REAL.py
```

### Opção 3: Background (para não travar terminal)

```bash
# Executar em background
nohup ./deploy_e_executar.sh > experimentos.log 2>&1 &

# Monitorar progresso
tail -f experimentos.log

# OU monitorar logs específicos
tail -f 04_hpmkd/logs/*.log
tail -f 06_ablation_studies/logs/*.log
```

---

## 📈 Monitoramento Durante Execução

### Monitorar GPU

```bash
# Em outro terminal
watch -n 1 nvidia-smi
```

### Monitorar Logs

```bash
# Experimento 4
tail -f 04_hpmkd/logs/hpmkd_real_*.log

# Experimento 6
tail -f 06_ablation_studies/logs/ablation_real_*.log
```

### Verificar Progresso

```bash
# Ver últimas linhas dos logs
tail -20 04_hpmkd/logs/*.log
tail -20 06_ablation_studies/logs/*.log

# Verificar se resultados foram gerados
ls -lh 04_hpmkd/results/
ls -lh 06_ablation_studies/results/
```

---

## 📊 Verificar Resultados

### Experimento 4 (HPM-KD)

```bash
# Ver resultados
cat 04_hpmkd/results/hpmkd_results_REAL.json | python -m json.tool | head -50

# Métricas principais
cat 04_hpmkd/results/hpmkd_results_REAL.json | grep -E "accuracy|retention|compression|speedup"
```

### Experimento 6 (Ablation)

```bash
# Ver resultados
cat 06_ablation_studies/results/ablation_study_REAL.json | python -m json.tool | head -50

# Comparação
cat 06_ablation_studies/results/ablation_study_REAL.json | grep -E "mean_seconds|speedup"
```

---

## ✅ Resultados Esperados

### Experimento 4: HPM-KD

```json
{
  "datasets": [
    {
      "teacher_accuracy": ~87%,
      "vanilla_kd_accuracy": ~82%,
      "takd_accuracy": ~84%,
      "auto_kd_accuracy": ~84%,
      "hpmkd_accuracy": ~86%,
      "compression_ratio": ~10×,
      "latency_speedup": ~10×,
      "hpmkd_retention": ~98%
    }
  ]
}
```

### Experimento 6: Ablation

```json
{
  "deepbridge_full": {
    "mean_seconds": ~36s,
    "num_runs": 10
  },
  "baseline_fragmented": {
    "mean_seconds": ~3.3s,
    "num_runs": 10
  },
  "comparison": {
    "speedup": ~0.09× (baseline mais rápido)
  }
}
```

**NOTA**: É normal (e correto) que o baseline seja mais rápido no Experimento 6.

---

## 🐛 Troubleshooting

### GPU não detectada

```bash
# Verificar driver
nvidia-smi

# Se falhar, instalar driver
sudo apt-get update
sudo apt-get install nvidia-driver-525
```

### PyTorch não encontra CUDA

```bash
source venv_gpu/bin/activate
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Se False, reinstalar
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"

```python
# Editar batch size em run_hpmkd_REAL.py linha 169
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # era 256
```

### Script travou

```bash
# Ver processos
ps aux | grep python

# Matar
kill -9 <PID>

# Limpar GPU
python -c "import torch; torch.cuda.empty_cache()"
```

### Erro de módulo não encontrado

```bash
# Reinstalar requirements
source venv_gpu/bin/activate
pip install -r requirements_gpu.txt

# Reinstalar DeepBridge
pip install -e /workspace/DeepBridge
```

---

## 📁 Estrutura de Arquivos

Após execução bem-sucedida:

```
experimentos/
├── 04_hpmkd/
│   ├── logs/
│   │   └── hpmkd_real_YYYYMMDD_HHMMSS.log  ← Ver para debug
│   └── results/
│       └── hpmkd_results_REAL.json          ← Resultados finais
│
├── 06_ablation_studies/
│   ├── logs/
│   │   └── ablation_real_YYYYMMDD_HHMMSS.log
│   └── results/
│       └── ablation_study_REAL.json
│
└── venv_gpu/  ← Ambiente virtual (criado automaticamente)
```

---

## ⏱️ Timeline

| Tarefa | Tempo |
|--------|-------|
| Setup (primeira vez) | 10-15 min |
| Teste de configuração | 1 min |
| Experimento 4 (HPM-KD) | 60 min |
| Experimento 6 (Ablation) | 10 min |
| **TOTAL** | **~1h 30min** |

---

## 📝 Checklist de Execução

- [ ] Script `deploy_e_executar.sh` tem permissão de execução
- [ ] GPU detectada com `nvidia-smi`
- [ ] Ambiente virtual criado
- [ ] PyTorch com CUDA instalado
- [ ] `test_gpu_setup.py` passou
- [ ] Experimento 4 executado
- [ ] Experimento 6 executado
- [ ] Arquivos JSON gerados em `results/`
- [ ] Logs sem erros em `logs/`
- [ ] Backup dos resultados feito

---

## 🆘 Suporte

Se tiver problemas:

1. Verificar logs em `logs/`
2. Executar `test_gpu_setup.py` para diagnosticar
3. Consultar `CORRECOES_APLICADAS.md` para bugs conhecidos
4. Consultar `GUIA_EXECUCAO_GPU.md` para troubleshooting detalhado

---

## 📖 Documentação Adicional

- `GUIA_EXECUCAO_GPU.md` - Guia completo e detalhado
- `CORRECOES_APLICADAS.md` - Bugs corrigidos e soluções
- `RESUMO_ATUALIZACOES_GPU.md` - Resumo de todas as mudanças
- `test_gpu_setup.py` - Script de teste de configuração
- `requirements_gpu.txt` - Dependências necessárias

---

**Autor**: Claude Code
**Data**: 2025-12-08
**Status**: ✅ **TESTADO E PRONTO**
