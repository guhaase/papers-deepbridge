# Guia de Execução - Experimentos com GPU

**Data**: 2025-12-08
**Status**: ✅ **PRONTO PARA EXECUÇÃO NO SERVIDOR GPU**

---

## 📋 Pré-requisitos

### Hardware Mínimo
- **GPU**: NVIDIA com CUDA 11.8+ (ex: RTX 3080, A100)
- **VRAM**: 12GB+ recomendado
- **RAM**: 32GB
- **Storage**: 50GB livre

### Software
- **OS**: Ubuntu 20.04+ ou similar
- **Python**: 3.10+
- **CUDA**: 11.8 ou 12.1
- **cuDNN**: 8.6+

---

## 🚀 Setup Inicial no Servidor GPU

### 1. Verificar GPU

```bash
# Verificar se GPU está disponível
nvidia-smi

# Verificar CUDA
nvcc --version
```

### 2. Criar Ambiente Virtual

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos

# Criar venv
python3 -m venv venv_gpu

# Ativar
source venv_gpu/bin/activate
```

### 3. Instalar PyTorch com CUDA

```bash
# Para CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar instalação
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### 4. Instalar Dependências

```bash
# Instalar requirements
pip install -r requirements_gpu.txt

# Instalar DeepBridge
pip install -e /home/guhaase/projetos/DeepBridge

# Instalar XGBoost e LightGBM com suporte GPU
pip install xgboost --upgrade
pip install lightgbm --upgrade --install-option=--gpu
```

### 5. Verificar Instalação

```bash
python -c "
import torch
import xgboost as xgb
import lightgbm as lgb
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'XGBoost: {xgb.__version__}')
print(f'LightGBM: {lgb.__version__}')
"
```

---

## 📊 Experimento 4: HPM-KD Framework

### Descrição
Implementa Knowledge Distillation real com ensemble teachers e student neural network.

### Tempo Estimado
- **Por dataset**: ~15-20 minutos
- **3 datasets**: ~1 hora
- **Com GPU**: Speedup de ~3-5× vs CPU

### Execução

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd

# Executar com 3 datasets
poetry run python scripts/run_hpmkd_REAL.py

# Ver resultados
cat results/hpmkd_results_REAL.json

# Ver logs
tail -f logs/hpmkd_real_*.log
```

### Monitoramento GPU

Em outro terminal:
```bash
watch -n 1 nvidia-smi
```

### Configurações Opcionais

Editar no código `run_hpmkd_REAL.py`:

```python
# Linha 432: Número de datasets
results = run_hpmkd_experiment(n_datasets=3, seed=42)

# Para mais datasets (mais lento, mas mais robusto):
results = run_hpmkd_experiment(n_datasets=5, seed=42)

# Epochs (linhas 175, 242):
# epochs=20  # Mais rápido
# epochs=50  # Mais preciso
```

---

## 📊 Experimento 6: Ablation Studies

### Descrição
Compara DeepBridge completo vs baseline fragmentado usando execução real.

### Tempo Estimado
- **Por run**: ~35-40 segundos (DeepBridge + Baseline)
- **10 runs**: ~6-7 minutos
- **Total**: ~10 minutos com overhead

### Execução

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/06_ablation_studies

# Executar com 10 runs
poetry run python scripts/run_ablation_REAL.py

# Ver resultados
cat results/ablation_study_REAL.json

# Ver logs
tail -f logs/ablation_real_*.log
```

### Configurações Opcionais

Editar no código `run_ablation_REAL.py`:

```python
# Linha 289: Número de runs
results = ablation.run_ablation_study(num_runs=10)

# Para teste rápido:
results = ablation.run_ablation_study(num_runs=3)

# Para mais robusto:
results = ablation.run_ablation_study(num_runs=20)
```

---

## 🔍 Troubleshooting

### Problema: "CUDA out of memory"

**Solução 1**: Reduzir batch size
```python
# Em run_hpmkd_REAL.py, linha 169:
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # era 256
```

**Solução 2**: Limpar cache
```python
import torch
torch.cuda.empty_cache()
```

### Problema: "No CUDA-capable device is detected"

**Solução**: Verificar driver
```bash
nvidia-smi
sudo apt-get install nvidia-driver-525  # ou versão adequada
```

### Problema: XGBoost não usa GPU

**Solução**: Compilar com suporte GPU
```bash
pip uninstall xgboost
pip install xgboost --no-binary :all: --config-settings=use_cuda=True
```

### Problema: LightGBM não usa GPU

**Solução**: Instalar com suporte GPU
```bash
pip uninstall lightgbm
pip install lightgbm --install-option=--gpu
```

---

## 📈 Monitoramento de Progresso

### Ver logs em tempo real

```bash
# Experimento 4
tail -f /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd/logs/hpmkd_real_*.log

# Experimento 6
tail -f /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/06_ablation_studies/logs/ablation_real_*.log
```

### Monitorar GPU

```bash
# Uso da GPU
watch -n 1 nvidia-smi

# Temperatura e power
nvidia-smi dmon -s pucvmet -c 100
```

### Checar progresso

```bash
# Experimento 4
ls -lh /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd/results/

# Experimento 6
ls -lh /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/06_ablation_studies/results/
```

---

## 🎯 Execução Completa (Ambos Experimentos)

### Script de Execução Automatizada

Criar `run_all_gpu.sh`:

```bash
#!/bin/bash
set -e

echo "========================================"
echo "Executando Experimentos com GPU"
echo "========================================"

# Ativar ambiente
source venv_gpu/bin/activate

# Experimento 4: HPM-KD
echo ""
echo "=== Experimento 4: HPM-KD ==="
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd
poetry run python scripts/run_hpmkd_REAL.py

# Experimento 6: Ablation
echo ""
echo "=== Experimento 6: Ablation Studies ==="
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/06_ablation_studies
poetry run python scripts/run_ablation_REAL.py

echo ""
echo "========================================"
echo "Experimentos concluídos!"
echo "========================================"

# Mostrar resultados
echo ""
echo "Resultados Experimento 4:"
cat /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd/results/hpmkd_results_REAL.json | head -50

echo ""
echo "Resultados Experimento 6:"
cat /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/06_ablation_studies/results/ablation_study_REAL.json | head -50
```

Executar:
```bash
chmod +x run_all_gpu.sh
nohup ./run_all_gpu.sh > experimentos_output.log 2>&1 &

# Monitorar
tail -f experimentos_output.log
```

---

## ⏱️ Timeline Estimado

| Tarefa | Tempo Estimado | Observações |
|--------|----------------|-------------|
| Setup inicial | 10-15 min | Instalação de dependências |
| Experimento 4 (3 datasets) | 1 hora | Com GPU |
| Experimento 6 (10 runs) | 10 min | Comparação rápida |
| **Total** | **~1h 30min** | **Com GPU disponível** |

---

## 📊 Resultados Esperados

### Experimento 4: HPM-KD

```json
{
  "datasets": [
    {
      "dataset_name": "adult_split_1",
      "teacher_accuracy": 87.2,
      "vanilla_kd_accuracy": 82.5,
      "takd_accuracy": 83.8,
      "auto_kd_accuracy": 84.4,
      "hpmkd_accuracy": 85.8,
      "compression_ratio": 10.3,
      "latency_speedup": 10.4,
      "hpmkd_retention": 98.4
    },
    ...
  ]
}
```

### Experimento 6: Ablation

```json
{
  "deepbridge_full": {
    "mean_seconds": 35.9,
    "std_seconds": 1.8,
    "num_runs": 10
  },
  "baseline_fragmented": {
    "mean_seconds": 3.3,
    "std_seconds": 0.2,
    "num_runs": 10
  },
  "comparison": {
    "speedup": 0.09,
    "interpretation": "baseline_faster"
  }
}
```

---

## ✅ Checklist de Execução

Antes de executar no servidor:

- [ ] GPU disponível e funcionando (`nvidia-smi`)
- [ ] CUDA instalado (`nvcc --version`)
- [ ] Python 3.10+ disponível
- [ ] Ambiente virtual criado
- [ ] PyTorch com CUDA instalado e testado
- [ ] XGBoost e LightGBM com GPU instalados
- [ ] DeepBridge instalado (`pip install -e ...`)
- [ ] Requirements instalados (`pip install -r requirements_gpu.txt`)
- [ ] Espaço em disco suficiente (50GB+)

Após execução:

- [ ] Verificar logs sem erros
- [ ] Verificar arquivos JSON gerados
- [ ] Validar métricas fazem sentido
- [ ] Backup dos resultados

---

## 📞 Suporte

Em caso de problemas:

1. Verificar logs detalhados em `logs/`
2. Testar GPU com script de teste
3. Verificar versões das bibliotecas
4. Consultar documentação do PyTorch/XGBoost

---

**Autor**: Claude Code
**Data**: 2025-12-08
**Versão**: 1.0
**Status**: ✅ **PRONTO PARA EXECUÇÃO**
