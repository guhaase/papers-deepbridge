# Correções Aplicadas - Experimentos GPU

**Data**: 2025-12-08
**Status**: ✅ **CORRIGIDO E PRONTO**

---

## 🐛 Bug Corrigido

### Problema
```
AttributeError: 'str' object has no attribute 'mkdir'
```

### Causa
A função `setup_logging()` tinha os parâmetros na ordem errada nos arquivos `utils.py`.

### Solução Aplicada

#### Arquivo 1: `04_hpmkd/scripts/utils.py`
```python
# ANTES (linha 16)
def setup_logging(experiment_name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

# DEPOIS (corrigido)
def setup_logging(log_dir, experiment_name: str) -> logging.Logger:
    log_dir = Path(log_dir)  # Ensure it's a Path object
    log_dir.mkdir(parents=True, exist_ok=True)
```

#### Arquivo 2: `06_ablation_studies/scripts/utils.py`
```python
# ANTES (linha 19)
def setup_logging(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(exist_ok=True, parents=True)

# DEPOIS (corrigido)
def setup_logging(log_dir, name: str) -> logging.Logger:
    log_dir = Path(log_dir)  # Ensure it's a Path object
    log_dir.mkdir(exist_ok=True, parents=True)
```

---

## ✅ Validação

Após as correções, os scripts devem executar sem erros:

```bash
# Teste rápido
python 04_hpmkd/scripts/run_hpmkd_REAL.py
python 06_ablation_studies/scripts/run_ablation_REAL.py
```

---

## 🚀 Instruções Atualizadas para Servidor

### Setup Completo (com correções)

```bash
# 1. No servidor GPU
cd /workspace/papers-deepbridge/00_DeepBridge_Overview/experimentos

# 2. Criar ambiente (se ainda não criou)
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# 3. Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Instalar requirements
pip install -r requirements_gpu.txt

# 5. Instalar DeepBridge
pip install -e /workspace/DeepBridge

# 6. Testar configuração
python test_gpu_setup.py
```

### Executar Experimentos

```bash
# Experimento 4: HPM-KD (~1 hora)
cd 04_hpmkd
python scripts/run_hpmkd_REAL.py

# Monitorar logs (outro terminal)
tail -f logs/hpmkd_real_*.log

# Ver resultados
cat results/hpmkd_results_REAL.json
```

```bash
# Experimento 6: Ablation (~10 minutos)
cd 06_ablation_studies
python scripts/run_ablation_REAL.py

# Monitorar logs (outro terminal)
tail -f logs/ablation_real_*.log

# Ver resultados
cat results/ablation_study_REAL.json
```

---

## 🔍 Troubleshooting Adicional

### Se encontrar "ModuleNotFoundError"

```bash
# Verificar instalações
pip list | grep -E "torch|xgboost|lightgbm|deepbridge"

# Reinstalar se necessário
pip install --upgrade torch xgboost lightgbm
pip install -e /workspace/DeepBridge
```

### Se GPU não for detectada

```bash
# Verificar CUDA
nvidia-smi
nvcc --version

# Testar PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Se CUDA=False, reinstalar PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Se experimento travar

```bash
# Ver processos
ps aux | grep python

# Matar se necessário
kill -9 <PID>

# Limpar memória GPU
python -c "import torch; torch.cuda.empty_cache()"

# Reduzir batch size
# Editar run_hpmkd_REAL.py linha 169: batch_size=128 (em vez de 256)
```

---

## 📊 Estrutura de Resultados

Após execução bem-sucedida:

```
experimentos/
├── 04_hpmkd/
│   ├── logs/
│   │   └── hpmkd_real_YYYYMMDD_HHMMSS.log
│   └── results/
│       └── hpmkd_results_REAL.json
│
└── 06_ablation_studies/
    ├── logs/
    │   └── ablation_real_YYYYMMDD_HHMMSS.log
    └── results/
        └── ablation_study_REAL.json
```

---

## ✅ Checklist de Execução

- [x] Bug de `setup_logging` corrigido
- [ ] Ambiente virtual criado
- [ ] PyTorch com CUDA instalado
- [ ] Requirements instalados
- [ ] DeepBridge instalado
- [ ] `test_gpu_setup.py` passou
- [ ] Experimento 4 executado
- [ ] Experimento 6 executado
- [ ] Resultados JSON gerados
- [ ] Logs sem erros
- [ ] Backup dos resultados

---

**Autor**: Claude Code
**Data**: 2025-12-08
**Status**: ✅ **PRONTO PARA EXECUÇÃO NO SERVIDOR**
