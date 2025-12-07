# Referência Rápida - Execução de Experimentos

## 🚀 Início Rápido

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos

# 1. Executar todos os experimentos (em sequência)
./run_all_experiments.sh

# 2. Monitorar progresso (em outro terminal)
./monitor_experiments.sh --follow
```

---

## 📜 Scripts Disponíveis

### 1. `run_all_experiments.sh` - Script Master de Execução

**Executa todos os experimentos em sequência (exceto Exp 4 - GPU)**

```bash
# Executar todos
./run_all_experiments.sh

# Ver ajuda
./run_all_experiments.sh --help

# Pular experimentos específicos
./run_all_experiments.sh --skip-exp1 --skip-exp2

# Dry run (simular sem executar)
./run_all_experiments.sh --dry-run
```

**Tempo estimado**: 21-23 horas (sequencial)

**Opções**:
- `--skip-exp1` - Pular Experimento 1 (Benchmarks - 3-4h)
- `--skip-exp2` - Pular Experimento 2 (Estudos de Caso - 2-3h)
- `--skip-exp3` - Pular Experimento 3 (Usabilidade - 30min)
- `--skip-exp5` - Pular Experimento 5 (Conformidade - 1h)
- `--skip-exp6` - Pular Experimento 6 (Ablation - 14h)
- `--dry-run` - Mostrar o que seria executado sem executar

---

### 2. `monitor_experiments.sh` - Monitor de Progresso

**Monitora execução em tempo real**

```bash
# Snapshot (executa uma vez)
./monitor_experiments.sh

# Modo contínuo (atualiza a cada 5s)
./monitor_experiments.sh --follow

# Monitorar log específico
./monitor_experiments.sh --log logs_master/master_20251206_120000.log
```

**Mostra**:
- Status de cada experimento (Executando/Completo/Pendente)
- Últimos logs
- Processos Python ativos
- Uso de disco

---

### 3. `utils_experiments.sh` - Utilitários

**Funções auxiliares para gerenciar experimentos**

```bash
# Ver todos os comandos
./utils_experiments.sh help

# Validar que tudo está OK
./utils_experiments.sh validate

# Ver todos os resultados
./utils_experiments.sh list-results

# Verificar espaço em disco
./utils_experiments.sh check-space

# Fazer backup
./utils_experiments.sh backup

# Limpar logs (mantém resultados)
./utils_experiments.sh clean-logs

# Limpar tudo (CUIDADO!)
./utils_experiments.sh clean

# Matar processos travados
./utils_experiments.sh kill-all
```

**Comandos disponíveis**:
- `validate` - Validar estrutura de experimentos
- `list-results` - Listar todos os arquivos de resultado
- `check-space` - Verificar uso de disco
- `backup` - Criar backup de resultados
- `clean-logs` - Limpar apenas logs
- `clean-results` - Limpar apenas resultados
- `clean` - Limpar tudo (resultados + logs)
- `kill-all` - Matar todos os processos de experimentos

---

## 📊 Experimentos Individuais

### Experimento 1: Benchmarks de Tempo (3-4 horas)

```bash
cd 01_benchmarks_tempo
echo "y" | python3 scripts/run_experiment.py
```

**Saídas**:
- `results/deepbridge_benchmark_*.json`
- `results/fragmented_benchmark_*.json`
- `figures/benchmark_comparison.pdf`
- `tables/benchmark_results.tex`

---

### Experimento 2: Estudos de Caso (2-3 horas)

```bash
cd 02_estudos_de_caso
python3 scripts/run_all_cases.py
```

**Saídas**:
- `results/credit_results.json`
- `results/hiring_results.json`
- `results/insurance_results.json`
- `figures/*.pdf`

---

### Experimento 3: Usabilidade (~30 min)

```bash
cd 03_usabilidade
python3 scripts/generate_mock_data.py
python3 scripts/analyze_usability.py
python3 scripts/generate_visualizations.py
```

**Saídas**:
- `results/usability_metrics.json`
- `figures/usability_*.pdf`
- `tables/usability_comparison.tex`

---

### Experimento 4: HPMKD (8-12 horas) - ⚠️ REQUER GPU

```bash
# EXECUTAR EM SERVIDOR COM GPU
cd 04_hpmkd
python3 scripts/run_hpmkd.py
```

**Requisitos**: GPU NVIDIA, CUDA, 8GB+ VRAM

---

### Experimento 5: Conformidade (~1 hora) - ⚠️ MOCK

```bash
cd 05_conformidade
python3 scripts/run_demo.py
```

**Status**: Versão mock (implementação real pendente)

---

### Experimento 6: Ablation Studies (~14 horas) - ⚠️ MOCK

```bash
cd 06_ablation_studies
python3 scripts/run_demo.py
```

**Status**: Versão mock (implementação real pendente)

---

## 📂 Estrutura de Arquivos

```
experimentos/
├── run_all_experiments.sh         # Script master
├── monitor_experiments.sh         # Monitor de progresso
├── utils_experiments.sh           # Utilitários
├── README_EXECUTION.md            # Guia completo
├── QUICK_REFERENCE.md             # Este arquivo
│
├── logs_master/                   # Logs consolidados
│   ├── master_YYYYMMDD_HHMMSS.log    # Log master
│   ├── summary_YYYYMMDD_HHMMSS.txt   # Resumo execução
│   └── exp*_YYYYMMDD_HHMMSS.log      # Logs individuais
│
├── 01_benchmarks_tempo/
├── 02_estudos_de_caso/
├── 03_usabilidade/
├── 04_hpmkd/
├── 05_conformidade/
└── 06_ablation_studies/
```

---

## 🔍 Verificações Pós-Execução

```bash
# 1. Ver resumo
cat logs_master/summary_*.txt

# 2. Contar resultados
find . -name "*.json" -path "*/results/*" | wc -l

# 3. Contar figuras
find . -name "*.pdf" -path "*/figures/*" | wc -l

# 4. Verificar erros
grep -i "error\|exception" logs_master/master_*.log

# 5. Ver tempo total
tail logs_master/master_*.log | grep "Tempo total"
```

---

## 🆘 Troubleshooting Rápido

### Experimento travado?

```bash
# Ver processos
ps aux | grep python3

# Matar processo específico
kill -9 <PID>

# Ou matar todos
./utils_experiments.sh kill-all
```

### Sem espaço em disco?

```bash
# Verificar espaço
./utils_experiments.sh check-space

# Fazer backup e limpar
./utils_experiments.sh backup
./utils_experiments.sh clean-logs
```

### Erro de permissão?

```bash
chmod +x *.sh
```

### Módulo Python não encontrado?

```bash
# Verificar ambiente
which python3

# Instalar dependências
pip install -r requirements.txt
```

---

## 📝 Logs

### Localização

Todos os logs ficam em `logs_master/`:

```
logs_master/
├── master_20251206_143022.log      # Log completo
├── summary_20251206_143022.txt     # Resumo tabular
├── exp1_20251206_143022.log        # Log Experimento 1
├── exp2_20251206_143022.log        # Log Experimento 2
└── ...
```

### Ver logs em tempo real

```bash
# Último log master
tail -f logs_master/master_*.log

# Log específico
tail -f logs_master/exp1_*.log

# Ou usar o monitor
./monitor_experiments.sh --follow
```

---

## ⏱️ Estimativas de Tempo

| Experimento | Tempo Estimado | Status |
|-------------|----------------|--------|
| 1. Benchmarks | 3-4 horas | Real |
| 2. Estudos de Caso | 2-3 horas | Real |
| 3. Usabilidade | 30 min | Real |
| 4. HPMKD | 8-12 horas | GPU (separado) |
| 5. Conformidade | 1 hora | Mock |
| 6. Ablation | 14 horas | Mock |
| **TOTAL** | **21-23 horas** | Sequencial |

---

## ✅ Checklist

**Antes de executar**:
- [ ] Scripts executáveis (`chmod +x *.sh`)
- [ ] Requirements instalados
- [ ] ~50GB espaço livre
- [ ] Tempo disponível (~21-23h)

**Durante execução**:
- [ ] Monitor rodando (`./monitor_experiments.sh --follow`)
- [ ] Verificar logs periodicamente
- [ ] Monitorar espaço em disco

**Após execução**:
- [ ] Ver resumo (`cat logs_master/summary_*.txt`)
- [ ] Validar resultados (`./utils_experiments.sh list-results`)
- [ ] Fazer backup (`./utils_experiments.sh backup`)
- [ ] Copiar figuras/tabelas para paper

---

## 📞 Comandos Úteis

```bash
# Status geral
./monitor_experiments.sh

# Validar estrutura
./utils_experiments.sh validate

# Ver uso de disco
./utils_experiments.sh check-space

# Listar resultados
./utils_experiments.sh list-results

# Backup
./utils_experiments.sh backup

# Processos ativos
ps aux | grep python3 | grep -E "experiment|demo"

# Espaço livre
df -h /home/guhaase/projetos/DeepBridge

# Última execução
ls -lt logs_master/ | head -5
```

---

**Última atualização**: 2025-12-06
**Versão**: 1.0
