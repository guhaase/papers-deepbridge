# Quick Start Guide - Experimento 01

## Setup Rápido (5 minutos)

```bash
# 1. Navegar para o diretório
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/01_benchmarks_tempo

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Testar instalação
cd scripts
python -c "import utils; print('✓ Setup OK!')"
```

## Teste Rápido (30 min)

Execute experimento com apenas 1 run para testar:

```bash
cd scripts
python run_experiment.py --quick
```

Isso vai:
1. ✓ Executar 1 run do benchmark DeepBridge (~15 min)
2. ✓ Executar 1 run do benchmark Fragmentado (~15 min)
3. ✓ Analisar resultados
4. ✓ Gerar todas as figuras

## Verificar Resultados

```bash
# Ver resumo dos resultados
cat ../results/comparison_summary.csv

# Ver figuras geradas
ls -lh ../figures/

# Ver tabela LaTeX
cat ../tables/time_benchmarks.tex
```

## Exemplo de Output Esperado

```
MAIN FINDINGS:
  Total time DeepBridge: 17.0 min
  Total time Fragmented: 150.0 min
  Overall Speedup: 8.8×
  Overall Reduction: 88.7%

TARGET CHECK:
  Target reduction: 89%
  Actual reduction: 88.7%
  ✓ TARGET MET!
```

## Estrutura dos Outputs

```
01_benchmarks_tempo/
├── results/
│   ├── deepbridge_times.json        # Tempos do DeepBridge
│   ├── fragmented_times.json        # Tempos do Fragmentado
│   ├── comparison_summary.csv       # Resumo comparativo
│   └── analysis_results.json        # Análise estatística
│
├── figures/
│   ├── time_comparison_barplot.pdf  # Comparação de tempos
│   ├── speedup_by_task.pdf          # Speedup por tarefa
│   ├── reduction_percentage.pdf     # Redução percentual
│   ├── boxplot_comparison.pdf       # Distribuições
│   └── total_time_breakdown.pdf     # Breakdown do tempo
│
├── tables/
│   └── time_benchmarks.tex          # Tabela LaTeX para paper
│
└── logs/
    └── *.log                        # Logs de execução
```

## Próximos Passos

### Para experimento completo (5-6 horas):

```bash
# Executar 10 runs de cada benchmark
python run_experiment.py --all
```

### Para executar partes específicas:

```bash
# Apenas DeepBridge
python run_experiment.py --deepbridge

# Apenas Fragmentado
python run_experiment.py --fragmented

# Apenas análise (requer resultados prévios)
python run_experiment.py --analyze

# Apenas figuras (requer resultados prévios)
python run_experiment.py --figures
```

## Troubleshooting

### Erro: "No module named 'utils'"

```bash
# Certifique-se de estar no diretório scripts/
cd scripts
python run_experiment.py --quick
```

### Erro: "FileNotFoundError: Config file not found"

```bash
# Verificar que config.yaml existe
ls -lh ../config/config.yaml

# Se não existir, está no diretório errado
cd /path/to/01_benchmarks_tempo/scripts
```

### Teste muito lento

Ajuste tempos em `config/config.yaml`:

```yaml
tests:
  fairness:
    expected_time_deepbridge: 0.1  # 6 segundos
    expected_time_fragmented: 0.5  # 30 segundos
```

## Customização

Edite `config/config.yaml`:

```yaml
general:
  num_runs: 5        # Número de runs (default: 10)
  seed: 42           # Seed para reprodutibilidade

dataset:
  name: "adult_income"
  test_size: 0.2

model:
  type: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
```

## Verificação de Sanidade

Execute testes básicos:

```bash
# Testar utils
python utils.py

# Testar config
python -c "from utils import load_config; print(load_config())"

# Testar sistema
python -c "from utils import get_system_info; import json; print(json.dumps(get_system_info(), indent=2))"
```

## Notas Importantes

⚠️ **MODO SIMULAÇÃO**: Os scripts atualmente **simulam** os tempos usando `time.sleep()`.

Quando DeepBridge estiver implementado, substituir em `benchmark_deepbridge.py`:

```python
# De:
time.sleep(expected_time)

# Para:
from deepbridge import DBDataset, Experiment
exp = Experiment(dataset, tests='all')
results = exp.run_tests()
```

## Ajuda

Para mais detalhes, consulte `README.md` completo.

---

**Última atualização**: 2025-12-05
