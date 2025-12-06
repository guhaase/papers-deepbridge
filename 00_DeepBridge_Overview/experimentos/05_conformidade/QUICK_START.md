# Quick Start - Experimento 5: Conformidade

## Instalação

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/05_conformidade
pip install -r requirements.txt
```

## Execução Rápida (Demo Mock)

```bash
# Executar demo com resultados simulados (~30 segundos)
python scripts/run_demo.py
```

**Outputs**:
- `results/compliance_demo_results.json` - Resultados completos
- `tables/compliance_results.tex` - Tabela LaTeX
- Summary no terminal

## Resultados Esperados (Demo)

```
COMPLIANCE VALIDATION - DEMO RESULTS
================================================================================

CONFUSION MATRICES:
--------------------------------------------------------------------------------

Deepbridge:
  TP: 25  FP:  0
  FN:  0  TN: 25
  Precision: 100.0%
  Recall:    100.0%
  F1-Score:  100.0%

Baseline:
  TP: 20  FP:  3
  FN:  5  TN: 22
  Precision: 87.0%
  Recall:    80.0%
  F1-Score:  83.3%

--------------------------------------------------------------------------------
FEATURE COVERAGE:
--------------------------------------------------------------------------------
  DeepBridge:  10/10 attributes (100% coverage)
  Baseline:     2/10 attributes ( 20% coverage)

--------------------------------------------------------------------------------
AUDIT TIME:
--------------------------------------------------------------------------------
  DeepBridge:  48 min
  Baseline:    285 min
  Reduction:   83%

================================================================================
SUMMARY
================================================================================
✓ DeepBridge achieves 100% precision (target: 100%)
✓ DeepBridge achieves 100% recall (target: 100%)
✓ DeepBridge achieves 100% F1-score (target: 100%)
✓ Feature coverage: 100% (target: 10/10 attributes)
✓ Audit time reduction: 83% (target: 70%)
================================================================================
```

## Execução Completa (Real - Pendente)

### 1. Gerar Ground Truth

```bash
# Gerar 50 casos de teste com violações conhecidas
python scripts/generate_ground_truth.py
```

Outputs:
- `data/case_01.csv` até `data/case_50.csv`
- `results/compliance_ground_truth.json`

### 2. Validar com DeepBridge (Pendente)

```bash
python scripts/validate_deepbridge.py
```

### 3. Validar com Baseline (Pendente)

```bash
python scripts/validate_baseline.py
```

### 4. Análise Completa (Pendente)

```bash
python scripts/analyze_results.py
```

## Estrutura de Outputs

```
results/
├── compliance_ground_truth.json          # Ground truth dos 50 casos
├── compliance_demo_results.json          # Resultados demo
├── compliance_deepbridge_results.json    # Resultados DeepBridge (pendente)
├── compliance_baseline_results.json      # Resultados baseline (pendente)
└── compliance_statistical_tests.json     # Testes estatísticos (pendente)

tables/
└── compliance_results.tex                # Tabela LaTeX

figures/ (pendente)
├── compliance_confusion_matrix.pdf
├── compliance_precision_recall.pdf
├── compliance_feature_coverage.pdf
└── compliance_audit_time.pdf
```

## Tempo de Execução

- **Demo (mock)**: ~30 segundos
- **Gerar ground truth**: ~2 minutos
- **Validação DeepBridge**: ~17 minutos (50 casos)
- **Validação baseline**: ~4-5 horas (manual + ferramentas)
- **Total**: ~1 dia

## Próximo Passo

Após rodar o demo, execute:

```bash
python scripts/generate_ground_truth.py
```

Para criar os 50 casos de teste reais.
