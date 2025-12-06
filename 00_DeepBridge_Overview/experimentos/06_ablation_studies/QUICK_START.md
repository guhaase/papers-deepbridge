# Quick Start - Experimento 6: Ablation Studies

## Instalação

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/06_ablation_studies
pip install -r requirements.txt
```

## Execução Rápida (Demo Mock)

```bash
# Executar demo com resultados simulados (~30 segundos)
python scripts/run_demo.py
```

**Outputs**:
- `results/ablation_demo_results.json` - Resultados completos
- `tables/ablation_results.tex` - Tabela LaTeX
- Summary no terminal

## Resultados Esperados (Demo)

```
ABLATION STUDY - DEMO RESULTS
================================================================================

EXECUTION TIMES BY CONFIGURATION:
--------------------------------------------------------------------------------
Configuração                   Tempo (min)      Ganho
--------------------------------------------------------------------------------
DeepBridge Completo                   17.0          -
Sem API Unificada                     83.0      +66.0
Sem Paralelização                     57.0      +40.0
Sem Caching                           30.0      +13.0
Sem Automação Relatórios              30.0      +13.0
--------------------------------------------------------------------------------
Workflow Fragmentado                 150.0     +133.0

--------------------------------------------------------------------------------
COMPONENT CONTRIBUTIONS:
--------------------------------------------------------------------------------
Componente                     Ganho (min)   % do Total
--------------------------------------------------------------------------------
API Unificada                         66.0          50%
Paralelização                         40.0          30%
Caching                               13.0          10%
Automação Relatórios                  13.0          10%
--------------------------------------------------------------------------------
TOTAL                                133.0         100%

================================================================================
SUMMARY
================================================================================
✓ Total time reduction: 133.0 min (150.0 → 17.0 min)
✓ Overall speedup: 8.8×
✓ API Unificada: 50% (target: 50%)
✓ Paralelização: 30% (target: 30%)
✓ Caching: 10% (target: 10%)
✓ Automação Relatórios: 10% (target: 10%)
================================================================================
```

## Execução Completa (Real - Pendente)

### 1. Implementar Configurações

```bash
# Implementar configurações de ablação no DeepBridge
# (código pendente)
```

### 2. Executar Ablação

```bash
# Executar experimento completo (10 runs × 6 configs)
python scripts/run_ablation.py
```

### 3. Análise Estatística

```bash
# Análise completa com ANOVA e visualizações
python scripts/analyze_results.py
```

## Estrutura de Outputs

```
results/
├── ablation_demo_results.json          # Resultados demo
├── ablation_all_configs.json           # Resultados completos (pendente)
├── ablation_contributions.json         # Contribuições (pendente)
└── ablation_anova.json                 # Análise estatística (pendente)

tables/
└── ablation_results.tex                # Tabela LaTeX

figures/ (pendente)
├── ablation_waterfall.pdf
├── ablation_stacked_bar.pdf
└── ablation_boxplot.pdf
```

## Tempo de Execução

- **Demo (mock)**: ~30 segundos
- **Execução real**: ~14 horas (10 runs × 6 configs × ~140 min médio)
- **Análise**: ~1 hora

## Próximo Passo

Após rodar o demo, implemente as configurações reais de ablação no DeepBridge.
