# Quick Start - Experimento 4: HPM-KD Framework

## Instalação Rápida

```bash
cd /home/guhaase/projetos/DeepBridge/papers/00_DeepBridge_Overview/experimentos/04_hpmkd
pip install -r requirements.txt
```

## Execução Rápida (Demo Mock)

```bash
# Teste rápido com dados simulados (~2 min)
python scripts/run_demo.py
```

**Outputs**:
- Métricas simuladas para 3 datasets
- Comparação HPM-KD vs baselines
- Visualizações
- Tabela LaTeX

## Resultados Esperados (Mock)

```
Método          Acurácia  Compressão  Latência  Retenção
---------------------------------------------------------
Teacher         87.2%     1.0×        125ms     100.0%
Vanilla KD      82.5%     10.2×       12ms      94.7%
TAKD            83.8%     10.1×       13ms      96.1%
Auto-KD         84.4%     10.3×       12ms      96.8%
HPM-KD          85.8%     10.3×       12ms      98.4%
```

**Todas as métricas atendem às metas!** ✅

## Estrutura de Outputs

```
results/
├── hpmkd_demo_results.json         # Resultados mock
├── hpmkd_statistical_tests.json    # Testes estatísticos
└── hpmkd_ablation_results.json     # Ablation study

figures/
├── hpmkd_accuracy_comparison.pdf
├── hpmkd_retention_rates.pdf
├── hpmkd_compression_latency.pdf
└── hpmkd_ablation_study.pdf

tables/
└── hpmkd_results.tex
```

## Comandos Úteis

```bash
# Ver resultados
cat results/hpmkd_demo_results.json | python -m json.tool

# Ver figuras
xdg-open figures/*.pdf  # Linux
open figures/*.pdf      # macOS

# Ver tabela LaTeX
cat tables/hpmkd_results.tex
```

## Próximo Passo

Para execução real (não mock):
1. Implementar HPM-KD em PyTorch
2. Baixar 20 datasets reais
3. Treinar teachers
4. Executar experimento completo (~3-4 semanas)

---

**Nota**: Implementação atual é **mock** para validar infraestrutura. Resultados reais requerem implementação completa do HPM-KD framework.
