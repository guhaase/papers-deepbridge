# Paper 13: Interpretable ML Validation Framework

## Título
**Framework de Validação de ML Interpretável para Ambientes Regulados: Equilibrando Acurácia e Conformidade Regulatória**

## Resumo
Este paper apresenta um framework integrado que combina Knowledge Distillation para modelos interpretáveis (Decision Trees e GAMs) com uma suite completa de validação (fairness, robustness, uncertainty) para permitir deployment de ML em domínios regulados.

## Estrutura do Paper

- **Seção 1: Introdução** - Motivação e problema central
- **Seção 2: Background** - Panorama regulatório e trabalhos relacionados
- **Seção 3: Design do Framework** - Arquitetura e componentes (KDDT, GAM, Validation Suite)
- **Seção 4: Implementação** - Detalhes técnicos da implementação no DeepBridge
- **Seção 5: Avaliação Empírica** - Experimentos em HELOC, Adult, COMPAS datasets
- **Seção 6: Discussão** - Limitações, considerações práticas, implicações éticas
- **Seção 7: Conclusão** - Síntese de contribuições e trabalho futuro

## Contribuições Principais

1. **KDDT (Knowledge Distillation for Decision Trees)**: Primeira implementação de destilação especificamente para decision trees com garantias matemáticas
2. **GAM-Based Distillation**: Extensão de GAMs para aceitar soft labels de teachers complexos
3. **Compliance-Aware Validation Suite**: Suite integrada de validação multi-dimensional para modelos interpretáveis
4. **Performance-Interpretability Trade-off Analysis**: Quantificação empírica de Pareto frontiers
5. **Regulatory Mapping**: Mapeamento entre métricas técnicas e requisitos regulatórios

## Resultados Principais

- **Performance**: KDDT atinge 95-97% da AUC de ensembles complexos
- **Compliance**: 91% compliance score médio (vs. 68% de XGBoost)
- **Fairness**: 100% de auditorias ECOA passadas
- **Trade-off**: 3-5% de perda de AUC por compliance robusto

## Compilação

Para compilar o paper:

```bash
./compile.sh
```

Ou manualmente:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Requisitos

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Pacotes: acmart, babel, inputenc, graphicx, booktabs, amsmath, listings, algorithm

## Conferências Alvo

- **Journal of Machine Learning Research (JMLR)** - PRINCIPAL
- Journal of Finance
- Journal of Banking & Finance
- FAccT (ACM Conference on Fairness, Accountability, and Transparency)
- AAAI (Responsible AI track)

## Arquivos

```
POR/
├── main.tex                    # Arquivo principal
├── sections/                   # Seções do paper
│   ├── 01_introduction.tex
│   ├── 02_background.tex
│   ├── 03_design.tex
│   ├── 04_implementation.tex
│   ├── 05_evaluation.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
├── bibliography/
│   └── references.bib         # Referências bibliográficas
├── acmart.cls                 # Classe ACM
├── compile.sh                 # Script de compilação
└── README.md                  # Este arquivo
```

## Contato

Para questões sobre o paper ou framework DeepBridge, consulte a documentação em https://deepbridge.readthedocs.io
