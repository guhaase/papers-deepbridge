# Paper 12: Tutorial on Production ML Validation

## Título
**Do Desenvolvimento ao Deploy: Guia Prático de Validação de Modelos de Machine Learning em Produção**

## Informações Básicas

- **Tipo**: Tutorial hands-on
- **Conferência Alvo**: KDD (Tutorial track), ICML (Tutorial)
- **Duração**: 3 horas (6 módulos de 30 minutos)
- **Idioma**: Português
- **Biblioteca**: DeepBridge

## Estrutura do Tutorial

1. **Introdução** (30 min)
   - Motivação: desafios de validação em produção
   - Casos reais de falhas (Amazon Hiring AI, Healthcare Risk, Credit Scoring)
   - Overview do DeepBridge
   - Setup e instalação

2. **Robustness Testing** (30 min)
   - Perturbation tests (Gaussian noise, quantile-based)
   - Weakspot detection
   - Overfitting analysis
   - Hands-on: Breast Cancer dataset

3. **Fairness Testing** (30 min)
   - 15 métricas de fairness (pre/post-training)
   - Auto-detecção de atributos sensíveis
   - Age grouping regulatório (ADEA/ECOA)
   - Threshold analysis
   - Hands-on: Adult Income dataset

4. **Uncertainty Quantification** (30 min)
   - CRQR (Conformalized Residual Quantile Regression)
   - Intervalos de predição calibrados
   - Reliability analysis por features
   - Coverage guarantees
   - Hands-on: California Housing dataset

5. **Integration & Reporting** (30 min)
   - Experiment class (orquestração)
   - Sistema de relatórios HTML
   - Integração CI/CD (GitHub Actions)
   - Monitoramento contínuo
   - Pipeline end-to-end

6. **Q&A e Discussão** (30 min)
   - Perguntas e respostas
   - Casos de uso específicos
   - Demonstração ao vivo

## Contribuições Principais

1. **Framework prático**: Tutorial hands-on com 3 notebooks executáveis
2. **Validação sistemática**: Cobertura de robustness, fairness e uncertainty
3. **Production-ready**: Integração CI/CD e monitoramento contínuo
4. **Regulatory compliance**: Alinhamento com EEOC, ECOA
5. **Impacto demonstrado**:
   - 73% redução em false negatives via weakspot mitigation
   - Detecção de 4 violações EEOC em modelo de hiring
   - Intervalos com 92% coverage real vs. 90% esperada

## Estrutura de Arquivos

```
POR/
├── main.tex                    # Arquivo principal LaTeX
├── acmart.cls                  # Classe ACM
├── compile.sh                  # Script de compilação
├── sections/                   # Seções do paper
│   ├── 01_introduction.tex
│   ├── 02_robustness.tex
│   ├── 03_fairness.tex
│   ├── 04_uncertainty.tex
│   ├── 05_integration.tex
│   └── 06_conclusion.tex
├── bibliography/
│   └── references.bib          # Referências bibliográficas
└── README.md                   # Este arquivo
```

## Compilação

### Usando o script (recomendado)

```bash
chmod +x compile.sh
./compile.sh
```

### Manual

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

O PDF será gerado como `main.pdf`.

## Requisitos

- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Pacotes necessários (instalados automaticamente na maioria das distribuições):
  - acmart
  - babel (portuguese)
  - inputenc, fontenc
  - graphicx, booktabs
  - amsmath
  - listings, xcolor
  - hyperref

## Datasets Utilizados

1. **Breast Cancer** (scikit-learn)
   - Task: Binary classification
   - Use case: Robustness testing
   - Samples: 569

2. **Adult Income** (UCI)
   - Task: Binary classification
   - Use case: Fairness testing
   - Samples: 48,842

3. **California Housing** (scikit-learn)
   - Task: Regression
   - Use case: Uncertainty quantification
   - Samples: 20,640

## Notebooks Complementares

Os notebooks práticos estão disponíveis em:
- `notebooks/01_setup_installation.ipynb`
- `notebooks/02_robustness_exercise.ipynb`
- `notebooks/03_fairness_exercise.ipynb`
- `notebooks/04_uncertainty_exercise.ipynb`
- `notebooks/05_integration_exercise.ipynb`

## Limitações

- **Tamanho máximo**: 10 páginas (ACM format)
- **Foco**: Tutorial prático, não pesquisa teórica
- **Escopo**: Validação (não inclui treino ou deployment)

## Contato

Para questões sobre o tutorial ou DeepBridge:
- Email: tutorial@deepbridge.ai
- GitHub: https://github.com/deepbridge/deepbridge
- Documentação: https://deepbridge.readthedocs.io

## Citação

Se utilizar este material, por favor cite:

```bibtex
@inproceedings{deepbridge2024tutorial,
  title={Do Desenvolvimento ao Deploy: Guia Prático de Validação de Modelos de Machine Learning em Produção},
  author={[Autores]},
  booktitle={KDD Tutorial Track},
  year={2024}
}
```

## Licença

Este material está disponível sob licença [a definir].

## Changelog

- **2024-12-06**: Versão inicial criada
  - 6 seções completas
  - 3 hands-on modules
  - 25+ referências bibliográficas
  - Exemplos de código executável
