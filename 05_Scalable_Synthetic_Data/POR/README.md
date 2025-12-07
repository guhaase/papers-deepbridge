# Geracao Escalavel de Dados Sinteticos com Preservacao de Privacidade

**Paper 5 da série DeepBridge**

## Descricao

Este paper apresenta um framework **escalavel** para geracao de dados sinteticos baseado em Copulas Gaussianas distribuidas via Dask, que processa datasets de 100GB+ preservando qualidade estatistica e privacidade.

**Conferencia alvo**: SIGKDD 2026

**Paginas**: ~10

**Referencias bibliograficas**: ~15

## Estrutura

```
05_Scalable_Synthetic_Data/POR/
├── main.tex                    # Documento principal
├── sections/
│   ├── 01_introduction.tex     # Introducao e motivacao
│   ├── 02_background.tex       # Fundamentos e trabalhos relacionados
│   ├── 03_architecture.tex     # Arquitetura distribuida
│   ├── 04_implementation.tex   # Implementacao memory-efficient
│   ├── 05_evaluation.tex       # Avaliacao experimental
│   ├── 06_discussion.tex       # Discussao e limitacoes
│   └── 07_conclusion.tex       # Conclusao e trabalhos futuros
├── bibliography/
│   └── references.bib          # Referencias bibliograficas
├── acmart.cls                  # Classe LaTeX ACM (corrigida)
├── compile.sh                  # Script de compilacao
└── README.md                   # Este arquivo
```

## Compilacao

### Metodo Manual

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Nota**: Devido a problemas com caracteres UTF-8 especiais em captions de lstlisting,
alguns acentos foram removidos. Para compilar corretamente, certifique-se de que:
- Captions de lstlisting não contenham caracteres acentuados
- Comentarios em codigo Python usem ASCII simples

## Principais Contribuicoes

1. **Arquitetura Distribuida**:
   - Chunk-based processing para datasets beyond-memory
   - Incremental fitting via streaming algorithms
   - Parallel sampling distribuido
   - Memory-efficient: -95% uso de memoria vs baselines

2. **Implementacao Memory-Efficient**:
   - Welford's algorithm: Online mean/variance em O(1) memoria
   - Two-pass covariance: Exato, O(d²) memoria
   - Lazy evaluation via Dask task graphs
   - Adaptive chunking baseado em RAM disponivel

3. **Preservacao de Qualidade e Privacidade**:
   - Statistical similarity: 98% (KS, JSD, correlacao)
   - ML utility: -2pp degradacao (train synth, test real)
   - k-anonymity: Zero copias exatas
   - Differential Privacy: Opcao com epsilon configuravel

4. **Avaliacao Abrangente**:
   - Scalability tests: 1GB a 100GB
   - Comparacao: SDV, CTGAN, TVAE
   - 3 estudos de caso: Healthcare (10M), Finance (50M), E-commerce (100M)

## Resultados Principais

**Escalabilidade**:
- **50x speedup** vs SDV em datasets 100GB
- **-95% memoria**: 8GB vs 64GB+ (baselines)
- **100GB+**: Unica solucao que completa (SDV/CTGAN OOM)
- **115 min** para fit em 100GB

**Qualidade**:
- **98% similarity** em metricas estatisticas
- **-1.9pp ML utility** (87.3% synthetic vs 89.2% real)
- **96% correlation agreement**
- **Comparavel a SDV**, ligeiramente inferior a CTGAN (esperado)

**Privacidade**:
- **k-anonymity > 5**: Nenhuma copia exata
- **NND = 0.17**: Nearest neighbor distance segura
- **DP option**: epsilon=1 aumenta NND para 0.28, degrada utility -3pp

## Metodos Implementados

### Gaussian Copula Synthesis

**Workflow**:
1. **Fit marginais**: Estima distribuicao de cada feature
2. **Compute correlacao**: Matriz de correlacao via two-pass algorithm
3. **Transform to Gaussian**: CDF + inverse normal
4. **Sample**: Normal multivariada + inverse transform

**Vantagens**:
- 10-20x mais rapido que GANs
- -90% memoria
- Mais estavel (sem mode collapse)
- Interpretavel (correlation matrix)

**Limitacoes**:
- Assume dependencia linear (correlacao)
- Nao captura tail dependence assimetrica
- Menos expressivo para padroes nao-lineares complexos

## Estudos de Caso

### Case 1: Healthcare (10M Pacientes)

**Dataset**: Electronic Health Records (EHR), 80 features

**Resultados**:
- Fitting time: 18 min (vs SDV OOM)
- Statistical similarity: KS = 0.028
- ML utility: Readmissao prediction 84% (synth) vs 86% (real) = -2pp
- Privacy: k-anonymity > 5, NND = 0.21

### Case 2: Finance (50M Transacoes)

**Dataset**: Transacoes de cartao de credito, 40 features

**Objetivo**: Data augmentation para fraud detection

**Resultados**:
- Fraud detection F1: 0.72 (real only) → 0.78 (real + synth) = +0.06
- Fitting time: 52 min
- Preservacao de correlacoes: 97% agreement

### Case 3: E-commerce (100M Interacoes)

**Dataset**: User-item interactions, 25 features

**Objetivo**: Compartilhar dados de comportamento sem expor usuarios

**Resultados**:
- Fitting time: 115 min (100GB dataset)
- CTR prediction: 0.89 AUC (synth) vs 0.91 (real) = -0.02
- Privacy: Zero copies, NND = 0.19

## Quando Usar Copula vs Deep Learning

**Gaussian Copula (DeepBridge)** ideal para:
- Dados tabulares: Features numericas e categoricas misturadas
- Correlacoes lineares: Relacoes primariamente lineares
- Escalabilidade: Datasets > 10GB
- Interpretabilidade: Correlation matrix transparente
- Velocidade: Fit/sample rapido

**Deep Learning (CTGAN/TVAE)** preferivel para:
- Relacoes complexas: Non-linear, interacoes de alta ordem
- Imagens/Text: Dados nao-tabulares
- Datasets pequenos/medios: < 5GB com GPU disponivel
- Maxima qualidade: Disposto a trocar tempo por quality

**Recomendacao**: Para dados tabulares > 10GB, Gaussian Copula. Para padroes complexos < 5GB, CTGAN.

## Privacy-Utility Trade-Off

**Spectrum**:
1. **Sem DP**: Maxima utility, privacidade basica (k-anonymity)
2. **DP baixo** (ε=10): Utility -1pp, privacy moderada
3. **DP medio** (ε=1): Utility -3pp, privacy forte
4. **DP alto** (ε=0.1): Utility -10pp+, privacy muito forte

**Guideline**:
- Research sharing: ε=1 (balanco razoavel)
- Public release: ε=0.1 (conservador)
- Internal testing: Sem DP (k-anonymity suficiente)

## Implementacao

### Stack Tecnologico

**Core Dependencies**:
- Dask: Distributed computing
- NumPy/Pandas: Numerical operations
- SciPy: Statistical distributions, linear algebra
- Scikit-learn: ML utility evaluation

### Algoritmos Memory-Efficient

**Welford's Online Algorithm**:
- Compute mean e variance em one pass
- O(1) memoria por feature

**Two-Pass Covariance**:
- Pass 1: Compute means
- Pass 2: Compute covariance matrix
- O(d²) memoria

### Otimizacoes

1. **Correlation Matrix Sparsification**: Threshold |ρ| < 0.05 → 0
2. **Parallel Inverse Transform**: Paralelizado por feature
3. **Adaptive Chunk Sizing**: Auto-tune baseado em RAM disponivel

## Boas Praticas

1. **Escolha Chunk Size Apropriado**: Regra: 10% da RAM disponivel
2. **Validacao de Qualidade**: Sempre compute metricas (KS, JSD)
3. **Privacy Assessment**: Compute NND, verifique k-anonymity
4. **Iterative Refinement**: Se quality baixa, investigue features problematics
5. **Documentacao**: Report method, metrics, privacy guarantees

## Limitacoes e Trabalhos Futuros

**Limitacoes Atuais**:
- Correlacoes nao-lineares: Assume dependencia linear
- Categorical high cardinality: 1000+ categorias problematico
- Temporal dependencies: Nao captura series temporais
- Rare events: Freq < 0.1% podem nao aparecer

**Trabalhos Futuros**:
1. Vine Copulas: Relacoes nao-lineares
2. GPU Acceleration: cuDF para 5-10x speedup
3. Federated Synthesis: Multi-party data sem centralizacao
4. Conditional Sampling: Sample com constraints
5. Time Series Support: VAR + Copula hybrid

## Comparacao: Copula vs GAN

| Aspecto | Gaussian Copula | CTGAN |
|---------|----------------|-------|
| Fitting time (10GB) | 12 min | 240 min |
| Memory (10GB) | 4GB | 48GB |
| Quality (KS) | 0.024 | 0.019 |
| ML Utility degradation | -1.9pp | -1.1pp |
| Max dataset size | 100GB+ | ~5GB |
| Interpretabilidade | Alta | Baixa |
| Hyperparameter tuning | Minimo | Extensivo |
| Stability | Alta | Media |

**Takeaway**: Copula e 10-20x mais rapido e -90% memoria, com 5-10% menos quality.

## Integracao com DeepBridge

```python
from deepbridge import Experiment, DBDataset
from deepbridge.synthetic import GaussianCopulaSynthesizer

dataset = DBDataset(df, target='label', model=model)

# Generate synthetic
synthesizer = GaussianCopulaSynthesizer()
synthesizer.fit(dataset.data)
synthetic_df = synthesizer.sample(n_rows=10000)

# Validate synthetic quality
exp = Experiment(
    dataset=dataset,
    tests=['synthetic_quality'],
    synthetic_data=synthetic_df
)
results = exp.run_tests()
```

## Dependencias

- LaTeX (TeXLive 2020+)
- Pacotes: acmart, babel (portuguese), listings, graphicx, booktabs, amsmath
- BibTeX

## Notas Tecnicas

- Arquivo acmart.cls inclui correcao para bug hyperref/hyperxmp
- Compilacao requer 3 passes pdflatex + 1 bibtex
- Caracteres acentuados em captions de lstlisting podem causar erros UTF-8
- Para ambiente de producao, use encoding ASCII em comentarios de codigo

## Autores

Paper desenvolvido como parte da serie DeepBridge sobre validacao de modelos ML.

## Licenca

Conteudo academico - todos os direitos reservados aos autores.
