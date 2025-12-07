# Detecção de Weakspots em Modelos de Machine Learning

**Paper 4 da série DeepBridge**

## Descrição

Este paper apresenta um framework sistemático para **detecção automática de weakspots** em modelos de Machine Learning—regiões do espaço de features onde performance degrada significativamente. Nossa abordagem combina três estratégias complementares de slicing (quantile-based, uniform, tree-based) com classificação de severidade e análise de interações.

**Conferência alvo**: AISTATS 2026

**Páginas**: 10 (máximo)

**Referências bibliográficas**: 10

## Estrutura

```
04_Weakspot_Detection/POR/
├── main.tex                    # Documento principal
├── sections/
│   ├── 01_introduction.tex     # Introdução e motivação
│   ├── 02_background.tex       # Trabalhos relacionados
│   ├── 03_framework.tex        # Framework de detecção
│   ├── 04_strategies.tex       # Estratégias de slicing
│   ├── 05_evaluation.tex       # Avaliação experimental
│   ├── 06_discussion.tex       # Discussão e limitações
│   └── 07_conclusion.tex       # Conclusão e trabalhos futuros
├── bibliography/
│   └── references.bib          # Referências bibliográficas
├── acmart.cls                  # Classe LaTeX ACM (corrigida)
├── compile.sh                  # Script de compilação
└── README.md                   # Este arquivo
```

## Compilação

### Método 1: Script automatizado (recomendado)

```bash
chmod +x compile.sh
./compile.sh
```

O script executa:
1. Limpeza de arquivos temporários
2. Compilação LaTeX (3 passes + BibTeX)
3. Verificação de páginas

### Método 2: Manual

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Principais Contribuições

1. **Framework Multi-Estratégia**:
   - Quantile-based slicing: Robusto a distribuições skewed
   - Uniform slicing: Interpretável para features com domínio conhecido
   - Tree-based slicing: Data-driven, descobre boundaries ótimos
   - Interaction detection: Weakspots multi-dimensionais

2. **Classificação de Severidade**:
   - Thresholds configuráveis (Low/Medium/High/Critical)
   - Testes de significância estatística
   - Requisitos de tamanho mínimo de amostra
   - Filtros anti-overfitting

3. **Avaliação Abrangente**:
   - Datasets sintéticos: 94% F1 na detecção
   - 8 datasets reais: 127 weakspots detectados
   - 3 estudos de caso (credit, healthcare, fraud)
   - Comparação com baselines: 2.8x cobertura, 80x speedup

## Resultados Principais

**Eficácia de Detecção**:
- **94% F1 score** em datasets sintéticos
- **127 weakspots únicos** em 8 datasets reais
- **Degradações de até 35pp** em subgrupos específicos
- **6% false discovery rate** (vs. 12-22% em baselines)

**Eficiência Computacional**:
- **80x speedup**: 3 min vs. 4 horas (análise manual)
- **2.8x cobertura**: 127 vs. 45 weakspots (manual)
- **Escalável**: 0.5-8 min para datasets de 300 a 500K amostras

**Insights de Produção**:
- **Credit scoring**: Degradação de 28pp em jovens de baixa renda
- **Healthcare**: AUC 21pp menor para jovens obesos
- **Fraud detection**: F1 30pp menor em transações grandes de madrugada

## Estratégias de Slicing

### Quantile-Based
- Divide features em quantis (P10, P25, P50, P75, P90)
- Vantagem: Tamanhos equilibrados, robusto a outliers
- Uso: Exploração inicial, features com distribuição desconhecida

### Uniform
- Bins uniformes no range da feature
- Vantagem: Interpretação direta, útil para features com semântica de range
- Uso: Features com domínio conhecido (idade, score)

### Tree-Based
- Decision tree identifica splits ótimos
- Vantagem: Descobre interações automaticamente
- Uso: Descoberta de boundaries não-lineares

**Recomendação**: Use estratégia combinada (padrão do framework) para máxima cobertura.

## Estudos de Caso

### Credit Scoring (German Credit)
- **Weakspot crítico**: Jovens (<25) com empréstimos longos (>24 meses)
- **Degradação**: 48% accuracy vs. 76% global (28pp)
- **Ação**: Data augmentation + feature engineering

### Healthcare (Diabetes)
- **Weakspot crítico**: Jovens (<30) obesos (BMI>35)
- **Degradação**: AUC 0.62 vs. 0.83 global (21pp)
- **Preocupação**: Potencial bias demográfico

### Fraud Detection
- **Weakspot crítico**: Transações grandes (>\$10K) de madrugada (2-5AM)
- **Degradação**: F1 0.58 vs. 0.88 global (30pp)
- **Ação**: Oversampling + ensemble model

## Verificação

Para verificar se o PDF foi gerado corretamente:

```bash
# Verificar número de páginas
pdfinfo main.pdf | grep Pages

# Verificar referências
grep "\\bibitem" main.bbl | wc -l

# Visualizar PDF
xdg-open main.pdf  # Linux
# ou
open main.pdf      # macOS
```

**Saída esperada**:
- Pages: 10
- Referências: ~10

## Integração com DeepBridge

Weakspot detection é uma das 5 dimensões do DeepBridge (Paper 3):

```python
from deepbridge import Experiment, DBDataset

dataset = DBDataset(X, y, model=model)

exp = Experiment(
    dataset=dataset,
    tests=['robustness', 'weakspots'],  # Inclui weakspot detection
    config='medium'
)

results = exp.run_tests()

# Weakspots detectados automaticamente
print(results.weakspots.summary)
```

Também disponível standalone:

```python
from deepbridge.weakspots import WeakspotDetector

detector = WeakspotDetector(
    model=trained_model,
    metric='accuracy',
    strategies=['quantile', 'uniform', 'tree']
)

weakspots = detector.detect(X_test, y_test)
```

## Dependências

- LaTeX (TeXLive 2020 ou superior)
- Pacotes: acmart, babel (portuguese), listings, graphicx, booktabs, amsmath
- BibTeX

## Notas

- O arquivo `acmart.cls` inclui correção para o bug de ordem de carregamento hyperref/hyperxmp
- Compilação requer 3 passes do pdflatex + 1 do bibtex
- Alguns avisos de referências faltantes são esperados (referências não adicionadas ainda ao .bib)

## Limitações e Trabalhos Futuros

**Limitações Atuais**:
- Features categóricas com alta cardinalidade (>100 categorias)
- Datasets muito pequenos (n < 1000)
- Interações limitadas a 2-way
- Otimizado para modelos tabulares

**Trabalhos Futuros**:
1. Automated remediation (sugerir fixes automaticamente)
2. Deep Learning support (slicing em embedding spaces)
3. Temporal tracking (detectar novos weakspots via drift)
4. Causal analysis (identificar root causes)
5. Multi-model comparison (selecionar modelo com menos weakspots)

## Autores

Paper desenvolvido como parte da série DeepBridge sobre validação de modelos ML.

## Licença

Conteúdo acadêmico - todos os direitos reservados aos autores.
