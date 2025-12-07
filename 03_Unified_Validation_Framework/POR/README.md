# DeepBridge: Um Framework Unificado para Validação Abrangente de Modelos de Machine Learning

**Paper 3 da série DeepBridge**

## Descrição

Este paper apresenta o **DeepBridge**, o primeiro framework unificado para validação multi-dimensional de modelos de Machine Learning. O framework integra 5 dimensões de validação (robustez, fairness, incerteza, resiliência, hiperparâmetros) em uma API consistente tipo scikit-learn.

**Conferência alvo**: MLSys 2026

**Páginas**: 8

**Referências bibliográficas**: 18

## Estrutura

```
03_Unified_Validation_Framework/POR/
├── main.tex                    # Documento principal
├── sections/
│   ├── 01_introduction.tex     # Introdução e motivação
│   ├── 02_background.tex       # Trabalhos relacionados
│   ├── 03_architecture.tex     # Arquitetura do DeepBridge
│   ├── 04_implementation.tex   # Detalhes de implementação
│   ├── 05_validation.tex       # Estudos de validação
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
3. Verificação de páginas e referências

### Método 2: Manual

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Principais Contribuições

1. **Framework Unificado**: Integra 5 dimensões de validação em API única
   - DBDataset: Container com auto-inferência
   - Experiment: Orquestrador com execução paralela
   - 5 Suítes integradas: Robustness, Uncertainty, Resilience, Fairness, Hyperparameters

2. **Otimizações de Performance**:
   - Lazy loading: -42% uso de memória
   - Execução paralela: -40% tempo total
   - 6.2x speedup vs. ferramentas manuais

3. **Avaliação Empírica**:
   - 4 estudos de caso (credit, healthcare, e-commerce, fraud)
   - Comparação com 5+ ferramentas especializadas
   - Estudo de usabilidade (SUS Score 87.5)

## Resultados Principais

- **89% redução** no tempo de validação (17 min vs. 150 min)
- **3.2x mais dimensões** testadas por organizações
- **2.4x mais problemas** detectados (127 vs. 53 issues)
- **SUS Score 87.5** (top 10%, classificação "excelente")

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
- Pages: 8
- Referências: 18

## Dependências

- LaTeX (TeXLive 2020 ou superior)
- Pacotes: acmart, babel (portuguese), listings, graphicx, booktabs, amsmath
- BibTeX

## Notas

- O arquivo `acmart.cls` inclui correção para o bug de ordem de carregamento hyperref/hyperxmp
- Compilação requer 3 passes do pdflatex + 1 do bibtex para resolver referências
- Avisos sobre "empty journal" em algumas referências são esperados (proceedings não têm journal)

## Autores

Paper desenvolvido como parte da série DeepBridge sobre validação de modelos ML.

## Licença

Conteúdo acadêmico - todos os direitos reservados aos autores.
