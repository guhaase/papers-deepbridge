# Paper 2: DeepBridge Fairness Framework

**Título**: DeepBridge Fairness: Da Pesquisa à Regulação -- Um Framework Pronto para Produção para Teste de Fairness Algorítmica

**Conferência Alvo**: FAccT 2026

**Status**: Draft completo (17 páginas)

## Estrutura do Paper

### Arquivos Principais
- `main.tex` - Documento principal (ACM sigconf format)
- `main.pdf` - PDF compilado final (17 páginas)
- `acmart.cls` - Classe LaTeX ACM

### Seções (em português)
1. `sections/01_introduction.tex` - Introdução e contribuições
2. `sections/02_related_work.tex` - Background e trabalhos relacionados
3. `sections/03_architecture.tex` - Arquitetura do DeepBridge Fairness
4. `sections/04_case_studies.tex` - 4 estudos de caso (COMPAS, German Credit, Adult, Healthcare)
5. `sections/05_evaluation.tex` - Avaliação (usabilidade, performance, métricas)
6. `sections/06_discussion.tex` - Discussão (limitações, ética, boas práticas)
7. `sections/07_conclusion.tex` - Conclusão e trabalhos futuros

### Bibliografia
- `bibliography/references.bib` - Todas as referências (740 linhas, ~60 citações)

### Figuras
- `figures/architecture_simple.tex` - Diagrama de arquitetura (TikZ)
- `figures/architecture_simple.pdf` - Diagrama compilado

## Compilação

### Método 1: Script Automatizado (Recomendado)

```bash
./compile.sh
```

Este script:
- Limpa arquivos temporários
- Compila 3 vezes (necessário para referências)
- Processa bibliografia com BibTeX
- Verifica se PDF tem 17 páginas e 24 referências
- Limpa arquivos temporários ao final

### Método 2: Manual

```bash
# Limpar arquivos antigos
rm -f main.aux main.bbl main.blg main.log main.out

# Primeira compilação (gera .aux)
pdflatex main.tex

# Processar bibliografia (gera .bbl)
bibtex main

# Segunda compilação (inclui bibliografia)
pdflatex main.tex

# Terceira compilação (resolve referências cruzadas)
pdflatex main.tex

# Verificar resultado
pdfinfo main.pdf | grep Pages
# Deve mostrar: Pages: 17
```

**⚠️ Importante**: É necessário compilar **3 vezes** após executar `bibtex` para que as referências bibliográficas apareçam corretamente no PDF.

### Verificar Referências Bibliográficas

```bash
# Ver número de páginas (deve ser 17)
pdfinfo main.pdf | grep Pages

# Contar referências (deve ser 24)
pdftotext main.pdf - | grep "^\[" | wc -l

# Ver primeiras referências
pdftotext -f 16 -l 17 main.pdf - | grep "^\[" | head -5
```

**Saída esperada**:
```
[1] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. Machine bias.
[2] Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and machine...
[3] Rachel KE Bellamy, Kuntal Dey, Michael Hind, et al. AI Fairness 360...
...
```

As referências aparecem nas **páginas 16-17** do PDF, listadas numericamente de [1] a [24].

## Contribuições Principais

1. **15 métricas integradas** (4 pré + 11 pós-treinamento)
2. **Auto-detecção de atributos sensíveis** (F1: 0.90)
3. **Verificação EEOC/ECOA automática** (regra 80%, Question 21)
4. **Otimização de threshold** (Pareto frontier)
5. **Visualizações audit-ready** (6 tipos)

## Resultados

- **SUS Score**: 85.2 (top 15%, "excelente")
- **Speedup**: 2.9x vs. ferramentas manuais
- **Cobertura**: 87% mais métricas que AI Fairness 360
- **Taxa de sucesso**: 95% (estudo com 20 participantes)

## Correções Aplicadas

✅ **Problema do hyperref/hyperxmp resolvido**:
- Corrigido bug da classe `acmart.cls` (v1.79) que carregava pacotes na ordem errada
- `hyperref` agora é carregado antes de `hyperxmp` (linha 494 do acmart.cls)

✅ **Bibliografia completa**:
- Adicionado `mitchell2019model` - Model Cards for Model Reporting (FAT* 2019)
- Corrigido `buolamwini2018gender` - mudado de @article para @inproceedings

✅ **Compilação sem erros**: PDF gera perfeitamente com 17 páginas
