# Makefile para compilação do paper
# Survey sobre Validação de Modelos ML

# Variáveis
MAIN = main
PDF = $(MAIN).pdf
TEX = $(MAIN).tex
BIB = references.bib

# Compilador LaTeX
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEX_FLAGS = -interaction=nonstopmode

# Alvo padrão
all: $(PDF)

# Compilação completa
$(PDF): $(TEX) $(BIB)
	@echo "===================================="
	@echo "Compilando $(MAIN)..."
	@echo "===================================="
	@echo "[1/5] Primeira compilação..."
	@$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "[2/5] Processando referências..."
	@$(BIBTEX) $(MAIN) || true
	@echo "[3/5] Segunda compilação..."
	@$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "[4/5] Terceira compilação..."
	@$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "[5/5] Verificando resultado..."
	@if [ -f $(PDF) ]; then \
		echo "✓ PDF gerado com sucesso!"; \
		du -h $(PDF); \
		if command -v pdfinfo > /dev/null; then \
			pdfinfo $(PDF) | grep Pages; \
		fi \
	else \
		echo "✗ Erro ao gerar PDF!"; \
		exit 1; \
	fi
	@echo "===================================="
	@echo "Compilação concluída!"
	@echo "===================================="

# Compilação rápida (sem BibTeX)
quick: $(TEX)
	@echo "Compilação rápida..."
	@$(LATEX) $(LATEX_FLAGS) $(TEX)

# Limpar arquivos auxiliares
clean:
	@echo "Limpando arquivos auxiliares..."
	@rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
	@rm -f *.fls *.fdb_latexmk *.synctex.gz *.nav *.snm *.vrb
	@echo "✓ Arquivos auxiliares removidos"

# Limpar tudo (incluindo PDF)
distclean: clean
	@echo "Removendo PDF..."
	@rm -f $(PDF)
	@echo "✓ Limpeza completa"

# Visualizar PDF (Linux)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v evince > /dev/null; then \
		evince $(PDF) &; \
	elif command -v okular > /dev/null; then \
		okular $(PDF) &; \
	else \
		echo "Nenhum visualizador de PDF encontrado"; \
	fi

# Contar palavras (aproximado)
wordcount: $(TEX)
	@echo "Contagem aproximada de palavras:"
	@detex $(TEX) 2>/dev/null | wc -w || \
		echo "Instale 'detex' para contagem de palavras"

# Verificar número de páginas
pages: $(PDF)
	@if command -v pdfinfo > /dev/null; then \
		pdfinfo $(PDF) | grep Pages; \
	else \
		echo "Instale 'poppler-utils' para ver número de páginas"; \
	fi

# Ajuda
help:
	@echo "Targets disponíveis:"
	@echo "  make         - Compilação completa (padrão)"
	@echo "  make quick   - Compilação rápida (sem BibTeX)"
	@echo "  make clean   - Limpar arquivos auxiliares"
	@echo "  make distclean - Limpar tudo (incluindo PDF)"
	@echo "  make view    - Visualizar PDF"
	@echo "  make wordcount - Contar palavras"
	@echo "  make pages   - Verificar número de páginas"
	@echo "  make help    - Mostrar esta ajuda"

.PHONY: all quick clean distclean view wordcount pages help
