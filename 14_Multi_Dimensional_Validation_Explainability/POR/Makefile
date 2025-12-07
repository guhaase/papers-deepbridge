# Makefile para Paper 14 - Multi-Dimensional Validation with Explainability
# Uso:
#   make        - Compila o paper
#   make clean  - Remove arquivos temporarios
#   make view   - Abre o PDF (Linux)
#   make count  - Conta paginas do PDF

.PHONY: all clean view count help

# Arquivo principal
MAIN = main
PDF = $(MAIN).pdf
TEX = $(MAIN).tex

# Comando LaTeX
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

all: $(PDF)

$(PDF): $(TEX) bibliography/references.bib sections/*.tex
	@echo "========================================="
	@echo "Compilando Paper 14: Multi-Dimensional Validation"
	@echo "========================================="
	@echo ""
	@echo "[1/5] Primeira compilacao..."
	@$(LATEX) $(MAIN) > /dev/null 2>&1 || (echo "ERRO! Veja $(MAIN).log"; exit 1)
	@echo "[2/5] Processando bibliografia..."
	@$(BIBTEX) $(MAIN) > /dev/null 2>&1 || echo "AVISO: bibtex warnings"
	@echo "[3/5] Segunda compilacao..."
	@$(LATEX) $(MAIN) > /dev/null 2>&1 || (echo "ERRO! Veja $(MAIN).log"; exit 1)
	@echo "[4/5] Terceira compilacao..."
	@$(LATEX) $(MAIN) > /dev/null 2>&1 || (echo "ERRO! Veja $(MAIN).log"; exit 1)
	@echo ""
	@echo "========================================="
	@echo "✓ Compilacao concluida!"
	@echo "========================================="
	@echo "PDF gerado: $(PDF)"
	@if command -v du >/dev/null 2>&1; then \
		echo "Tamanho: $$(du -h $(PDF) | cut -f1)"; \
	fi
	@if command -v pdfinfo >/dev/null 2>&1; then \
		echo "Paginas: $$(pdfinfo $(PDF) 2>/dev/null | grep Pages | awk '{print $$2}')"; \
	fi
	@echo ""

clean:
	@echo "Limpando arquivos temporarios..."
	@rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	@rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk
	@rm -f $(MAIN).synctex.gz
	@echo "✓ Arquivos temporarios removidos"

cleanall: clean
	@echo "Removendo PDF..."
	@rm -f $(PDF)
	@echo "✓ PDF removido"

view: $(PDF)
	@echo "Abrindo PDF..."
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(PDF) & \
	elif command -v open >/dev/null 2>&1; then \
		open $(PDF); \
	else \
		echo "ERRO: Nao foi possivel encontrar comando para abrir PDF"; \
		exit 1; \
	fi

count: $(PDF)
	@if command -v pdfinfo >/dev/null 2>&1; then \
		echo "Numero de paginas: $$(pdfinfo $(PDF) 2>/dev/null | grep Pages | awk '{print $$2}')"; \
	else \
		echo "ERRO: pdfinfo nao encontrado. Instale com: apt-get install poppler-utils"; \
		exit 1; \
	fi

help:
	@echo "Makefile para Paper 14 - Multi-Dimensional Validation"
	@echo ""
	@echo "Uso:"
	@echo "  make          - Compila o paper (padrao)"
	@echo "  make clean    - Remove arquivos temporarios (mantem PDF)"
	@echo "  make cleanall - Remove arquivos temporarios e PDF"
	@echo "  make view     - Abre o PDF gerado"
	@echo "  make count    - Conta numero de paginas"
	@echo "  make help     - Mostra esta mensagem"
	@echo ""
