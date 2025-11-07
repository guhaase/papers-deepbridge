# Makefile for Article 1: Are Historical Redlining Effects Immutable?
# Author: Gustavo Coelho Haase
# Date: January 2025

# Main file name (without extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Output directory
BUILDDIR = build

# Flags
LATEXFLAGS = -interaction=nonstopmode -output-directory=$(BUILDDIR)

# Default target - always rebuild
.PHONY: all
all: clean-pdf
	@echo "Starting fresh compilation..."
	$(MAKE) build

# Create build directory if it doesn't exist
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Incremental build (checks timestamps)
.PHONY: build
build: $(BUILDDIR)/$(MAIN).pdf

# Compile the document
$(BUILDDIR)/$(MAIN).pdf: $(MAIN).tex sections/*.tex bibliography/references.bib | $(BUILDDIR)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	cp *.bst $(BUILDDIR)/ 2>/dev/null || true
	cp bibliography/references.bib $(BUILDDIR)/ 2>/dev/null || true
	cd $(BUILDDIR) && $(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "Compilation complete: $(BUILDDIR)/$(MAIN).pdf"

# Remove only the PDF to force rebuild
.PHONY: clean-pdf
clean-pdf:
	@rm -f $(BUILDDIR)/$(MAIN).pdf

# Quick compile (no bibliography update)
.PHONY: quick
quick: | $(BUILDDIR)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean auxiliary files
.PHONY: clean
clean:
	rm -f $(BUILDDIR)/*.aux $(BUILDDIR)/*.log $(BUILDDIR)/*.out
	rm -f $(BUILDDIR)/*.bbl $(BUILDDIR)/*.blg $(BUILDDIR)/*.toc
	rm -f $(BUILDDIR)/*.lot $(BUILDDIR)/*.lof $(BUILDDIR)/*.idx
	rm -f $(BUILDDIR)/*.synctex.gz

# Clean everything including PDF
.PHONY: distclean
distclean: clean
	rm -f $(BUILDDIR)/$(MAIN).pdf

# View the PDF
.PHONY: view
view: $(BUILDDIR)/$(MAIN).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(BUILDDIR)/$(MAIN).pdf; \
	elif command -v open > /dev/null; then \
		open $(BUILDDIR)/$(MAIN).pdf; \
	else \
		echo "Cannot detect PDF viewer. Please open $(BUILDDIR)/$(MAIN).pdf manually."; \
	fi

# Count words (approximate, excludes LaTeX commands)
.PHONY: wordcount
wordcount:
	@echo "Approximate word count (excludes LaTeX commands):"
	@detex sections/*.tex | wc -w

# Check for TODO items
.PHONY: todos
todos:
	@echo "Remaining TODOs:"
	@grep -r "TODO" sections/ || echo "No TODOs found!"

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make          - Force full recompilation (always rebuilds)"
	@echo "  make build    - Incremental build (only if sources changed)"
	@echo "  make quick    - Quick compile (no bibliography update)"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean- Remove all generated files including PDF"
	@echo "  make view     - Open the compiled PDF"
	@echo "  make wordcount- Approximate word count"
	@echo "  make todos    - List remaining TODO items"
	@echo "  make help     - Show this help message"
