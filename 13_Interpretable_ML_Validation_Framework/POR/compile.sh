#!/bin/bash

# Script para compilar o paper LaTeX

echo "Compilando paper..."

# Primeira compilacao (gera arquivos auxiliares)
pdflatex -interaction=nonstopmode main.tex

# Compilar bibliografia
bibtex main

# Segunda compilacao (resolve referencias)
pdflatex -interaction=nonstopmode main.tex

# Terceira compilacao (finaliza referencias cruzadas)
pdflatex -interaction=nonstopmode main.tex

# Limpar arquivos auxiliares (opcional)
# rm -f *.aux *.log *.out *.bbl *.blg

echo "Compilacao concluida! PDF gerado: main.pdf"

# Mostrar numero de paginas
if command -v pdfinfo &> /dev/null; then
    PAGES=$(pdfinfo main.pdf | grep "Pages:" | awk '{print $2}')
    echo "Numero de paginas: $PAGES"
    if [ "$PAGES" -gt 10 ]; then
        echo "AVISO: Paper tem mais de 10 paginas!"
    fi
fi

# Abrir PDF (se estiver em ambiente com display)
if [ -n "$DISPLAY" ]; then
    xdg-open main.pdf 2>/dev/null &
fi
