#!/bin/bash

# Script para compilar o paper LaTeX
# Uso: ./compile.sh

echo "=== Compilando Paper 16: XAI-Driven Distillation ==="

# Limpar arquivos auxiliares antigos
echo "Limpando arquivos auxiliares..."
rm -f main.aux main.bbl main.blg main.log main.out main.toc

# Primeira compilacao
echo "Primeira compilacao (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex

# Compilar bibliografia
echo "Compilando bibliografia (bibtex)..."
bibtex main

# Segunda compilacao (para referencias)
echo "Segunda compilacao (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex

# Terceira compilacao (para garantir referencias corretas)
echo "Terceira compilacao (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex

# Verificar se PDF foi gerado
if [ -f main.pdf ]; then
    echo ""
    echo "=== Compilacao concluida com sucesso! ==="
    echo "PDF gerado: main.pdf"

    # Contar paginas
    PAGES=$(pdfinfo main.pdf | grep "Pages:" | awk '{print $2}')
    echo "Numero de paginas: $PAGES"

    if [ "$PAGES" -gt 10 ]; then
        echo "AVISO: O paper tem mais de 10 paginas! ($PAGES paginas)"
        echo "Por favor, reduza o conteudo para atender o limite de 10 paginas."
    else
        echo "OK: Paper esta dentro do limite de 10 paginas."
    fi

    # Abrir PDF (se estiver em ambiente com display)
    if [ -n "$DISPLAY" ]; then
        xdg-open main.pdf 2>/dev/null || open main.pdf 2>/dev/null || echo "PDF gerado, mas nao foi possivel abrir automaticamente."
    fi
else
    echo ""
    echo "=== ERRO: Compilacao falhou! ==="
    echo "Verifique o arquivo main.log para detalhes."
fi

echo ""
echo "=== Fim da compilacao ==="
