#!/bin/bash

# Script de compilacao para Paper 07 - Threshold Optimization for Fairness
# Compila o documento LaTeX em PDF usando pdflatex e bibtex

set -e  # Para em caso de erro

echo "========================================="
echo "Compilando Paper 07: Threshold Optimization"
echo "========================================="

# Remove arquivos temporarios anteriores
echo ""
echo "[1/5] Limpando arquivos temporarios anteriores..."
rm -f main.aux main.bbl main.blg main.log main.out main.pdf

# Primeira compilacao - gera referencias
echo ""
echo "[2/5] Primeira compilacao (gerando referencias)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "ERRO na primeira compilacao! Veja main.log para detalhes"
    tail -n 30 main.log
    exit 1
}

# Processa bibliografia
echo ""
echo "[3/5] Processando bibliografia com bibtex..."
bibtex main > /dev/null 2>&1 || {
    echo "AVISO: bibtex gerou warnings (normal se nao houver citacoes)"
}

# Segunda compilacao - incorpora referencias
echo ""
echo "[4/5] Segunda compilacao (incorporando referencias)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "ERRO na segunda compilacao! Veja main.log para detalhes"
    tail -n 30 main.log
    exit 1
}

# Terceira compilacao - finaliza cross-references
echo ""
echo "[5/5] Terceira compilacao (finalizando cross-references)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "ERRO na terceira compilacao! Veja main.log para detalhes"
    tail -n 30 main.log
    exit 1
}

# Verifica se PDF foi gerado
if [ -f "main.pdf" ]; then
    echo ""
    echo "========================================="
    echo "✓ Compilacao concluida com sucesso!"
    echo "========================================="
    echo ""
    echo "PDF gerado: main.pdf"
    echo "Tamanho: $(du -h main.pdf | cut -f1)"
    echo "Paginas: $(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ ERRO: PDF nao foi gerado!"
    echo "========================================="
    echo "Verifique main.log para detalhes do erro"
    exit 1
fi

# Limpa arquivos auxiliares (opcional)
echo "Deseja limpar arquivos auxiliares? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Limpando arquivos auxiliares..."
    rm -f main.aux main.bbl main.blg main.log main.out
    echo "✓ Arquivos auxiliares removidos"
fi

echo ""
echo "Concluido!"
