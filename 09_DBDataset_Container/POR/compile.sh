#!/bin/bash

# Script de compilacao automatica do paper DBDataset
# Autor: [A definir]
# Data: Dezembro 2025

set -e  # Exit on error

echo "========================================="
echo "Compilando Paper DBDataset"
echo "========================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funcao para mensagens
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar se LaTeX esta instalado
if ! command -v pdflatex &> /dev/null; then
    error "pdflatex nao encontrado. Instale texlive-full:"
    echo "  sudo apt-get install texlive-full"
    exit 1
fi

# Verificar se bibtex esta instalado
if ! command -v bibtex &> /dev/null; then
    error "bibtex nao encontrado. Instale texlive-full:"
    echo "  sudo apt-get install texlive-full"
    exit 1
fi

# Limpar arquivos antigos
info "Limpando arquivos auxiliares antigos..."
rm -f main.aux main.log main.out main.bbl main.blg main.toc main.lof main.lot

# Primeira compilacao
info "Primeira compilacao (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    error "Erro na primeira compilacao. Verifique main.log para detalhes."
    exit 1
}

# Compilar bibliografia
info "Compilando bibliografia (bibtex)..."
bibtex main > /dev/null 2>&1 || {
    warn "Aviso: bibtex reportou warnings. Continuando..."
}

# Segunda compilacao (resolver referencias)
info "Segunda compilacao (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    error "Erro na segunda compilacao. Verifique main.log para detalhes."
    exit 1
}

# Terceira compilacao (garantir referencias corretas)
info "Terceira compilacao (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    error "Erro na terceira compilacao. Verifique main.log para detalhes."
    exit 1
}

# Verificar se PDF foi gerado
if [ ! -f main.pdf ]; then
    error "PDF nao foi gerado!"
    exit 1
fi

# Informacoes sobre o PDF gerado
info "Compilacao concluida com sucesso!"
echo ""
echo "========================================="
echo "Informacoes do PDF:"
echo "========================================="
echo "Arquivo: main.pdf"
echo "Tamanho: $(du -h main.pdf | cut -f1)"

if command -v pdfinfo &> /dev/null; then
    PAGES=$(pdfinfo main.pdf | grep Pages | awk '{print $2}')
    echo "Paginas: $PAGES"
else
    warn "pdfinfo nao encontrado. Instale poppler-utils para ver numero de paginas."
fi

# Opcao para limpar arquivos auxiliares
echo ""
read -p "Limpar arquivos auxiliares? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Limpando arquivos auxiliares..."
    rm -f main.aux main.log main.out main.bbl main.blg main.toc main.lof main.lot
    info "Arquivos auxiliares removidos."
fi

echo ""
echo "========================================="
echo "✓ Paper compilado: main.pdf"
echo "========================================="
