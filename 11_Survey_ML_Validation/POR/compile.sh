#!/bin/bash

# Script de compilação do paper
# Survey sobre Validação de Modelos ML

echo "=================================="
echo "Compilando Paper: Survey ML Validation"
echo "=================================="

# Limpar arquivos auxiliares anteriores
echo ""
echo "[1/6] Limpando arquivos auxiliares..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Primeira compilação
echo ""
echo "[2/6] Primeira compilação (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERRO: Primeira compilação falhou!"
    exit 1
fi

# Compilar referências
echo ""
echo "[3/6] Compilando referências (bibtex)..."
bibtex main
if [ $? -ne 0 ]; then
    echo "AVISO: BibTeX reportou problemas (pode ser normal se não houver citações)"
fi

# Segunda compilação (resolver referências)
echo ""
echo "[4/6] Segunda compilação (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERRO: Segunda compilação falhou!"
    exit 1
fi

# Terceira compilação (garantir todas referências)
echo ""
echo "[5/6] Terceira compilação (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERRO: Terceira compilação falhou!"
    exit 1
fi

# Verificar resultado
echo ""
echo "[6/6] Verificando PDF gerado..."
if [ -f main.pdf ]; then
    # Obter informações do PDF
    file_size=$(du -h main.pdf | cut -f1)

    # Tentar obter número de páginas (se pdfinfo estiver disponível)
    if command -v pdfinfo &> /dev/null; then
        num_pages=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        echo "✓ PDF gerado com sucesso!"
        echo "  Arquivo: main.pdf"
        echo "  Tamanho: $file_size"
        echo "  Páginas: $num_pages"

        # Verificar se está dentro do limite de 10 páginas
        if [ ! -z "$num_pages" ] && [ "$num_pages" -gt 10 ]; then
            echo ""
            echo "⚠ AVISO: Paper tem $num_pages páginas (máximo: 10)"
            echo "  Por favor, reduza o conteúdo para atender o limite."
        fi
    else
        echo "✓ PDF gerado com sucesso!"
        echo "  Arquivo: main.pdf"
        echo "  Tamanho: $file_size"
        echo "  (Instale 'poppler-utils' para ver número de páginas)"
    fi

    # Limpar arquivos auxiliares (opcional)
    echo ""
    read -p "Deseja limpar arquivos auxiliares? (s/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "Limpando arquivos auxiliares..."
        rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
        echo "✓ Arquivos auxiliares removidos"
    fi
else
    echo "✗ ERRO: PDF não foi gerado!"
    echo "Verifique os erros acima."
    exit 1
fi

echo ""
echo "=================================="
echo "Compilação concluída!"
echo "=================================="
