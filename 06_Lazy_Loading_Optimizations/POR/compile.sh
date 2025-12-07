#!/bin/bash
# Script de compilacao do paper DeepBridge Lazy Loading Optimizations
# Garante que as referencias bibliograficas sejam processadas corretamente

echo "🔨 Iniciando compilação do paper..."
echo ""

# Limpar arquivos temporários anteriores
echo "🧹 Limpando arquivos temporários..."
rm -f main.aux main.bbl main.blg main.log main.out main.pdf
echo "✅ Limpeza concluída"
echo ""

# Primeira compilação (gera .aux com citações)
echo "📝 Primeira compilação LaTeX..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Primeira compilação OK"
else
    echo "❌ Erro na primeira compilação"
    exit 1
fi
echo ""

# Processar bibliografia (gera .bbl)
echo "📚 Processando bibliografia..."
bibtex main 2>&1 | grep -v "^The "
if [ $? -eq 0 ]; then
    echo "✅ Bibliografia processada"
else
    echo "❌ Erro ao processar bibliografia"
    exit 1
fi
echo ""

# Segunda compilação (inclui bibliografia)
echo "📝 Segunda compilação LaTeX..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Segunda compilação OK"
else
    echo "❌ Erro na segunda compilação"
    exit 1
fi
echo ""

# Terceira compilação (resolve referências cruzadas)
echo "📝 Terceira compilação LaTeX (final)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Terceira compilação OK"
else
    echo "❌ Erro na terceira compilação"
    exit 1
fi
echo ""

# Verificar resultado
if [ -f "main.pdf" ]; then
    PAGES=$(pdfinfo main.pdf | grep "Pages:" | awk '{print $2}')
    SIZE=$(ls -lh main.pdf | awk '{print $5}')
    echo "🎉 Compilação completa bem-sucedida!"
    echo ""
    echo "📄 PDF gerado: main.pdf"
    echo "   - Páginas: $PAGES"
    echo "   - Tamanho: $SIZE"
    echo ""

    # Verificar se referências foram incluídas
    REF_COUNT=$(grep "\\\\bibitem" main.bbl | wc -l)
    echo "📚 Referências bibliográficas: $REF_COUNT"

    if [ "$PAGES" -eq 10 ]; then
        echo "✅ Paper completo (10 páginas - máximo permitido)!"
    else
        echo "⚠️  Verificar: esperado no máximo 10 páginas, gerado $PAGES páginas"
    fi
else
    echo "❌ Erro: main.pdf não foi gerado"
    exit 1
fi

echo ""
echo "🧹 Limpando arquivos temporários..."
rm -f main.aux main.bbl main.blg main.log main.out
echo "✅ Limpeza concluída"
echo ""
echo "✨ Processo finalizado com sucesso!"
