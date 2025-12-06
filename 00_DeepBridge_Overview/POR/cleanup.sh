#!/bin/bash
# Script de Limpeza Segura - POR Directory
# Remove arquivos duplicados e temporários, mantendo V1 e V2

set -e  # Exit on error

echo "=============================================="
echo "  Limpeza do Diretório POR"
echo "=============================================="
echo ""

# Verificar se estamos no diretório correto
if [ ! -d "V1" ] || [ ! -d "V2" ]; then
    echo "❌ ERRO: Este script deve ser executado no diretório POR!"
    echo "   Diretórios V1 e V2 não encontrados."
    exit 1
fi

echo "✅ Diretórios V1 e V2 encontrados."
echo ""

# 1. BACKUP automático
BACKUP_DIR="../BACKUP_POR_$(date +%Y%m%d_%H%M%S)"
echo "📦 Criando backup em: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR/"
echo "   ✅ Backup criado com sucesso!"
echo ""

# 2. Mostrar o que será removido
echo "📋 Arquivos/Diretórios que serão REMOVIDOS:"
echo ""
echo "   Duplicados em V1/V2:"
echo "   - bibliography/"
echo "   - figures/"
echo "   - sections/"
echo "   - main.tex, main.pdf, main.spl"
echo "   - elsarticle.cls, elsarticle-*.bst"
echo ""
echo "   Temporários de compilação:"
echo "   - main.aux, main.log, main.out, main.bbl, main.blg"
echo ""
echo "   Diretórios vazios:"
echo "   - experiments/, supplementary/, tables/"
echo ""
echo "   Build intermediário:"
echo "   - build/ (972KB de arquivos temporários)"
echo ""

# Perguntar confirmação
read -p "⚠️  Deseja continuar com a limpeza? (s/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "❌ Limpeza CANCELADA pelo usuário."
    echo "   Backup mantido em: $BACKUP_DIR"
    exit 0
fi

echo ""
echo "🗑️  Iniciando limpeza..."
echo ""

# 3. Remover duplicados
echo "   Removendo diretórios duplicados..."
rm -rf bibliography figures sections
echo "   ✅ Diretórios removidos"

echo "   Removendo arquivos principais duplicados..."
rm -f main.tex main.pdf main.spl
rm -f elsarticle.cls elsarticle-*.bst
echo "   ✅ Arquivos principais removidos"

# 4. Remover temporários de compilação
echo "   Removendo arquivos temporários de compilação..."
rm -f main.aux main.log main.out main.bbl main.blg
echo "   ✅ Temporários removidos"

# 5. Remover diretórios vazios
echo "   Removendo diretórios vazios..."
rmdir experiments supplementary tables 2>/dev/null || true
echo "   ✅ Diretórios vazios removidos"

# 6. Remover build/
echo "   Removendo build/ (arquivos intermediários)..."
rm -rf build/
echo "   ✅ Build removido"

echo ""
echo "=============================================="
echo "  ✅ Limpeza CONCLUÍDA com sucesso!"
echo "=============================================="
echo ""
echo "📁 Estrutura final:"
ls -lh | grep -E "^d|^-" | awk '{print "   " $9}'
echo ""
echo "💾 Backup disponível em: $BACKUP_DIR"
echo "   (Você pode removê-lo manualmente após verificar)"
echo ""
echo "✨ Espaço liberado: ~1.4 MB"
echo ""
