#!/bin/bash
#
# Script de setup para experimentos DeepBridge Fairness
#
# Uso:
#   chmod +x setup.sh
#   ./setup.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Setup - Experimentos DeepBridge Fairness"
echo "=========================================="

# Check Python version
echo ""
echo "🐍 Verificando Python..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python $python_version"

if ! python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ é necessário"
    exit 1
fi

echo "✅ Python OK"

# Create virtual environment
echo ""
echo "📦 Criando ambiente virtual..."

if [ -d "venv" ]; then
    echo "⚠️  venv já existe. Removendo..."
    rm -rf venv
fi

python -m venv venv
echo "✅ Ambiente virtual criado"

# Activate virtual environment
echo ""
echo "🔄 Ativando ambiente virtual..."
source venv/bin/activate
echo "✅ Ambiente ativado"

# Upgrade pip
echo ""
echo "⬆️  Atualizando pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "✅ pip atualizado"

# Install dependencies
echo ""
echo "📥 Instalando dependências..."
echo "   (isso pode levar alguns minutos...)"

pip install -r requirements.txt --quiet

echo "✅ Dependências instaladas"

# Verify installation
echo ""
echo "🔍 Verificando instalação..."

python -c "from deepbridge import DBDataset; print('✅ DeepBridge')"
python -c "import pandas; print('✅ Pandas')"
python -c "import numpy; print('✅ NumPy')"
python -c "import sklearn; print('✅ scikit-learn')"
python -c "import matplotlib; print('✅ Matplotlib')"

echo ""
echo "✅ Todas as dependências verificadas!"

# Create necessary directories
echo ""
echo "📁 Criando diretórios..."

mkdir -p data/case_studies
mkdir -p data/synthetic
mkdir -p data/annotations
mkdir -p results/auto_detection
mkdir -p results/eeoc_validation
mkdir -p results/case_studies
mkdir -p results/usability
mkdir -p results/performance
mkdir -p results/comparison
mkdir -p reports/figures

echo "✅ Diretórios criados"

# Test quick experiment
echo ""
echo "🧪 Testando instalação com experimento rápido..."
cd scripts
python exp1_auto_detection.py --quick

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SETUP CONCLUÍDO COM SUCESSO!"
    echo "=========================================="
    echo ""
    echo "Próximos passos:"
    echo "  1. Ative o ambiente: source venv/bin/activate"
    echo "  2. Leia RESUMO_EXECUTIVO.md"
    echo "  3. Execute experimentos: cd scripts && python exp1_auto_detection.py"
    echo ""
else
    echo ""
    echo "❌ Erro ao executar experimento de teste"
    echo "   Verifique os logs acima"
    exit 1
fi
