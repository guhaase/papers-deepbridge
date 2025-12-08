#!/bin/bash
# Deploy e Execução Automatizada dos Experimentos GPU
# Autor: Claude Code
# Data: 2025-12-08

set -e  # Exit on error

echo "========================================================================"
echo "Deploy e Execução - Experimentos GPU com Dados REAIS"
echo "========================================================================"
echo ""

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para printar com cor
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Verificar se está no diretório correto
if [ ! -f "requirements_gpu.txt" ]; then
    print_error "Execute este script do diretório: experimentos/"
    exit 1
fi

# 1. Verificar GPU
echo ""
echo "=== Verificando GPU ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print_status "GPU detectada"
else
    print_warning "nvidia-smi não encontrado (GPU pode não estar disponível)"
fi

# 2. Criar ambiente virtual (se não existir)
echo ""
echo "=== Configurando Ambiente Virtual ==="
if [ ! -d "venv_gpu" ]; then
    print_status "Criando venv_gpu..."
    python3 -m venv venv_gpu
else
    print_status "venv_gpu já existe"
fi

# Ativar ambiente
source venv_gpu/bin/activate
print_status "Ambiente ativado"

# 3. Instalar PyTorch com CUDA
echo ""
echo "=== Instalando PyTorch com CUDA ==="
python -c "import torch" 2>/dev/null && HAS_TORCH=1 || HAS_TORCH=0

if [ $HAS_TORCH -eq 0 ]; then
    print_status "Instalando PyTorch com CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "PyTorch já instalado"
fi

# 4. Instalar requirements
echo ""
echo "=== Instalando Requirements ==="
print_status "Instalando pacotes..."
pip install -q -r requirements_gpu.txt

# 5. Instalar DeepBridge
echo ""
echo "=== Instalando DeepBridge ==="
if [ -d "/workspace/DeepBridge" ]; then
    DEEPBRIDGE_PATH="/workspace/DeepBridge"
elif [ -d "/home/guhaase/projetos/DeepBridge" ]; then
    DEEPBRIDGE_PATH="/home/guhaase/projetos/DeepBridge"
else
    print_error "DeepBridge não encontrado. Ajuste o path manualmente."
    exit 1
fi

print_status "Instalando DeepBridge de: $DEEPBRIDGE_PATH"
pip install -q -e $DEEPBRIDGE_PATH

# 6. Testar configuração
echo ""
echo "=== Testando Configuração ==="
python test_gpu_setup.py

# Verificar se teste passou
if [ $? -eq 0 ]; then
    print_status "Configuração OK!"
else
    print_error "Teste de configuração falhou. Verifique os erros acima."
    exit 1
fi

# 7. Perguntar qual experimento executar
echo ""
echo "========================================================================"
echo "Qual experimento deseja executar?"
echo "========================================================================"
echo "1. Experimento 4: HPM-KD Framework (~1 hora)"
echo "2. Experimento 6: Ablation Studies (~10 minutos)"
echo "3. Ambos (sequencial, ~1h 10min)"
echo "4. Sair"
echo ""
read -p "Escolha (1-4): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "=== Executando Experimento 4: HPM-KD ==="
        cd 04_hpmkd
        python scripts/run_hpmkd_REAL.py
        print_status "Experimento 4 concluído!"
        echo "Resultados: $(pwd)/results/hpmkd_results_REAL.json"
        ;;
    2)
        echo ""
        echo "=== Executando Experimento 6: Ablation ==="
        cd 06_ablation_studies
        python scripts/run_ablation_REAL.py
        print_status "Experimento 6 concluído!"
        echo "Resultados: $(pwd)/results/ablation_study_REAL.json"
        ;;
    3)
        echo ""
        echo "=== Executando Ambos Experimentos ==="

        echo ""
        echo "--- Experimento 4: HPM-KD ---"
        cd 04_hpmkd
        python scripts/run_hpmkd_REAL.py
        print_status "Experimento 4 concluído!"

        echo ""
        echo "--- Experimento 6: Ablation ---"
        cd ../06_ablation_studies
        python scripts/run_ablation_REAL.py
        print_status "Experimento 6 concluído!"

        echo ""
        print_status "Ambos experimentos concluídos!"
        echo "Resultados:"
        echo "  - Exp 4: 04_hpmkd/results/hpmkd_results_REAL.json"
        echo "  - Exp 6: 06_ablation_studies/results/ablation_study_REAL.json"
        ;;
    4)
        echo "Saindo..."
        exit 0
        ;;
    *)
        print_error "Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Execução Completa!"
echo "========================================================================"
echo ""
echo "Próximos passos:"
echo "1. Verificar logs em: logs/"
echo "2. Ver resultados em: results/"
echo "3. Validar métricas geradas"
echo "4. Fazer backup dos resultados"
echo ""
