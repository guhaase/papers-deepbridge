#!/bin/bash
# Script para verificar se o treinamento está rodando

echo "========================================"
echo "STATUS DO TREINAMENTO"
echo "========================================"

echo ""
echo "1. Processos Python:"
PROC=$(ps aux | grep "compression_efficiency.py" | grep -v grep | wc -l)
if [ $PROC -gt 0 ]; then
    echo "   ✅ Script está RODANDO"
    ps aux | grep "compression_efficiency.py" | grep -v grep
else
    echo "   ❌ Script NÃO está rodando"
fi

echo ""
echo "2. Uso de GPU:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "   ⚠️ nvidia-smi não disponível"

echo ""
echo "3. Arquivos recentes (últimos 5 min):"
RECENT=$(find "/content/drive/MyDrive/HPM-KD_Results/results_full_20251112_111138/exp_01_01_compression_efficiency/" -type f -mmin -5 2>/dev/null | wc -l)
if [ $RECENT -gt 0 ]; then
    echo "   ✅ $RECENT arquivo(s) modificado(s) recentemente"
    find "/content/drive/MyDrive/HPM-KD_Results/results_full_20251112_111138/exp_01_01_compression_efficiency/" -type f -mmin -5 -exec ls -lh {} \;
else
    echo "   ⚠️ Nenhum arquivo modificado nos últimos 5 minutos"
fi

echo ""
echo "4. Último modelo criado/modificado:"
ls -lth "/content/drive/MyDrive/HPM-KD_Results/results_full_20251112_111138/exp_01_01_compression_efficiency/models/" 2>/dev/null | head -3

echo ""
echo "========================================"
