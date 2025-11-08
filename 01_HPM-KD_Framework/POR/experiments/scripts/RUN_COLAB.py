#!/usr/bin/env python3
"""
🚀 SCRIPT SIMPLIFICADO PARA GOOGLE COLAB
========================================

Execute TODOS os experimentos HPM-KD com um único comando!

USO RÁPIDO NO COLAB:
-------------------

# Modo Quick (3-4 horas, dataset pequeno)
!python RUN_COLAB.py

# Modo Full (8-10 horas, dataset completo)
!python RUN_COLAB.py --full

# Customizar dataset
!python RUN_COLAB.py --dataset CIFAR10

# Múltiplos datasets (apenas Exp 1)
!python RUN_COLAB.py --datasets MNIST CIFAR10

DATASETS DISPONÍVEIS:
- MNIST (padrão, rápido)
- FashionMNIST
- CIFAR10
- CIFAR100

O QUE FAZ:
- ✅ Executa os 4 experimentos em sequência
- ✅ Usa DeepBridge HPM-KD completo
- ✅ Salva resultados, figuras e modelos
- ✅ Gera relatório final consolidado
- ✅ Mostra progresso em tempo real

RESULTADOS:
- Salvos em: results_quick_YYYYMMDD_HHMMSS/
- Relatório: RELATORIO_FINAL.md
- Logs: run_all_experiments.log
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Parse simple arguments
    args = sys.argv[1:]

    # Build command for main script
    script_dir = Path(__file__).parent
    main_script = script_dir / 'run_all_experiments.py'

    cmd = [sys.executable, str(main_script)]

    # Default mode: quick
    mode = 'quick'

    # Parse arguments
    if '--full' in args:
        mode = 'full'
        args.remove('--full')

    cmd.extend(['--mode', mode])

    # Check for GPU (Colab usually has GPU)
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(['--gpu', '0'])
            print("✅ GPU detectada! Usando GPU 0")
        else:
            print("ℹ️  GPU não detectada. Usando CPU")
    except ImportError:
        print("⚠️  PyTorch não encontrado")

    # Add remaining arguments
    cmd.extend(args)

    # Print info
    print("="*80)
    print("🚀 EXECUTANDO TODOS OS EXPERIMENTOS HPM-KD".center(80))
    print("="*80)
    print(f"\nModo: {mode.upper()}")
    print(f"Comando: {' '.join(cmd)}\n")
    print("="*80)
    print()

    # Run
    result = subprocess.run(cmd)

    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
