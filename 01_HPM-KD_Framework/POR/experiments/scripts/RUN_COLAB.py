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
- ✅ Monta Google Drive automaticamente
- ✅ Salva resultados NO GOOGLE DRIVE (persistente!)
- ✅ Gera relatório final consolidado
- ✅ Mostra progresso em tempo real

RESULTADOS SALVOS NO DRIVE:
- Pasta: /content/drive/MyDrive/HPM-KD_Results/results_YYYYMMDD_HHMMSS/
- Relatório: RELATORIO_FINAL.md
- Logs: run_all_experiments.log
- Modelos, figuras e dados salvos permanentemente!
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def mount_google_drive():
    """Monta o Google Drive automaticamente"""
    try:
        # Verifica se já está montado
        drive_path = Path('/content/drive')
        if drive_path.exists() and (drive_path / 'MyDrive').exists():
            print("✅ Google Drive já está montado!")
            return True

        # Tenta montar
        print("📁 Montando Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive montado com sucesso!")
        return True

    except ImportError:
        print("⚠️  Não está rodando no Google Colab")
        print("   Resultados serão salvos localmente em /content/")
        return False
    except Exception as e:
        print(f"⚠️  Erro ao montar Drive: {e}")
        print("   Resultados serão salvos localmente em /content/")
        return False


def get_output_dir(mode: str, use_drive: bool) -> str:
    """Define o diretório de saída"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"results_{mode}_{timestamp}"

    if use_drive:
        # Salvar no Google Drive
        base_path = Path('/content/drive/MyDrive/HPM-KD_Results')
        base_path.mkdir(parents=True, exist_ok=True)
        output_dir = base_path / dirname

        print(f"💾 Resultados serão salvos NO GOOGLE DRIVE:")
        print(f"   {output_dir}")
        print(f"   ✅ Persistente - não será perdido ao fechar o Colab!")
    else:
        # Salvar localmente (temporário)
        output_dir = Path('/content') / dirname

        print(f"⚠️  ATENÇÃO: Resultados serão salvos LOCALMENTE (temporário):")
        print(f"   {output_dir}")
        print(f"   ❌ Será perdido ao fechar o Colab!")
        print(f"   💡 Recomendado: montar o Google Drive primeiro")

    return str(output_dir)


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

    # Print banner
    print("="*80)
    print("🚀 EXECUTANDO TODOS OS EXPERIMENTOS HPM-KD".center(80))
    print("="*80)
    print(f"\nModo: {mode.upper()}")
    print()

    # Mount Google Drive
    use_drive = mount_google_drive()
    print()

    # Set output directory
    output_dir = get_output_dir(mode, use_drive)
    cmd.extend(['--output-dir', output_dir])
    print()

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

    print()

    # Add remaining arguments
    cmd.extend(args)

    # Show final command
    print("Comando completo:")
    print(f"  {' '.join(cmd)}")
    print()
    print("="*80)
    print()

    # Confirmation
    if use_drive:
        print("💾 Seus resultados estarão seguros no Google Drive!")
        print("   Você poderá acessá-los mesmo depois de fechar o Colab.")
    else:
        print("⚠️  LEMBRE-SE: Faça backup dos resultados antes de fechar o Colab!")
        print("   Use: !zip -r results.zip /content/results_*")
        print("   E depois: from google.colab import files; files.download('results.zip')")

    print()
    print("="*80)
    print()

    # Run
    result = subprocess.run(cmd)

    # Final message
    if result.returncode == 0:
        print()
        print("="*80)
        print("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO!".center(80))
        print("="*80)
        print()
        if use_drive:
            print(f"📁 Resultados salvos em:")
            print(f"   {output_dir}")
            print()
            print("💡 Acesse pelo Google Drive ou navegue diretamente:")
            print(f"   /content/drive/MyDrive/HPM-KD_Results/")
        else:
            print("⚠️  Faça backup dos resultados AGORA:")
            print()
            print("!zip -r results.zip " + output_dir)
            print("from google.colab import files")
            print("files.download('results.zip')")
        print()
        print("="*80)

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
