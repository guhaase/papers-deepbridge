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

🔄 RETOMAR EXECUÇÃO (SE O COLAB DESCONECTAR):
---------------------------------------------

# Retomar automaticamente de onde parou
!python RUN_COLAB.py --resume --output /content/drive/MyDrive/HPM-KD_Results/results_quick_YYYYMMDD_HHMMSS

# Começar de um experimento específico (ex: experimento 3)
!python RUN_COLAB.py --start-from 3 --output /content/drive/MyDrive/HPM-KD_Results/results_quick_YYYYMMDD_HHMMSS

# Executar apenas experimentos específicos
!python RUN_COLAB.py --only 2 3 4

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
- ✅ Sistema de checkpoint automático após cada experimento
- ✅ Retoma de onde parou se o Colab desconectar
- ✅ Gera relatório final consolidado
- ✅ Mostra progresso em tempo real

RESULTADOS SALVOS NO DRIVE:
- Pasta: /content/drive/MyDrive/HPM-KD_Results/results_YYYYMMDD_HHMMSS/
- Relatório: RELATORIO_FINAL.md
- Logs: run_all_experiments.log
- Checkpoint: checkpoint.json (para retomar)
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


def find_latest_checkpoint(use_drive: bool) -> tuple:
    """Find the latest checkpoint directory"""
    if use_drive:
        base_path = Path('/content/drive/MyDrive/HPM-KD_Results')
    else:
        base_path = Path('/content')

    if not base_path.exists():
        return None, None

    # Find all result directories
    result_dirs = sorted(base_path.glob('results_*'), key=lambda p: p.stat().st_mtime, reverse=True)

    for result_dir in result_dirs:
        checkpoint_file = result_dir / 'checkpoint.json'
        if checkpoint_file.exists():
            return result_dir, checkpoint_file

    return None, None


def get_output_dir(mode: str, use_drive: bool, resume: bool = False) -> str:
    """Define o diretório de saída"""

    # Check for existing checkpoint
    latest_dir, checkpoint_file = find_latest_checkpoint(use_drive)

    if resume and latest_dir:
        print(f"♻️  RETOMANDO EXECUÇÃO ANTERIOR:")
        print(f"   {latest_dir}")
        print(f"   Checkpoint: {checkpoint_file}")
        return str(latest_dir)

    # Create new directory
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


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint info"""
    import json
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    # Parse simple arguments
    args = sys.argv[1:]

    # Build command for main script
    script_dir = Path(__file__).parent
    main_script = script_dir / 'run_all_experiments.py'

    cmd = [sys.executable, str(main_script)]

    # Default mode: quick
    mode = 'quick'

    # Check for resume flag
    resume = '--resume' in args
    if resume:
        args.remove('--resume')

    # Parse arguments
    if '--full' in args:
        mode = 'full'
        args.remove('--full')

    # Mount Google Drive first (needed to check checkpoints)
    use_drive = mount_google_drive()
    print()

    # Check for existing checkpoint
    checkpoint_info = None
    if resume or ('--output' not in ' '.join(args)):
        latest_dir, checkpoint_file = find_latest_checkpoint(use_drive)
        if checkpoint_file:
            checkpoint_info = load_checkpoint_info(checkpoint_file)

    # If resuming, restore mode from checkpoint
    if resume and checkpoint_info:
        if 'mode' in checkpoint_info and checkpoint_info['mode']:
            mode = checkpoint_info['mode']
            print(f"♻️  Modo restaurado do checkpoint: {mode.upper()}")

    # Only add --mode if NOT resuming (let run_all_experiments.py restore it)
    if not resume:
        cmd.extend(['--mode', mode])

    # Print banner
    print("="*80)
    if resume:
        print("♻️  RETOMANDO EXPERIMENTOS HPM-KD".center(80))
    else:
        print("🚀 EXECUTANDO TODOS OS EXPERIMENTOS HPM-KD".center(80))
    print("="*80)
    print(f"\nModo: {mode.upper()}")
    if resume:
        print("Retomando: SIM ♻️")
        if checkpoint_info:
            if 'datasets' in checkpoint_info:
                print(f"Datasets: {', '.join(checkpoint_info.get('datasets', []))}")
            if 'completed_experiments' in checkpoint_info:
                print(f"Experimentos concluídos: {checkpoint_info['completed_experiments']}")
    print()

    # Check for existing checkpoint and suggest resume
    if not resume and '--output' not in ' '.join(args):
        latest_dir, checkpoint_file = find_latest_checkpoint(use_drive)
        if latest_dir:
            print("="*80)
            print("💡 CHECKPOINT DETECTADO!")
            print("="*80)
            print(f"\nEncontrado checkpoint de execução anterior em:")
            print(f"   {latest_dir}")
            if checkpoint_info:
                print(f"   Modo: {checkpoint_info.get('mode', 'unknown').upper()}")
                print(f"   Concluídos: {checkpoint_info.get('completed_experiments', [])}")
            print()
            print("Para RETOMAR de onde parou, execute:")
            print(f"   !python RUN_COLAB.py --resume")
            print()
            print("="*80)
            print()

    # Set output directory
    output_dir = get_output_dir(mode, use_drive, resume)
    cmd.extend(['--output', output_dir])
    print()

    # Add resume flag if needed
    if resume:
        cmd.append('--resume')

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
