#!/usr/bin/env python3
"""
Script para verificar o estado do Experimento 2 e retomar se necessário.

Uso no Google Colab:
    !python check_and_resume_exp2.py --drive-path "/content/drive/MyDrive/HPM-KD_Results/exp02_ablation_full"
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from datetime import datetime

def check_checkpoint(checkpoint_path: Path) -> dict:
    """
    Verifica se um checkpoint existe e é válido.

    Returns:
        dict com informações do checkpoint ou None se inválido
    """
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Verificar se tem os campos necessários
        required_fields = ['model_state_dict', 'accuracy', 'train_time']
        if not all(field in checkpoint for field in required_fields):
            return None

        return {
            'path': str(checkpoint_path),
            'accuracy': checkpoint['accuracy'],
            'train_time': checkpoint['train_time'],
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'metadata': checkpoint.get('metadata', {}),
            'valid': True
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def check_experiment_csvs(output_dir: Path) -> dict:
    """
    Verifica quais experimentos já foram completados.

    Returns:
        dict com status de cada experimento
    """
    experiments = {
        'exp05_component_ablation.csv': 'Experiment 5: Component Ablation',
        'exp06_component_interactions.csv': 'Experiment 6: Component Interactions',
        'exp07_hyperparameter_sensitivity.csv': 'Experiment 7: Hyperparameter Sensitivity',
        'exp08_progressive_chain.csv': 'Experiment 8: Progressive Chain Length',
        'exp09_num_teachers.csv': 'Experiment 9: Number of Teachers'
    }

    status = {}
    for csv_file, description in experiments.items():
        csv_path = output_dir / csv_file
        if csv_path.exists():
            # Verificar tamanho do arquivo
            size = csv_path.stat().st_size
            status[csv_file] = {
                'completed': True,
                'description': description,
                'size': size,
                'path': str(csv_path)
            }
        else:
            status[csv_file] = {
                'completed': False,
                'description': description
            }

    return status


def main():
    parser = argparse.ArgumentParser(
        description='Verificar estado do Experimento 2 e retomar se necessário'
    )
    parser.add_argument(
        '--drive-path',
        type=str,
        required=True,
        help='Caminho para a pasta de resultados no Google Drive'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'],
        help='Dataset usado no experimento'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Retomar experimento automaticamente se possível'
    )

    args = parser.parse_args()

    output_dir = Path(args.drive_path)

    print("=" * 80)
    print("VERIFICAÇÃO DO ESTADO DO EXPERIMENTO 2")
    print("=" * 80)
    print(f"Diretório: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Verificar se o diretório existe
    if not output_dir.exists():
        print(f"\n❌ ERRO: Diretório não encontrado: {output_dir}")
        print("\nVerifique se:")
        print("  1. O Google Drive está montado")
        print("  2. O caminho está correto")
        print("\nPara montar o Drive:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        sys.exit(1)

    print("\n✅ Diretório encontrado")

    # 2. Verificar checkpoint do Teacher
    print("\n" + "=" * 80)
    print("CHECKPOINT DO PROFESSOR (TEACHER)")
    print("=" * 80)

    teacher_checkpoint_path = output_dir / 'models' / f'teacher_{args.dataset}.pt'
    teacher_info = check_checkpoint(teacher_checkpoint_path)

    if teacher_info and teacher_info.get('valid', False):
        print(f"✅ Checkpoint do Teacher encontrado e válido!")
        print(f"   📍 Caminho: {teacher_info['path']}")
        print(f"   🎯 Acurácia: {teacher_info['accuracy']:.2f}%")
        print(f"   ⏱️  Tempo de treino: {teacher_info['train_time']:.2f}s")
        print(f"   📅 Timestamp: {teacher_info['timestamp']}")
        if teacher_info.get('metadata'):
            print(f"   📝 Metadata: {teacher_info['metadata']}")

        teacher_ready = True
    else:
        if teacher_info and not teacher_info.get('valid', True):
            print(f"❌ Checkpoint do Teacher CORROMPIDO!")
            print(f"   Erro: {teacher_info.get('error', 'unknown')}")
            print(f"\n   Solução: Delete o arquivo e retreine:")
            print(f"   !rm {teacher_checkpoint_path}")
        else:
            print(f"⚠️  Checkpoint do Teacher NÃO encontrado")
            print(f"   Esperado em: {teacher_checkpoint_path}")
            print(f"\n   O script treinará o teacher automaticamente.")

        teacher_ready = False

    # 3. Verificar experimentos completados
    print("\n" + "=" * 80)
    print("EXPERIMENTOS COMPLETADOS")
    print("=" * 80)

    experiments_status = check_experiment_csvs(output_dir)
    completed_count = sum(1 for exp in experiments_status.values() if exp['completed'])
    total_count = len(experiments_status)

    for csv_file, info in experiments_status.items():
        if info['completed']:
            print(f"✅ {info['description']}")
            print(f"   📄 Arquivo: {csv_file} ({info['size']} bytes)")
        else:
            print(f"⏳ {info['description']}")
            print(f"   ⚠️  Ainda não completado")

    print(f"\nProgresso: {completed_count}/{total_count} experimentos completados")

    # 4. Verificar figuras
    print("\n" + "=" * 80)
    print("FIGURAS GERADAS")
    print("=" * 80)

    figures_dir = output_dir / 'figures'
    if figures_dir.exists():
        figures = list(figures_dir.glob('*.png'))
        if figures:
            print(f"✅ {len(figures)} figuras encontradas:")
            for fig in figures:
                print(f"   📊 {fig.name}")
        else:
            print("⚠️  Nenhuma figura gerada ainda")
    else:
        print("⚠️  Pasta 'figures/' não existe")

    # 5. Resumo e recomendações
    print("\n" + "=" * 80)
    print("RESUMO E RECOMENDAÇÕES")
    print("=" * 80)

    if teacher_ready and completed_count == total_count:
        print("\n🎉 EXPERIMENTO COMPLETO!")
        print("\nTodos os experimentos foram concluídos com sucesso.")
        print("\nPróximos passos:")
        print("  1. Analisar os resultados CSV")
        print("  2. Visualizar as figuras geradas")
        print("  3. Gerar o relatório final")

    elif teacher_ready and completed_count > 0:
        print(f"\n⚠️  EXPERIMENTO PARCIALMENTE COMPLETO")
        print(f"\n✅ Teacher treinado: SIM")
        print(f"✅ Experimentos completados: {completed_count}/{total_count}")
        print(f"\nO script pode continuar de onde parou!")

        if args.resume:
            print("\n🚀 Retomando experimento...")
            # Importar e executar o script principal
            # (isso será implementado se necessário)
            print("   Execute manualmente:")
            print(f"   !cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \\")
            print(f"   python3 02_ablation_studies.py \\")
            print(f"       --mode full \\")
            print(f"       --dataset {args.dataset} \\")
            print(f"       --gpu 0 \\")
            print(f"       --output \"{output_dir}\"")

    else:
        print(f"\n⚠️  EXPERIMENTO NÃO INICIADO OU INCOMPLETO")
        print(f"\n✅ Teacher treinado: {'SIM' if teacher_ready else 'NÃO'}")
        print(f"✅ Experimentos completados: {completed_count}/{total_count}")

        if teacher_ready:
            print(f"\n💡 Você já tem o teacher treinado! Economizará ~30-45 minutos.")
        else:
            print(f"\n⏱️  O script treinará o teacher primeiro (~30-45 minutos).")

        print("\n🚀 Para iniciar/continuar:")
        print(f"   !cd /content/papers-deepbridge/01_HPM-KD_Framework/POR/experiments/scripts && \\")
        print(f"   python3 02_ablation_studies.py \\")
        print(f"       --mode full \\")
        print(f"       --dataset {args.dataset} \\")
        print(f"       --gpu 0 \\")
        print(f"       --output \"{output_dir}\"")

    print("\n" + "=" * 80)
    print("Verificação concluída!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
