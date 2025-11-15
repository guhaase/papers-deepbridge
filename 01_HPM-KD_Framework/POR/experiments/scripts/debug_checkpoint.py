#!/usr/bin/env python3
"""
Script de diagnóstico para verificar a estrutura dos checkpoints
"""
import torch
from pathlib import Path

# Caminho do checkpoint
checkpoint_path = Path("/content/drive/MyDrive/HPM-KD_Results/results_full_20251112_111138/exp_01_01_compression_efficiency/models/teacher_CIFAR10.pt")

print("="*80)
print("DIAGNÓSTICO DE CHECKPOINT")
print("="*80)

# Verificar se existe
print(f"\n1. Arquivo existe? {checkpoint_path.exists()}")
print(f"   Caminho: {checkpoint_path}")

if checkpoint_path.exists():
    # Verificar tamanho
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"   Tamanho: {size_mb:.2f} MB")

    # Tentar carregar
    try:
        print("\n2. Carregando checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("   ✅ Checkpoint carregado com sucesso!")

        # Verificar estrutura
        print(f"\n3. Chaves no checkpoint:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                print(f"   ✅ {key}: (dict com {len(checkpoint[key])} parâmetros)")
            else:
                print(f"   ✅ {key}: {checkpoint[key]}")

        # Verificar se tem as chaves necessárias
        print(f"\n4. Verificações:")
        print(f"   - model_state_dict presente? {'model_state_dict' in checkpoint}")
        print(f"   - accuracy presente? {'accuracy' in checkpoint}")
        print(f"   - train_time presente? {'train_time' in checkpoint}")

        if 'model_state_dict' in checkpoint and 'accuracy' in checkpoint:
            print(f"\n✅ CHECKPOINT VÁLIDO!")
            print(f"   Acurácia: {checkpoint['accuracy']:.2f}%")
            if 'train_time' in checkpoint:
                print(f"   Tempo de treino: {checkpoint['train_time']:.1f}s")
        else:
            print(f"\n❌ CHECKPOINT INVÁLIDO - Faltam chaves necessárias")

    except Exception as e:
        print(f"\n❌ Erro ao carregar checkpoint:")
        print(f"   {type(e).__name__}: {e}")
else:
    print("\n❌ Arquivo não encontrado!")
    print("\nVerifique:")
    print("1. Google Drive está montado?")
    print("2. O caminho está correto?")
    print("3. Você tem permissão para acessar o arquivo?")

print("\n" + "="*80)
