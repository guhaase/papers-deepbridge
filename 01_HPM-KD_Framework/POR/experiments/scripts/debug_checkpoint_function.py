#!/usr/bin/env python3
"""
Testar a função model_checkpoint_exists() do script
"""
import torch
from pathlib import Path

def model_checkpoint_exists(checkpoint_path: Path) -> bool:
    """Check if model checkpoint exists and is valid."""
    print(f"\n🔍 Testando: {checkpoint_path.name}")
    print(f"   Caminho completo: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"   ❌ Arquivo não existe")
        return False

    print(f"   ✅ Arquivo existe")

    try:
        # Try to load to ensure it's not corrupted
        print(f"   Carregando checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"   ✅ Checkpoint carregado")

        has_state = 'model_state_dict' in checkpoint
        has_acc = 'accuracy' in checkpoint

        print(f"   - model_state_dict: {has_state}")
        print(f"   - accuracy: {has_acc}")

        result = has_state and has_acc
        print(f"   Resultado: {result}")

        return result

    except Exception as e:
        print(f"   ⚠️ Checkpoint corrupted: {checkpoint_path.name} - {e}")
        return False


# Testar o caminho exato
output_dir = Path("/content/drive/MyDrive/HPM-KD_Results/results_full_20251112_111138/exp_01_01_compression_efficiency")
dataset_name = "CIFAR10"

# Construir o caminho da mesma forma que o script faz
models_dir = output_dir / 'models'
teacher_checkpoint_path = models_dir / f"teacher_{dataset_name}.pt"

print("="*80)
print("TESTANDO FUNÇÃO model_checkpoint_exists()")
print("="*80)

result = model_checkpoint_exists(teacher_checkpoint_path)

print(f"\n{'='*80}")
print(f"RESULTADO FINAL: {result}")
print(f"{'='*80}")

if result:
    print("\n✅ A função deveria pular o treinamento do Teacher!")
else:
    print("\n❌ A função vai treinar o Teacher novamente (BUG!)")
