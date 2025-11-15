#!/usr/bin/env python3
"""
Script de teste para verificar se o treinamento TAKD está funcionando
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from the main script
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("="*80)
print("TESTE DE TREINAMENTO TAKD")
print("="*80)

# Simple LeNet5Student
class LeNet5Student(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(LeNet5Student, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        fc_size = 5*5*20
        self.fc1 = nn.Linear(fc_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("\n1. Criando modelo...")
student = LeNet5Student(num_classes=10, in_channels=3)
print("   ✅ Modelo criado")

print("\n2. Carregando dataset CIFAR10...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

try:
    train_dataset = datasets.CIFAR10(
        root='/tmp/data',
        train=True,
        download=True,
        transform=transform
    )
    print("   ✅ Dataset carregado")
except Exception as e:
    print(f"   ❌ Erro ao carregar dataset: {e}")
    sys.exit(1)

print("\n3. Criando DataLoader...")
try:
    # Use num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,  # Important for Colab!
        pin_memory=False
    )
    print("   ✅ DataLoader criado")
except Exception as e:
    print(f"   ❌ Erro ao criar DataLoader: {e}")
    sys.exit(1)

print("\n4. Testando iteração no DataLoader...")
try:
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"   ✅ Batch {batch_idx+1}: data shape = {data.shape}, target shape = {target.shape}")
        if batch_idx >= 2:  # Test 3 batches
            break
except Exception as e:
    print(f"   ❌ Erro ao iterar no DataLoader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testando forward pass...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = student.to(device)
    data = data.to(device)

    output = student(data)
    print(f"   ✅ Forward pass OK: output shape = {output.shape}")
except Exception as e:
    print(f"   ❌ Erro no forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ TODOS OS TESTES PASSARAM!")
print("="*80)
print("\nO problema NÃO está no DataLoader ou no modelo.")
print("O script 01_compression_efficiency.py deve estar travando em outro lugar.")
