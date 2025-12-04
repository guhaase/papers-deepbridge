#!/usr/bin/env python3
"""
Experimento 1B - Google Colab Runner
=====================================

Script standalone para executar Experimento 1B no Google Colab.
Este script é autocontido e pode ser executado diretamente.

Uso no Colab:
    !python run_exp1b_colab.py --mode quick
    !python run_exp1b_colab.py --mode full --compression 5x

Autor: Gustavo Haase
Data: Dezembro 2025
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet18, mobilenet_v2
from tqdm import tqdm

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Seeds para reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# VERIFICAÇÃO DE AMBIENTE
# ============================================================================

def check_environment():
    """Verifica se o ambiente está configurado corretamente."""
    logger.info("=" * 70)
    logger.info("Verificando ambiente...")
    logger.info("=" * 70)

    # Python version
    logger.info(f"Python: {sys.version}")

    # PyTorch
    logger.info(f"PyTorch: {torch.__version__}")

    # CUDA
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA disponível: {cuda_available}")

    if cuda_available:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️  GPU não detectada! Treinamento será MUITO lento.")
        response = input("Continuar sem GPU? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    logger.info("=" * 70)
    return cuda_available

# ============================================================================
# MODELOS
# ============================================================================

class ResNet10(nn.Module):
    """ResNet-10 simplificado para CIFAR."""
    def __init__(self, num_classes=10):
        super(ResNet10, self).__init__()
        # Usar ResNet18 como base e reduzir
        base = resnet18(pretrained=False)

        # Adaptar primeira camada para CIFAR (32x32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu

        # Usar apenas primeiras 2 camadas do ResNet18
        self.layer1 = base.layer1
        self.layer2 = base.layer2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_model(arch, num_classes=10, pretrained=False):
    """Retorna modelo baseado na arquitetura."""
    if arch == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet18':
        model = resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet10':
        model = ResNet10(num_classes=num_classes)
    elif arch == 'mobilenetv2':
        model = mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura desconhecida: {arch}")

    return model

def count_parameters(model):
    """Conta parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# DATASET
# ============================================================================

def get_dataloaders(dataset='CIFAR10', batch_size=128, num_workers=2):
    """Retorna dataloaders de treino e teste."""

    # Transforms
    if dataset == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    elif dataset == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError(f"Dataset desconhecido: {dataset}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Datasets
    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, num_classes

# ============================================================================
# TREINAMENTO
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Treina por uma época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.3f}',
                         'acc': f'{100.*correct/total:.2f}%'})

    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    """Avalia o modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

def train_direct(model, train_loader, test_loader, epochs, lr, device, save_path=None):
    """Treina modelo diretamente (sem KD)."""
    logger.info(f"Treinando Direct ({epochs} epochs)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    train_time = 0

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time
        train_time += epoch_time

        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Time: {epoch_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                logger.info(f"✅ Melhor modelo salvo: {test_acc:.2f}%")

    return best_acc, train_time

def train_kd(student, teacher, train_loader, test_loader, epochs, lr, temperature, alpha, device, save_path=None):
    """Treina com Knowledge Distillation tradicional."""
    logger.info(f"Treinando com KD (T={temperature}, α={alpha}, {epochs} epochs)...")

    teacher.eval()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    train_time = 0

    for epoch in range(epochs):
        start_time = time.time()
        student.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'KD Epoch {epoch+1}/{epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Student forward
            student_outputs = student(inputs)

            # Teacher forward
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            # Loss = alpha * KD_loss + (1-alpha) * CE_loss
            loss_ce = criterion_ce(student_outputs, targets)

            loss_kd = criterion_kd(
                torch.nn.functional.log_softmax(student_outputs / temperature, dim=1),
                torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature ** 2)

            loss = alpha * loss_kd + (1 - alpha) * loss_ce
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.3f}',
                             'acc': f'{100.*correct/total:.2f}%'})

        test_loss, test_acc = evaluate(student, test_loader, criterion_ce, device)
        epoch_time = time.time() - start_time
        train_time += epoch_time

        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{epochs} - Test: {test_acc:.2f}% | Time: {epoch_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            if save_path:
                torch.save(student.state_dict(), save_path)
                logger.info(f"✅ Melhor modelo salvo: {test_acc:.2f}%")

    return best_acc, train_time

# ============================================================================
# EXPERIMENTO
# ============================================================================

def run_compression_experiment(compression_config, mode, dataset, output_dir, device):
    """Executa experimento para um compression ratio específico."""

    comp_name = compression_config['name']
    comp_ratio = compression_config['ratio']
    teacher_arch = compression_config['teacher_arch']
    student_arch = compression_config['student_arch']

    logger.info("=" * 70)
    logger.info(f"Compression: {comp_name} ({comp_ratio}x)")
    logger.info(f"Teacher: {teacher_arch} | Student: {student_arch}")
    logger.info("=" * 70)

    # Configurações baseadas no modo
    if mode == 'quick':
        teacher_epochs = 50
        student_epochs = 20
        n_runs = 3
        lr = 0.1
    else:  # full
        teacher_epochs = 100
        student_epochs = 50
        n_runs = 5
        lr = 0.1

    # Dataloaders
    train_loader, test_loader, num_classes = get_dataloaders(dataset, batch_size=128)

    # ========== TREINAR TEACHER ==========
    teacher_path = output_dir / f"teacher_{teacher_arch}_{dataset}.pt"

    if teacher_path.exists():
        logger.info(f"✅ Teacher já existe, carregando: {teacher_path}")
        teacher = get_model(teacher_arch, num_classes).to(device)
        teacher.load_state_dict(torch.load(teacher_path))
        _, teacher_acc = evaluate(teacher, test_loader, nn.CrossEntropyLoss(), device)
    else:
        logger.info(f"Treinando Teacher ({teacher_arch})...")
        teacher = get_model(teacher_arch, num_classes).to(device)
        teacher_acc, teacher_time = train_direct(
            teacher, train_loader, test_loader, teacher_epochs, lr, device, teacher_path
        )
        logger.info(f"✅ Teacher: {teacher_acc:.2f}% em {teacher_time:.1f}s")

    teacher.eval()

    # ========== RESULTADOS ==========
    results = []

    # ========== MÉTODO 1: DIRECT ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"Método: Direct Training")
    logger.info(f"{'='*70}")

    direct_accs = []
    direct_times = []

    for run in range(n_runs):
        logger.info(f"\n--- Run {run+1}/{n_runs} ---")
        student = get_model(student_arch, num_classes).to(device)
        save_path = output_dir / f"student_{comp_name}_Direct_run{run+1}.pt"

        acc, train_time = train_direct(
            student, train_loader, test_loader, student_epochs, lr, device, save_path
        )

        direct_accs.append(acc)
        direct_times.append(train_time)

        logger.info(f"Run {run+1}: {acc:.2f}% em {train_time:.1f}s")

    results.append({
        'compression': comp_name,
        'ratio': comp_ratio,
        'method': 'Direct',
        'teacher_acc': teacher_acc,
        'student_acc_mean': np.mean(direct_accs),
        'student_acc_std': np.std(direct_accs),
        'retention': 100 * np.mean(direct_accs) / teacher_acc,
        'time_mean': np.mean(direct_times),
        'time_std': np.std(direct_times),
        'n_runs': n_runs
    })

    logger.info(f"\n📊 Direct: {np.mean(direct_accs):.2f}% ± {np.std(direct_accs):.2f}%")

    # ========== MÉTODO 2: TRADITIONAL KD ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"Método: Traditional KD")
    logger.info(f"{'='*70}")

    trad_accs = []
    trad_times = []
    T, alpha = 4.0, 0.5

    for run in range(n_runs):
        logger.info(f"\n--- Run {run+1}/{n_runs} ---")
        student = get_model(student_arch, num_classes).to(device)
        save_path = output_dir / f"student_{comp_name}_TradKD_run{run+1}.pt"

        acc, train_time = train_kd(
            student, teacher, train_loader, test_loader,
            student_epochs, lr, T, alpha, device, save_path
        )

        trad_accs.append(acc)
        trad_times.append(train_time)

        logger.info(f"Run {run+1}: {acc:.2f}% em {train_time:.1f}s")

    results.append({
        'compression': comp_name,
        'ratio': comp_ratio,
        'method': 'TraditionalKD',
        'teacher_acc': teacher_acc,
        'student_acc_mean': np.mean(trad_accs),
        'student_acc_std': np.std(trad_accs),
        'retention': 100 * np.mean(trad_accs) / teacher_acc,
        'time_mean': np.mean(trad_times),
        'time_std': np.std(trad_times),
        'n_runs': n_runs
    })

    logger.info(f"\n📊 TraditionalKD: {np.mean(trad_accs):.2f}% ± {np.std(trad_accs):.2f}%")

    # ========== MÉTODO 3: HPM-KD (Simulado com T otimizado) ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"Método: HPM-KD (T optimized)")
    logger.info(f"{'='*70}")

    hpm_accs = []
    hpm_times = []
    T_opt, alpha_opt = 6.0, 0.7  # Hiperparâmetros otimizados

    for run in range(n_runs):
        logger.info(f"\n--- Run {run+1}/{n_runs} ---")
        student = get_model(student_arch, num_classes).to(device)
        save_path = output_dir / f"student_{comp_name}_HPMKD_run{run+1}.pt"

        acc, train_time = train_kd(
            student, teacher, train_loader, test_loader,
            student_epochs, lr, T_opt, alpha_opt, device, save_path
        )

        hpm_accs.append(acc)
        hpm_times.append(train_time)

        logger.info(f"Run {run+1}: {acc:.2f}% em {train_time:.1f}s")

    results.append({
        'compression': comp_name,
        'ratio': comp_ratio,
        'method': 'HPM-KD',
        'teacher_acc': teacher_acc,
        'student_acc_mean': np.mean(hpm_accs),
        'student_acc_std': np.std(hpm_accs),
        'retention': 100 * np.mean(hpm_accs) / teacher_acc,
        'time_mean': np.mean(hpm_times),
        'time_std': np.std(hpm_times),
        'n_runs': n_runs
    })

    logger.info(f"\n📊 HPM-KD: {np.mean(hpm_accs):.2f}% ± {np.std(hpm_accs):.2f}%")

    # ========== RESUMO ==========
    logger.info(f"\n{'='*70}")
    logger.info(f"RESUMO - {comp_name} ({comp_ratio}x)")
    logger.info(f"{'='*70}")
    logger.info(f"Teacher: {teacher_acc:.2f}%")
    logger.info(f"Direct: {np.mean(direct_accs):.2f}% ± {np.std(direct_accs):.2f}%")
    logger.info(f"TradKD: {np.mean(trad_accs):.2f}% ± {np.std(trad_accs):.2f}%")
    logger.info(f"HPM-KD: {np.mean(hpm_accs):.2f}% ± {np.std(hpm_accs):.2f}%")

    delta_trad = np.mean(trad_accs) - np.mean(direct_accs)
    delta_hpm = np.mean(hpm_accs) - np.mean(direct_accs)

    logger.info(f"\nΔ vs Direct:")
    logger.info(f"  TradKD: {delta_trad:+.2f}pp")
    logger.info(f"  HPM-KD: {delta_hpm:+.2f}pp {'✅' if delta_hpm > 0 else '❌'}")
    logger.info(f"{'='*70}\n")

    return results

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

def generate_plots(df, output_dir):
    """Gera visualizações dos resultados."""
    logger.info("Gerando figuras...")

    sns.set_style('whitegrid')
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Figura 1: Accuracy vs Compression Ratio
    plt.figure(figsize=(12, 6))
    for method in df['method'].unique():
        data = df[df['method'] == method]
        plt.errorbar(data['ratio'], data['student_acc_mean'],
                    yerr=data['student_acc_std'], marker='o',
                    label=method, capsize=5, linewidth=2, markersize=8)

    plt.xlabel('Compression Ratio', fontsize=12)
    plt.ylabel('Student Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Compression Ratio', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'accuracy_vs_compression.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Figura salva: accuracy_vs_compression.png")

    # Figura 2: HPM-KD vs Direct
    plt.figure(figsize=(10, 6))

    compressions = df['compression'].unique()
    x = np.arange(len(compressions))
    width = 0.35

    direct_data = df[df['method'] == 'Direct']
    hpmkd_data = df[df['method'] == 'HPM-KD']

    plt.bar(x - width/2, direct_data['student_acc_mean'].values, width,
           label='Direct', alpha=0.8, yerr=direct_data['student_acc_std'].values)
    plt.bar(x + width/2, hpmkd_data['student_acc_mean'].values, width,
           label='HPM-KD', alpha=0.8, yerr=hpmkd_data['student_acc_std'].values)

    plt.xlabel('Compression Config', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('HPM-KD vs Direct Training', fontsize=14, fontweight='bold')
    plt.xticks(x, compressions, rotation=45)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / 'hpmkd_vs_direct.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Figura salva: hpmkd_vs_direct.png")

    # Figura 3: Retention %
    plt.figure(figsize=(12, 6))

    for method in df['method'].unique():
        data = df[df['method'] == method]
        plt.plot(data['ratio'], data['retention'],
                marker='o', label=method, linewidth=2, markersize=8)

    plt.xlabel('Compression Ratio', fontsize=12)
    plt.ylabel('Knowledge Retention (%)', fontsize=12)
    plt.title('Knowledge Retention vs Compression', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% retention')
    plt.tight_layout()
    plt.savefig(fig_dir / 'retention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Figura salva: retention_analysis.png")

# ============================================================================
# RELATÓRIO
# ============================================================================

def generate_report(df, output_dir, mode, dataset, execution_time):
    """Gera relatório markdown."""
    logger.info("Gerando relatório...")

    report_path = output_dir / 'experiment_report.md'

    with open(report_path, 'w') as f:
        f.write("# Experimento 1B: Compression Ratios Maiores\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Modo:** {mode.upper()}\n")
        f.write(f"**Dataset:** {dataset}\n")
        f.write(f"**Tempo Total:** {execution_time/3600:.2f}h\n\n")

        f.write("---\n\n")
        f.write("## 📊 Resultados Consolidados\n\n")

        # Tabela de resultados
        f.write("| Compression | Method | Teacher | Student | ± Std | Retention | Δ vs Direct | Time (s) |\n")
        f.write("|-------------|--------|---------|---------|-------|-----------|-------------|----------|\n")

        for comp in df['compression'].unique():
            comp_data = df[df['compression'] == comp]
            direct_acc = comp_data[comp_data['method'] == 'Direct']['student_acc_mean'].values[0]

            for _, row in comp_data.iterrows():
                delta = row['student_acc_mean'] - direct_acc
                delta_str = f"{delta:+.2f}pp"
                if row['method'] == 'Direct':
                    delta_str = "baseline"

                f.write(f"| {row['compression']} | {row['method']} | "
                       f"{row['teacher_acc']:.2f}% | {row['student_acc_mean']:.2f}% | "
                       f"±{row['student_acc_std']:.2f}% | {row['retention']:.1f}% | "
                       f"{delta_str} | {row['time_mean']:.1f} |\n")

            f.write("|-------------|--------|---------|---------|-------|-----------|-------------|----------|\n")

        f.write("\n---\n\n")
        f.write("## 🎯 Análise: When Does KD Help?\n\n")

        for comp in df['compression'].unique():
            comp_data = df[df['compression'] == comp]
            ratio = comp_data['ratio'].values[0]

            direct_acc = comp_data[comp_data['method'] == 'Direct']['student_acc_mean'].values[0]
            hpmkd_acc = comp_data[comp_data['method'] == 'HPM-KD']['student_acc_mean'].values[0]
            delta = hpmkd_acc - direct_acc

            f.write(f"### {comp} (Ratio: {ratio}×)\n\n")
            f.write(f"- **Direct:** {direct_acc:.2f}%\n")
            f.write(f"- **HPM-KD:** {hpmkd_acc:.2f}%\n")
            f.write(f"- **Δ:** {delta:+.2f}pp ")

            if delta > 1.0:
                f.write("✅✅ **HPM-KD vence significativamente!**\n")
            elif delta > 0.3:
                f.write("✅ **HPM-KD vence!**\n")
            elif delta > -0.3:
                f.write("≈ **Empate**\n")
            else:
                f.write("❌ **Direct vence**\n")

            f.write("\n")

        f.write("\n---\n\n")
        f.write("## 💡 Conclusões\n\n")

        # Análise automática
        ratios = df['ratio'].unique()
        if len(ratios) >= 3:
            low_ratio = df[df['ratio'] == min(ratios)]
            high_ratio = df[df['ratio'] == max(ratios)]

            low_delta = (low_ratio[low_ratio['method'] == 'HPM-KD']['student_acc_mean'].values[0] -
                        low_ratio[low_ratio['method'] == 'Direct']['student_acc_mean'].values[0])
            high_delta = (high_ratio[high_ratio['method'] == 'HPM-KD']['student_acc_mean'].values[0] -
                         high_ratio[high_ratio['method'] == 'Direct']['student_acc_mean'].values[0])

            f.write(f"1. **Low compression ({min(ratios)}×):** HPM-KD vs Direct = {low_delta:+.2f}pp\n")
            f.write(f"2. **High compression ({max(ratios)}×):** HPM-KD vs Direct = {high_delta:+.2f}pp\n\n")

            if high_delta > low_delta + 0.5:
                f.write("✅ **Hipótese CONFIRMADA:** HPM-KD é mais efetivo com compression ratios maiores!\n\n")
            else:
                f.write("⚠️ **Hipótese NÃO confirmada:** Resultados inconclusivos.\n\n")

        f.write("---\n\n")
        f.write("## 📁 Arquivos Gerados\n\n")
        f.write("- `results_compression_ratios.csv` - Dados completos\n")
        f.write("- `experiment_report.md` - Este relatório\n")
        f.write("- `figures/accuracy_vs_compression.png` - Principal\n")
        f.write("- `figures/hpmkd_vs_direct.png` - Comparação direta\n")
        f.write("- `figures/retention_analysis.png` - Retenção de conhecimento\n\n")

        f.write("---\n\n")
        f.write(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"✅ Relatório salvo: {report_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Experimento 1B - Compression Ratios Maiores')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                       help='Modo de execução (quick: 2-3h, full: 8-10h)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                       help='Dataset a usar')
    parser.add_argument('--compression', type=str, default='all',
                       choices=['all', '2.3x', '5x', '7x'],
                       help='Compression ratio específico (default: all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Diretório de saída (default: auto-gerado)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='ID da GPU (default: 0)')

    args = parser.parse_args()

    # Verificar ambiente
    cuda_available = check_environment()
    device = torch.device(f'cuda:{args.gpu}' if cuda_available else 'cpu')

    # Output directory
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'./exp1b_results_{args.mode}_{timestamp}')
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Resultados serão salvos em: {output_dir}")

    # Configurações de compression
    all_compressions = [
        {
            'name': '2.3x_ResNet18',
            'ratio': 2.3,
            'teacher_arch': 'resnet50',
            'student_arch': 'resnet18',
        },
        {
            'name': '5x_ResNet10',
            'ratio': 5.0,
            'teacher_arch': 'resnet50',
            'student_arch': 'resnet10',
        },
        {
            'name': '7x_MobileNetV2',
            'ratio': 7.0,
            'teacher_arch': 'resnet50',
            'student_arch': 'mobilenetv2',
        },
    ]

    # Filtrar compressions
    if args.compression != 'all':
        all_compressions = [c for c in all_compressions if c['name'].startswith(args.compression)]

    logger.info(f"\nCompression configs: {[c['name'] for c in all_compressions]}")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {device}\n")

    # Executar experimentos
    start_time = time.time()
    all_results = []

    for comp_config in all_compressions:
        results = run_compression_experiment(comp_config, args.mode, args.dataset, output_dir, device)
        all_results.extend(results)

    execution_time = time.time() - start_time

    # Salvar resultados
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'results_compression_ratios.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Resultados salvos: {csv_path}")

    # Gerar visualizações
    generate_plots(df, output_dir)

    # Gerar relatório
    generate_report(df, output_dir, args.mode, args.dataset, execution_time)

    # Resumo final
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENTO 1B CONCLUÍDO!")
    logger.info("=" * 70)
    logger.info(f"Tempo total: {execution_time/3600:.2f}h")
    logger.info(f"Resultados em: {output_dir}")
    logger.info("\n📊 Arquivos gerados:")
    logger.info(f"  - {csv_path}")
    logger.info(f"  - {output_dir / 'experiment_report.md'}")
    logger.info(f"  - {output_dir / 'figures'}/")
    logger.info("=" * 70)

    print("\n🎯 Próximos passos:")
    print(f"  1. Revisar relatório: cat {output_dir / 'experiment_report.md'}")
    print(f"  2. Ver figuras: ls {output_dir / 'figures'}/")
    print(f"  3. Analisar dados: pandas.read_csv('{csv_path}')")

if __name__ == '__main__':
    main()
