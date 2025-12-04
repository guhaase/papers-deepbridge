#!/usr/bin/env python3
"""
Experimento 1B - Kaggle Runner
===============================

Script otimizado para executar no Kaggle Notebooks.
Inclui sistema robusto de checkpoints para sessões longas.

Características Kaggle:
- Sessões: até 9-12 horas (vs 90min Colab)
- GPU: P100 (16GB) ou T4 (16GB)
- RAM: 16GB
- Storage: /kaggle/working/ (outputs salvos automaticamente)

Uso no Kaggle:
    python run_exp1b_kaggle.py --mode quick
    python run_exp1b_kaggle.py --mode full --resume

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
from typing import Dict, List, Tuple, Optional
import pickle

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
# CONFIGURAÇÃO KAGGLE
# ============================================================================

# Paths do Kaggle
KAGGLE_WORKING_DIR = Path('/kaggle/working')
KAGGLE_INPUT_DIR = Path('/kaggle/input')

# Detectar se está no Kaggle
IS_KAGGLE = KAGGLE_WORKING_DIR.exists()

if IS_KAGGLE:
    OUTPUT_BASE = KAGGLE_WORKING_DIR
else:
    OUTPUT_BASE = Path('./kaggle_outputs')
    OUTPUT_BASE.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_BASE / 'experiment.log')
    ]
)
logger = logging.getLogger(__name__)

# Seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# CHECKPOINT MANAGER (Robusto para Kaggle)
# ============================================================================

class CheckpointManager:
    """Gerenciador de checkpoints para retomar execução."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / 'experiment_state.pkl'

    def save_state(self, state: dict):
        """Salva estado do experimento."""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"✅ Estado salvo: {self.state_file}")

    def load_state(self) -> Optional[dict]:
        """Carrega estado do experimento."""
        if self.state_file.exists():
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            logger.info(f"✅ Estado carregado: {self.state_file}")
            return state
        return None

    def save_model(self, model, name: str):
        """Salva modelo."""
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(model.state_dict(), path)
        logger.info(f"💾 Modelo salvo: {path}")
        return path

    def load_model(self, model, name: str) -> bool:
        """Carrega modelo se existir."""
        path = self.checkpoint_dir / f"{name}.pt"
        if path.exists():
            model.load_state_dict(torch.load(path))
            logger.info(f"✅ Modelo carregado: {path}")
            return True
        return False

    def model_exists(self, name: str) -> bool:
        """Verifica se modelo existe."""
        return (self.checkpoint_dir / f"{name}.pt").exists()

# ============================================================================
# VERIFICAÇÃO DE AMBIENTE
# ============================================================================

def check_kaggle_environment():
    """Verifica ambiente Kaggle."""
    logger.info("=" * 70)
    logger.info("VERIFICANDO AMBIENTE KAGGLE")
    logger.info("=" * 70)

    # Kaggle
    logger.info(f"Rodando no Kaggle: {IS_KAGGLE}")
    if IS_KAGGLE:
        logger.info(f"Working dir: {KAGGLE_WORKING_DIR}")
        logger.info(f"Input dir: {KAGGLE_INPUT_DIR}")

    # Python
    logger.info(f"Python: {sys.version}")

    # PyTorch
    logger.info(f"PyTorch: {torch.__version__}")

    # CUDA
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA disponível: {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {vram:.1f} GB")

        # Detectar tipo de GPU Kaggle
        if 'P100' in gpu_name:
            logger.info("🚀 GPU: Tesla P100 (16GB) - EXCELENTE!")
        elif 'T4' in gpu_name:
            logger.info("✅ GPU: Tesla T4 (16GB) - BOM!")
        elif 'K80' in gpu_name:
            logger.warning("⚠️ GPU: Tesla K80 (12GB) - LENTO, considere restartar para P100/T4")
    else:
        logger.error("❌ GPU não detectada!")
        logger.error("No Kaggle: Settings → Accelerator → GPU")
        return False

    # RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    logger.info(f"RAM: {ram_gb:.1f} GB")

    # Disk space
    if IS_KAGGLE:
        disk_usage = os.popen('df -h /kaggle/working').read()
        logger.info(f"Disk space:\n{disk_usage}")

    logger.info("=" * 70)
    return True

# ============================================================================
# MODELOS
# ============================================================================

class ResNet10(nn.Module):
    """ResNet-10 para CIFAR."""
    def __init__(self, num_classes=10):
        super(ResNet10, self).__init__()
        base = resnet18(pretrained=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
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

def get_model(arch, num_classes=10):
    """Retorna modelo."""
    if arch == 'resnet50':
        model = resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet18':
        model = resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet10':
        model = ResNet10(num_classes=num_classes)
    elif arch == 'mobilenetv2':
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura desconhecida: {arch}")

    return model

def count_parameters(model):
    """Conta parâmetros."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# DATASET
# ============================================================================

def get_dataloaders(dataset='CIFAR10', batch_size=128, num_workers=4):
    """Retorna dataloaders."""

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

    # Download para /kaggle/working/data
    data_dir = OUTPUT_BASE / 'data'

    if dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    else:
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, test_loader, num_classes

# ============================================================================
# TREINAMENTO
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch_desc="Training"):
    """Treina uma época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=epoch_desc, leave=False)
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
    """Avalia modelo."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

def train_direct(model, train_loader, test_loader, epochs, lr, device, ckpt_mgr, model_name):
    """Treina modelo direto."""
    logger.info(f"Treinando Direct: {model_name} ({epochs} epochs)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    train_time = 0

    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch_desc=f"Direct Epoch {epoch+1}/{epochs}"
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.time() - start_time
        train_time += epoch_time

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_mgr.save_model(model, model_name)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    logger.info(f"✅ Direct concluído: {best_acc:.2f}% em {train_time:.1f}s")
    return best_acc, train_time

def train_kd(student, teacher, train_loader, test_loader, epochs, lr, temperature, alpha, device, ckpt_mgr, model_name):
    """Treina com KD."""
    logger.info(f"Treinando KD: {model_name} (T={temperature}, α={alpha}, {epochs} epochs)")

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

        pbar = tqdm(train_loader, desc=f'KD Epoch {epoch+1}/{epochs}', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            student_outputs = student(inputs)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

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

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_mgr.save_model(student, model_name)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    logger.info(f"✅ KD concluído: {best_acc:.2f}% em {train_time:.1f}s")
    return best_acc, train_time

# ============================================================================
# EXPERIMENTO COM CHECKPOINTS
# ============================================================================

def run_compression_experiment_with_checkpoints(
    compression_config, mode, dataset, output_dir, device, ckpt_mgr, resume=False
):
    """Executa experimento com sistema de checkpoints."""

    comp_name = compression_config['name']
    comp_ratio = compression_config['ratio']
    teacher_arch = compression_config['teacher_arch']
    student_arch = compression_config['student_arch']

    logger.info("=" * 70)
    logger.info(f"Compression: {comp_name} ({comp_ratio}x)")
    logger.info(f"Teacher: {teacher_arch} | Student: {student_arch}")
    logger.info("=" * 70)

    # Configurações
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

    # Estado de retomada
    state_key = f"{comp_name}_{dataset}"
    experiment_state = None

    if resume:
        saved_state = ckpt_mgr.load_state()
        if saved_state and state_key in saved_state:
            experiment_state = saved_state[state_key]
            logger.info(f"✅ Retomando experimento: {state_key}")
        else:
            logger.info(f"⚠️ Nenhum checkpoint encontrado para {state_key}, iniciando do zero")
            experiment_state = {'teacher_done': False, 'results': []}
    else:
        experiment_state = {'teacher_done': False, 'results': []}

    # ========== TEACHER ==========
    teacher_name = f"teacher_{teacher_arch}_{dataset}"

    if experiment_state.get('teacher_done', False) or ckpt_mgr.model_exists(teacher_name):
        logger.info(f"✅ Teacher já treinado, carregando...")
        teacher = get_model(teacher_arch, num_classes).to(device)
        ckpt_mgr.load_model(teacher, teacher_name)
        _, teacher_acc = evaluate(teacher, test_loader, nn.CrossEntropyLoss(), device)
        logger.info(f"Teacher accuracy: {teacher_acc:.2f}%")
        experiment_state['teacher_done'] = True
        experiment_state['teacher_acc'] = teacher_acc
    else:
        logger.info(f"Treinando Teacher ({teacher_arch})...")
        teacher = get_model(teacher_arch, num_classes).to(device)
        teacher_acc, teacher_time = train_direct(
            teacher, train_loader, test_loader, teacher_epochs, lr, device,
            ckpt_mgr, teacher_name
        )
        experiment_state['teacher_done'] = True
        experiment_state['teacher_acc'] = teacher_acc

        # Salvar estado
        all_state = ckpt_mgr.load_state() or {}
        all_state[state_key] = experiment_state
        ckpt_mgr.save_state(all_state)

    teacher.eval()
    teacher_acc = experiment_state['teacher_acc']

    # ========== MÉTODOS ==========
    methods = [
        {'name': 'Direct', 'type': 'direct'},
        {'name': 'TraditionalKD', 'type': 'kd', 'T': 4.0, 'alpha': 0.5},
        {'name': 'HPM-KD', 'type': 'kd', 'T': 6.0, 'alpha': 0.7},
    ]

    results = experiment_state.get('results', [])

    for method in methods:
        method_name = method['name']

        # Verificar se já foi feito
        if any(r['method'] == method_name for r in results):
            logger.info(f"✅ {method_name} já executado, pulando...")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Método: {method_name}")
        logger.info(f"{'='*70}")

        method_accs = []
        method_times = []

        for run in range(n_runs):
            model_name = f"student_{comp_name}_{method_name}_run{run+1}"

            # Verificar se run já foi feito
            if ckpt_mgr.model_exists(model_name):
                logger.info(f"✅ Run {run+1}/{n_runs} já treinado, carregando...")
                student = get_model(student_arch, num_classes).to(device)
                ckpt_mgr.load_model(student, model_name)
                _, acc = evaluate(student, test_loader, nn.CrossEntropyLoss(), device)
                method_accs.append(acc)
                method_times.append(0)  # Tempo não registrado
                continue

            logger.info(f"\n--- Run {run+1}/{n_runs} ---")
            student = get_model(student_arch, num_classes).to(device)

            if method['type'] == 'direct':
                acc, train_time = train_direct(
                    student, train_loader, test_loader, student_epochs, lr, device,
                    ckpt_mgr, model_name
                )
            else:  # kd
                acc, train_time = train_kd(
                    student, teacher, train_loader, test_loader,
                    student_epochs, lr, method['T'], method['alpha'], device,
                    ckpt_mgr, model_name
                )

            method_accs.append(acc)
            method_times.append(train_time)

            logger.info(f"Run {run+1}: {acc:.2f}% em {train_time:.1f}s")

            # Salvar estado após cada run
            all_state = ckpt_mgr.load_state() or {}
            all_state[state_key] = experiment_state
            ckpt_mgr.save_state(all_state)

        # Adicionar resultado
        result = {
            'compression': comp_name,
            'ratio': comp_ratio,
            'method': method_name,
            'teacher_acc': teacher_acc,
            'student_acc_mean': np.mean(method_accs),
            'student_acc_std': np.std(method_accs),
            'retention': 100 * np.mean(method_accs) / teacher_acc,
            'time_mean': np.mean([t for t in method_times if t > 0]),
            'time_std': np.std([t for t in method_times if t > 0]),
            'n_runs': n_runs
        }
        results.append(result)
        experiment_state['results'] = results

        # Salvar estado
        all_state = ckpt_mgr.load_state() or {}
        all_state[state_key] = experiment_state
        ckpt_mgr.save_state(all_state)

        logger.info(f"\n📊 {method_name}: {np.mean(method_accs):.2f}% ± {np.std(method_accs):.2f}%")

    # Resumo
    logger.info(f"\n{'='*70}")
    logger.info(f"RESUMO - {comp_name} ({comp_ratio}x)")
    logger.info(f"{'='*70}")
    logger.info(f"Teacher: {teacher_acc:.2f}%")

    for result in results:
        logger.info(f"{result['method']}: {result['student_acc_mean']:.2f}% ± {result['student_acc_std']:.2f}%")

    direct_acc = next(r['student_acc_mean'] for r in results if r['method'] == 'Direct')
    hpm_acc = next(r['student_acc_mean'] for r in results if r['method'] == 'HPM-KD')
    delta = hpm_acc - direct_acc

    logger.info(f"\nΔ HPM-KD vs Direct: {delta:+.2f}pp {'✅' if delta > 0 else '❌'}")
    logger.info(f"{'='*70}\n")

    return results

# ============================================================================
# VISUALIZAÇÃO E RELATÓRIO (igual ao anterior)
# ============================================================================

def generate_plots(df, output_dir):
    """Gera visualizações."""
    logger.info("Gerando figuras...")

    sns.set_style('whitegrid')
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Figura 1: Accuracy vs Compression
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

    logger.info(f"✅ Figuras salvas em: {fig_dir}")

def generate_report(df, output_dir, mode, dataset, execution_time):
    """Gera relatório."""
    report_path = output_dir / 'experiment_report.md'

    with open(report_path, 'w') as f:
        f.write("# Experimento 1B: Compression Ratios (Kaggle)\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Modo:** {mode.upper()}\n")
        f.write(f"**Dataset:** {dataset}\n")
        f.write(f"**Plataforma:** Kaggle Notebooks\n")
        f.write(f"**Tempo Total:** {execution_time/3600:.2f}h\n\n")

        f.write("---\n\n## 📊 Resultados\n\n")
        f.write("| Compression | Method | Student | ± Std | Retention | Δ vs Direct |\n")
        f.write("|-------------|--------|---------|-------|-----------|-------------|\n")

        for comp in df['compression'].unique():
            comp_data = df[df['compression'] == comp]
            direct_acc = comp_data[comp_data['method'] == 'Direct']['student_acc_mean'].values[0]

            for _, row in comp_data.iterrows():
                delta = row['student_acc_mean'] - direct_acc
                delta_str = f"{delta:+.2f}pp" if row['method'] != 'Direct' else "baseline"

                f.write(f"| {row['compression']} | {row['method']} | "
                       f"{row['student_acc_mean']:.2f}% | ±{row['student_acc_std']:.2f}% | "
                       f"{row['retention']:.1f}% | {delta_str} |\n")

        f.write("\n---\n\n## 🎯 When Does KD Help?\n\n")

        for comp in df['compression'].unique():
            comp_data = df[df['compression'] == comp]
            ratio = comp_data['ratio'].values[0]
            direct_acc = comp_data[comp_data['method'] == 'Direct']['student_acc_mean'].values[0]
            hpmkd_acc = comp_data[comp_data['method'] == 'HPM-KD']['student_acc_mean'].values[0]
            delta = hpmkd_acc - direct_acc

            f.write(f"### {comp} ({ratio}×)\n")
            f.write(f"- HPM-KD vs Direct: {delta:+.2f}pp ")

            if delta > 1.5:
                f.write("✅✅ **HPM-KD vence significativamente!**\n")
            elif delta > 0.5:
                f.write("✅ **HPM-KD vence!**\n")
            elif delta > -0.3:
                f.write("≈ **Empate**\n")
            else:
                f.write("❌ **Direct vence**\n")
            f.write("\n")

        f.write(f"\n---\n**Gerado em:** {datetime.now()}\n")

    logger.info(f"✅ Relatório salvo: {report_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Experimento 1B - Kaggle')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--compression', type=str, default='all', choices=['all', '2.3x', '5x', '7x'])
    parser.add_argument('--resume', action='store_true', help='Retomar execução')

    args = parser.parse_args()

    # Verificar ambiente
    if not check_kaggle_environment():
        logger.error("Ambiente não configurado corretamente!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_BASE / f'exp1b_{args.mode}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Outputs em: {output_dir}")

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(output_dir / 'checkpoints')

    # Compressions
    all_compressions = [
        {'name': '2.3x_ResNet18', 'ratio': 2.3, 'teacher_arch': 'resnet50', 'student_arch': 'resnet18'},
        {'name': '5x_ResNet10', 'ratio': 5.0, 'teacher_arch': 'resnet50', 'student_arch': 'resnet10'},
        {'name': '7x_MobileNetV2', 'ratio': 7.0, 'teacher_arch': 'resnet50', 'student_arch': 'mobilenetv2'},
    ]

    if args.compression != 'all':
        all_compressions = [c for c in all_compressions if c['name'].startswith(args.compression)]

    logger.info(f"\n🚀 Iniciando Experimento 1B")
    logger.info(f"Compressions: {[c['name'] for c in all_compressions]}")
    logger.info(f"Modo: {args.mode}")
    logger.info(f"Resume: {args.resume}\n")

    # Executar
    start_time = time.time()
    all_results = []

    for comp_config in all_compressions:
        results = run_compression_experiment_with_checkpoints(
            comp_config, args.mode, args.dataset, output_dir, device, ckpt_mgr, args.resume
        )
        all_results.extend(results)

    execution_time = time.time() - start_time

    # Salvar
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Resultados: {csv_path}")

    # Visualizações
    generate_plots(df, output_dir)

    # Relatório
    generate_report(df, output_dir, args.mode, args.dataset, execution_time)

    # Resumo
    logger.info("\n" + "=" * 70)
    logger.info("✅ EXPERIMENTO 1B CONCLUÍDO!")
    logger.info("=" * 70)
    logger.info(f"Tempo total: {execution_time/3600:.2f}h")
    logger.info(f"Resultados em: {output_dir}")
    logger.info("=" * 70)

    logger.info("\n📊 Próximos passos:")
    logger.info("  1. Ver relatório: experiment_report.md")
    logger.info("  2. Ver figuras: figures/")
    logger.info("  3. Baixar outputs (Kaggle Output tab)")

if __name__ == '__main__':
    main()
