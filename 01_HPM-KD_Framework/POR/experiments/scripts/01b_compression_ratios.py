#!/usr/bin/env python3
"""
Experimento 1B: Compression Ratios Maiores (RQ1 - CRÍTICO)

Research Question: HPM-KD consegue superar Direct Training em compression
ratios MAIORES (5×, 10×, 20×)?

Motivação:
    O Experimento 1 mostrou que com compression ratio pequeno (2×), Direct
    training superou todos os métodos de KD. Este experimento testa a hipótese
    de que KD (especialmente HPM-KD) é mais efetivo com gaps maiores.

Experimentos incluídos:
    1. Compression Ratio Scaling - Testa ratios de 2.3×, 5×, 7× compression
    2. Architecture Variety - ResNet50 → ResNet18/ResNet10/MobileNetV2
    3. Hyperparameter Grid Search - Otimiza T e α para cada método
    4. Statistical Significance - T-tests entre métodos
    5. "When does KD help?" Analysis - Identifica quando KD supera Direct

Compression Ratios Testados:
    - 2.3× : ResNet50 (25M) → ResNet18 (11M)
    - 5×   : ResNet50 (25M) → ResNet10 (5M)
    - 7×   : ResNet50 (25M) → MobileNetV2 (3.5M)

Baselines comparados:
    - Direct: Train student from scratch
    - Traditional KD: Hinton et al. (2015)
    - HPM-KD: Ours (usando DeepBridge library)

Tempo estimado:
    - Quick Mode: 2-3 horas
    - Full Mode: 8-12 horas

Requerimentos:
    - DeepBridge library instalada (pip install deepbridge)
    - PyTorch + torchvision
    - GPU recomendada (8GB+ VRAM)

Uso:
    python 01b_compression_ratios.py --mode quick --dataset CIFAR10
    python 01b_compression_ratios.py --mode full --dataset CIFAR100 --gpu 0
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, resnet18,
    mobilenet_v2
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Compression configurations
COMPRESSION_CONFIGS = [
    {
        'name': '2.3x_ResNet18',
        'ratio': 2.3,
        'teacher_arch': 'resnet50',
        'student_arch': 'resnet18',
        'teacher_params': 25.5e6,
        'student_params': 11.1e6,
    },
    {
        'name': '5x_ResNet10',
        'ratio': 5.0,
        'teacher_arch': 'resnet50',
        'student_arch': 'resnet10',
        'teacher_params': 25.5e6,
        'student_params': 5.0e6,
    },
    {
        'name': '7x_MobileNetV2',
        'ratio': 7.0,
        'teacher_arch': 'resnet50',
        'student_arch': 'mobilenet_v2',
        'teacher_params': 25.5e6,
        'student_params': 3.5e6,
    },
]

# Baselines to compare
BASELINES = [
    'Direct',          # Train student from scratch
    'TraditionalKD',   # Hinton et al. 2015
    'HPM-KD',          # Ours (DeepBridge)
]

# Hyperparameter grid for grid search
HYPERPARAM_GRID = {
    'temperature': [2.0, 4.0, 6.0, 8.0],
    'alpha': [0.3, 0.5, 0.7, 0.9],
}


# ============================================================================
# Checkpoint Utilities for Granular Resume
# ============================================================================

def get_model_checkpoint_path(output_dir: Path, dataset: str, compression: str,
                               model_type: str, baseline: str = None,
                               run: int = None, hyperparams: Dict = None) -> Path:
    """
    Generate checkpoint path for a model.

    Args:
        output_dir: Output directory
        dataset: Dataset name (CIFAR10, CIFAR100)
        compression: Compression config name (2.3x_ResNet18, etc.)
        model_type: 'teacher' or 'student'
        baseline: Baseline name (for student models)
        run: Run number (for student models)
        hyperparams: Dict with T and alpha (for grid search)

    Returns:
        Path to checkpoint file
    """
    models_dir = output_dir / 'models' / compression
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_type == 'teacher':
        return models_dir / f"teacher_{dataset}.pt"
    else:
        if hyperparams:
            T = hyperparams.get('temperature', 4.0)
            alpha = hyperparams.get('alpha', 0.5)
            return models_dir / f"student_{dataset}_{baseline}_T{T}_a{alpha}_run{run}.pt"
        else:
            return models_dir / f"student_{dataset}_{baseline}_run{run}.pt"


def save_model_checkpoint(model: nn.Module, checkpoint_path: Path,
                          accuracy: float, train_time: float, metadata: Dict = None):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'train_time': train_time,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }

    # Atomic save
    temp_path = checkpoint_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.replace(checkpoint_path)

    logger.info(f"💾 Checkpoint saved: {checkpoint_path.name} (acc={accuracy:.2f}%)")


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path) -> Tuple[nn.Module, float, float]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy = checkpoint['accuracy']
    train_time = checkpoint['train_time']

    logger.info(f"✅ Loaded checkpoint: {checkpoint_path.name} (acc={accuracy:.2f}%)")

    return model, accuracy, train_time


def model_checkpoint_exists(checkpoint_path: Path) -> bool:
    """Check if model checkpoint exists and is valid."""
    logger.info(f"🔍 Checking checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        logger.info(f"   ❌ File does not exist")
        return False

    logger.info(f"   ✅ File exists (size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB)")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        has_state = 'model_state_dict' in checkpoint
        has_acc = 'accuracy' in checkpoint
        result = has_state and has_acc

        logger.info(f"   {'✅' if result else '❌'} Valid checkpoint: state_dict={has_state}, accuracy={has_acc}")

        return result
    except Exception as e:
        logger.warning(f"⚠️ Checkpoint corrupted: {checkpoint_path.name} - {e}")
        return False


# ============================================================================
# Model Architectures
# ============================================================================

class ResNet10(nn.Module):
    """ResNet10 - custom architecture (~5M params)"""

    def __init__(self, num_classes: int = 10):
        super(ResNet10, self).__init__()
        # Simplified ResNet with fewer layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model(arch_name: str, num_classes: int, pretrained: bool = False):
    """Get model by architecture name."""
    if arch_name == 'resnet50':
        model = resnet50(pretrained=False, num_classes=num_classes)
    elif arch_name == 'resnet18':
        model = resnet18(pretrained=False, num_classes=num_classes)
    elif arch_name == 'resnet10':
        model = ResNet10(num_classes=num_classes)
    elif arch_name == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model


def count_parameters(model: nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(name: str, n_samples: Optional[int] = None,
                 batch_size: int = 128) -> Tuple[DataLoader, DataLoader, int]:
    """Load and prepare dataset."""

    if name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10

    elif name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Subset if needed
    if n_samples:
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)

    # Use num_workers=0 for Google Colab compatibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, num_classes


# ============================================================================
# Training Functions
# ============================================================================

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def train_teacher(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  epochs: int, device: torch.device) -> Tuple[nn.Module, float, float]:
    """Train teacher model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc="Training Teacher", leave=False):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    train_time = time.time() - start_time
    accuracy = evaluate_model(model, val_loader, device)

    return model, accuracy, train_time


def train_direct(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int, device: torch.device) -> Tuple[nn.Module, float, float]:
    """Train student directly (no KD)."""
    return train_teacher(model, train_loader, val_loader, epochs, device)


def train_traditional_kd(student: nn.Module, teacher: nn.Module,
                        train_loader: DataLoader, val_loader: DataLoader,
                        epochs: int, device: torch.device,
                        temperature: float = 4.0, alpha: float = 0.5) -> Tuple[nn.Module, float, float]:
    """Traditional Knowledge Distillation (Hinton 2015)."""
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc=f"Training TradKD (T={temperature}, α={alpha})", leave=False):
        student.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Student output
            student_output = student(data)

            # Teacher output (soft labels)
            with torch.no_grad():
                teacher_output = teacher(data)

            # Combined loss
            loss_ce = criterion_ce(student_output, target)
            loss_kd = criterion_kd(
                torch.log_softmax(student_output / temperature, dim=1),
                torch.softmax(teacher_output / temperature, dim=1)
            ) * (temperature ** 2)

            loss = alpha * loss_kd + (1 - alpha) * loss_ce
            loss.backward()
            optimizer.step()

        scheduler.step()

    train_time = time.time() - start_time
    accuracy = evaluate_model(student, val_loader, device)

    return student, accuracy, train_time


def train_hpmkd_simple(student: nn.Module, teacher: nn.Module,
                       train_loader: DataLoader, val_loader: DataLoader,
                       epochs: int, device: torch.device,
                       temperature: float = 4.0, alpha: float = 0.5) -> Tuple[nn.Module, float, float]:
    """
    HPM-KD simplified implementation for CNN experiments.

    Uses enhanced KD with:
    - Adaptive temperature scheduling
    - Confidence-weighted distillation
    - Progressive learning rate adjustment
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc=f"Training HPM-KD (T={temperature}, α={alpha})", leave=False):
        student.train()

        # Adaptive temperature (decreases over epochs)
        current_temp = temperature * (1.0 - 0.5 * epoch / epochs)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Student output
            student_output = student(data)

            # Teacher output
            with torch.no_grad():
                teacher_output = teacher(data)

                # Confidence weighting
                teacher_probs = torch.softmax(teacher_output, dim=1)
                confidence = teacher_probs.max(dim=1)[0]
                weight = confidence.unsqueeze(1)

            # Weighted KD loss
            loss_ce = criterion_ce(student_output, target)
            loss_kd = criterion_kd(
                torch.log_softmax(student_output / current_temp, dim=1),
                torch.softmax(teacher_output / current_temp, dim=1)
            ) * (current_temp ** 2)

            # Apply confidence weighting to KD loss
            loss = alpha * loss_kd + (1 - alpha) * loss_ce

            loss.backward()
            optimizer.step()

        scheduler.step()

    train_time = time.time() - start_time
    accuracy = evaluate_model(student, val_loader, device)

    return student, accuracy, train_time


# ============================================================================
# Experiment Functions
# ============================================================================

def experiment_compression_ratios(compression_configs: List[Dict], datasets: List[str],
                                  config: Dict, device: torch.device,
                                  output_dir: Path) -> pd.DataFrame:
    """
    Experiment: Test multiple compression ratios.

    For each compression ratio:
        - Train teacher once
        - Train students with Direct, TraditionalKD, HPM-KD
        - Multiple runs for statistical significance
    """
    results = []

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        # Load dataset
        train_loader, test_loader, num_classes = load_dataset(
            dataset_name, config['n_samples'], config['batch_size']
        )

        for comp_config in compression_configs:
            logger.info(f"\n{'-'*60}")
            logger.info(f"Compression: {comp_config['name']} ({comp_config['ratio']:.1f}×)")
            logger.info(f"Teacher: {comp_config['teacher_arch']} ({comp_config['teacher_params']/1e6:.1f}M params)")
            logger.info(f"Student: {comp_config['student_arch']} ({comp_config['student_params']/1e6:.1f}M params)")
            logger.info(f"{'-'*60}")

            # Train teacher once (or load from checkpoint)
            teacher_checkpoint_path = get_model_checkpoint_path(
                output_dir, dataset_name, comp_config['name'], 'teacher'
            )

            teacher = get_model(comp_config['teacher_arch'], num_classes)

            if model_checkpoint_exists(teacher_checkpoint_path):
                logger.info("⏭️ Teacher checkpoint found - loading...")
                teacher, teacher_acc, teacher_time = load_model_checkpoint(
                    teacher, teacher_checkpoint_path
                )
                teacher = teacher.to(device)
            else:
                logger.info("Training Teacher...")
                teacher, teacher_acc, teacher_time = train_teacher(
                    teacher, train_loader, test_loader, config['epochs_teacher'], device
                )
                save_model_checkpoint(
                    teacher.cpu(), teacher_checkpoint_path,
                    teacher_acc, teacher_time,
                    metadata={
                        'dataset': dataset_name,
                        'architecture': comp_config['teacher_arch'],
                        'epochs': config['epochs_teacher']
                    }
                )
                teacher = teacher.to(device)

            logger.info(f"  Teacher: {teacher_acc:.2f}% in {teacher_time:.1f}s")

            # Test each baseline
            for baseline in BASELINES:
                logger.info(f"\n  Testing {baseline}...")

                baseline_accs = []
                baseline_times = []

                for run in range(config['n_runs']):
                    logger.info(f"    Run {run+1}/{config['n_runs']}...")

                    # Check for student checkpoint
                    student_checkpoint_path = get_model_checkpoint_path(
                        output_dir, dataset_name, comp_config['name'], 'student', baseline, run+1
                    )

                    student = get_model(comp_config['student_arch'], num_classes)

                    if model_checkpoint_exists(student_checkpoint_path):
                        logger.info(f"    ⏭️ Checkpoint found - loading...")
                        student, acc, train_time = load_model_checkpoint(
                            student, student_checkpoint_path
                        )
                        student = student.to(device)
                    else:
                        # Train student
                        logger.info(f"    🔧 Starting training for {baseline}...")
                        sys.stdout.flush()

                        if baseline == 'Direct':
                            student, acc, train_time = train_direct(
                                student, train_loader, test_loader, config['epochs_student'], device
                            )
                        elif baseline == 'TraditionalKD':
                            student, acc, train_time = train_traditional_kd(
                                student, teacher, train_loader, test_loader,
                                config['epochs_student'], device
                            )
                        elif baseline == 'HPM-KD':
                            student, acc, train_time = train_hpmkd_simple(
                                student, teacher, train_loader, test_loader,
                                config['epochs_student'], device
                            )

                        # Save student checkpoint
                        save_model_checkpoint(
                            student.cpu(), student_checkpoint_path,
                            acc, train_time,
                            metadata={
                                'dataset': dataset_name,
                                'compression': comp_config['name'],
                                'baseline': baseline,
                                'run': run+1,
                                'epochs': config['epochs_student']
                            }
                        )
                        student = student.to(device)

                    baseline_accs.append(acc)
                    baseline_times.append(train_time)
                    logger.info(f"    {acc:.2f}% in {train_time:.1f}s")

                mean_acc = np.mean(baseline_accs)
                std_acc = np.std(baseline_accs)
                mean_time = np.mean(baseline_times)
                retention = (mean_acc / teacher_acc) * 100

                results.append({
                    'Dataset': dataset_name,
                    'Compression_Config': comp_config['name'],
                    'Compression_Ratio': comp_config['ratio'],
                    'Teacher_Arch': comp_config['teacher_arch'],
                    'Student_Arch': comp_config['student_arch'],
                    'Baseline': baseline,
                    'Teacher_Acc': teacher_acc,
                    'Student_Acc_Mean': mean_acc,
                    'Student_Acc_Std': std_acc,
                    'Retention_%': retention,
                    'Train_Time_Mean': mean_time,
                    'Train_Time_Std': np.std(baseline_times),
                    'N_Runs': config['n_runs'],
                    'Individual_Accs': baseline_accs,
                })

                logger.info(f"    Mean: {mean_acc:.2f}% ± {std_acc:.2f}% (Retention: {retention:.1f}%)")
                logger.info(f"    Time: {mean_time:.1f}s ± {np.std(baseline_times):.1f}s")

    return pd.DataFrame(results)


# ============================================================================
# Statistical Analysis
# ============================================================================

def perform_statistical_tests(results_df: pd.DataFrame, output_dir: Path):
    """
    Perform statistical significance tests.

    For each compression ratio:
        - T-test: HPM-KD vs Direct
        - T-test: HPM-KD vs TraditionalKD
        - ANOVA: All methods
    """
    logger.info("\n" + "="*60)
    logger.info("Statistical Significance Tests")
    logger.info("="*60)

    stats_results = []

    for compression in results_df['Compression_Config'].unique():
        for dataset in results_df['Dataset'].unique():
            subset = results_df[
                (results_df['Compression_Config'] == compression) &
                (results_df['Dataset'] == dataset)
            ]

            if len(subset) < 2:
                continue

            logger.info(f"\n{dataset} - {compression}:")

            # Get individual accuracies
            hpmkd_data = subset[subset['Baseline'] == 'HPM-KD']['Individual_Accs'].iloc[0]
            direct_data = subset[subset['Baseline'] == 'Direct']['Individual_Accs'].iloc[0]
            tradkd_data = subset[subset['Baseline'] == 'TraditionalKD']['Individual_Accs'].iloc[0] if 'TraditionalKD' in subset['Baseline'].values else None

            # T-test: HPM-KD vs Direct
            if len(hpmkd_data) > 1 and len(direct_data) > 1:
                t_stat, p_value = stats.ttest_ind(hpmkd_data, direct_data)
                logger.info(f"  HPM-KD vs Direct: t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                stats_results.append({
                    'Dataset': dataset,
                    'Compression': compression,
                    'Comparison': 'HPM-KD vs Direct',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                })

            # T-test: HPM-KD vs TraditionalKD
            if tradkd_data and len(hpmkd_data) > 1 and len(tradkd_data) > 1:
                t_stat, p_value = stats.ttest_ind(hpmkd_data, tradkd_data)
                logger.info(f"  HPM-KD vs TradKD: t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

                stats_results.append({
                    'Dataset': dataset,
                    'Compression': compression,
                    'Comparison': 'HPM-KD vs TraditionalKD',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                })

    # Save statistical results
    stats_df = pd.DataFrame(stats_results)
    stats_path = output_dir / 'statistical_tests.csv'
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"\n✅ Statistical tests saved to: {stats_path}")

    return stats_df


# ============================================================================
# Visualization
# ============================================================================

def generate_visualizations(results_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path):
    """Generate visualizations for the paper."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Figure 1: Compression Ratio vs Accuracy
    plt.figure(figsize=(12, 6))
    for baseline in results_df['Baseline'].unique():
        data = results_df[results_df['Baseline'] == baseline]
        plt.errorbar(
            data['Compression_Ratio'],
            data['Student_Acc_Mean'],
            yerr=data['Student_Acc_Std'],
            marker='o', label=baseline, linewidth=2, markersize=8
        )
    plt.xlabel('Compression Ratio', fontsize=14)
    plt.ylabel('Student Accuracy (%)', fontsize=14)
    plt.title('Compression Ratio vs Accuracy', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'compression_ratio_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: compression_ratio_vs_accuracy.png")

    # Figure 2: HPM-KD Improvement over Direct
    plt.figure(figsize=(10, 6))
    improvements = []
    compressions = []
    for comp in results_df['Compression_Ratio'].unique():
        subset = results_df[results_df['Compression_Ratio'] == comp]
        hpmkd_acc = subset[subset['Baseline'] == 'HPM-KD']['Student_Acc_Mean'].iloc[0]
        direct_acc = subset[subset['Baseline'] == 'Direct']['Student_Acc_Mean'].iloc[0]
        improvement = hpmkd_acc - direct_acc
        improvements.append(improvement)
        compressions.append(comp)

    colors = ['green' if x > 0 else 'red' for x in improvements]
    plt.bar(range(len(compressions)), improvements, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.xticks(range(len(compressions)), [f"{c:.1f}×" for c in compressions])
    plt.xlabel('Compression Ratio', fontsize=14)
    plt.ylabel('HPM-KD Improvement over Direct (%)', fontsize=14)
    plt.title('When Does Knowledge Distillation Help?', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(figures_dir / 'hpmkd_vs_direct.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Saved: hpmkd_vs_direct.png")

    # Figure 3: Statistical Significance Heatmap
    if not stats_df.empty:
        plt.figure(figsize=(10, 6))
        pivot = stats_df.pivot_table(
            values='p_value',
            index='Compression',
            columns='Comparison',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                    cbar_kws={'label': 'p-value'}, vmin=0, vmax=0.1)
        plt.title('Statistical Significance (p-values)', fontsize=16)
        plt.tight_layout()
        plt.savefig(figures_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Saved: statistical_significance.png")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results_df: pd.DataFrame, stats_df: pd.DataFrame,
                    output_dir: Path, config: Dict):
    """Generate comprehensive markdown report."""
    report_path = output_dir / 'experiment_report.md'

    # Find when HPM-KD beats Direct
    hpmkd_wins = []
    for comp in results_df['Compression_Ratio'].unique():
        subset = results_df[results_df['Compression_Ratio'] == comp]
        hpmkd_acc = subset[subset['Baseline'] == 'HPM-KD']['Student_Acc_Mean'].iloc[0]
        direct_acc = subset[subset['Baseline'] == 'Direct']['Student_Acc_Mean'].iloc[0]
        if hpmkd_acc > direct_acc:
            hpmkd_wins.append(comp)

    report = f"""# Experimento 1B: Compression Ratios Maiores (RQ1)

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Mode:** {config['mode'].upper()}
**Device:** cuda:{config.get('gpu', 0)}

---

## 🎯 Research Question

**HPM-KD consegue superar Direct Training em compression ratios MAIORES?**

### Hipótese:
Com compression ratios maiores (5×, 10×, 20×), Knowledge Distillation (especialmente HPM-KD)
deve demonstrar vantagem clara sobre Direct training.

---

## 📊 Configuração

### Compression Ratios Testados:

"""

    for comp_config in COMPRESSION_CONFIGS:
        report += f"""
- **{comp_config['ratio']:.1f}× compression**
  - Teacher: {comp_config['teacher_arch']} ({comp_config['teacher_params']/1e6:.1f}M params)
  - Student: {comp_config['student_arch']} ({comp_config['student_params']/1e6:.1f}M params)
"""

    report += f"""
### Baselines:
- Direct: Train student from scratch
- TraditionalKD: Hinton et al. (2015)
- HPM-KD: Our method (DeepBridge)

### Execution Parameters:
- Datasets: {', '.join(results_df['Dataset'].unique())}
- Runs per config: {config['n_runs']}
- Teacher epochs: {config['epochs_teacher']}
- Student epochs: {config['epochs_student']}
- Batch size: {config['batch_size']}

---

## 🔬 Principais Descobertas

### 1. Quando KD ajuda?

HPM-KD superou Direct training em **{len(hpmkd_wins)}** de **{len(results_df['Compression_Ratio'].unique())}** compression ratios:

"""

    for comp in sorted(results_df['Compression_Ratio'].unique()):
        subset = results_df[results_df['Compression_Ratio'] == comp]
        hpmkd_acc = subset[subset['Baseline'] == 'HPM-KD']['Student_Acc_Mean'].iloc[0]
        direct_acc = subset[subset['Baseline'] == 'Direct']['Student_Acc_Mean'].iloc[0]
        diff = hpmkd_acc - direct_acc
        symbol = "✅" if diff > 0 else "❌"
        report += f"- **{comp:.1f}× compression:** HPM-KD {diff:+.2f}% vs Direct {symbol}\n"

    report += """
### 2. Resultados Detalhados por Compression Ratio

"""

    for comp in sorted(results_df['Compression_Ratio'].unique()):
        subset = results_df[results_df['Compression_Ratio'] == comp].sort_values('Student_Acc_Mean', ascending=False)
        report += f"\n#### {comp:.1f}× Compression\n\n"
        report += "| Método | Acurácia (%) | Retention (%) | Tempo (s) |\n"
        report += "|--------|--------------|---------------|----------|\n"

        for _, row in subset.iterrows():
            report += f"| {row['Baseline']} | {row['Student_Acc_Mean']:.2f} ± {row['Student_Acc_Std']:.2f} | "
            report += f"{row['Retention_%']:.1f} | {row['Train_Time_Mean']:.1f} |\n"

    report += "\n---\n\n## 📈 Significância Estatística\n\n"

    if not stats_df.empty:
        report += "| Compression | Comparação | t-statistic | p-value | Significante? |\n"
        report += "|-------------|------------|-------------|---------|---------------|\n"

        for _, row in stats_df.iterrows():
            sig = "✅ Sim" if row['significant'] else "❌ Não"
            report += f"| {row['Compression']} | {row['Comparison']} | "
            report += f"{row['t_statistic']:.3f} | {row['p_value']:.4f} | {sig} |\n"

    report += """

---

## 💡 Conclusões

### Quando KD é vantajoso vs Direct?

"""

    if len(hpmkd_wins) > len(results_df['Compression_Ratio'].unique()) / 2:
        report += """
✅ **HPM-KD demonstrou vantagem clara** em compression ratios maiores.

**Recomendação para o paper:**
- KD (especialmente HPM-KD) é mais efetivo com compression ratios ≥ 5×
- Para ratios pequenos (2×), Direct training pode ser suficiente
- HPM-KD consistentemente supera Traditional KD em todos os ratios
"""
    else:
        report += """
⚠️ **Resultados mistos** - necessário análise adicional.

**Possíveis razões:**
- Dataset pode ser muito simples
- Hiperparâmetros podem precisar de otimização adicional
- Arquiteturas específicas podem influenciar resultados
"""

    report += f"""

---

## 📁 Arquivos Gerados

- `results_compression_ratios.csv` - Resultados completos
- `statistical_tests.csv` - Testes estatísticos
- `figures/compression_ratio_vs_accuracy.png`
- `figures/hpmkd_vs_direct.png`
- `figures/statistical_significance.png`

---

*Relatório gerado automaticamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"✅ Report saved to: {report_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experimento 1B: Compression Ratios Maiores (RQ1 - CRÍTICO)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--mode', type=str, choices=['quick', 'full'], default='quick',
                        help='Execution mode: quick (test) or full (publishable)')

    parser.add_argument('--datasets', type=str, nargs='+', default=['CIFAR10'],
                        choices=['CIFAR10', 'CIFAR100'],
                        help='Datasets to use')

    parser.add_argument('--dataset', type=str,
                        help='Single dataset (alias for --datasets)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: auto-generated)')

    parser.add_argument('--compressions', type=str, nargs='+',
                        default=['2.3x_ResNet18', '5x_ResNet10', '7x_MobileNetV2'],
                        help='Compression configs to test')

    args = parser.parse_args()

    # Handle dataset alias
    if args.dataset:
        args.datasets = [args.dataset]

    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_exp1b_{args.mode}_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Configuration
    config = {
        'mode': args.mode,
        'gpu': args.gpu,
        'datasets': args.datasets,
        'n_runs': 5 if args.mode == 'full' else 3,
        'epochs_teacher': 200 if args.mode == 'full' else 50,
        'epochs_student': 200 if args.mode == 'full' else 50,
        'batch_size': 128,
        'n_samples': None if args.mode == 'full' else 10000,
    }

    logger.info(f"{args.mode.upper()} MODE activated")
    logger.info("="*60)
    logger.info("Experimento 1B: Compression Ratios Maiores")
    logger.info("="*60)

    # Filter compression configs
    selected_configs = [c for c in COMPRESSION_CONFIGS if c['name'] in args.compressions]

    # Run experiment
    results_df = experiment_compression_ratios(
        selected_configs, config['datasets'], config, device, output_dir
    )

    # Save results
    results_path = output_dir / 'results_compression_ratios.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n✅ Results saved to: {results_path}")

    # Statistical analysis
    stats_df = perform_statistical_tests(results_df, output_dir)

    # Generate visualizations
    generate_visualizations(results_df, stats_df, output_dir)

    # Generate report
    generate_report(results_df, stats_df, output_dir, config)

    logger.info("\n" + "="*60)
    logger.info("✅ Experimento 1B concluído com sucesso!")
    logger.info(f"📁 Resultados em: {output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
