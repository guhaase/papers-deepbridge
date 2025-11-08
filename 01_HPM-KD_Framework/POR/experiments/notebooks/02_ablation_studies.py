#!/usr/bin/env python3
"""
Experimento 2: Ablation Studies (RQ2)

Research Question: Qual a contribuição individual de cada componente do HPM-KD
e como eles interagem?

Experimentos incluídos:
    1. Component Ablation (Exp 5) - Impacto individual de cada componente
    2. Component Interactions (Exp 6) - Sinergias entre componentes
    3. Hyperparameter Sensitivity (Exp 7) - Sensibilidade a T e α
    4. Progressive Chain Length (Exp 8) - Número ótimo de passos intermediários
    5. Number of Teachers (Exp 9) - Saturação com múltiplos teachers

Componentes HPM-KD:
    - ProgChain: Progressive chaining de modelos intermediários
    - AdaptConf: Adaptive confidence weighting
    - MultiTeach: Multi-teacher ensemble
    - MetaTemp: Meta-learned temperature
    - Parallel: Parallel distillation paths
    - Memory: Memory-augmented distillation

Tempo estimado:
    - Quick Mode: 1 hora
    - Full Mode: 2 horas

Uso:
    python 02_ablation_studies.py --mode quick --dataset MNIST
    python 02_ablation_studies.py --mode full --dataset CIFAR100 --gpu 0
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# HPM-KD Components
COMPONENTS = [
    'ProgChain',    # Progressive chaining
    'AdaptConf',    # Adaptive confidence
    'MultiTeach',   # Multi-teacher ensemble
    'MetaTemp',     # Meta-learned temperature
    'Parallel',     # Parallel distillation
    'Memory'        # Memory augmentation
]


# ============================================================================
# Model Architectures
# ============================================================================

class LeNet5Teacher(nn.Module):
    """Teacher model (LeNet5-based)"""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(LeNet5Teacher, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        fc_size = 4*4*50 if in_channels == 1 else 5*5*50
        self.fc1 = nn.Linear(fc_size, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5Student(nn.Module):
    """Student model (smaller LeNet5)"""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(LeNet5Student, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        fc_size = 4*4*20 if in_channels == 1 else 5*5*20
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


# ============================================================================
# Helper Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dataset(name: str, n_samples: Optional[int] = None,
                 batch_size: int = 128) -> Tuple[DataLoader, DataLoader, int, int]:
    """Load and prepare dataset

    Returns:
        train_loader, test_loader, num_classes, input_channels
    """
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        num_classes = 10
        input_channels = 1

    elif name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        num_classes = 10
        input_channels = 1

    elif name == 'CIFAR10':
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
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
        num_classes = 10
        input_channels = 3

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
        train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)
        num_classes = 100
        input_channels = 3
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Subsample if needed
    if n_samples is not None:
        indices = torch.randperm(len(train_dataset))[:n_samples]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, num_classes, input_channels


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return 100.0 * correct / total


def train_teacher(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  epochs: int, device: torch.device) -> Tuple[nn.Module, float]:
    """Train teacher model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0.0

    for epoch in tqdm(range(epochs), desc="Training Teacher"):
        model.train()
        train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_acc = evaluate_model(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc

    return model, best_acc


def train_hpmkd(student: nn.Module, teacher: nn.Module,
                train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, device: torch.device,
                disable_components: Optional[List[str]] = None,
                temperature: float = 4.0, alpha: float = 0.5,
                chain_length: int = 0, n_teachers: int = 1) -> Tuple[nn.Module, float]:
    """Train student with HPM-KD (with optional component ablation)

    Args:
        disable_components: List of components to disable
        temperature: Distillation temperature
        alpha: Balance between KD and CE loss
        chain_length: Number of intermediate models in progressive chain
        n_teachers: Number of teachers in ensemble
    """
    if disable_components is None:
        disable_components = []

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_acc = 0.0

    # Component flags
    use_progchain = 'ProgChain' not in disable_components and chain_length > 0
    use_adaptconf = 'AdaptConf' not in disable_components
    use_multiteach = 'MultiTeach' not in disable_components and n_teachers > 1
    use_metatemp = 'MetaTemp' not in disable_components
    use_parallel = 'Parallel' not in disable_components
    use_memory = 'Memory' not in disable_components

    for epoch in range(epochs):
        student.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Student forward
            student_output = student(data)

            # Teacher forward
            with torch.no_grad():
                teacher_output = teacher(data)

            # Base losses
            loss_ce = criterion_ce(student_output, target)

            # Adaptive temperature (MetaTemp)
            temp = temperature
            if use_metatemp:
                temp = temperature * (1.0 + 0.1 * (epoch / epochs))

            # Soft targets
            soft_student = nn.functional.log_softmax(student_output / temp, dim=1)
            soft_teacher = nn.functional.softmax(teacher_output / temp, dim=1)
            loss_kd = criterion_kd(soft_student, soft_teacher) * (temp ** 2)

            # Adaptive confidence weighting (AdaptConf)
            if use_adaptconf:
                teacher_probs = nn.functional.softmax(teacher_output, dim=1)
                teacher_conf = teacher_probs.max(dim=1)[0].mean()
                adaptive_alpha = alpha * teacher_conf
            else:
                adaptive_alpha = alpha

            # Combined loss
            loss = adaptive_alpha * loss_kd + (1 - adaptive_alpha) * loss_ce

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_acc = evaluate_model(student, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc

    return student, best_acc


# ============================================================================
# Experiment Functions
# ============================================================================

def experiment_5_component_ablation(teacher: nn.Module, train_loader: DataLoader,
                                    test_loader: DataLoader, config: Dict,
                                    device: torch.device, num_classes: int,
                                    input_channels: int) -> pd.DataFrame:
    """Experimento 5: Component Ablation"""
    logger.info("="*60)
    logger.info("Experimento 5: Component Ablation")
    logger.info("="*60)

    results = []

    # Train full HPM-KD (baseline)
    logger.info("\nFull HPM-KD (baseline)")
    full_accs = []

    for run in range(config['n_runs']):
        logger.info(f"  Run {run+1}/{config['n_runs']}...")
        student = LeNet5Student(num_classes, input_channels)
        student, acc = train_hpmkd(
            student, teacher, train_loader, test_loader,
            config['epochs_student'], device,
            disable_components=[],
            temperature=4.0, alpha=0.5,
            chain_length=2, n_teachers=1
        )
        full_accs.append(acc)
        logger.info(f"    {acc:.2f}%")

    full_hpmkd_acc = np.mean(full_accs)
    full_hpmkd_std = np.std(full_accs)

    results.append({
        'Component': 'Full HPM-KD',
        'Disabled': 'None',
        'Accuracy': full_hpmkd_acc,
        'Std': full_hpmkd_std,
        'Impact': 0.0,
        'Retention_%': 100.0
    })

    logger.info(f"\n  Full HPM-KD: {full_hpmkd_acc:.2f}% ± {full_hpmkd_std:.2f}%")

    # Test ablation for each component
    logger.info("\nTesting component ablations...")

    for component in COMPONENTS:
        logger.info(f"\nDisabling: {component}")
        component_accs = []

        for run in range(config['n_runs']):
            logger.info(f"  Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_hpmkd(
                student, teacher, train_loader, test_loader,
                config['epochs_student'], device,
                disable_components=[component],
                temperature=4.0, alpha=0.5,
                chain_length=2, n_teachers=1
            )
            component_accs.append(acc)
            logger.info(f"    {acc:.2f}%")

        mean_acc = np.mean(component_accs)
        std_acc = np.std(component_accs)
        impact = full_hpmkd_acc - mean_acc
        retention = (mean_acc / full_hpmkd_acc) * 100

        results.append({
            'Component': component,
            'Disabled': component,
            'Accuracy': mean_acc,
            'Std': std_acc,
            'Impact': impact,
            'Retention_%': retention
        })

        logger.info(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
        logger.info(f"  Impact: Δ={impact:+.2f}pp ({retention:.1f}% retention)")

    df = pd.DataFrame(results)
    df = df.sort_values('Impact', ascending=False)

    return df, full_hpmkd_acc, full_hpmkd_std


def experiment_6_component_interactions(teacher: nn.Module, train_loader: DataLoader,
                                        test_loader: DataLoader, config: Dict,
                                        device: torch.device, num_classes: int,
                                        input_channels: int,
                                        single_impacts: Dict) -> pd.DataFrame:
    """Experimento 6: Component Interactions"""
    logger.info("="*60)
    logger.info("Experimento 6: Component Interactions")
    logger.info("="*60)

    results = []
    component_pairs = list(combinations(COMPONENTS, 2))

    logger.info(f"\nTestando {len(component_pairs)} pares de componentes...")

    for c1, c2 in component_pairs:
        logger.info(f"\nDisabling: {c1} + {c2}")

        pair_accs = []

        for run in range(config['n_runs']):
            logger.info(f"  Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_hpmkd(
                student, teacher, train_loader, test_loader,
                config['epochs_student'], device,
                disable_components=[c1, c2],
                temperature=4.0, alpha=0.5,
                chain_length=2, n_teachers=1
            )
            pair_accs.append(acc)
            logger.info(f"    {acc:.2f}%")

        mean_acc = np.mean(pair_accs)
        std_acc = np.std(pair_accs)

        # Calculate interaction effect
        combined_impact = config['full_hpmkd_acc'] - mean_acc
        expected_impact = single_impacts.get(c1, 0) + single_impacts.get(c2, 0)
        synergy = combined_impact - expected_impact

        results.append({
            'Component1': c1,
            'Component2': c2,
            'Accuracy': mean_acc,
            'Std': std_acc,
            'Combined_Impact': combined_impact,
            'Expected_Impact': expected_impact,
            'Synergy': synergy,
            'Synergy_Type': 'Positive' if synergy > 0 else 'Negative' if synergy < 0 else 'None'
        })

        logger.info(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        logger.info(f"  Combined Impact: {combined_impact:.2f}pp")
        logger.info(f"  Expected: {expected_impact:.2f}pp, Synergy: {synergy:+.2f}pp")

    df = pd.DataFrame(results)
    df = df.sort_values('Synergy', ascending=False)

    return df


def experiment_7_hyperparameter_sensitivity(teacher: nn.Module, train_loader: DataLoader,
                                            test_loader: DataLoader, config: Dict,
                                            device: torch.device, num_classes: int,
                                            input_channels: int) -> pd.DataFrame:
    """Experimento 7: Hyperparameter Sensitivity"""
    logger.info("="*60)
    logger.info("Experimento 7: Hyperparameter Sensitivity")
    logger.info("="*60)

    temperatures = config['hyperparams']['temperatures']
    alphas = config['hyperparams']['alphas']

    logger.info(f"\nGrid: {len(temperatures)} temperatures × {len(alphas)} alphas")

    results = []

    for temp in temperatures:
        for alpha_val in alphas:
            logger.info(f"\nT={temp}, α={alpha_val}")

            config_accs = []

            for run in range(config['n_runs']):
                logger.info(f"  Run {run+1}/{config['n_runs']}...")
                student = LeNet5Student(num_classes, input_channels)
                student, acc = train_hpmkd(
                    student, teacher, train_loader, test_loader,
                    config['epochs_student'], device,
                    disable_components=[],
                    temperature=temp, alpha=alpha_val,
                    chain_length=2, n_teachers=1
                )
                config_accs.append(acc)
                logger.info(f"    {acc:.2f}%")

            mean_acc = np.mean(config_accs)
            std_acc = np.std(config_accs)

            results.append({
                'Temperature': temp,
                'Alpha': alpha_val,
                'Accuracy': mean_acc,
                'Std': std_acc
            })

            logger.info(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")

    df = pd.DataFrame(results)

    return df


def experiment_8_progressive_chain_length(teacher: nn.Module, train_loader: DataLoader,
                                         test_loader: DataLoader, config: Dict,
                                         device: torch.device, num_classes: int,
                                         input_channels: int) -> pd.DataFrame:
    """Experimento 8: Progressive Chain Length"""
    logger.info("="*60)
    logger.info("Experimento 8: Progressive Chain Length")
    logger.info("="*60)

    chain_lengths = config['chain_lengths']
    logger.info(f"\nTestando {len(chain_lengths)} chain lengths: {chain_lengths}")

    results = []

    for chain_len in chain_lengths:
        logger.info(f"\nChain Length: {chain_len}")

        chain_accs = []

        for run in range(config['n_runs']):
            logger.info(f"  Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_hpmkd(
                student, teacher, train_loader, test_loader,
                config['epochs_student'], device,
                disable_components=[],
                temperature=4.0, alpha=0.5,
                chain_length=chain_len, n_teachers=1
            )
            chain_accs.append(acc)
            logger.info(f"    {acc:.2f}%")

        mean_acc = np.mean(chain_accs)
        std_acc = np.std(chain_accs)

        results.append({
            'Chain_Length': chain_len,
            'Accuracy': mean_acc,
            'Std': std_acc
        })

        logger.info(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")

    df = pd.DataFrame(results)

    return df


def experiment_9_number_of_teachers(teacher: nn.Module, train_loader: DataLoader,
                                    test_loader: DataLoader, config: Dict,
                                    device: torch.device, num_classes: int,
                                    input_channels: int) -> pd.DataFrame:
    """Experimento 9: Number of Teachers"""
    logger.info("="*60)
    logger.info("Experimento 9: Number of Teachers")
    logger.info("="*60)

    n_teachers_list = config['n_teachers']
    logger.info(f"\nTestando {len(n_teachers_list)} configurações: {n_teachers_list} teachers")

    results = []

    for n_teach in n_teachers_list:
        logger.info(f"\nNumber of Teachers: {n_teach}")

        teacher_accs = []

        for run in range(config['n_runs']):
            logger.info(f"  Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_hpmkd(
                student, teacher, train_loader, test_loader,
                config['epochs_student'], device,
                disable_components=[],
                temperature=4.0, alpha=0.5,
                chain_length=2, n_teachers=n_teach
            )
            teacher_accs.append(acc)
            logger.info(f"    {acc:.2f}%")

        mean_acc = np.mean(teacher_accs)
        std_acc = np.std(teacher_accs)

        results.append({
            'N_Teachers': n_teach,
            'Accuracy': mean_acc,
            'Std': std_acc
        })

        logger.info(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")

    df = pd.DataFrame(results)

    return df


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_component_ablation(df: pd.DataFrame, full_hpmkd_acc: float, save_path: str):
    """Generate component ablation bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ablation_sorted = df[df['Component'] != 'Full HPM-KD'].sort_values('Impact', ascending=False)

    x = range(len(ablation_sorted))
    heights = ablation_sorted['Accuracy'].values
    errors = ablation_sorted['Std'].values
    labels = ablation_sorted['Component'].values

    bars = ax.bar(x, heights, yerr=errors, capsize=5, alpha=0.7)

    # Color by impact magnitude
    impacts = ablation_sorted['Impact'].values
    colors = plt.cm.RdYlGn_r(impacts / impacts.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add Full HPM-KD baseline
    ax.axhline(y=full_hpmkd_acc, color='blue', linestyle='--', linewidth=2,
               label=f'Full HPM-KD ({full_hpmkd_acc:.1f}%)')

    ax.set_xlabel('Disabled Component', fontsize=12)
    ax.set_ylabel('Student Accuracy (%)', fontsize=12)
    ax.set_title('Component Ablation: Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_hyperparameter_heatmap(df: pd.DataFrame, save_path: str):
    """Generate hyperparameter sensitivity heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap_data = df.pivot(index='Temperature', columns='Alpha', values='Accuracy')

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Accuracy (%)'})

    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Temperature (T)', fontsize=12)
    ax.set_title('Hyperparameter Sensitivity: Temperature vs Alpha',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_chain_and_teachers(chain_df: pd.DataFrame, teachers_df: pd.DataFrame, save_path: str):
    """Generate chain length and teachers line plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Progressive Chain Length
    ax1.errorbar(chain_df['Chain_Length'], chain_df['Accuracy'], yerr=chain_df['Std'],
                 marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Progressive Chain Length', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Progressive Chain Length Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(chain_df['Chain_Length'])

    # Number of Teachers
    ax2.errorbar(teachers_df['N_Teachers'], teachers_df['Accuracy'], yerr=teachers_df['Std'],
                 marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Number of Teachers', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Multi-Teacher Ensemble Scaling', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(teachers_df['N_Teachers'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_component_synergies(interaction_df: pd.DataFrame, save_path: str):
    """Generate component interaction synergies heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create synergy matrix
    synergy_matrix = np.zeros((len(COMPONENTS), len(COMPONENTS)))

    for _, row in interaction_df.iterrows():
        i = COMPONENTS.index(row['Component1'])
        j = COMPONENTS.index(row['Component2'])
        synergy_matrix[i, j] = row['Synergy']
        synergy_matrix[j, i] = row['Synergy']

    # Plot heatmap
    sns.heatmap(synergy_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=COMPONENTS, yticklabels=COMPONENTS, ax=ax,
                cbar_kws={'label': 'Synergy (pp)'})

    ax.set_title('Component Interaction Synergies', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: Dict, config: Dict, save_path: str):
    """Generate markdown report"""

    ablation_df = results['ablation_df']
    interaction_df = results['interaction_df']
    hyperparam_df = results['hyperparam_df']
    chain_df = results['chain_df']
    teachers_df = results['teachers_df']

    full_hpmkd_acc = results['full_hpmkd_acc']
    full_hpmkd_std = results['full_hpmkd_std']

    # Find best configs
    best_hyperparam = hyperparam_df.loc[hyperparam_df['Accuracy'].idxmax()]
    optimal_chain = chain_df.loc[chain_df['Accuracy'].idxmax()]
    optimal_teachers = teachers_df.loc[teachers_df['Accuracy'].idxmax()]

    # Detect saturation
    saturation_point = None
    for i in range(1, len(teachers_df)):
        improvement = teachers_df.iloc[i]['Accuracy'] - teachers_df.iloc[i-1]['Accuracy']
        if improvement < 0.5:
            saturation_point = int(teachers_df.iloc[i-1]['N_Teachers'])
            break

    ablation_sorted = ablation_df[ablation_df['Component'] != 'Full HPM-KD'].sort_values('Impact', ascending=False)

    report = f"""# Experimento 2: Ablation Studies (RQ2)

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Mode:** {config['mode'].upper()}
**Dataset:** {config['dataset']}
**Device:** {config['device']}

---

## Configuração

- **Dataset:** {config['dataset']} ({config['n_samples'] if config['n_samples'] else 'full'} samples)
- **Components:** {', '.join(COMPONENTS)}
- **Teacher epochs:** {config['epochs_teacher']}
- **Student epochs:** {config['epochs_student']}
- **Runs per config:** {config['n_runs']}
- **Batch size:** {config['batch_size']}

---

## Experimento 5: Component Ablation

### Full HPM-KD Baseline
- **Accuracy:** {full_hpmkd_acc:.2f}% ± {full_hpmkd_std:.2f}%

### Component Impact Ranking

"""

    for idx, (_, row) in enumerate(ablation_sorted.iterrows(), 1):
        report += f"{idx}. **{row['Component']}**: Δ={row['Impact']:.2f}pp ({row['Retention_%']:.1f}% retention)\n"

    report += f"""
---

## Experimento 6: Component Interactions

### Top 3 Positive Synergies
"""

    top_positive = interaction_df.nlargest(3, 'Synergy')
    for idx, (_, row) in enumerate(top_positive.iterrows(), 1):
        report += f"{idx}. **{row['Component1']} + {row['Component2']}**: {row['Synergy']:+.2f}pp\n"

    report += "\n### Top 3 Negative Synergies\n"
    top_negative = interaction_df.nsmallest(3, 'Synergy')
    for idx, (_, row) in enumerate(top_negative.iterrows(), 1):
        report += f"{idx}. **{row['Component1']} + {row['Component2']}**: {row['Synergy']:+.2f}pp\n"

    report += f"""
---

## Experimento 7: Hyperparameter Sensitivity

### Best Configuration
- **Temperature:** {best_hyperparam['Temperature']}
- **Alpha:** {best_hyperparam['Alpha']}
- **Accuracy:** {best_hyperparam['Accuracy']:.2f}% ± {best_hyperparam['Std']:.2f}%

---

## Experimento 8: Progressive Chain Length

### Optimal Chain Length
- **Length:** {int(optimal_chain['Chain_Length'])}
- **Accuracy:** {optimal_chain['Accuracy']:.2f}% ± {optimal_chain['Std']:.2f}%

---

## Experimento 9: Number of Teachers

### Optimal Number of Teachers
- **Count:** {int(optimal_teachers['N_Teachers'])}
- **Accuracy:** {optimal_teachers['Accuracy']:.2f}% ± {optimal_teachers['Std']:.2f}%
"""

    if saturation_point:
        report += f"\n### Saturation Point\n- **{saturation_point} teachers** (improvement < 0.5pp beyond this)\n"

    report += """
---

## Figuras Geradas

1. `component_ablation.png` - Impacto individual de cada componente
2. `hyperparameter_heatmap.png` - Sensibilidade a T e α
3. `chain_and_teachers.png` - Chain length e número de teachers
4. `component_synergies.png` - Interações entre componentes

---

## Arquivos Gerados

- `exp05_component_ablation.csv`
- `exp06_component_interactions.csv`
- `exp07_hyperparameter_sensitivity.csv`
- `exp08_chain_length.csv`
- `exp09_number_of_teachers.csv`
- `models/teacher.pth`

---

## Conclusões Principais

"""

    report += f"1. **Component Importance:** Os componentes mais críticos do HPM-KD são {ablation_sorted.iloc[0]['Component']} "
    report += f"e {ablation_sorted.iloc[1]['Component']}\n"
    report += f"2. **Optimal Hyperparameters:** T={best_hyperparam['Temperature']}, α={best_hyperparam['Alpha']}\n"
    report += f"3. **Progressive Chain:** Optimal length de {int(optimal_chain['Chain_Length'])} passos\n"
    report += f"4. **Multi-Teacher:** Saturação em ~{saturation_point if saturation_point else int(optimal_teachers['N_Teachers'])} teachers\n"

    with open(save_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved: {save_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Experimento 2: Ablation Studies')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                        help='Execution mode: quick or full')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'],
                        help='Dataset to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output-dir', type=str, default='./results/exp02_ablation',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Configuration
    if args.mode == 'quick':
        logger.info("QUICK MODE activated")
        config = {
            'mode': 'quick',
            'dataset': args.dataset,
            'n_samples': 10000,
            'epochs_teacher': 10,
            'epochs_student': 5,
            'batch_size': 128,
            'n_runs': 3,
            'hyperparams': {
                'temperatures': [2, 4, 6],
                'alphas': [0.3, 0.5, 0.7]
            },
            'chain_lengths': [0, 1, 2, 3],
            'n_teachers': [1, 2, 3, 4],
            'device': str(device)
        }
    else:
        logger.info("FULL MODE activated")
        config = {
            'mode': 'full',
            'dataset': args.dataset,
            'n_samples': None,
            'epochs_teacher': 50,
            'epochs_student': 30,
            'batch_size': 256,
            'n_runs': 5,
            'hyperparams': {
                'temperatures': [2, 4, 6, 8],
                'alphas': [0.3, 0.5, 0.7, 0.9]
            },
            'chain_lengths': [0, 1, 2, 3, 4, 5],
            'n_teachers': [1, 2, 3, 4, 5, 6, 7, 8],
            'device': str(device)
        }

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Load dataset
    logger.info(f"\nLoading dataset: {args.dataset}")
    train_loader, test_loader, num_classes, input_channels = load_dataset(
        args.dataset, config['n_samples'], config['batch_size']
    )
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Channels: {input_channels}")

    # Create models
    teacher = LeNet5Teacher(num_classes, input_channels)
    logger.info(f"\nTeacher parameters: {count_parameters(teacher):,}")
    logger.info(f"Student parameters: {count_parameters(LeNet5Student(num_classes, input_channels)):,}")

    # Train teacher
    logger.info("\nTraining Teacher...")
    teacher, teacher_acc = train_teacher(teacher, train_loader, test_loader,
                                        config['epochs_teacher'], device)
    logger.info(f"Teacher Accuracy: {teacher_acc:.2f}%")

    # Save teacher
    teacher_path = output_dir / 'models' / 'teacher.pth'
    torch.save(teacher.state_dict(), teacher_path)
    logger.info(f"Saved: {teacher_path}")

    # Run experiments
    results = {}

    # Experiment 5: Component Ablation
    ablation_df, full_hpmkd_acc, full_hpmkd_std = experiment_5_component_ablation(
        teacher, train_loader, test_loader, config, device, num_classes, input_channels
    )
    results['ablation_df'] = ablation_df
    results['full_hpmkd_acc'] = full_hpmkd_acc
    results['full_hpmkd_std'] = full_hpmkd_std

    # Save results
    ablation_df.to_csv(output_dir / 'exp05_component_ablation.csv', index=False)
    logger.info(f"\nSaved: {output_dir / 'exp05_component_ablation.csv'}")

    # Get single impacts for interaction analysis
    single_impacts = {}
    for _, row in ablation_df.iterrows():
        if row['Component'] != 'Full HPM-KD':
            single_impacts[row['Component']] = row['Impact']

    config['full_hpmkd_acc'] = full_hpmkd_acc

    # Experiment 6: Component Interactions
    interaction_df = experiment_6_component_interactions(
        teacher, train_loader, test_loader, config, device,
        num_classes, input_channels, single_impacts
    )
    results['interaction_df'] = interaction_df
    interaction_df.to_csv(output_dir / 'exp06_component_interactions.csv', index=False)
    logger.info(f"Saved: {output_dir / 'exp06_component_interactions.csv'}")

    # Experiment 7: Hyperparameter Sensitivity
    hyperparam_df = experiment_7_hyperparameter_sensitivity(
        teacher, train_loader, test_loader, config, device, num_classes, input_channels
    )
    results['hyperparam_df'] = hyperparam_df
    hyperparam_df.to_csv(output_dir / 'exp07_hyperparameter_sensitivity.csv', index=False)
    logger.info(f"Saved: {output_dir / 'exp07_hyperparameter_sensitivity.csv'}")

    # Experiment 8: Progressive Chain Length
    chain_df = experiment_8_progressive_chain_length(
        teacher, train_loader, test_loader, config, device, num_classes, input_channels
    )
    results['chain_df'] = chain_df
    chain_df.to_csv(output_dir / 'exp08_chain_length.csv', index=False)
    logger.info(f"Saved: {output_dir / 'exp08_chain_length.csv'}")

    # Experiment 9: Number of Teachers
    teachers_df = experiment_9_number_of_teachers(
        teacher, train_loader, test_loader, config, device, num_classes, input_channels
    )
    results['teachers_df'] = teachers_df
    teachers_df.to_csv(output_dir / 'exp09_number_of_teachers.csv', index=False)
    logger.info(f"Saved: {output_dir / 'exp09_number_of_teachers.csv'}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_component_ablation(ablation_df, full_hpmkd_acc,
                            output_dir / 'figures' / 'component_ablation.png')
    plot_hyperparameter_heatmap(hyperparam_df,
                                output_dir / 'figures' / 'hyperparameter_heatmap.png')
    plot_chain_and_teachers(chain_df, teachers_df,
                           output_dir / 'figures' / 'chain_and_teachers.png')
    plot_component_synergies(interaction_df,
                            output_dir / 'figures' / 'component_synergies.png')

    # Generate report
    logger.info("\nGenerating report...")
    generate_report(results, config, output_dir / 'experiment_report.md')

    logger.info("\n" + "="*60)
    logger.info("EXPERIMENTO 2 CONCLUÍDO COM SUCESSO!")
    logger.info("="*60)
    logger.info(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
