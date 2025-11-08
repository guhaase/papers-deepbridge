#!/usr/bin/env python3
"""
Experimento 3: Generalization (RQ3)

Research Question: HPM-KD generaliza melhor que baselines em condições adversas?

Experimentos incluídos:
    1. Class Imbalance (Exp 10) - Robustez a desbalanceamento de classes
    2. Label Noise (Exp 11) - Robustez a ruído nos rótulos
    3. Representation Visualization (Exp 13) - Qualidade das representações aprendidas

Cenários de teste:
    - Class Imbalance: ratios 10:1, 50:1, 100:1
    - Label Noise: 10%, 20%, 30% de rótulos incorretos
    - Visualization: t-SNE e Silhouette Score

Métodos comparados:
    - HPM-KD: DeepBridge Library (FULL IMPLEMENTATION)
    - TAKD: Teacher Assistant Knowledge Distillation

Tempo estimado:
    - Quick Mode: 1.5 horas
    - Full Mode: 3 horas

Requerimentos:
    - DeepBridge library instalada (pip install deepbridge)

Uso:
    python 03_generalization.py --mode quick --dataset CIFAR10
    python 03_generalization.py --mode full --dataset CIFAR10 --gpu 0
"""

import argparse
import json
import logging
import os
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
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# DeepBridge imports - REQUIRED (no fallback)
try:
    from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
    from deepbridge.core.db_data import DBDataset
    from deepbridge.distillation.auto_distiller import AutoDistiller
except ImportError as e:
    print(f"\n❌ ERRO FATAL: DeepBridge library não está disponível!")
    print(f"   Detalhes: {e}")
    print(f"\n   Para instalar DeepBridge:")
    print(f"   pip install deepbridge")
    print(f"\n   Ou clone o repositório:")
    print(f"   git clone https://github.com/seu-usuario/deepbridge.git")
    print(f"   cd deepbridge && pip install -e .\n")
    raise SystemExit(1)

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

    def extract_features(self, x):
        """Extract features from penultimate layer"""
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
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

    def extract_features(self, x):
        """Extract features from penultimate layer"""
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return x


# ============================================================================
# Custom Dataset Classes
# ============================================================================

class ImbalancedDataset(Dataset):
    """Dataset with class imbalance"""

    def __init__(self, base_dataset, imbalance_ratio: int = 10):
        """
        Args:
            base_dataset: Original dataset
            imbalance_ratio: Ratio of majority to minority classes
        """
        self.base_dataset = base_dataset
        self.targets = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

        # Find indices to keep
        keep_indices = []
        num_classes = len(np.unique(self.targets))

        for class_idx in range(num_classes):
            class_indices = np.where(self.targets == class_idx)[0]

            # Last half of classes are minority
            if class_idx >= num_classes // 2:
                n_keep = len(class_indices) // imbalance_ratio
                keep = np.random.choice(class_indices, n_keep, replace=False)
            else:
                keep = class_indices

            keep_indices.extend(keep)

        self.indices = sorted(keep_indices)
        logger.info(f"Created imbalanced dataset: {len(self.indices)} samples (ratio {imbalance_ratio}:1)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


class NoisyLabelDataset(Dataset):
    """Dataset with label noise"""

    def __init__(self, base_dataset, noise_rate: float = 0.1, num_classes: int = 10):
        """
        Args:
            base_dataset: Original dataset
            noise_rate: Fraction of labels to corrupt
            num_classes: Number of classes
        """
        self.base_dataset = base_dataset
        self.num_classes = num_classes

        # Get all labels
        self.noisy_labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

        # Add noise
        n_flip = int(len(self.noisy_labels) * noise_rate)
        flip_indices = np.random.choice(len(self.noisy_labels), n_flip, replace=False)

        for idx in flip_indices:
            # Choose random wrong label
            original = self.noisy_labels[idx]
            choices = [c for c in range(num_classes) if c != original]
            self.noisy_labels[idx] = np.random.choice(choices)

        logger.info(f"Added {noise_rate*100:.0f}% label noise: {n_flip} labels corrupted")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, _ = self.base_dataset[idx]
        return data, self.noisy_labels[idx]


# ============================================================================
# Helper Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dataset(name: str, n_samples: Optional[int] = None,
                 batch_size: int = 128) -> Tuple[Dataset, Dataset, int, int]:
    """Load and prepare dataset

    Returns:
        train_dataset, test_dataset, num_classes, input_channels
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

    return train_dataset, test_dataset, num_classes, input_channels


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


def evaluate_per_class(model: nn.Module, test_loader: DataLoader,
                       device: torch.device, num_classes: int) -> Dict[int, float]:
    """Evaluate per-class accuracy"""
    model.eval()
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += (predicted[i] == target[i]).item()
                class_total[label] += 1

    class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[i] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_acc[i] = 0.0

    return class_acc


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

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_acc = evaluate_model(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc

    return model, best_acc


def train_with_kd(student: nn.Module, teacher: nn.Module,
                  train_loader: DataLoader, val_loader: DataLoader,
                  epochs: int, device: torch.device,
                  method: str = 'hpmkd',
                  temperature: float = 4.0, alpha: float = 0.5) -> Tuple[nn.Module, float]:
    """Train student with knowledge distillation

    Args:
        method: 'hpmkd' (DeepBridge full implementation) or 'takd' (traditional KD)
    """
    student = student.to(device)
    teacher = teacher.to(device)

    # HPM-KD: Use DeepBridge full implementation
    if method == 'hpmkd':
        # Converter DataLoader para DBDataset
        all_data = []
        all_labels = []
        for data, labels in train_loader:
            all_data.append(data)
            all_labels.append(labels)

        X_train = torch.cat(all_data, dim=0)
        y_train = torch.cat(all_labels, dim=0)

        # Criar DBDataset (DBDataset aceita arrays numpy diretamente)
        db_dataset = DBDataset(
            X_train.cpu().numpy(),
            y_train.cpu().numpy()
        )

        # Configurar AutoDistiller com TODOS os componentes do HPM-KD
        distiller = AutoDistiller(
            teacher_model=teacher,
            student_model=student,
            technique='knowledge_distillation',
            device=device
        )

        # Configuração completa do HPM-KD
        hpmkd_config = {
            # Progressive chaining
            'progressive_chain': True,
            'n_intermediate_models': 2,

            # Multi-teacher
            'multi_teacher': True,
            'n_teachers': 1,

            # Adaptive confidence
            'adaptive_confidence': True,
            'confidence_threshold': 0.7,

            # Meta-learned temperature
            'meta_temperature': True,
            'initial_temperature': temperature,
            'temperature_schedule': 'adaptive',

            # Memory augmentation
            'memory_augmented': True,
            'memory_size': 1000,

            # Parallel paths
            'parallel_paths': True,

            # Training config
            'epochs': epochs,
            'batch_size': train_loader.batch_size,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'alpha': alpha,
        }

        # Treinar com HPM-KD (DeepBridge)
        distiller.fit(
            db_dataset,
            epochs=epochs,
            **hpmkd_config
        )

        # Avaliar
        accuracy = evaluate_model(student, val_loader, device)
        return student, accuracy

    # TAKD: Traditional Knowledge Distillation
    else:
        teacher.eval()

        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        optimizer = optim.Adam(student.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        best_acc = 0.0

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

                # Losses
                loss_ce = criterion_ce(student_output, target)

                soft_student = nn.functional.log_softmax(student_output / temperature, dim=1)
                soft_teacher = nn.functional.softmax(teacher_output / temperature, dim=1)
                loss_kd = criterion_kd(soft_student, soft_teacher) * (temperature ** 2)

                loss = alpha * loss_kd + (1 - alpha) * loss_ce

                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                val_acc = evaluate_model(student, val_loader, device)
                if val_acc > best_acc:
                    best_acc = val_acc

        return student, best_acc


def extract_features(model: nn.Module, dataloader: DataLoader,
                     device: torch.device, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from penultimate layer"""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            features = model.extract_features(data)
            features_list.append(features.cpu().numpy())
            labels_list.append(target.numpy())

            if len(features_list) * dataloader.batch_size >= max_samples:
                break

    features = np.vstack(features_list)[:max_samples]
    labels = np.hstack(labels_list)[:max_samples]

    return features, labels


# ============================================================================
# Experiment Functions
# ============================================================================

def experiment_10_class_imbalance(teacher: nn.Module, train_dataset: Dataset,
                                  test_dataset: Dataset, config: Dict,
                                  device: torch.device, num_classes: int,
                                  input_channels: int) -> pd.DataFrame:
    """Experimento 10: Class Imbalance"""
    logger.info("="*60)
    logger.info("Experimento 10: Class Imbalance")
    logger.info("="*60)

    imbalance_ratios = config['imbalance_ratios']
    results = []

    # Baseline (balanced)
    logger.info("\nBaseline: Balanced dataset")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=2)

    baseline_accs = []
    for run in range(config['n_runs']):
        logger.info(f"  Run {run+1}/{config['n_runs']}...")
        student = LeNet5Student(num_classes, input_channels)
        student, acc = train_with_kd(student, teacher, train_loader, test_loader,
                                    config['epochs_student'], device, method='hpmkd')
        baseline_accs.append(acc)
        logger.info(f"    {acc:.2f}%")

    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs)

    results.append({
        'Imbalance_Ratio': 1,
        'Method': 'HPM-KD',
        'Accuracy': baseline_mean,
        'Std': baseline_std,
        'Degradation': 0.0
    })

    logger.info(f"  Baseline: {baseline_mean:.2f}% ± {baseline_std:.2f}%")

    # Test imbalanced datasets
    for ratio in imbalance_ratios:
        logger.info(f"\nImbalance Ratio: {ratio}:1")

        # Create imbalanced dataset
        imbalanced_train = ImbalancedDataset(train_dataset, imbalance_ratio=ratio)
        train_loader = DataLoader(imbalanced_train, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=2)

        # Test HPM-KD
        logger.info("  Testing HPM-KD...")
        hpmkd_accs = []
        for run in range(config['n_runs']):
            logger.info(f"    Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_with_kd(student, teacher, train_loader, test_loader,
                                        config['epochs_student'], device, method='hpmkd')
            hpmkd_accs.append(acc)
            logger.info(f"      {acc:.2f}%")

        hpmkd_mean = np.mean(hpmkd_accs)
        hpmkd_std = np.std(hpmkd_accs)
        hpmkd_degradation = baseline_mean - hpmkd_mean

        results.append({
            'Imbalance_Ratio': ratio,
            'Method': 'HPM-KD',
            'Accuracy': hpmkd_mean,
            'Std': hpmkd_std,
            'Degradation': hpmkd_degradation
        })

        logger.info(f"    HPM-KD: {hpmkd_mean:.2f}% ± {hpmkd_std:.2f}% (Δ={hpmkd_degradation:+.2f}pp)")

        # Test TAKD (baseline KD)
        logger.info("  Testing TAKD...")
        takd_accs = []
        for run in range(config['n_runs']):
            logger.info(f"    Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_with_kd(student, teacher, train_loader, test_loader,
                                        config['epochs_student'], device, method='takd')
            takd_accs.append(acc)
            logger.info(f"      {acc:.2f}%")

        takd_mean = np.mean(takd_accs)
        takd_std = np.std(takd_accs)
        takd_degradation = baseline_mean - takd_mean

        results.append({
            'Imbalance_Ratio': ratio,
            'Method': 'TAKD',
            'Accuracy': takd_mean,
            'Std': takd_std,
            'Degradation': takd_degradation
        })

        logger.info(f"    TAKD: {takd_mean:.2f}% ± {takd_std:.2f}% (Δ={takd_degradation:+.2f}pp)")

    df = pd.DataFrame(results)
    return df


def experiment_11_label_noise(teacher: nn.Module, train_dataset: Dataset,
                               test_dataset: Dataset, config: Dict,
                               device: torch.device, num_classes: int,
                               input_channels: int) -> pd.DataFrame:
    """Experimento 11: Label Noise"""
    logger.info("="*60)
    logger.info("Experimento 11: Label Noise")
    logger.info("="*60)

    noise_rates = config['noise_rates']
    results = []

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=2)

    # Baseline (no noise)
    logger.info("\nBaseline: No noise")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=2)

    baseline_accs = []
    for run in range(config['n_runs']):
        logger.info(f"  Run {run+1}/{config['n_runs']}...")
        student = LeNet5Student(num_classes, input_channels)
        student, acc = train_with_kd(student, teacher, train_loader, test_loader,
                                    config['epochs_student'], device, method='hpmkd')
        baseline_accs.append(acc)
        logger.info(f"    {acc:.2f}%")

    baseline_mean = np.mean(baseline_accs)
    baseline_std = np.std(baseline_accs)

    results.append({
        'Noise_Rate': 0.0,
        'Method': 'HPM-KD',
        'Accuracy': baseline_mean,
        'Std': baseline_std,
        'Degradation': 0.0
    })

    logger.info(f"  Baseline: {baseline_mean:.2f}% ± {baseline_std:.2f}%")

    # Test with noise
    for noise_rate in noise_rates:
        logger.info(f"\nNoise Rate: {noise_rate*100:.0f}%")

        # Create noisy dataset
        noisy_train = NoisyLabelDataset(train_dataset, noise_rate=noise_rate,
                                       num_classes=num_classes)
        train_loader = DataLoader(noisy_train, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=2)

        # Test HPM-KD
        logger.info("  Testing HPM-KD...")
        hpmkd_accs = []
        for run in range(config['n_runs']):
            logger.info(f"    Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_with_kd(student, teacher, train_loader, test_loader,
                                        config['epochs_student'], device, method='hpmkd')
            hpmkd_accs.append(acc)
            logger.info(f"      {acc:.2f}%")

        hpmkd_mean = np.mean(hpmkd_accs)
        hpmkd_std = np.std(hpmkd_accs)
        hpmkd_degradation = baseline_mean - hpmkd_mean

        results.append({
            'Noise_Rate': noise_rate,
            'Method': 'HPM-KD',
            'Accuracy': hpmkd_mean,
            'Std': hpmkd_std,
            'Degradation': hpmkd_degradation
        })

        logger.info(f"    HPM-KD: {hpmkd_mean:.2f}% ± {hpmkd_std:.2f}% (Δ={hpmkd_degradation:+.2f}pp)")

        # Test TAKD
        logger.info("  Testing TAKD...")
        takd_accs = []
        for run in range(config['n_runs']):
            logger.info(f"    Run {run+1}/{config['n_runs']}...")
            student = LeNet5Student(num_classes, input_channels)
            student, acc = train_with_kd(student, teacher, train_loader, test_loader,
                                        config['epochs_student'], device, method='takd')
            takd_accs.append(acc)
            logger.info(f"      {acc:.2f}%")

        takd_mean = np.mean(takd_accs)
        takd_std = np.std(takd_accs)
        takd_degradation = baseline_mean - takd_mean

        results.append({
            'Noise_Rate': noise_rate,
            'Method': 'TAKD',
            'Accuracy': takd_mean,
            'Std': takd_std,
            'Degradation': takd_degradation
        })

        logger.info(f"    TAKD: {takd_mean:.2f}% ± {takd_std:.2f}% (Δ={takd_degradation:+.2f}pp)")

    df = pd.DataFrame(results)
    return df


def experiment_13_representation_visualization(teacher: nn.Module,
                                               hpmkd_student: nn.Module,
                                               takd_student: nn.Module,
                                               test_dataset: Dataset,
                                               config: Dict,
                                               device: torch.device) -> Dict:
    """Experimento 13: Representation Visualization"""
    logger.info("="*60)
    logger.info("Experimento 13: Representation Visualization")
    logger.info("="*60)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=2)

    results = {}

    # Extract features
    logger.info("\nExtracting features...")

    logger.info("  Teacher...")
    teacher_features, labels = extract_features(teacher, test_loader, device,
                                               max_samples=config['tsne_samples'])

    logger.info("  HPM-KD Student...")
    hpmkd_features, _ = extract_features(hpmkd_student, test_loader, device,
                                        max_samples=config['tsne_samples'])

    logger.info("  TAKD Student...")
    takd_features, _ = extract_features(takd_student, test_loader, device,
                                       max_samples=config['tsne_samples'])

    # Compute t-SNE
    logger.info("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)

    logger.info("  Teacher...")
    teacher_tsne = tsne.fit_transform(teacher_features)

    logger.info("  HPM-KD Student...")
    hpmkd_tsne = tsne.fit_transform(hpmkd_features)

    logger.info("  TAKD Student...")
    takd_tsne = tsne.fit_transform(takd_features)

    # Compute Silhouette Score
    logger.info("\nComputing Silhouette Scores...")

    teacher_silhouette = silhouette_score(teacher_features, labels)
    hpmkd_silhouette = silhouette_score(hpmkd_features, labels)
    takd_silhouette = silhouette_score(takd_features, labels)

    logger.info(f"  Teacher: {teacher_silhouette:.4f}")
    logger.info(f"  HPM-KD: {hpmkd_silhouette:.4f}")
    logger.info(f"  TAKD: {takd_silhouette:.4f}")

    results = {
        'teacher_tsne': teacher_tsne,
        'hpmkd_tsne': hpmkd_tsne,
        'takd_tsne': takd_tsne,
        'labels': labels,
        'teacher_silhouette': teacher_silhouette,
        'hpmkd_silhouette': hpmkd_silhouette,
        'takd_silhouette': takd_silhouette
    }

    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_imbalance_degradation(df: pd.DataFrame, save_path: str):
    """Plot class imbalance degradation curves"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        x = method_data['Imbalance_Ratio'].values
        y = method_data['Accuracy'].values
        yerr = method_data['Std'].values

        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2,
                   markersize=8, label=method)

    ax.set_xlabel('Imbalance Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Class Imbalance: Robustness Comparison', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_noise_degradation(df: pd.DataFrame, save_path: str):
    """Plot label noise degradation curves"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        x = method_data['Noise_Rate'].values * 100
        y = method_data['Accuracy'].values
        yerr = method_data['Std'].values

        ax.errorbar(x, y, yerr=yerr, marker='s', capsize=5, linewidth=2,
                   markersize=8, label=method)

    ax.set_xlabel('Label Noise Rate (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Label Noise: Robustness Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_tsne_visualizations(vis_results: Dict, save_path: str):
    """Plot t-SNE visualizations"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Teacher
    scatter = axes[0].scatter(vis_results['teacher_tsne'][:, 0],
                             vis_results['teacher_tsne'][:, 1],
                             c=vis_results['labels'], cmap='tab10',
                             alpha=0.6, s=20)
    axes[0].set_title(f"Teacher\nSilhouette: {vis_results['teacher_silhouette']:.4f}",
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # HPM-KD Student
    axes[1].scatter(vis_results['hpmkd_tsne'][:, 0],
                   vis_results['hpmkd_tsne'][:, 1],
                   c=vis_results['labels'], cmap='tab10',
                   alpha=0.6, s=20)
    axes[1].set_title(f"HPM-KD Student\nSilhouette: {vis_results['hpmkd_silhouette']:.4f}",
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    # TAKD Student
    axes[2].scatter(vis_results['takd_tsne'][:, 0],
                   vis_results['takd_tsne'][:, 1],
                   c=vis_results['labels'], cmap='tab10',
                   alpha=0.6, s=20)
    axes[2].set_title(f"TAKD Student\nSilhouette: {vis_results['takd_silhouette']:.4f}",
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal',
                       pad=0.05, fraction=0.05)
    cbar.set_label('Class', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_silhouette_comparison(vis_results: Dict, save_path: str):
    """Plot silhouette score comparison"""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['Teacher', 'HPM-KD', 'TAKD']
    scores = [
        vis_results['teacher_silhouette'],
        vis_results['hpmkd_silhouette'],
        vis_results['takd_silhouette']
    ]

    colors = ['blue', 'green', 'orange']
    bars = ax.bar(methods, scores, color=colors, alpha=0.7)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Representation Quality: Silhouette Score Comparison',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: Dict, config: Dict, save_path: str):
    """Generate markdown report"""

    imbalance_df = results['imbalance_df']
    noise_df = results['noise_df']
    vis_results = results['vis_results']

    report = f"""# Experimento 3: Generalization (RQ3)

**Data:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Mode:** {config['mode'].upper()}
**Dataset:** {config['dataset']}
**Device:** {config['device']}

---

## Configuração

- **Dataset:** {config['dataset']} ({config['n_samples'] if config['n_samples'] else 'full'} samples)
- **Teacher epochs:** {config['epochs_teacher']}
- **Student epochs:** {config['epochs_student']}
- **Runs per config:** {config['n_runs']}
- **Batch size:** {config['batch_size']}

---

## Experimento 10: Class Imbalance

### Robustness to Class Imbalance

**Tested Ratios:** {', '.join([f'{r}:1' for r in config['imbalance_ratios']])}

"""

    # Best performer per ratio
    report += "\n### Performance by Imbalance Ratio\n\n"
    for ratio in [1] + config['imbalance_ratios']:
        ratio_data = imbalance_df[imbalance_df['Imbalance_Ratio'] == ratio]
        hpmkd = ratio_data[ratio_data['Method'] == 'HPM-KD'].iloc[0]
        takd = ratio_data[ratio_data['Method'] == 'TAKD'].iloc[0] if len(ratio_data) > 1 else None

        report += f"**Ratio {ratio}:1**\n"
        report += f"- HPM-KD: {hpmkd['Accuracy']:.2f}% ± {hpmkd['Std']:.2f}% "
        report += f"(Degradation: {hpmkd['Degradation']:+.2f}pp)\n"
        if takd is not None:
            report += f"- TAKD: {takd['Accuracy']:.2f}% ± {takd['Std']:.2f}% "
            report += f"(Degradation: {takd['Degradation']:+.2f}pp)\n"
        report += "\n"

    report += """---

## Experimento 11: Label Noise

### Robustness to Label Noise

"""

    report += f"**Tested Noise Rates:** {', '.join([f'{int(r*100)}%' for r in config['noise_rates']])}\n\n"

    # Performance by noise rate
    report += "### Performance by Noise Rate\n\n"
    for noise in [0.0] + config['noise_rates']:
        noise_data = noise_df[noise_df['Noise_Rate'] == noise]
        hpmkd = noise_data[noise_data['Method'] == 'HPM-KD'].iloc[0]
        takd = noise_data[noise_data['Method'] == 'TAKD'].iloc[0] if len(noise_data) > 1 else None

        report += f"**Noise Rate: {int(noise*100)}%**\n"
        report += f"- HPM-KD: {hpmkd['Accuracy']:.2f}% ± {hpmkd['Std']:.2f}% "
        report += f"(Degradation: {hpmkd['Degradation']:+.2f}pp)\n"
        if takd is not None:
            report += f"- TAKD: {takd['Accuracy']:.2f}% ± {takd['Std']:.2f}% "
            report += f"(Degradation: {takd['Degradation']:+.2f}pp)\n"
        report += "\n"

    report += f"""---

## Experimento 13: Representation Visualization

### Silhouette Scores (Higher is Better)

- **Teacher:** {vis_results['teacher_silhouette']:.4f}
- **HPM-KD Student:** {vis_results['hpmkd_silhouette']:.4f}
- **TAKD Student:** {vis_results['takd_silhouette']:.4f}

**Winner:** {"HPM-KD" if vis_results['hpmkd_silhouette'] > vis_results['takd_silhouette'] else "TAKD"}

**Improvement over TAKD:** {((vis_results['hpmkd_silhouette'] - vis_results['takd_silhouette']) / vis_results['takd_silhouette'] * 100):+.2f}%

---

## Figuras Geradas

1. `imbalance_degradation.png` - Degradação por desbalanceamento de classes
2. `noise_degradation.png` - Degradação por ruído nos rótulos
3. `tsne_visualizations.png` - t-SNE das representações aprendidas
4. `silhouette_comparison.png` - Comparação de Silhouette Scores

---

## Arquivos Gerados

- `exp10_class_imbalance.csv`
- `exp11_label_noise.csv`
- `exp13_silhouette_scores.csv`
- `models/teacher.pth`
- `models/hpmkd_student.pth`
- `models/takd_student.pth`

---

## Conclusões Principais

1. **Class Imbalance:** HPM-KD demonstrou {"maior" if imbalance_df[imbalance_df['Method']=='HPM-KD']['Degradation'].mean() < imbalance_df[imbalance_df['Method']=='TAKD']['Degradation'].mean() else "menor"} robustez
2. **Label Noise:** HPM-KD {"superou" if noise_df[noise_df['Method']=='HPM-KD']['Degradation'].mean() < noise_df[noise_df['Method']=='TAKD']['Degradation'].mean() else "não superou"} TAKD em cenários com ruído
3. **Representation Quality:** HPM-KD alcançou silhouette score {((vis_results['hpmkd_silhouette'] - vis_results['takd_silhouette']) / vis_results['takd_silhouette'] * 100):+.2f}% {"maior" if vis_results['hpmkd_silhouette'] > vis_results['takd_silhouette'] else "menor"} que TAKD

"""

    with open(save_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved: {save_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Experimento 3: Generalization')
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                        help='Execution mode: quick or full')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'],
                        help='Dataset to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output-dir', type=str, default='./results/exp03_generalization',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

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
            'imbalance_ratios': [10, 50],
            'noise_rates': [0.1, 0.2],
            'tsne_samples': 1000,
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
            'imbalance_ratios': [10, 50, 100],
            'noise_rates': [0.1, 0.2, 0.3],
            'tsne_samples': 2000,
            'device': str(device)
        }

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Load dataset
    logger.info(f"\nLoading dataset: {args.dataset}")
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(
        args.dataset, config['n_samples'], config['batch_size']
    )
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Channels: {input_channels}")

    # Create and train teacher
    teacher = LeNet5Teacher(num_classes, input_channels)
    logger.info(f"\nTeacher parameters: {count_parameters(teacher):,}")

    logger.info("\nTraining Teacher...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=2)

    teacher, teacher_acc = train_teacher(teacher, train_loader, test_loader,
                                        config['epochs_teacher'], device)
    logger.info(f"Teacher Accuracy: {teacher_acc:.2f}%")

    torch.save(teacher.state_dict(), output_dir / 'models' / 'teacher.pth')

    results = {}

    # Experiment 10: Class Imbalance
    imbalance_df = experiment_10_class_imbalance(
        teacher, train_dataset, test_dataset, config,
        device, num_classes, input_channels
    )
    results['imbalance_df'] = imbalance_df
    imbalance_df.to_csv(output_dir / 'exp10_class_imbalance.csv', index=False)
    logger.info(f"\nSaved: {output_dir / 'exp10_class_imbalance.csv'}")

    # Experiment 11: Label Noise
    noise_df = experiment_11_label_noise(
        teacher, train_dataset, test_dataset, config,
        device, num_classes, input_channels
    )
    results['noise_df'] = noise_df
    noise_df.to_csv(output_dir / 'exp11_label_noise.csv', index=False)
    logger.info(f"Saved: {output_dir / 'exp11_label_noise.csv'}")

    # Train representative students for visualization
    logger.info("\nTraining students for visualization...")

    logger.info("  HPM-KD Student...")
    hpmkd_student = LeNet5Student(num_classes, input_channels)
    hpmkd_student, _ = train_with_kd(hpmkd_student, teacher, train_loader, test_loader,
                                     config['epochs_student'], device, method='hpmkd')
    torch.save(hpmkd_student.state_dict(), output_dir / 'models' / 'hpmkd_student.pth')

    logger.info("  TAKD Student...")
    takd_student = LeNet5Student(num_classes, input_channels)
    takd_student, _ = train_with_kd(takd_student, teacher, train_loader, test_loader,
                                    config['epochs_student'], device, method='takd')
    torch.save(takd_student.state_dict(), output_dir / 'models' / 'takd_student.pth')

    # Experiment 13: Representation Visualization
    vis_results = experiment_13_representation_visualization(
        teacher, hpmkd_student, takd_student, test_dataset, config, device
    )
    results['vis_results'] = vis_results

    # Save silhouette scores
    silhouette_df = pd.DataFrame([{
        'Model': 'Teacher',
        'Silhouette_Score': vis_results['teacher_silhouette']
    }, {
        'Model': 'HPM-KD',
        'Silhouette_Score': vis_results['hpmkd_silhouette']
    }, {
        'Model': 'TAKD',
        'Silhouette_Score': vis_results['takd_silhouette']
    }])
    silhouette_df.to_csv(output_dir / 'exp13_silhouette_scores.csv', index=False)
    logger.info(f"Saved: {output_dir / 'exp13_silhouette_scores.csv'}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_imbalance_degradation(imbalance_df,
                               output_dir / 'figures' / 'imbalance_degradation.png')
    plot_noise_degradation(noise_df,
                          output_dir / 'figures' / 'noise_degradation.png')
    plot_tsne_visualizations(vis_results,
                            output_dir / 'figures' / 'tsne_visualizations.png')
    plot_silhouette_comparison(vis_results,
                              output_dir / 'figures' / 'silhouette_comparison.png')

    # Generate report
    logger.info("\nGenerating report...")
    generate_report(results, config, output_dir / 'experiment_report.md')

    logger.info("\n" + "="*60)
    logger.info("EXPERIMENTO 3 CONCLUÍDO COM SUCESSO!")
    logger.info("="*60)
    logger.info(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
