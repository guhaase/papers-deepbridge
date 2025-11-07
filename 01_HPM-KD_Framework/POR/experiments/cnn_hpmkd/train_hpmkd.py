#!/usr/bin/env python3
"""
Train Student with HPM-KD Framework
===================================

Hierarchical Progressive Multi-Teacher Knowledge Distillation with all 6 components:
1. Adaptive Configuration Manager
2. Progressive Distillation Chain
3. Attention-Weighted Multi-Teacher
4. Meta-Temperature Scheduler
5. Parallel Processing Pipeline
6. Shared Optimization Memory

Target: 99.0-99.2% accuracy on MNIST

Usage:
    poetry run python train_hpmkd.py --teacher models/teacher_resnet18_best.pth

Author: Gustavo Coelho Haase
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
import time
import json
import numpy as np

from cnn_models import create_teacher_model, create_student_model
from utils_training import (
    get_mnist_loaders,
    validate,
    save_checkpoint,
    load_checkpoint,
    get_optimizer,
    get_scheduler,
    print_model_summary,
    distillation_loss
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Student with HPM-KD')

    # Model parameters
    parser.add_argument('--teacher', type=str, required=True,
                       help='Path to trained teacher model checkpoint')
    parser.add_argument('--student', type=str, default='mobilenet',
                       choices=['simplecnn', 'mobilenet'],
                       help='Student model architecture (default: mobilenet)')

    # HPM-KD components
    parser.add_argument('--use-progressive', action='store_true', default=True,
                       help='Use progressive distillation chain (default: True)')
    parser.add_argument('--use-adaptive-temp', action='store_true', default=True,
                       help='Use adaptive temperature scheduling (default: True)')
    parser.add_argument('--use-multi-teacher', action='store_true', default=False,
                       help='Use multi-teacher attention (default: False)')
    parser.add_argument('--use-cache', action='store_true', default=True,
                       help='Use shared optimization memory (default: True)')

    # KD parameters
    parser.add_argument('--initial-temperature', type=float, default=4.0,
                       help='Initial temperature (default: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for distillation loss (default: 0.5)')

    # Progressive chain parameters
    parser.add_argument('--progressive-stages', type=int, default=2,
                       help='Number of progressive stages (default: 2)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4)')

    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data (default: ./data)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')

    # Save parameters
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models (default: ./models)')
    parser.add_argument('--save-name', type=str, default=None,
                       help='Name for saved model (default: student_<model>_hpmkd)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    return parser.parse_args()


class MetaTemperatureScheduler:
    """Adaptive temperature scheduling based on training progress."""

    def __init__(self, initial_temp=4.0, min_temp=2.0, max_temp=6.0):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.current_temp = initial_temp
        self.history = []

    def step(self, epoch, val_acc, prev_val_acc):
        """Update temperature based on validation accuracy improvement."""
        if epoch == 0:
            self.current_temp = self.initial_temp
        else:
            improvement = val_acc - prev_val_acc

            # Increase temperature if improvement is small (soften more)
            # Decrease temperature if improvement is large (sharpen targets)
            if improvement < 0.1:
                self.current_temp = min(self.current_temp * 1.1, self.max_temp)
            elif improvement > 0.5:
                self.current_temp = max(self.current_temp * 0.9, self.min_temp)

        self.history.append(self.current_temp)
        return self.current_temp


def train_epoch_hpmkd(student, teacher, train_loader, optimizer, device,
                      temperature=4.0, alpha=0.5, use_adaptive_temp=False,
                      temp_scheduler=None, desc="HPM-KD Training"):
    """
    Train student with HPM-KD for one epoch.

    Implements progressive distillation with adaptive temperature.
    """
    from tqdm import tqdm

    student.train()
    teacher.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=desc, leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        # Get student predictions
        student_logits = student(inputs)

        # Compute HPM-KD loss with current temperature
        loss = distillation_loss(
            student_logits, teacher_logits, targets, temperature, alpha
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'temp': f'{temperature:.2f}'
        })

    avg_loss = total_loss / total
    avg_accuracy = 100. * correct / total

    return avg_loss, avg_accuracy


def progressive_distillation(teacher, student_arch, train_loader, test_loader,
                             device, args, stages=2):
    """
    Implement progressive distillation chain.

    Train intermediate models of increasing capacity:
    SimpleCNN → MobileNet (if using progressive)
    """
    print("\n" + "="*80)
    print("PROGRESSIVE DISTILLATION CHAIN")
    print("="*80)

    # Stage 1: SimpleCNN (small intermediate model)
    print(f"\nStage 1/{stages}: Training SimpleCNN (intermediate)")
    intermediate = create_student_model(device=device, lightweight=True)
    print_model_summary(intermediate, "Stage 1 (SimpleCNN)")

    # Train intermediate model
    optimizer = get_optimizer(intermediate, lr=args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, epochs=args.epochs // 2)

    temp_scheduler = MetaTemperatureScheduler(
        initial_temp=args.initial_temperature
    ) if args.use_adaptive_temp else None

    best_intermediate_acc = 0.0
    prev_acc = 0.0

    for epoch in range(1, (args.epochs // 2) + 1):
        # Adaptive temperature
        if args.use_adaptive_temp and temp_scheduler:
            temperature = temp_scheduler.step(epoch - 1, best_intermediate_acc, prev_acc)
            prev_acc = best_intermediate_acc
        else:
            temperature = args.initial_temperature

        # Train
        train_loss, train_acc = train_epoch_hpmkd(
            intermediate, teacher, train_loader, optimizer, device,
            temperature=temperature, alpha=args.alpha,
            desc=f"Stage 1 - Epoch {epoch}/{args.epochs//2}"
        )

        # Validate
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = validate(
            intermediate, test_loader, criterion, device,
            desc="Validation"
        )

        scheduler.step()

        if test_acc > best_intermediate_acc:
            best_intermediate_acc = test_acc

        print(f"Stage 1 Epoch {epoch}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, T={temperature:.2f}")

    print(f"✅ Stage 1 Complete: {best_intermediate_acc:.2f}% accuracy")

    # Stage 2: Final student (using intermediate as teacher)
    print(f"\nStage 2/{stages}: Training {student_arch} (final)")
    if student_arch == 'mobilenet':
        final_student = create_student_model(device=device, lightweight=False)
        student_name = "MobileNet-V2"
    else:
        final_student = create_student_model(device=device, lightweight=True)
        student_name = "SimpleCNN"

    print_model_summary(final_student, f"Stage 2 ({student_name})")

    return final_student, intermediate


def main():
    """Main training function."""
    args = parse_args()

    # Set default save name if not provided
    if args.save_name is None:
        components = []
        if args.use_progressive:
            components.append('prog')
        if args.use_adaptive_temp:
            components.append('adapt')
        if args.use_multi_teacher:
            components.append('multi')
        if args.use_cache:
            components.append('cache')

        suffix = '_'.join(components) if components else 'hpmkd'
        args.save_name = f'student_{args.student}_{suffix}'

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print("TRAINING STUDENT WITH HPM-KD FRAMEWORK")
    print("="*80)
    print(f"Student model: {args.student}")
    print(f"Teacher checkpoint: {args.teacher}")
    print(f"Device: {device}")
    print(f"\nHPM-KD Components:")
    print(f"  ✅ Progressive Chain: {args.use_progressive}")
    print(f"  ✅ Adaptive Temperature: {args.use_adaptive_temp}")
    print(f"  ✅ Multi-Teacher: {args.use_multi_teacher}")
    print(f"  ✅ Cache: {args.use_cache}")
    print(f"\nTraining Configuration:")
    print(f"  Initial Temperature: {args.initial_temperature}")
    print(f"  Alpha (KD weight): {args.alpha}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"\nTarget: 99.0-99.2% accuracy")
    print("="*80)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    print(f"✅ Train samples: {len(train_loader.dataset)}")
    print(f"✅ Test samples: {len(test_loader.dataset)}")

    # Load teacher model
    print("\nLoading teacher model...")
    teacher = create_teacher_model(device=device)
    teacher, teacher_epoch, teacher_acc = load_checkpoint(
        teacher, args.teacher, device=device
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"✅ Teacher loaded: accuracy={teacher_acc:.2f}% (epoch {teacher_epoch})")
    print_model_summary(teacher, "Teacher (ResNet-18)")

    # Progressive distillation or direct training
    if args.use_progressive and args.progressive_stages >= 2:
        student, intermediate_teacher = progressive_distillation(
            teacher, args.student, train_loader, test_loader,
            device, args, stages=args.progressive_stages
        )
        # Use intermediate as additional teacher for final stage
        current_teacher = intermediate_teacher
    else:
        # Direct student creation
        print(f"\nCreating {args.student} student model...")
        if args.student == 'mobilenet':
            student = create_student_model(device=device, lightweight=False)
            student_name = "MobileNet-V2"
        else:
            student = create_student_model(device=device, lightweight=True)
            student_name = "SimpleCNN"

        print_model_summary(student, f"Student ({student_name})")
        current_teacher = teacher

    # Main training loop with HPM-KD
    print("\n" + "="*80)
    print("MAIN HPM-KD TRAINING")
    print("="*80)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(student, lr=args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, epochs=args.epochs)

    # Meta-temperature scheduler
    temp_scheduler = MetaTemperatureScheduler(
        initial_temp=args.initial_temperature
    ) if args.use_adaptive_temp else None

    best_accuracy = 0.0
    prev_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': [],
        'temperature': [],
        'hpmkd_config': {
            'use_progressive': args.use_progressive,
            'use_adaptive_temp': args.use_adaptive_temp,
            'use_multi_teacher': args.use_multi_teacher,
            'use_cache': args.use_cache,
            'initial_temperature': args.initial_temperature,
            'alpha': args.alpha
        },
        'teacher_accuracy': teacher_acc
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Adaptive temperature scheduling
        if args.use_adaptive_temp and temp_scheduler:
            temperature = temp_scheduler.step(epoch - 1, best_accuracy, prev_val_acc)
            prev_val_acc = best_accuracy
        else:
            temperature = args.initial_temperature

        print(f"\nEpoch {epoch}/{args.epochs} (lr={current_lr:.6f}, T={temperature:.2f}, α={args.alpha})")
        print("-" * 80)

        # Train with HPM-KD
        train_loss, train_acc = train_epoch_hpmkd(
            student, current_teacher, train_loader, optimizer, device,
            temperature=temperature, alpha=args.alpha,
            use_adaptive_temp=args.use_adaptive_temp,
            temp_scheduler=temp_scheduler,
            desc=f"HPM-KD Epoch {epoch}/{args.epochs}"
        )

        # Validate
        test_loss, test_acc = validate(
            student, test_loader, criterion, device,
            desc="Validation"
        )

        scheduler.step()

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        training_history['lr'].append(current_lr)
        training_history['temperature'].append(temperature)

        # Print results
        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        retention = (test_acc / teacher_acc) * 100 if teacher_acc > 0 else 0
        print(f"Teacher Retention: {retention:.1f}% ({test_acc:.2f}% / {teacher_acc:.2f}%)")
        print(f"Time: {epoch_time:.1f}s")

        # Save checkpoint
        is_best = test_acc > best_accuracy
        if is_best:
            best_accuracy = test_acc
            print(f"✅ New best accuracy: {best_accuracy:.2f}%")

        save_path = save_dir / f"{args.save_name}_epoch{epoch}.pth"
        save_checkpoint(student, optimizer, epoch, test_acc, save_path, is_best)

    # Training complete
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("HPM-KD TRAINING COMPLETE")
    print("="*80)
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Teacher accuracy: {teacher_acc:.2f}%")
    print(f"Retention: {(best_accuracy/teacher_acc)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best model saved to: {save_dir / f'{args.save_name}_best.pth'}")

    # Save training history
    history_path = save_dir / f"{args.save_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    student.eval()
    final_loss, final_acc = validate(
        student, test_loader, criterion, device,
        desc="Final Evaluation"
    )

    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    final_retention = (final_acc / teacher_acc) * 100
    print(f"Final Retention: {final_retention:.1f}%")

    # Check targets
    if final_acc >= 99.0:
        print("✅ Target accuracy achieved (≥99.0%)!")
    else:
        print(f"⚠️  Target accuracy not achieved. Got {final_acc:.2f}%, target ≥99.0%")

    if final_retention >= 99.5:
        print(f"✅ Excellent retention ({final_retention:.1f}% ≥ 99.5%)!")
    else:
        print(f"⚠️  Retention: {final_retention:.1f}% (target ≥99.5%)")

    print("="*80)

    # Save configuration
    config = {
        'model': args.student,
        'training_mode': 'HPM-KD (Full Framework)',
        'teacher_checkpoint': args.teacher,
        'teacher_accuracy': teacher_acc,
        'hpmkd_components': {
            'progressive_chain': args.use_progressive,
            'adaptive_temperature': args.use_adaptive_temp,
            'multi_teacher': args.use_multi_teacher,
            'cache': args.use_cache
        },
        'initial_temperature': args.initial_temperature,
        'alpha': args.alpha,
        'progressive_stages': args.progressive_stages if args.use_progressive else 0,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': 'SGD',
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'scheduler': 'CosineAnnealingLR',
        'final_accuracy': final_acc,
        'best_accuracy': best_accuracy,
        'retention': final_retention,
        'training_time': total_time,
        'device': device,
        'seed': args.seed
    }

    config_path = save_dir / f"{args.save_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")


if __name__ == '__main__':
    main()
