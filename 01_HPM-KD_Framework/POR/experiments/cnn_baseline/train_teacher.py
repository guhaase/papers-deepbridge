#!/usr/bin/env python3
"""
Train Teacher Model (ResNet-18) for MNIST
==========================================

Train a large teacher model to use in knowledge distillation experiments.

Target: 99.3-99.5% accuracy on MNIST

Usage:
    poetry run python train_teacher.py [--epochs 20] [--batch-size 128] [--lr 0.1]

Author: Gustavo Coelho Haase
Date: November 2025
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import time
import json

from cnn_models import create_teacher_model
from utils_training import (
    get_mnist_loaders,
    train_epoch,
    validate,
    save_checkpoint,
    get_optimizer,
    get_scheduler,
    print_model_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Teacher Model for MNIST')

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
    parser.add_argument('--save-name', type=str, default='teacher_resnet18',
                       help='Name for saved model (default: teacher_resnet18)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print("TRAINING TEACHER MODEL (ResNet-18)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Target: 99.3-99.5% accuracy")
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

    # Create model
    print("\nCreating teacher model...")
    model = create_teacher_model(device=device)
    print_model_summary(model, "Teacher (ResNet-18)")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = get_optimizer(
        model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, epochs=args.epochs)

    # Training loop
    print("\nStarting training...")
    print("="*80)

    best_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.epochs} (lr={current_lr:.6f})")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch}/{args.epochs}"
        )

        # Validate
        test_loss, test_acc = validate(
            model, test_loader, criterion, device,
            desc=f"Validation"
        )

        # Update learning rate
        scheduler.step()

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        training_history['lr'].append(current_lr)

        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s")

        # Save checkpoint
        is_best = test_acc > best_accuracy
        if is_best:
            best_accuracy = test_acc
            print(f"✅ New best accuracy: {best_accuracy:.2f}%")

        save_path = save_dir / f"{args.save_name}_epoch{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, test_acc, save_path, is_best)

    # Training complete
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best accuracy: {best_accuracy:.2f}%")
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

    model.eval()
    final_loss, final_acc = validate(
        model, test_loader, criterion, device,
        desc="Final Evaluation"
    )

    print(f"Final Test Accuracy: {final_acc:.2f}%")

    # Check if target accuracy achieved
    if final_acc >= 99.3:
        print("✅ Target accuracy achieved (≥99.3%)!")
    else:
        print(f"⚠️  Target accuracy not achieved. Got {final_acc:.2f}%, target ≥99.3%")

    print("="*80)

    # Save final configuration
    config = {
        'model': 'ResNet-18 (adapted for MNIST)',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'optimizer': 'SGD',
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'scheduler': 'CosineAnnealingLR',
        'final_accuracy': final_acc,
        'best_accuracy': best_accuracy,
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
