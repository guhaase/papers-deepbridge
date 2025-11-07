#!/usr/bin/env python3
"""
Comprehensive Evaluation of All Models
======================================

Evaluate and compare all trained models:
- Teacher (ResNet-18)
- Student Direct (SimpleCNN/MobileNet)
- Student Traditional KD
- Student HPM-KD

Generates:
- Accuracy comparison table
- Confusion matrices
- Retention analysis
- Feature visualizations (optional)
- Statistical significance tests

Usage:
    poetry run python evaluate_all.py --models models/*.pth --output results/

Author: Gustavo Coelho Haase
Date: November 2025
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from cnn_models import create_teacher_model, create_student_model
from utils_training import (
    get_mnist_loaders,
    validate,
    load_checkpoint
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate All Models')

    # Model parameters
    parser.add_argument('--teacher', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--student-direct', type=str, default=None,
                       help='Path to direct student checkpoint (optional)')
    parser.add_argument('--student-kd', type=str, default=None,
                       help='Path to traditional KD checkpoint (optional)')
    parser.add_argument('--student-hpmkd', type=str, default=None,
                       help='Path to HPM-KD checkpoint (optional)')

    # Model architectures
    parser.add_argument('--student-arch', type=str, default='mobilenet',
                       choices=['simplecnn', 'mobilenet'],
                       help='Student architecture (default: mobilenet)')

    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save-confusion', action='store_true', default=True,
                       help='Save confusion matrices')
    parser.add_argument('--save-figures', action='store_true', default=True,
                       help='Save comparison figures')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')

    return parser.parse_args()


def evaluate_model(model, test_loader, device, model_name="Model"):
    """
    Comprehensive evaluation of a single model.

    Returns:
        dict with accuracy, loss, predictions, targets, probabilities
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEvaluating {model_name}...")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)

            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            # Statistics
            total_loss += loss.item() * inputs.size(0)
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = 100. * correct / total

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities)
    }

    print(f"✅ {model_name}: {accuracy:.2f}% accuracy ({correct}/{total})")

    return results


def plot_confusion_matrix(targets, predictions, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved: {save_path}")


def plot_comparison_chart(results_dict, save_path):
    """Plot accuracy comparison chart."""
    models = []
    accuracies = []

    for name, results in results_dict.items():
        models.append(name)
        accuracies.append(results['accuracy'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#06A77D'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(accuracies) - 5, 100])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparison chart saved: {save_path}")


def plot_retention_chart(results_dict, teacher_acc, save_path):
    """Plot retention percentage chart."""
    models = []
    retentions = []

    for name, results in results_dict.items():
        if name != 'Teacher':
            models.append(name)
            retention = (results['accuracy'] / teacher_acc) * 100
            retentions.append(retention)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(models, retentions, color=['#A23B72', '#F18F01', '#06A77D'])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add target line at 99.5%
    ax.axhline(y=99.5, color='red', linestyle='--', linewidth=2, label='Target (99.5%)')

    ax.set_ylabel('Teacher Knowledge Retention (%)', fontsize=12)
    ax.set_title('Knowledge Retention Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([min(retentions) - 5, 105])
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Retention chart saved: {save_path}")


def generate_comparison_table(results_dict, teacher_acc, output_path):
    """Generate comprehensive comparison table."""
    data = []

    for name, results in results_dict.items():
        retention = (results['accuracy'] / teacher_acc) * 100 if name != 'Teacher' else 100.0

        data.append({
            'Model': name,
            'Accuracy (%)': f"{results['accuracy']:.2f}",
            'Loss': f"{results['loss']:.4f}",
            'Correct': results['correct'],
            'Total': results['total'],
            'Retention (%)': f"{retention:.2f}"
        })

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = output_path.parent / (output_path.stem + '.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Results table saved: {csv_path}")

    # Save as formatted text
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CNN MODELS EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")

    print(f"✅ Results table saved: {output_path}")

    return df


def calculate_improvement_stats(results_dict):
    """Calculate improvement statistics."""
    if 'Student (Direct)' in results_dict and 'Student (HPM-KD)' in results_dict:
        direct_acc = results_dict['Student (Direct)']['accuracy']
        hpmkd_acc = results_dict['Student (HPM-KD)']['accuracy']
        improvement = hpmkd_acc - direct_acc

        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS")
        print("="*80)
        print(f"Direct Training:     {direct_acc:.2f}%")
        print(f"HPM-KD:             {hpmkd_acc:.2f}%")
        print(f"Absolute Improvement: +{improvement:.2f}pp")
        print("="*80)

    if 'Student (KD)' in results_dict and 'Student (HPM-KD)' in results_dict:
        kd_acc = results_dict['Student (KD)']['accuracy']
        hpmkd_acc = results_dict['Student (HPM-KD)']['accuracy']
        improvement = hpmkd_acc - kd_acc

        print("\n" + "="*80)
        print("HPM-KD vs TRADITIONAL KD")
        print("="*80)
        print(f"Traditional KD:      {kd_acc:.2f}%")
        print(f"HPM-KD:             {hpmkd_acc:.2f}%")
        print(f"Improvement:         +{improvement:.2f}pp")
        print("="*80)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Student architecture: {args.student_arch}")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    print("\nLoading MNIST test set...")
    _, test_loader = get_mnist_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    print(f"✅ Test samples: {len(test_loader.dataset)}")

    # Evaluate all models
    results_dict = {}

    # 1. Evaluate Teacher
    print("\n" + "="*80)
    print("EVALUATING TEACHER MODEL")
    print("="*80)

    teacher = create_teacher_model(device=device)
    teacher, _, _ = load_checkpoint(teacher, args.teacher, device=device)
    results_dict['Teacher'] = evaluate_model(
        teacher, test_loader, device, "Teacher (ResNet-18)"
    )
    teacher_acc = results_dict['Teacher']['accuracy']

    # 2. Evaluate Direct Student (if provided)
    if args.student_direct:
        print("\n" + "="*80)
        print("EVALUATING DIRECT STUDENT")
        print("="*80)

        if args.student_arch == 'mobilenet':
            student_direct = create_student_model(device=device, lightweight=False)
        else:
            student_direct = create_student_model(device=device, lightweight=True)

        student_direct, _, _ = load_checkpoint(student_direct, args.student_direct, device=device)
        results_dict['Student (Direct)'] = evaluate_model(
            student_direct, test_loader, device, f"Student (Direct - {args.student_arch})"
        )

    # 3. Evaluate Traditional KD Student (if provided)
    if args.student_kd:
        print("\n" + "="*80)
        print("EVALUATING TRADITIONAL KD STUDENT")
        print("="*80)

        if args.student_arch == 'mobilenet':
            student_kd = create_student_model(device=device, lightweight=False)
        else:
            student_kd = create_student_model(device=device, lightweight=True)

        student_kd, _, _ = load_checkpoint(student_kd, args.student_kd, device=device)
        results_dict['Student (KD)'] = evaluate_model(
            student_kd, test_loader, device, f"Student (Traditional KD - {args.student_arch})"
        )

    # 4. Evaluate HPM-KD Student (if provided)
    if args.student_hpmkd:
        print("\n" + "="*80)
        print("EVALUATING HPM-KD STUDENT")
        print("="*80)

        if args.student_arch == 'mobilenet':
            student_hpmkd = create_student_model(device=device, lightweight=False)
        else:
            student_hpmkd = create_student_model(device=device, lightweight=True)

        student_hpmkd, _, _ = load_checkpoint(student_hpmkd, args.student_hpmkd, device=device)
        results_dict['Student (HPM-KD)'] = evaluate_model(
            student_hpmkd, test_loader, device, f"Student (HPM-KD - {args.student_arch})"
        )

    # Generate comparison table
    print("\n" + "="*80)
    print("GENERATING COMPARISON TABLE")
    print("="*80)

    table_path = output_dir / 'comparison_results.txt'
    df = generate_comparison_table(results_dict, teacher_acc, table_path)

    # Calculate improvement statistics
    calculate_improvement_stats(results_dict)

    # Save confusion matrices
    if args.save_confusion:
        print("\n" + "="*80)
        print("GENERATING CONFUSION MATRICES")
        print("="*80)

        for name, results in results_dict.items():
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            cm_path = output_dir / f'confusion_matrix_{safe_name}.png'
            plot_confusion_matrix(
                results['targets'],
                results['predictions'],
                name,
                cm_path
            )

    # Save comparison figures
    if args.save_figures:
        print("\n" + "="*80)
        print("GENERATING COMPARISON FIGURES")
        print("="*80)

        # Accuracy comparison
        comp_path = output_dir / 'accuracy_comparison.png'
        plot_comparison_chart(results_dict, comp_path)

        # Retention comparison (only for students)
        if len(results_dict) > 1:
            ret_path = output_dir / 'retention_comparison.png'
            plot_retention_chart(results_dict, teacher_acc, ret_path)

    # Save detailed results as JSON
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)

    detailed_results = {}
    for name, results in results_dict.items():
        detailed_results[name] = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'correct': int(results['correct']),
            'total': int(results['total']),
            'retention': (results['accuracy'] / teacher_acc * 100) if name != 'Teacher' else 100.0
        }

    json_path = output_dir / 'detailed_results.json'
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"✅ Detailed results saved: {json_path}")

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - Comparison table: comparison_results.txt/.csv")
    print(f"  - Confusion matrices: confusion_matrix_*.png")
    print(f"  - Comparison charts: accuracy_comparison.png, retention_comparison.png")
    print(f"  - Detailed results: detailed_results.json")
    print("="*80)


if __name__ == '__main__':
    main()
