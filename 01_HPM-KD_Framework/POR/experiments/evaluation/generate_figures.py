#!/usr/bin/env python3
"""
HPM-KD Results Visualization
============================

Generate publication-quality figures from experimental results.

Author: Gustavo Coelho Haase
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_palette("colorblind")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_results():
    """Load experimental results."""
    quick_path = Path(__file__).parent / "experiment_results" / "hpmkd_results.csv"
    full_path = Path(__file__).parent / "experiment_results_full" / "hpmkd_results.csv"

    quick_df = pd.read_csv(quick_path)
    full_df = pd.read_csv(full_path)

    quick_df['dataset'] = '10k samples'
    full_df['dataset'] = '70k samples'

    return quick_df, full_df


def figure1_performance_comparison(quick_df, full_df):
    """Figure 1: Performance comparison across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Quick test
    methods = quick_df['method'].values
    accuracies = quick_df['test_accuracy'].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars1 = ax1.bar(range(len(methods)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Quick Test (10k samples)')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Full MNIST
    methods = full_df['method'].values
    accuracies = full_df['test_accuracy'].values

    bars2 = ax2.bar(range(len(methods)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Full MNIST (70k samples)')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Highlight HPM-KD
    bars1[2].set_edgecolor('green')
    bars1[2].set_linewidth(3)
    bars2[2].set_edgecolor('green')
    bars2[2].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure1_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure1_performance_comparison.pdf", bbox_inches='tight')
    print(f"✅ Saved: figure1_performance_comparison.png/pdf")
    plt.close()


def figure2_improvement_over_baseline(quick_df, full_df):
    """Figure 2: Improvement over Traditional KD."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate improvements
    quick_trad_kd = quick_df[quick_df['method'] == 'Traditional KD']['test_accuracy'].values[0]
    quick_hpmkd = quick_df[quick_df['method'] == 'HPM-KD']['test_accuracy'].values[0]
    quick_improvement = quick_hpmkd - quick_trad_kd

    full_trad_kd = full_df[full_df['method'] == 'Traditional KD']['test_accuracy'].values[0]
    full_hpmkd = full_df[full_df['method'] == 'HPM-KD']['test_accuracy'].values[0]
    full_improvement = full_hpmkd - full_trad_kd

    datasets = ['10k samples', '70k samples']
    improvements = [quick_improvement, full_improvement]
    colors = ['#2ca02c', '#228b22']

    bars = ax.bar(datasets, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Improvement over Traditional KD (pp)')
    ax.set_title('HPM-KD Improvement Over Traditional Knowledge Distillation', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 30])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'+{imp:.2f}pp', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add paper expectation line
    ax.axhline(y=3, color='red', linestyle='--', linewidth=2, label='Paper expectation (3-7pp)')
    ax.axhline(y=7, color='red', linestyle='--', linewidth=2)
    ax.fill_between([-0.5, 1.5], 3, 7, color='red', alpha=0.1)

    ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure2_improvement_over_baseline.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure2_improvement_over_baseline.pdf", bbox_inches='tight')
    print(f"✅ Saved: figure2_improvement_over_baseline.png/pdf")
    plt.close()


def figure3_retention_comparison(quick_df, full_df):
    """Figure 3: Teacher accuracy retention."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    data = []
    for df, dataset in [(quick_df, '10k samples'), (full_df, '70k samples')]:
        for _, row in df.iterrows():
            if pd.notna(row['retention']):
                data.append({
                    'Dataset': dataset,
                    'Method': row['method'],
                    'Retention': row['retention']
                })

    df_retention = pd.DataFrame(data)

    # Plot grouped bars
    x = np.arange(len(df_retention['Dataset'].unique()))
    width = 0.35

    trad_kd_10k = df_retention[(df_retention['Dataset'] == '10k samples') & (df_retention['Method'] == 'Traditional KD')]['Retention'].values[0]
    hpmkd_10k = df_retention[(df_retention['Dataset'] == '10k samples') & (df_retention['Method'] == 'HPM-KD')]['Retention'].values[0]
    trad_kd_70k = df_retention[(df_retention['Dataset'] == '70k samples') & (df_retention['Method'] == 'Traditional KD')]['Retention'].values[0]
    hpmkd_70k = df_retention[(df_retention['Dataset'] == '70k samples') & (df_retention['Method'] == 'HPM-KD')]['Retention'].values[0]

    bars1 = ax.bar(x - width/2, [trad_kd_10k, trad_kd_70k], width, label='Traditional KD',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, [hpmkd_10k, hpmkd_70k], width, label='HPM-KD',
                   color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=2)

    ax.set_ylabel('Teacher Accuracy Retention (%)')
    ax.set_title('Teacher Accuracy Retention Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['10k samples', '70k samples'])
    ax.set_ylim([0, 100])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars1, [trad_kd_10k, trad_kd_70k]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    for bar, val in zip(bars2, [hpmkd_10k, hpmkd_70k]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure3_retention_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure3_retention_comparison.pdf", bbox_inches='tight')
    print(f"✅ Saved: figure3_retention_comparison.png/pdf")
    plt.close()


def figure4_scaling_analysis(quick_df, full_df):
    """Figure 4: Scaling with dataset size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    dataset_sizes = [10000, 70000]

    methods_data = {}
    for method in ['Direct Training', 'Traditional KD', 'HPM-KD']:
        quick_acc = quick_df[quick_df['method'] == method]['test_accuracy'].values[0]
        full_acc = full_df[full_df['method'] == method]['test_accuracy'].values[0]
        methods_data[method] = [quick_acc, full_acc]

    # Plot lines
    colors = {'Direct Training': '#1f77b4', 'Traditional KD': '#ff7f0e', 'HPM-KD': '#2ca02c'}
    markers = {'Direct Training': 'o', 'Traditional KD': 's', 'HPM-KD': '^'}

    for method, accs in methods_data.items():
        ax.plot(dataset_sizes, accs, marker=markers[method], color=colors[method],
                linewidth=2.5, markersize=10, label=method, alpha=0.8)

        # Add improvement annotation for HPM-KD
        if method == 'HPM-KD':
            improvement = accs[1] - accs[0]
            mid_x = np.mean(dataset_sizes)
            mid_y = np.mean(accs)
            ax.annotate(f'+{improvement:.2f}pp', xy=(mid_x, mid_y),
                       xytext=(mid_x, mid_y + 5),
                       fontsize=12, fontweight='bold', color='green',
                       ha='center')

    ax.set_xlabel('Dataset Size (samples)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Scaling Analysis: Performance vs Dataset Size', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks([10000, 70000])
    ax.set_xticklabels(['10k', '70k'])
    ax.set_ylim([60, 95])
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure4_scaling_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure4_scaling_analysis.pdf", bbox_inches='tight')
    print(f"✅ Saved: figure4_scaling_analysis.png/pdf")
    plt.close()


def figure5_training_time_comparison(quick_df, full_df):
    """Figure 5: Training time comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Quick test
    methods = quick_df['method'].values
    times = quick_df['training_time'].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars1 = ax1.bar(range(len(methods)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Quick Test (10k samples)')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

    # Full MNIST
    methods = full_df['method'].values
    times = full_df['training_time'].values

    bars2 = ax2.bar(range(len(methods)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Full MNIST (70k samples)')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure5_training_time.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure5_training_time.pdf", bbox_inches='tight')
    print(f"✅ Saved: figure5_training_time.png/pdf")
    plt.close()


def figure6_comprehensive_comparison(quick_df, full_df):
    """Figure 6: Comprehensive comparison matrix."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for heatmap
    metrics = ['Test Acc (%)', 'Retention (%)', 'Time (s)']
    methods = ['Direct\nTraining', 'Traditional\nKD', 'HPM-KD']

    # Quick test data
    quick_data = []
    for method in ['Direct Training', 'Traditional KD', 'HPM-KD']:
        row = quick_df[quick_df['method'] == method].iloc[0]
        quick_data.append([
            row['test_accuracy'],
            row['retention'] if pd.notna(row['retention']) else 0,
            row['training_time']
        ])

    # Full MNIST data
    full_data = []
    for method in ['Direct Training', 'Traditional KD', 'HPM-KD']:
        row = full_df[full_df['method'] == method].iloc[0]
        full_data.append([
            row['test_accuracy'],
            row['retention'] if pd.notna(row['retention']) else 0,
            row['training_time']
        ])

    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(quick_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_title('Quick Test (10k samples)', fontsize=14, fontweight='bold')

    # Add values
    for i in range(len(methods)):
        for j in range(len(metrics)):
            value = quick_data[i][j]
            if j == 2:  # Time metric
                text = ax1.text(j, i, f'{value:.1f}s', ha='center', va='center',
                              fontweight='bold', fontsize=11)
            else:
                text = ax1.text(j, i, f'{value:.1f}', ha='center', va='center',
                              fontweight='bold', fontsize=11)

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(full_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metrics)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.set_title('Full MNIST (70k samples)', fontsize=14, fontweight='bold')

    # Add values
    for i in range(len(methods)):
        for j in range(len(metrics)):
            value = full_data[i][j]
            if j == 2:  # Time metric
                text = ax2.text(j, i, f'{value:.1f}s', ha='center', va='center',
                              fontweight='bold', fontsize=11)
            else:
                text = ax2.text(j, i, f'{value:.1f}', ha='center', va='center',
                              fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure6_comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure6_comprehensive_comparison.pdf", bbox_inches='tight')
    print(f"✅ Saved: figure6_comprehensive_comparison.png/pdf")
    plt.close()


def main():
    """Generate all figures."""
    print("="*80)
    print("HPM-KD RESULTS VISUALIZATION")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Load results
    print("Loading experimental results...")
    quick_df, full_df = load_results()
    print(f"✅ Loaded quick test results: {len(quick_df)} methods")
    print(f"✅ Loaded full MNIST results: {len(full_df)} methods")
    print()

    # Generate figures
    print("Generating figures...")
    print()

    figure1_performance_comparison(quick_df, full_df)
    figure2_improvement_over_baseline(quick_df, full_df)
    figure3_retention_comparison(quick_df, full_df)
    figure4_scaling_analysis(quick_df, full_df)
    figure5_training_time_comparison(quick_df, full_df)
    figure6_comprehensive_comparison(quick_df, full_df)

    print()
    print("="*80)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"Location: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    for fig_file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {fig_file.name}")
    print()


if __name__ == '__main__':
    main()
