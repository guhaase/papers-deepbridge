"""
Generate Visualizations for Usability Study

Creates publication-quality figures:
- SUS score distribution
- NASA TLX dimensions radar chart
- Task completion times boxplot
- Success rate by task
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

FIGURES_DIR.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def plot_sus_distribution(logger=None):
    """Plot SUS score distribution"""
    if logger:
        logger.info("Generating SUS score distribution plot...")

    sus_df = pd.read_csv(RESULTS_DIR / "03_usability_sus_scores.csv")
    sus_scores = sus_df['sus_score'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with KDE
    ax1.hist(sus_scores, bins=10, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(sus_scores)
    x_range = np.linspace(sus_scores.min(), sus_scores.max(), 100)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add mean line
    mean_sus = np.mean(sus_scores)
    ax1.axvline(mean_sus, color='darkred', linestyle='--', linewidth=2, label=f'Mean = {mean_sus:.1f}')

    # Add global average line
    ax1.axvline(68, color='gray', linestyle=':', linewidth=2, label='Global Avg = 68')

    ax1.set_xlabel('SUS Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('SUS Score Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Boxplot
    ax2.boxplot(sus_scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='darkred', linewidth=2))

    # Add individual points
    y = sus_scores
    x = np.random.normal(1, 0.04, size=len(y))
    ax2.scatter(x, y, alpha=0.3, color='navy', s=30)

    # Add reference lines
    ax2.axhline(68, color='gray', linestyle=':', linewidth=2, label='Global Avg')
    ax2.axhline(85, color='green', linestyle='--', linewidth=2, label='Excellent (85)')

    ax2.set_ylabel('SUS Score', fontweight='bold')
    ax2.set_title('SUS Score Boxplot', fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['DeepBridge'])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_file = FIGURES_DIR / "sus_score_distribution.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved to {output_file}")


def plot_nasa_tlx_dimensions(logger=None):
    """Plot NASA TLX dimensions radar chart"""
    if logger:
        logger.info("Generating NASA TLX dimensions plot...")

    tlx_df = pd.read_csv(RESULTS_DIR / "03_usability_nasa_tlx.csv")

    # Calculate means for each dimension
    dimensions = ['Mental\nDemand', 'Physical\nDemand', 'Temporal\nDemand',
                  'Performance', 'Effort', 'Frustration']

    means = [
        tlx_df['mental_demand'].mean(),
        tlx_df['physical_demand'].mean(),
        tlx_df['temporal_demand'].mean(),
        100 - tlx_df['performance'].mean(),  # Invert performance (low is good)
        tlx_df['effort'].mean(),
        tlx_df['frustration'].mean()
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    means_plot = means + [means[0]]  # Complete the circle
    angles += angles[:1]

    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, means_plot, 'o-', linewidth=2, color='steelblue', label='DeepBridge')
    ax1.fill(angles, means_plot, alpha=0.25, color='steelblue')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(dimensions)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Score (0-100)', fontweight='bold')
    ax1.set_title('NASA TLX Dimensions', fontweight='bold', pad=20)
    ax1.grid(True)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Bar chart
    ax2 = plt.subplot(122)
    colors = ['steelblue' if m < 40 else 'orange' if m < 60 else 'red' for m in means]
    bars = ax2.barh(dimensions, means, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, means)):
        ax2.text(value + 2, i, f'{value:.1f}', va='center')

    # Add reference lines
    ax2.axvline(40, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low threshold')
    ax2.axvline(60, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate threshold')

    ax2.set_xlabel('Score (0-100)', fontweight='bold')
    ax2.set_title('NASA TLX Dimension Scores', fontweight='bold')
    ax2.set_xlim(0, 110)
    ax2.legend(fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_file = FIGURES_DIR / "nasa_tlx_dimensions.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved to {output_file}")


def plot_task_completion_times(logger=None):
    """Plot task completion times"""
    if logger:
        logger.info("Generating task completion times plot...")

    times_df = pd.read_csv(RESULTS_DIR / "03_usability_task_times.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot by task
    task_data = [
        times_df['task1_fairness_time'].dropna(),
        times_df['task2_report_time'].dropna(),
        times_df['task3_cicd_time'].dropna(),
        times_df['total_time'].dropna()
    ]

    bp = ax1.boxplot(task_data, labels=['Task 1\nFairness', 'Task 2\nReport', 'Task 3\nCI/CD', 'Total'],
                     patch_artist=True, showmeans=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Style
    for median in bp['medians']:
        median.set_color('darkred')
        median.set_linewidth(2)

    for mean in bp['means']:
        mean.set_marker('D')
        mean.set_markerfacecolor('blue')
        mean.set_markersize(6)

    ax1.set_ylabel('Time (minutes)', fontweight='bold')
    ax1.set_title('Task Completion Times', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add target line for total time
    ax1.axhline(15, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (15 min)')
    ax1.legend()

    # Cumulative distribution
    total_times_sorted = np.sort(times_df['total_time'].dropna())
    cumulative = np.arange(1, len(total_times_sorted) + 1) / len(total_times_sorted) * 100

    ax2.plot(total_times_sorted, cumulative, marker='o', linewidth=2, markersize=4, color='steelblue')
    ax2.axvline(15, color='red', linestyle='--', linewidth=2, label='Target (15 min)')
    ax2.axvline(np.median(total_times_sorted), color='green', linestyle='--', linewidth=2,
                label=f'Median ({np.median(total_times_sorted):.1f} min)')

    ax2.set_xlabel('Total Time (minutes)', fontweight='bold')
    ax2.set_ylabel('Cumulative % of Participants', fontweight='bold')
    ax2.set_title('Cumulative Distribution of Total Time', fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    output_file = FIGURES_DIR / "task_completion_times.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved to {output_file}")


def plot_success_rates(logger=None):
    """Plot success rates by task"""
    if logger:
        logger.info("Generating success rate plot...")

    times_df = pd.read_csv(RESULTS_DIR / "03_usability_task_times.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    tasks = ['Task 1\nFairness', 'Task 2\nReport', 'Task 3\nCI/CD', 'Overall']
    success_rates = [
        times_df['task1_success'].sum() / len(times_df) * 100,
        times_df['task2_success'].sum() / len(times_df) * 100,
        times_df['task3_success'].sum() / len(times_df) * 100,
        times_df['completed_all_tasks'].sum() / len(times_df) * 100
    ]

    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    bars = ax.bar(tasks, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add target line
    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')

    # Add 100% reference line
    ax.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate by Task', fontweight='bold', fontsize=16)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = FIGURES_DIR / "success_rate_by_task.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"  Saved to {output_file}")


def main():
    """Generate all visualizations"""
    logger = setup_logging("generate_visualizations", LOGS_DIR)

    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    plot_sus_distribution(logger)
    plot_nasa_tlx_dimensions(logger)
    plot_task_completion_times(logger)
    plot_success_rates(logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("ALL VISUALIZATIONS GENERATED")
    logger.info("=" * 80)
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
