# HPM-KD Implementation Guide
## Connecting Paper Theory to DeepBridge Code

**Date**: November 5, 2025
**Paper**: HPM-KD Framework
**Library**: DeepBridge v0.1.50+

---

## 🎯 Overview

This guide maps the **theoretical framework** described in the paper to the **actual implementation** in the DeepBridge library. Use this document to:

1. Understand how paper concepts are implemented
2. Run the experiments described in the paper
3. Reproduce the results for publication
4. Extend the framework with new components

---

## 📂 Code Structure

### HPM-KD Components Location

```
deepbridge/distillation/techniques/hpm/
├── __init__.py                 # Module exports
├── hpm_distiller.py           # Main HPM-KD class (Section 4 of paper)
├── adaptive_config.py         # Section 4.2: Adaptive Configuration Manager
├── progressive_chain.py       # Section 4.3: Progressive Distillation Chain
├── multi_teacher.py           # Section 4.4: Attention-Weighted Multi-Teacher
├── meta_scheduler.py          # Section 4.5: Meta-Temperature Scheduler
├── parallel_pipeline.py       # Section 4.6: Parallel Processing Pipeline
├── shared_memory.py           # Section 4.7: Shared Optimization Memory
└── cache_system.py            # Supporting cache infrastructure
```

---

## 🔗 Paper-to-Code Mapping

### Section 4.1: Framework Overview

**Paper**: "HPM-KD integrates six synergistic components..."

**Code**: `HPMDistiller` class in `hpm_distiller.py`

```python
from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig

# Initialize the full framework
config = HPMConfig(
    use_progressive=True,      # Enable Progressive Chain
    use_multi_teacher=True,    # Enable Multi-Teacher
    use_adaptive_temperature=True,  # Enable Meta-Temperature
    use_parallel=True,         # Enable Parallel Processing
    use_cache=True             # Enable Shared Memory
)

distiller = HPMDistiller(
    teacher_model=teacher,
    student_model_type=ModelType.SKLEARN_TREE,
    config=config
)
```

---

### Section 4.2: Adaptive Configuration Manager

**Paper Algorithm 1** → **Code**: `AdaptiveConfigurationManager`

**Paper Equation (2)**:
```
ĉ = g_ACM(m; Θ_ACM)
```

**Code Implementation**:
```python
from deepbridge.distillation.techniques.hpm.adaptive_config import (
    AdaptiveConfigurationManager
)

# Extract meta-features (Algorithm 1, Line 1)
acm = AdaptiveConfigurationManager(
    max_configs=16,
    initial_samples=8,
    exploration_ratio=0.3
)

# Get optimal configuration (Algorithm 1, Line 3-9)
meta_features = acm.extract_meta_features(X_train, y_train, teacher, student)
optimal_config = acm.get_optimal_config(meta_features)

# Returns: {
#   'temperature': 4.2,
#   'alpha': 0.7,
#   'learning_rate': 0.001,
#   'weight_decay': 0.0001,
#   'chain_threshold': 0.01
# }
```

**Meta-Features** (corresponds to paper Section 4.2.1):
- Dataset: `N_train`, `N_test`, `d`, `K`, `ρ`, `C_D`
- Model: `|θ_T|`, `|θ_S|`, `r`, architecture family, `Acc_T`

---

### Section 4.3: Progressive Distillation Chain

**Paper Algorithm 2** → **Code**: `ProgressiveDistillationChain`

**Paper Equation (5)**:
```
|θ_i| = |θ_T| · r^(i/L)
```

**Code Implementation**:
```python
from deepbridge.distillation.techniques.hpm.progressive_chain import (
    ProgressiveDistillationChain
)

# Initialize chain (Algorithm 2, Line 1-4)
chain = ProgressiveDistillationChain(
    min_improvement=0.01,  # ε in paper (Equation 6)
    max_chain_length=5,
    compression_ratio=10.0
)

# Construct and train chain (Algorithm 2, Line 5-17)
chain.build_chain(
    teacher=teacher,
    student_architecture=student_arch,
    X_train=X_train,
    y_train=y_train
)

# Get trained student (Algorithm 2, Line 18)
final_student = chain.get_final_student()

# Inspect chain structure
for i, model in enumerate(chain.intermediate_models):
    print(f"Step {i}: {model.n_parameters} params, acc={model.accuracy:.3f}")
```

**Automatic Chain Length** (Paper Section 4.3.2):
The implementation automatically determines `L` based on Equation (6):
```
(Acc_i - Acc_{i-1}) / Acc_{i-1} < ε
```

---

### Section 4.4: Attention-Weighted Multi-Teacher

**Paper Algorithm 3** → **Code**: `AttentionWeightedMultiTeacher`

**Paper Equation (11)**:
```
p_AWMT(x) = Σ a_m(x) · f_T^(m)(x)
```

**Code Implementation**:
```python
from deepbridge.distillation.techniques.hpm.multi_teacher import (
    AttentionWeightedMultiTeacher
)

# Setup multi-teacher (Algorithm 3, Line 1-3)
teachers = [teacher1, teacher2, teacher3, teacher4]

multi_teacher = AttentionWeightedMultiTeacher(
    teachers=teachers,
    attention_type='learned',  # Learned attention (Section 4.4.3)
    attention_hidden_size=128,
    attention_dropout=0.2,
    entropy_weight=0.01  # β in Equation (16)
)

# Train student with attention (Algorithm 3, Line 4-15)
multi_teacher.fit(
    X_train=X_train,
    y_train=y_train,
    student=student,
    epochs=150,
    batch_size=128
)

# Visualize learned attention weights
attention_weights = multi_teacher.get_attention_weights(X_test)
# Shape: (n_samples, n_teachers)

# Average attention per teacher
mean_attention = attention_weights.mean(axis=0)
print(f"Teacher contributions: {mean_attention}")
# Output: [0.35, 0.28, 0.22, 0.15] - teachers used differently
```

**Attention Network** (Paper Equation 12-13):
```python
# Internal implementation matches paper:
# h = MLP_1([x; t_1; ...; t_M])
# a(x) = softmax(MLP_2(h))
```

---

### Section 4.5: Meta-Temperature Scheduler

**Paper Equation (17)**:
```
T(t) = T_min + (T_max - T_min) · s(t)
```

**Code Implementation**:
```python
from deepbridge.distillation.techniques.hpm.meta_scheduler import (
    MetaTemperatureScheduler
)

# Initialize scheduler (Section 4.5.2)
scheduler = MetaTemperatureScheduler(
    initial_temperature=4.0,
    min_temperature=2.0,
    max_temperature=6.0,
    schedule_type='cosine'  # Options: 'cosine', 'loss_based', 'convergence_based'
)

# During training loop
for epoch in range(num_epochs):
    # Get current temperature (Equation 17)
    current_temp = scheduler.get_temperature(
        epoch=epoch,
        total_epochs=num_epochs,
        current_loss=loss,
        loss_gradient=loss_grad
    )

    # Use in distillation loss
    loss_kd = kl_divergence(
        student_logits / current_temp,
        teacher_logits / current_temp
    )

    # Update scheduler state
    scheduler.step(loss)

# Temperature trajectory
temp_history = scheduler.get_history()
# Decreases from T_max (early training) to T_min (late training)
```

**Three Scheduling Strategies** (Paper Section 4.5.2):
1. **Cosine Decay** (Equation 18): `schedule_type='cosine'`
2. **Loss-Based** (Equation 19): `schedule_type='loss_based'`
3. **Convergence-Based** (Equation 20): `schedule_type='convergence_based'`

---

### Section 4.6: Parallel Processing Pipeline

**Code**: `ParallelDistillationPipeline`

**Code Implementation**:
```python
from deepbridge.distillation.techniques.hpm.parallel_pipeline import (
    ParallelDistillationPipeline,
    WorkloadConfig
)

# Setup parallel processing (Section 4.6.1)
pipeline = ParallelDistillationPipeline(
    num_workers=4,  # Number of parallel workers
    load_balancing='dynamic'  # Dynamic load balancing
)

# Distribute multi-teacher computations
workloads = [
    WorkloadConfig(teacher=t, data=X_train, task_id=i)
    for i, t in enumerate(teachers)
]

# Execute in parallel (Equation 21)
results = pipeline.execute_parallel(workloads)

# Get speedup metrics
speedup = pipeline.get_speedup_factor()
print(f"Parallel speedup: {speedup:.2f}x")
# Expected: 3.2x with 4 workers (80% efficiency)
```

**Load Balancing** (Paper Equation 22):
```python
# Internal implementation:
# worker_assign = argmin_w (queue_time_w + est_time_task)
```

---

### Section 4.7: Shared Optimization Memory

**Code**: `SharedOptimizationMemory`

**Code Implementation**:
```python
from deepbridge.distillation.techniques.hpm.shared_memory import (
    SharedOptimizationMemory
)

# Initialize shared memory (Section 4.7.1)
memory = SharedOptimizationMemory(
    max_size_gb=2.0,  # Cache size limit
    eviction_policy='lru'  # LRU eviction
)

# Store configuration (Equation 23)
memory.store_config(
    meta_features=meta_features,
    config=optimal_config,
    performance=accuracy_retention
)

# Retrieve similar configuration
cached_config = memory.retrieve_config(
    meta_features=new_meta_features,
    top_k=5
)

# Get cache statistics
stats = memory.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
# Expected: 40-60% after 10+ experiments

# Clear cache if needed
memory.clear()
```

**Three Databases** (Paper Section 4.7.1):
1. **Configuration DB**: Stores `(m, c, perf)` tuples
2. **Teacher Embedding DB**: Caches teacher predictions
3. **Intermediate Model DB**: Stores pretrained intermediate models

---

## 🧪 Running Paper Experiments

### Experiment 1: Main Results (Section 5, Table 2-3)

```python
from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig
from deepbridge.core import DBDataset, Experiment
from deepbridge.utils import ModelType
from sklearn.datasets import fetch_openml

# Load dataset (CIFAR-10 from paper)
X, y = fetch_openml('CIFAR_10', return_X_y=True, parser='auto')

# Train teacher model
teacher = train_teacher_model(X, y, architecture='ResNet-56')
# Expected accuracy: 93.52% (from Table 2)

# Configure HPM-KD
config = HPMConfig(
    use_progressive=True,
    use_multi_teacher=True,
    use_adaptive_temperature=True,
    use_parallel=True,
    use_cache=True,
    min_improvement=0.01,
    initial_temperature=4.0,
    parallel_workers=4
)

# Run HPM-KD distillation
distiller = HPMDistiller(
    teacher_model=teacher,
    student_model_type=ModelType.NEURAL_NET,  # ResNet-20
    config=config
)

# Train student (corresponds to main experiment)
student = distiller.fit(X, y)

# Evaluate (Table 2, Row 6)
student_acc = evaluate_model(student, X_test, y_test)
print(f"Student accuracy: {student_acc:.2f}%")
# Expected: 92.34% (98.74% retention)

# Compare with baselines
baselines = {
    'Direct Training': train_direct(X, y, student_arch),
    'Traditional KD': train_traditional_kd(X, y, teacher, student_arch),
    'FitNets': train_fitnets(X, y, teacher, student_arch),
    'DML': train_dml(X, y, student_arch),
    'TAKD': train_takd(X, y, teacher, student_arch)
}

# Expected results from Table 2:
# Direct: 88.74%, Traditional KD: 91.37%, FitNets: 91.68%
# DML: 91.42%, TAKD: 91.85%, HPM-KD: 92.34% ✓
```

---

### Experiment 2: Ablation Studies (Section 6, Table 10)

```python
# Full HPM-KD
config_full = HPMConfig(
    use_progressive=True,
    use_multi_teacher=True,
    use_adaptive_temperature=True,
    use_parallel=True,
    use_cache=True
)
acc_full = run_distillation(config_full, X, y, teacher, student)
# Expected: 92.34% (Table 10, Row 1)

# Ablation: Remove Progressive Chain
config_no_chain = HPMConfig(
    use_progressive=False,  # ← Disabled
    use_multi_teacher=True,
    use_adaptive_temperature=True,
    use_parallel=True,
    use_cache=True
)
acc_no_chain = run_distillation(config_no_chain, X, y, teacher, student)
# Expected: 89.48% (-2.86 pp, Table 10, Row 4)

# Ablation: Remove Multi-Teacher Attention
config_no_multi = HPMConfig(
    use_progressive=True,
    use_multi_teacher=False,  # ← Disabled
    use_adaptive_temperature=True,
    use_parallel=True,
    use_cache=True
)
acc_no_multi = run_distillation(config_no_multi, X, y, teacher, student)
# Expected: 91.12% (-1.22 pp, Table 10, Row 5)

# ... repeat for all 6 components
```

---

### Experiment 3: Sensitivity Analysis (Section 6.3.1)

```python
# Temperature sensitivity (Figure 6)
temperatures = [2, 4, 6, 8]
results = []

for temp in temperatures:
    config = HPMConfig(initial_temperature=temp)
    acc = run_distillation(config, X, y, teacher, student)
    results.append({'temperature': temp, 'accuracy': acc})

# Plot sensitivity heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# Expected: HPM-KD shows lower sensitivity than Traditional KD

# Chain length sensitivity (Table 13)
chain_lengths = [0, 1, 2, 3, 4, 5]
for length in chain_lengths:
    config = HPMConfig(max_chain_length=length)
    acc = run_distillation(config, X, y, teacher, student)
    print(f"Chain {length}: {acc:.2f}%")

# Expected optimal: 3 steps (92.34%, Table 13, Row 4)
```

---

### Experiment 4: Robustness Tests (Section 6.4)

```python
# Class imbalance robustness (Table 14)
imbalance_ratios = [1, 10, 50, 100]  # majority:minority

for ratio in imbalance_ratios:
    # Create imbalanced dataset
    X_imb, y_imb = create_imbalanced_dataset(X, y, ratio=ratio)

    # Run HPM-KD
    config = HPMConfig()
    acc = run_distillation(config, X_imb, y_imb, teacher, student)

    print(f"Imbalance {ratio}:1 - Accuracy: {acc:.2f}%")

# Expected results from Table 14:
# Balanced: 98.74%, 10:1: 97.91%, 50:1: 95.86%, 100:1: 93.52%

# Label noise robustness (Table 15)
noise_levels = [0.0, 0.1, 0.2, 0.3]

for noise in noise_levels:
    # Inject random label noise
    X_noisy, y_noisy = inject_label_noise(X, y, noise_rate=noise)

    acc = run_distillation(config, X_noisy, y_noisy, teacher, student)
    print(f"Noise {noise*100:.0f}% - Accuracy: {acc:.2f}%")

# Expected degradation: -6.59 pp at 30% noise (better than TAKD: -7.38 pp)
```

---

## 📊 Reproducing Paper Figures

### Figure 1: Architecture Diagram

Create using TikZ or Python visualization:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# This corresponds to Figure 1 in Section 4.1
fig, ax = plt.subplots(figsize=(12, 8))

# Draw 6 component boxes
components = [
    'Adaptive\nConfiguration',
    'Progressive\nChain',
    'Multi-Teacher\nAttention',
    'Meta-Temperature\nScheduler',
    'Parallel\nProcessing',
    'Shared\nMemory'
]

# Position and draw components...
# (Full implementation in separate visualization script)

plt.savefig('figures/hpm_architecture.pdf', bbox_inches='tight')
```

### Figure 2: Generalization Radar Chart

```python
import numpy as np
import matplotlib.pyplot as plt

# Data from Table 5 (OpenML-CC18 results)
datasets = ['MNIST', 'Fashion', 'CIFAR-10', 'CIFAR-100',
            'Adult', 'Credit', 'Wine', 'OpenML Avg']
hpmkd_retention = [99.87, 99.24, 98.74, 96.13, 99.44, 98.94, 97.96, 97.8]
takd_retention = [99.75, 98.88, 98.21, 94.60, 99.13, 98.40, 96.90, 96.7]

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

hpmkd_retention = np.concatenate((hpmkd_retention, [hpmkd_retention[0]]))
takd_retention = np.concatenate((takd_retention, [takd_retention[0]]))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, hpmkd_retention, 'o-', linewidth=2, label='HPM-KD')
ax.plot(angles, takd_retention, 's-', linewidth=2, label='TAKD')
ax.fill(angles, hpmkd_retention, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(datasets)
ax.set_ylim(94, 100)
ax.legend()
ax.set_title('Accuracy Retention Across Datasets')
ax.grid(True)

plt.savefig('figures/generalization_radar.pdf', bbox_inches='tight')
```

### Figure 3: Compression Ratio Analysis

```python
# Data from Section 5.3
compression_ratios = [2, 3, 5, 10, 15, 20]

methods = {
    'Traditional KD': [98.5, 97.7, 96.2, 93.3, 91.2, 89.8],
    'TAKD': [98.8, 98.2, 96.9, 94.6, 92.5, 90.9],
    'HPM-KD': [99.1, 98.7, 97.8, 96.1, 95.2, 94.3]
}

plt.figure(figsize=(10, 6))
for method, retentions in methods.items():
    plt.plot(compression_ratios, retentions, 'o-', linewidth=2, label=method)

plt.xlabel('Compression Ratio')
plt.ylabel('Accuracy Retention (%)')
plt.title('Accuracy Retention vs Compression Ratio (CIFAR-10)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/compression_ratios.pdf', bbox_inches='tight')
```

---

## 🎯 Complete Experiment Pipeline

### Full Pipeline Script

```python
"""
Complete experimental pipeline for HPM-KD paper.
Runs all experiments from Sections 5 and 6.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig
from deepbridge.core import DBDataset, Experiment
from deepbridge.validation.metrics import classification_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_experiment_pipeline():
    """
    Run complete experimental pipeline matching paper structure.
    """

    # Section 5.1: Main Results
    logger.info("=" * 80)
    logger.info("SECTION 5.1: Main Compression Results")
    logger.info("=" * 80)

    results_main = {}
    datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100',
                'adult', 'credit', 'wine_quality']

    for dataset_name in datasets:
        logger.info(f"\nProcessing {dataset_name}...")

        # Load data
        X, y = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Train teacher
        teacher = train_teacher(X_train, y_train, dataset_name)
        teacher_acc = evaluate(teacher, X_test, y_test)
        logger.info(f"Teacher accuracy: {teacher_acc:.2f}%")

        # Run baselines + HPM-KD
        methods = {
            'Direct Training': run_direct_training,
            'Traditional KD': run_traditional_kd,
            'FitNets': run_fitnets,
            'DML': run_dml,
            'TAKD': run_takd,
            'HPM-KD': run_hpmkd
        }

        dataset_results = {}
        for method_name, method_func in methods.items():
            # Run 5 independent trials
            accs = []
            times = []

            for seed in range(5):
                student, elapsed = method_func(
                    X_train, y_train, teacher,
                    seed=42 + seed
                )
                acc = evaluate(student, X_test, y_test)
                accs.append(acc)
                times.append(elapsed)

            dataset_results[method_name] = {
                'accuracy_mean': np.mean(accs),
                'accuracy_std': np.std(accs),
                'time_mean': np.mean(times),
                'retention': np.mean(accs) / teacher_acc * 100
            }

            logger.info(f"{method_name}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}")

        results_main[dataset_name] = dataset_results

    # Save results to CSV (matches Table 2 and 3 in paper)
    save_results_table(results_main, 'results/table_2_3_main_results.csv')


    # Section 6.2: Ablation Studies
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 6.2: Ablation Studies")
    logger.info("=" * 80)

    results_ablation = {}

    ablation_configs = {
        'Full HPM-KD': HPMConfig(),
        'No Adaptive Config': HPMConfig(use_adaptive_config=False),
        'No Progressive Chain': HPMConfig(use_progressive=False),
        'No Multi-Teacher': HPMConfig(use_multi_teacher=False),
        'No Meta-Temperature': HPMConfig(use_adaptive_temperature=False),
        'No Parallel': HPMConfig(use_parallel=False),
        'No Shared Memory': HPMConfig(use_cache=False)
    }

    for dataset_name in ['cifar10', 'adult']:  # Representative datasets
        logger.info(f"\nAblation on {dataset_name}...")

        X, y = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        teacher = train_teacher(X_train, y_train, dataset_name)

        dataset_ablation = {}
        for config_name, config in ablation_configs.items():
            distiller = HPMDistiller(teacher, config=config)
            student, elapsed = distiller.fit(X_train, y_train)
            acc = evaluate(student, X_test, y_test)

            dataset_ablation[config_name] = {
                'accuracy': acc,
                'time': elapsed
            }
            logger.info(f"{config_name}: {acc:.2f}%")

        results_ablation[dataset_name] = dataset_ablation

    # Save ablation results (matches Table 10 in paper)
    save_results_table(results_ablation, 'results/table_10_ablation.csv')


    # Section 6.3: Sensitivity Analysis
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 6.3: Sensitivity Analysis")
    logger.info("=" * 80)

    # Temperature sensitivity
    temps = [2, 4, 6, 8]
    temp_results = []
    for temp in temps:
        config = HPMConfig(initial_temperature=temp)
        acc = run_sensitivity_test(config, 'cifar10')
        temp_results.append({'temperature': temp, 'accuracy': acc})
        logger.info(f"Temperature {temp}: {acc:.2f}%")

    # Chain length sensitivity
    lengths = [1, 2, 3, 4, 5]
    length_results = []
    for length in lengths:
        config = HPMConfig(max_chain_length=length)
        acc = run_sensitivity_test(config, 'cifar10')
        length_results.append({'chain_length': length, 'accuracy': acc})
        logger.info(f"Chain length {length}: {acc:.2f}%")

    # Save sensitivity results (matches Table 13 and Figure 6)
    pd.DataFrame(temp_results).to_csv('results/temperature_sensitivity.csv')
    pd.DataFrame(length_results).to_csv('results/chain_length_sensitivity.csv')


    # Section 6.4: Robustness Tests
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 6.4: Robustness Tests")
    logger.info("=" * 80)

    # Class imbalance
    imbalance_results = test_class_imbalance('cifar10', ratios=[1, 10, 50, 100])
    imbalance_results.to_csv('results/table_14_imbalance.csv')

    # Label noise
    noise_results = test_label_noise('cifar10', noise_levels=[0.0, 0.1, 0.2, 0.3])
    noise_results.to_csv('results/table_15_noise.csv')


    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETED")
    logger.info("=" * 80)
    logger.info("Results saved to results/ directory")
    logger.info("Ready for paper submission!")


if __name__ == '__main__':
    run_full_experiment_pipeline()
```

---

## 📝 Next Steps

### 1. Run Experiments (4-6 weeks)
```bash
# Run full pipeline
python experiments/run_paper_experiments.py

# Expected output:
# - results/table_2_3_main_results.csv (Section 5.1)
# - results/table_10_ablation.csv (Section 6.2)
# - results/temperature_sensitivity.csv (Section 6.3.1)
# - results/chain_length_sensitivity.csv (Section 6.3.2)
# - results/table_14_imbalance.csv (Section 6.4.1)
# - results/table_15_noise.csv (Section 6.4.2)
```

### 2. Generate Figures (1 week)
```bash
# Create all publication figures
python experiments/generate_figures.py

# Output: 13 PDF figures in figures/ directory
```

### 3. Update Paper (1 day)
```bash
# Replace placeholder data with real results
python experiments/update_paper_tables.py

# Recompile paper
cd papers/01_HPM-KD_Framework/POR
make
```

### 4. Submit! (1 week)
```bash
# Prepare submission package
python experiments/prepare_submission.py

# Creates:
# - submission/main.pdf (camera-ready)
# - submission/supplementary.pdf
# - submission/code.zip
# - submission/data.zip
```

---

## 🎓 Support

### Documentation
- **Paper**: `papers/01_HPM-KD_Framework/POR/build/main.pdf`
- **Code**: `deepbridge/distillation/techniques/hpm/`
- **Examples**: `examples/hpm_distillation_examples.py`

### Contact
- **Author**: Gustavo Coelho Haase
- **Email**: gustavohaase@ucb.edu.br
- **GitHub**: https://github.com/DeepBridge-Validation/DeepBridge

---

**Last Updated**: November 5, 2025
**Status**: ✅ Implementation Complete - Ready for Experiments
