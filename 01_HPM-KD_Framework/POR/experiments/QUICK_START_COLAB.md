# Quick Start: Executar Experimentos HPM-KD no Google Colab

**Última atualização:** 07/11/2025
**Objetivo:** Guia rápido para executar todos os experimentos HPM-KD no Google Colab

---

## 🚀 Setup Inicial (Uma Vez - 10 minutos)

### Passo 1: Abrir o Notebook de Setup

1. Acesse Google Colab: https://colab.research.google.com/
2. **Configure GPU**:
   - Menu: `Runtime` → `Change runtime type`
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (recomendado) ou **V100** (mais rápido)
   - Clique **Save**

3. **Upload do notebook de setup**:
   ```
   Arquivo: notebooks/00_setup_colab.ipynb
   ```

   **OU** execute diretamente no Colab:

```python
# Cole este código em uma célula nova no Colab:

# 1. Clone repositório
!git clone https://github.com/DeepBridge-Validation/DeepBridge.git
%cd DeepBridge

# 2. Instalar DeepBridge + dependências
!pip install -q -e .
!pip install -q jinja2 pyyaml seaborn tabulate

# 3. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 4. Criar diretório de resultados
import os
os.makedirs('/content/drive/MyDrive/HPM-KD-Results', exist_ok=True)

# 5. Verificar GPU
import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"✅ DeepBridge instalado!")
print(f"✅ Resultados serão salvos em: /content/drive/MyDrive/HPM-KD-Results")
```

✅ **Setup concluído!** Agora você pode executar os experimentos.

---

## 📊 Executar Experimentos (Sequência Recomendada)

### 🔹 Experimento 1: Sklearn Baseline (Rápido - 5 min)

**Objetivo:** Validar framework com sklearn (MNIST, 10K samples)

```python
# 📌 Execute em nova célula no Colab

import sys
sys.path.append('/content/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments')

from scripts.report_generator import ExperimentReporter
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

# ===== 1. CONFIGURAÇÃO =====
config = {
    'experiment_name': '01_sklearn_baseline',
    'dataset': 'MNIST',
    'n_samples': 10000,
    'teacher_model': 'RandomForest(n_estimators=500)',
    'student_model': 'DecisionTree(max_depth=10)',
    'seed': 42
}

reporter = ExperimentReporter(
    experiment_name=config['experiment_name'],
    output_dir='/content/drive/MyDrive/HPM-KD-Results',
    description='Baseline sklearn - validação rápida'
)
reporter.log_config(config)

# ===== 2. CARREGAR DADOS =====
print("📥 Carregando MNIST...")
X, y = fetch_openml('mnist_784', return_X_y=True, parser='auto')
X = X.values if hasattr(X, 'values') else X
y = y.values.astype(int) if hasattr(y, 'values') else y.astype(int)
X = X / 255.0  # Normalize

# Subset rápido
indices = np.random.choice(len(X), config['n_samples'], replace=False)
X, y = X[indices], y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 3. TREINAR TEACHER =====
print("🎓 Treinando Teacher (RandomForest)...")
start = time.time()
teacher = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
teacher.fit(X_train, y_train)
teacher_time = time.time() - start

teacher_acc = accuracy_score(y_test, teacher.predict(X_test))
print(f"✅ Teacher Accuracy: {teacher_acc:.4f} ({teacher_time:.1f}s)")

# ===== 4. TREINAR STUDENT (Direct) =====
print("🎒 Treinando Student (Direct Training)...")
start = time.time()
student_direct = DecisionTreeClassifier(max_depth=10, random_state=42)
student_direct.fit(X_train, y_train)
direct_time = time.time() - start

direct_acc = accuracy_score(y_test, student_direct.predict(X_test))
print(f"✅ Student Direct: {direct_acc:.4f} ({direct_time:.1f}s)")

# ===== 5. TREINAR STUDENT (KD - Soft Targets) =====
print("📚 Treinando Student (Traditional KD)...")
start = time.time()

# Get soft targets from teacher
teacher_probs = teacher.predict_proba(X_train)

# Train student with soft targets (weighted ensemble trick)
student_kd = DecisionTreeClassifier(max_depth=10, random_state=42)

# Pseudo-labeling approach (simple KD for trees)
soft_labels = np.argmax(teacher_probs, axis=1)
student_kd.fit(X_train, soft_labels)
kd_time = time.time() - start

kd_acc = accuracy_score(y_test, student_kd.predict(X_test))
print(f"✅ Student KD: {kd_acc:.4f} ({kd_time:.1f}s)")

# ===== 6. LOG RESULTADOS =====
reporter.log_metrics({
    'teacher_accuracy': teacher_acc,
    'student_direct_accuracy': direct_acc,
    'student_kd_accuracy': kd_acc,
    'improvement_kd_vs_direct': kd_acc - direct_acc,
    'retention_direct': (direct_acc / teacher_acc) * 100,
    'retention_kd': (kd_acc / teacher_acc) * 100,
    'teacher_time': teacher_time,
    'student_direct_time': direct_time,
    'student_kd_time': kd_time
})

# ===== 7. GERAR VISUALIZAÇÕES =====
comparison_data = {
    'Teacher': teacher_acc,
    'Student (Direct)': direct_acc,
    'Student (KD)': kd_acc
}
reporter.plot_comparison_bar(comparison_data, metric_name='Accuracy', filename='comparison.png')

# Observations
reporter.add_observation(f"Teacher → Student compression: ~{500/10:.0f}× (500 trees → 1 tree depth 10)")
reporter.add_observation(f"KD improved student by {(kd_acc - direct_acc)*100:.2f} percentage points")
reporter.add_observation(f"Retention: Direct={direct_acc/teacher_acc*100:.1f}%, KD={kd_acc/teacher_acc*100:.1f}%")

# ===== 8. GERAR RELATÓRIO =====
reporter.generate_markdown_report()
reporter.display_summary()

print(f"\n✅ Experimento concluído!")
print(f"📄 Relatório: /content/drive/MyDrive/HPM-KD-Results/01_sklearn_baseline/report.md")
```

---

### 🔹 Experimento 2: HPM-KD com Sklearn (10 min)

```python
# 📌 Execute em nova célula no Colab

import sys
sys.path.append('/content/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments')

from scripts.report_generator import ExperimentReporter
from deepbridge.core.knowledge_distillation import HPM_KD
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ===== 1. CONFIGURAÇÃO =====
config = {
    'experiment_name': '02_sklearn_hpmkd',
    'dataset': 'MNIST',
    'n_samples': 10000,
    'framework': 'HPM-KD',
    'components': ['adaptive_config', 'progressive_chain', 'multi_teacher'],
    'seed': 42
}

reporter = ExperimentReporter(
    experiment_name=config['experiment_name'],
    output_dir='/content/drive/MyDrive/HPM-KD-Results',
    description='HPM-KD completo com sklearn'
)
reporter.log_config(config)

# ===== 2. CARREGAR DADOS =====
print("📥 Carregando MNIST...")
X, y = fetch_openml('mnist_784', return_X_y=True, parser='auto')
X = X.values if hasattr(X, 'values') else X
y = y.values.astype(int) if hasattr(y, 'values') else y.astype(int)
X = X / 255.0

indices = np.random.choice(len(X), config['n_samples'], replace=False)
X, y = X[indices], y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== 3. EXECUTAR HPM-KD =====
print("🚀 Executando HPM-KD Framework...")

hpmkd = HPM_KD(
    use_adaptive_config=True,
    use_progressive_chain=True,
    use_multi_teacher=False,  # sklearn não suporta multi-teacher ainda
    verbose=True
)

# Train
results = hpmkd.fit_distill(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    teacher_config={'n_estimators': 500, 'max_depth': 20},
    student_config={'max_depth': 10}
)

# ===== 4. LOG RESULTADOS =====
reporter.log_metrics({
    'teacher_accuracy': results['teacher_accuracy'],
    'student_accuracy': results['student_accuracy'],
    'direct_accuracy': results.get('baseline_direct_accuracy', 0),
    'improvement_vs_direct': results['student_accuracy'] - results.get('baseline_direct_accuracy', 0),
    'retention': (results['student_accuracy'] / results['teacher_accuracy']) * 100,
    'training_time': results.get('total_time', 0),
    'chain_length': results.get('progressive_chain_length', 1)
})

# ===== 5. VISUALIZAÇÕES =====
if 'history' in results:
    reporter.plot_training_curves(results['history'])

comparison_data = {
    'Teacher': results['teacher_accuracy'],
    'Student (HPM-KD)': results['student_accuracy'],
    'Direct': results.get('baseline_direct_accuracy', 0)
}
reporter.plot_comparison_bar(comparison_data)

# Observations
reporter.add_observation(f"HPM-KD used {results.get('progressive_chain_length', 1)} progressive steps")
reporter.add_observation(f"Adaptive config selected automatically")
reporter.add_observation(f"Compression: {results.get('compression_ratio', 0):.1f}×")

# ===== 6. GERAR RELATÓRIO =====
reporter.generate_markdown_report()
reporter.display_summary()

print(f"\n✅ HPM-KD concluído!")
```

---

### 🔹 Experimento 3: CNN Teacher MNIST (30-40 min)

**Objetivo:** Treinar teacher ResNet18 para MNIST (target: 99.3-99.5%)

```python
# 📌 Código completo no notebook: notebooks/03_cnn_mnist_teacher.ipynb
# OU execute o script:

%cd /content/DeepBridge/papers/01_HPM-KD_Framework/POR/experiments/cnn_baseline
!python train_teacher.py --epochs 20 --batch-size 128 --lr 0.1 \
    --save-dir /content/drive/MyDrive/HPM-KD-Results/03_cnn_mnist_teacher

print("✅ Teacher treinado! Modelo salvo em: .../03_cnn_mnist_teacher/teacher_model.pth")
```

---

### 🔹 Experimento 4-10: Sequência Completa

Execute os notebooks na ordem:

```
04_cnn_mnist_baselines.ipynb    → Direct, KD, FitNets (45 min)
05_cnn_mnist_hpmkd.ipynb         → HPM-KD completo (60 min)
06_cifar10_experiments.ipynb     → CIFAR-10 (2-3 horas)
07_ablation_studies.ipynb        → Remover componentes (1 hora)
08_compression_analysis.ipynb    → Diferentes ratios (1 hora)
09_multi_dataset.ipynb           → UCI datasets (30 min)
10_generate_paper_results.ipynb  → Consolidar tudo (1 hora)
```

**Tempo total:** 12-16 horas de GPU

---

## 📊 Visualizar Resultados

### Durante a Execução

Cada experimento exibe um resumo ao final:

```
✅ Experimento Concluído: 01_sklearn_baseline
Duração: 5m 32s
GPU: Tesla T4

Métricas Principais:
| Métrica                   | Valor   |
|---------------------------|---------|
| teacher_accuracy          | 0.9420  |
| student_kd_accuracy       | 0.6830  |
| improvement_kd_vs_direct  | 0.0213  |
| retention_kd              | 72.52   |
```

### Relatórios Completos

Acesse no Google Drive:

```
/content/drive/MyDrive/HPM-KD-Results/
├── 01_sklearn_baseline/
│   ├── report.md          ← RELATÓRIO COMPLETO
│   ├── metrics.json
│   ├── results.csv
│   └── figures/
│       ├── comparison.png
│       └── ...
├── 02_sklearn_hpmkd/
│   ├── report.md
│   └── ...
└── ...
```

### Abrir Relatório no Colab

```python
# Visualizar report.md diretamente no Colab
from IPython.display import Markdown, display

with open('/content/drive/MyDrive/HPM-KD-Results/01_sklearn_baseline/report.md') as f:
    display(Markdown(f.read()))
```

---

## 🔧 Troubleshooting

### ❌ "GPU não disponível"

```python
# Verifique se GPU está ativa
import torch
print(torch.cuda.is_available())

# Se False:
# 1. Runtime → Change runtime type → GPU → Save
# 2. Runtime → Restart runtime
```

### ❌ "Memória insuficiente"

```python
# Reduzir batch size ou samples
config['batch_size'] = 64  # Era 128
config['n_samples'] = 5000  # Era 10000

# Limpar cache GPU
import torch
torch.cuda.empty_cache()
```

### ❌ "DeepBridge não encontrado"

```bash
# Reinstalar
!pip uninstall -y deepbridge
%cd /content/DeepBridge
!pip install -e .
```

---

## 📌 Checklist de Experimentos

Marque conforme completar:

- [ ] 00. Setup inicial (10 min)
- [ ] 01. Sklearn baseline (5 min)
- [ ] 02. HPM-KD sklearn (10 min)
- [ ] 03. CNN teacher MNIST (30 min)
- [ ] 04. CNN baselines MNIST (45 min)
- [ ] 05. HPM-KD CNN MNIST (60 min)
- [ ] 06. CIFAR-10 completo (2-3 horas)
- [ ] 07. Ablation studies (1 hora)
- [ ] 08. Compression analysis (1 hora)
- [ ] 09. Multi-dataset (30 min)
- [ ] 10. Gerar paper results (1 hora)

**Total:** ~12-16 horas GPU

---

## 📚 Recursos Adicionais

- **Plano Completo:** `experiments/REORGANIZATION_PLAN.md`
- **Report Generator:** `scripts/report_generator.py`
- **GitHub:** https://github.com/DeepBridge-Validation/DeepBridge
- **Docs:** https://deepbridge.readthedocs.io/

---

## 💾 Baixar Todos os Resultados

```python
# Compactar todos os resultados
!zip -r /content/HPM-KD-Results.zip /content/drive/MyDrive/HPM-KD-Results

# Download
from google.colab import files
files.download('/content/HPM-KD-Results.zip')
```

---

**✅ Pronto para começar!** Execute o notebook `00_setup_colab.ipynb` para iniciar.
