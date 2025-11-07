# CNN Implementation Plan for HPM-KD

**Status**: Ready to implement
**Requirement**: PyTorch installation
**Timeline**: 2 weeks
**Expected Results**: 98-99% accuracy on MNIST

---

## 📋 OVERVIEW

### Current Status

✅ **sklearn Implementation**: 91.67% accuracy (Full MNIST)
⏳ **CNN Implementation**: Not started (PyTorch not installed)

### Goal

Implement CNN-based HPM-KD to:
- Close gap to paper results (99.15% target)
- Use proper teacher/student architecture
- Validate framework with neural networks
- Generate paper-ready results

---

## 🔧 INSTALLATION REQUIREMENTS

### Step 1: Install PyTorch

```bash
# CPU version (for testing)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR GPU version (for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Install Additional Dependencies

```bash
pip install torchvision  # For dataset loaders
pip install tqdm         # For progress bars
```

---

## 🏗️ ARCHITECTURE DESIGN

### Teacher Model: ResNet-18 (Adapted for MNIST)

**Original ResNet-18**: 11M parameters (ImageNet, 224×224 RGB)
**Adapted for MNIST**: 4.2M parameters (MNIST, 28×28 grayscale)

```
Input: 28×28×1 (grayscale)
├── Conv1: 3×3, 64 channels
├── Layer1: 2× ResBlock(64 → 64)
├── Layer2: 2× ResBlock(64 → 128), stride=2
├── Layer3: 2× ResBlock(128 → 256), stride=2
├── Layer4: 2× ResBlock(256 → 512), stride=2
├── GlobalAvgPool: 512 → 512
└── FC: 512 → 10 classes
```

**Expected Performance**: 99.3-99.5% on MNIST

### Student Model: MobileNet-V2 (Adapted)

**Target**: 0.4M parameters (10.5× compression)

```
Input: 28×28×1 (grayscale)
├── Conv1: 3×3, 32 channels
├── DSConv1: 32 → 64
├── DSConv2: 64 → 128, stride=2
├── DSConv3: 128 → 128
├── DSConv4: 128 → 256, stride=2
├── DSConv5: 256 → 256
├── DSConv6: 256 → 512, stride=2
├── GlobalAvgPool: 512 → 512
└── FC: 512 → 10 classes
```

**Expected Performance**: 98.8-99.0% with HPM-KD

### Baseline Model: SimpleCNN

**Target**: 50k parameters (very small)

```
Input: 28×28×1
├── Conv1: 3×3, 32 channels → MaxPool
├── Conv2: 3×3, 64 channels → MaxPool
├── Conv3: 3×3, 64 channels → MaxPool
├── FC1: 576 → 128
└── FC2: 128 → 10 classes
```

**Expected Performance**: 98.5-98.8% (Direct Training)

---

## 📊 EXPECTED RESULTS

### Performance Targets

| Model | Params | Accuracy | Gap to Teacher |
|-------|--------|----------|----------------|
| **Teacher (ResNet-18)** | 4.2M | 99.3-99.5% | - |
| **Direct Training (SimpleCNN)** | 50k | 98.5-98.8% | -0.7% |
| **Traditional KD** | 0.4M | 98.9-99.1% | -0.4% |
| **HPM-KD (MobileNet)** | 0.4M | **99.0-99.2%** | **-0.2%** |

### Comparison with sklearn Results

| Method | sklearn (10k) | sklearn (70k) | CNN (70k) | Improvement |
|--------|--------------|--------------|-----------|-------------|
| Teacher | 94.10% | 96.57% | **99.40%** | **+2.83pp** |
| Direct | 65.20% | 65.54% | **98.70%** | **+33.16pp** |
| Trad KD | 67.35% | 68.54% | **99.00%** | **+30.46pp** |
| HPM-KD | 89.50% | 91.67% | **99.15%** | **+7.48pp** |

**Key**: CNN implementation should close the **-7.48% gap** completely!

---

## 🔬 IMPLEMENTATION STEPS

### Phase 1: Model Implementation (Days 1-2)

**Files to Create**:

1. **`cnn_models.py`** ✅ Already created!
   - TeacherResNet class
   - StudentMobileNet class
   - SimpleCNN baseline
   - Helper functions

2. **`cnn_training.py`** (Next)
   - Training loop
   - Validation loop
   - Learning rate scheduling
   - Model checkpointing

3. **`cnn_distillation.py`** (Next)
   - Knowledge distillation loss
   - Temperature-scaled softmax
   - Feature matching (optional)
   - Integration with HPM-KD

### Phase 2: Teacher Training (Days 3-4)

```python
# Train teacher model
python3 train_teacher.py \
    --model resnet18 \
    --dataset mnist \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.1 \
    --save-path models/teacher_resnet18.pth
```

**Expected**:
- Training time: 10-15 minutes (CPU) or 2-3 minutes (GPU)
- Final accuracy: 99.3-99.5%
- Save checkpoint for distillation

### Phase 3: Baseline Training (Days 5-6)

```python
# Direct training (no distillation)
python3 train_student.py \
    --model simplecnn \
    --dataset mnist \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.1 \
    --save-path models/student_direct.pth
```

**Expected**:
- Training time: 5-10 minutes
- Final accuracy: 98.5-98.8%

### Phase 4: Traditional KD (Days 7-8)

```python
# Traditional Knowledge Distillation
python3 train_kd.py \
    --teacher models/teacher_resnet18.pth \
    --student mobilenet \
    --dataset mnist \
    --epochs 20 \
    --batch-size 128 \
    --temperature 4.0 \
    --alpha 0.5 \
    --save-path models/student_kd.pth
```

**Expected**:
- Training time: 8-12 minutes
- Final accuracy: 98.9-99.1%

### Phase 5: HPM-KD Implementation (Days 9-12)

```python
# HPM-KD with all components
python3 train_hpmkd.py \
    --teacher models/teacher_resnet18.pth \
    --student mobilenet \
    --dataset mnist \
    --use-progressive \
    --use-adaptive-temp \
    --use-multi-teacher \
    --epochs 20 \
    --batch-size 128 \
    --save-path models/student_hpmkd.pth
```

**Expected**:
- Training time: 15-20 minutes
- Final accuracy: 99.0-99.2%
- +0.1-0.3pp improvement over Traditional KD

### Phase 6: Evaluation & Analysis (Days 13-14)

```python
# Comprehensive evaluation
python3 evaluate_all.py \
    --models models/*.pth \
    --dataset mnist \
    --output results/cnn_results.csv
```

**Generate**:
- Accuracy metrics
- Confusion matrices
- Feature visualizations (t-SNE)
- Confidence calibration plots

---

## 📝 CODE STRUCTURE

### Proposed File Organization

```
papers/01_HPM-KD_Framework/POR/
├── cnn_models.py              ✅ Created
├── cnn_training.py            ⏳ Next
├── cnn_distillation.py        ⏳ Next
├── train_teacher.py           ⏳ Next
├── train_student.py           ⏳ Next
├── train_kd.py                ⏳ Next
├── train_hpmkd.py             ⏳ Next
├── evaluate_all.py            ⏳ Next
├── models/                    (Saved checkpoints)
│   ├── teacher_resnet18.pth
│   ├── student_direct.pth
│   ├── student_kd.pth
│   └── student_hpmkd.pth
└── results/
    └── cnn_results.csv
```

---

## 🎯 INTEGRATION WITH HPM-KD FRAMEWORK

### Connecting to DeepBridge

The DeepBridge HPM-KD framework needs to be extended to support PyTorch models:

**Option 1: Wrapper Approach** (Recommended)
```python
from deepbridge.distillation.techniques.hpm import HPMDistiller

# Wrap PyTorch models for HPM-KD
teacher_wrapped = PyTorchModelWrapper(teacher_model)
student_wrapped = PyTorchModelWrapper(student_model)

# Use existing HPM-KD
distiller = HPMDistiller(
    teacher_model=teacher_wrapped,
    config=hpm_config
)
```

**Option 2: Standalone Implementation**
```python
# Implement HPM-KD components directly in PyTorch
from hpm_pytorch import (
    ProgressiveDistillationChain,
    AdaptiveConfigurationManager,
    MetaTemperatureScheduler
)
```

---

## 📊 VALIDATION CRITERIA

### Success Metrics

✅ **Teacher Performance**:
- Accuracy ≥ 99.3% on MNIST
- Matches paper specification

✅ **Direct Training Performance**:
- Accuracy ≥ 98.5%
- Establishes strong baseline

✅ **Traditional KD Performance**:
- Accuracy ≥ 98.9%
- Retention ≥ 99.5%

✅ **HPM-KD Performance**:
- Accuracy ≥ 99.0%
- Improvement over Traditional KD ≥ +0.1pp
- Closes gap to paper results

### Validation Tests

1. **Accuracy Test**: All models achieve target accuracies
2. **Compression Test**: Student has 10× fewer parameters
3. **Improvement Test**: HPM-KD > Traditional KD
4. **Retention Test**: HPM-KD retains ≥99.5% teacher accuracy
5. **Reproducibility Test**: Results consistent across 5 seeds

---

## 🔬 EXPERIMENTAL PROTOCOL

### Training Configuration

**Teacher Training**:
```yaml
Model: ResNet-18 (adapted)
Optimizer: SGD (momentum=0.9)
Learning Rate: 0.1 → 0.001 (cosine annealing)
Batch Size: 128
Epochs: 20
Weight Decay: 5e-4
Data Augmentation: Random crop, horizontal flip (disabled for MNIST)
```

**Student Training (Direct)**:
```yaml
Model: SimpleCNN
Optimizer: SGD (momentum=0.9)
Learning Rate: 0.1 → 0.001
Batch Size: 128
Epochs: 20
Weight Decay: 5e-4
```

**Knowledge Distillation (Traditional KD)**:
```yaml
Temperature: 4.0
Alpha (KD weight): 0.5
Optimizer: SGD (momentum=0.9)
Learning Rate: 0.1 → 0.001
Batch Size: 128
Epochs: 20
```

**HPM-KD Configuration**:
```yaml
Use Progressive: True
Progressive Stages: [SimpleCNN → MobileNet-Small → MobileNet]
Adaptive Temperature: True
Initial Temperature: 4.0
Multi-Teacher: False (single teacher)
Parallel: False (sequential)
Cache: True
```

---

## 📈 EXPECTED TIMELINE

### Two-Week Plan

**Week 1: Core Implementation**
- Days 1-2: Training infrastructure ✅
- Days 3-4: Teacher training ✅
- Days 5-6: Baseline student training ✅
- Day 7: Traditional KD implementation ✅

**Week 2: HPM-KD & Validation**
- Days 8-10: HPM-KD implementation ✅
- Days 11-12: Comprehensive experiments ✅
- Days 13-14: Analysis & figures ✅

**Total**: 14 days to complete CNN implementation

---

## 🎯 DELIVERABLES

### At Completion

1. ✅ **Trained Models** (4 checkpoints)
   - Teacher ResNet-18
   - Student Direct
   - Student Traditional KD
   - Student HPM-KD

2. ✅ **Experimental Results**
   - Accuracy comparison table
   - Training curves
   - Feature visualizations
   - Statistical significance tests

3. ✅ **Updated Figures**
   - Replace sklearn results with CNN results
   - Add new CNN-specific visualizations
   - Generate paper-ready plots

4. ✅ **Documentation**
   - Training logs
   - Hyperparameter configurations
   - Reproducibility guide

---

## 💡 NEXT IMMEDIATE STEPS

### To Start CNN Implementation

1. **Install PyTorch**:
   ```bash
   pip install torch torchvision
   ```

2. **Test CNN Models**:
   ```bash
   python3 cnn_models.py
   ```

3. **Create Training Script**:
   - Implement training loop
   - Add learning rate scheduling
   - Include validation monitoring

4. **Train Teacher**:
   - Run for 20 epochs
   - Save best checkpoint
   - Verify 99.3%+ accuracy

5. **Implement Distillation**:
   - Traditional KD first
   - Then HPM-KD integration
   - Compare results

---

## 📊 COMPARISON: sklearn vs CNN

### Why CNN Implementation Matters

| Aspect | sklearn (Current) | CNN (Planned) | Impact |
|--------|------------------|---------------|---------|
| **Teacher Capacity** | 94.10-96.57% | **99.3-99.5%** | +2.7-3.8pp |
| **Absolute Accuracy** | 91.67% | **99.0-99.2%** | +7.3-7.5pp |
| **Gap to Paper** | -7.48% | **~0%** | Close gap! |
| **Publication Value** | Good | **Excellent** | Top-tier |
| **Architecture** | sklearn trees | **Neural nets** | Proper KD |

---

## 🏆 SUCCESS CRITERIA

### Must Achieve

- [x] CNN models defined ✅
- [ ] PyTorch installed
- [ ] Teacher trained (≥99.3%)
- [ ] Baselines trained (≥98.5%)
- [ ] Traditional KD (≥98.9%)
- [ ] HPM-KD (≥99.0%)
- [ ] Improvement demonstrated (+0.1pp)
- [ ] Results reproducible

### Nice to Have

- [ ] Feature visualization (t-SNE)
- [ ] Attention maps
- [ ] Confidence calibration
- [ ] Adversarial robustness tests

---

## 📝 NOTES

### Advantages of CNN Implementation

1. **Paper Alignment**: Matches paper specification exactly
2. **Performance**: Expected 99%+ accuracy (vs 91.67% sklearn)
3. **Proper KD**: Neural network distillation (not tree→tree)
4. **Publication**: Top-tier venues expect CNN results
5. **Extensibility**: Can extend to CIFAR, ImageNet

### Challenges

1. **Setup**: Requires PyTorch installation
2. **Compute**: Needs GPU for faster training (optional)
3. **Time**: 2 weeks vs 2 hours for sklearn
4. **Complexity**: More code to maintain

### Mitigation

- Start with CPU version (works, just slower)
- Use Google Colab if no local GPU
- Reuse sklearn experiment infrastructure
- Follow modular design for maintainability

---

**Status**: ✅ **READY TO START**
**Next Action**: Install PyTorch and test cnn_models.py
**Timeline**: 2 weeks to completion
**Expected Result**: 99%+ accuracy, close gap to paper

**Created**: November 5, 2025, 07:20 BRT
**Author**: Claude Code + Gustavo Coelho Haase
