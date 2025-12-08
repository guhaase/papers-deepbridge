"""
Test GPU Setup - Verifica se tudo está configurado corretamente

Execute este script ANTES de rodar os experimentos completos.
"""

import sys
from pathlib import Path

print("="*70)
print("GPU Setup Test")
print("="*70)

# Test 1: Python version
print("\n1. Python Version:")
print(f"   {sys.version}")
if sys.version_info >= (3, 10):
    print("   ✓ Python 3.10+ OK")
else:
    print("   ✗ WARNING: Python 3.10+ recommended")

# Test 2: PyTorch and CUDA
print("\n2. PyTorch and CUDA:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("   ✓ GPU ready")

        # Test GPU computation
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✓ GPU computation test passed")
    else:
        print("   ✗ WARNING: CUDA not available, will run on CPU")
except ImportError:
    print("   ✗ ERROR: PyTorch not installed")

# Test 3: XGBoost
print("\n3. XGBoost:")
try:
    import xgboost as xgb
    print(f"   XGBoost version: {xgb.__version__}")
    print("   ✓ XGBoost installed")

    # Check GPU support
    try:
        import xgboost
        params = {'tree_method': 'hist', 'device': 'cuda'}
        dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
        bst = xgb.train(params, dtrain, num_boost_round=1)
        print("   ✓ XGBoost GPU support working")
    except Exception as e:
        print(f"   ⚠ XGBoost GPU support issue: {e}")
        print("   (Will fallback to CPU)")
except ImportError:
    print("   ✗ ERROR: XGBoost not installed")

# Test 4: LightGBM
print("\n4. LightGBM:")
try:
    import lightgbm as lgb
    print(f"   LightGBM version: {lgb.__version__}")
    print("   ✓ LightGBM installed")
except ImportError:
    print("   ✗ ERROR: LightGBM not installed")

# Test 5: Fairness libraries
print("\n5. Fairness Libraries:")
try:
    import aif360
    print(f"   AIF360 installed")
    print("   ✓ AIF360 OK")
except ImportError:
    print("   ✗ WARNING: AIF360 not installed")

try:
    import fairlearn
    print(f"   Fairlearn installed")
    print("   ✓ Fairlearn OK")
except ImportError:
    print("   ✗ WARNING: Fairlearn not installed")

# Test 6: DeepBridge
print("\n6. DeepBridge:")
try:
    sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')
    from deepbridge import DBDataset, Experiment
    print("   ✓ DeepBridge imported successfully")

    # Quick test
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    df = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'y': np.random.randint(0, 2, 100)
    })

    model = RandomForestClassifier(n_estimators=10)
    model.fit(df[['x1', 'x2']], df['y'])

    dataset = DBDataset(data=df, target_column='y', model=model)
    print("   ✓ DBDataset creation works")

except Exception as e:
    print(f"   ✗ ERROR: DeepBridge issue: {e}")

# Test 7: Disk space
print("\n7. Disk Space:")
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    print(f"   Total: {total // (2**30)} GB")
    print(f"   Used:  {used // (2**30)} GB")
    print(f"   Free:  {free // (2**30)} GB")

    if free > 50 * (2**30):
        print("   ✓ Sufficient disk space (>50GB)")
    else:
        print("   ✗ WARNING: Low disk space (<50GB)")
except:
    print("   ⚠ Could not check disk space")

# Test 8: Memory
print("\n8. System Memory:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   Total: {mem.total / (1024**3):.1f} GB")
    print(f"   Available: {mem.available / (1024**3):.1f} GB")
    print(f"   Used: {mem.percent}%")

    if mem.total >= 16 * (1024**3):
        print("   ✓ Sufficient RAM (>=16GB)")
    else:
        print("   ⚠ WARNING: Low RAM (<16GB)")
except ImportError:
    print("   ⚠ psutil not installed, cannot check memory")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)

# Count checks
checks = []
try:
    import torch
    checks.append(("PyTorch", torch.cuda.is_available()))
except:
    checks.append(("PyTorch", False))

try:
    import xgboost
    checks.append(("XGBoost", True))
except:
    checks.append(("XGBoost", False))

try:
    import lightgbm
    checks.append(("LightGBM", True))
except:
    checks.append(("LightGBM", False))

try:
    import aif360
    checks.append(("AIF360", True))
except:
    checks.append(("AIF360", False))

try:
    sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')
    from deepbridge import DBDataset
    checks.append(("DeepBridge", True))
except:
    checks.append(("DeepBridge", False))

passed = sum(1 for _, ok in checks if ok)
total = len(checks)

print(f"\nChecks passed: {passed}/{total}")

for name, ok in checks:
    status = "✓" if ok else "✗"
    print(f"  {status} {name}")

if passed == total:
    print("\n✅ ALL CHECKS PASSED - Ready to run experiments!")
elif passed >= total - 1:
    print("\n⚠️ MOSTLY OK - Can proceed but some features may not work")
else:
    print("\n❌ ISSUES FOUND - Please fix before running experiments")

print("\nNext steps:")
print("1. If GPU not available: Install NVIDIA drivers and CUDA")
print("2. If libraries missing: pip install -r requirements_gpu.txt")
print("3. If DeepBridge error: pip install -e /home/guhaase/projetos/DeepBridge")
print("\nThen run:")
print("  Experiment 4: poetry run python 04_hpmkd/scripts/run_hpmkd_REAL.py")
print("  Experiment 6: poetry run python 06_ablation_studies/scripts/run_ablation_REAL.py")

print("\n" + "="*70)
