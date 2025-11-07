# HPM-KD Experiments Comparison

## Quick Summary

| Experiment | Dataset Size | HPM-KD Accuracy | vs Traditional KD | Teacher Accuracy |
|------------|--------------|-----------------|-------------------|------------------|
| **Quick Test** | 10,000 samples | 89.50% | **+22.15pp** | 94.10% |
| **Full MNIST** | 70,000 samples | 91.67% | **+23.13pp** | 96.57% |
| **Improvement** | 7× more data | **+2.17pp** | **+0.98pp** | **+2.47pp** |

## Detailed Comparison

### Quick Test (10k samples)
```
Teacher:          94.10%
Direct Training:  65.20%
Traditional KD:   67.35% (71.6% retention)
HPM-KD:           89.50% (95.1% retention) → +22.15pp over Traditional KD
```

### Full MNIST (70k samples)
```
Teacher:          96.57% ← +2.47pp
Direct Training:  65.54%
Traditional KD:   68.54% (71.0% retention)
HPM-KD:           91.67% (94.9% retention) → +23.13pp over Traditional KD
```

## Key Findings

✅ **HPM-KD scales better**: +2.17pp improvement vs +1.19pp for Traditional KD
✅ **Maintains advantage**: +22→23pp improvement over baseline
✅ **Consistent behavior**: Always selected LogisticRegression automatically
✅ **Gap to paper closes**: -9.65% → -7.48% (2.17pp improvement)

## Files

- Quick test results: `experiment_results/hpmkd_results.csv`
- Full MNIST results: `experiment_results_full/hpmkd_results.csv`

**Status**: Both experiments successful ✅
**Date**: November 5, 2025
