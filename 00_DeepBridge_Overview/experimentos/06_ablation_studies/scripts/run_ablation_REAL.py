"""
Run Ablation Studies - REAL Implementation
Tests different DeepBridge configurations to quantify component contributions

Configurações testadas:
1. full: DeepBridge completo (todos componentes habilitados)
2. baseline: Workflow fragmentado (AIF360 + Fairlearn + sklearn manual)

Autor: DeepBridge Team
Data: 2025-12-08
Versão: REAL (não mock)
"""

import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Adicionar path do DeepBridge
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge import DBDataset, Experiment

# Imports para baseline fragmentado
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    HAVE_AIF360 = True
except ImportError:
    HAVE_AIF360 = False
    print("WARNING: AIF360 not available. Install with: pip install aif360")

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    HAVE_FAIRLEARN = True
except ImportError:
    HAVE_FAIRLEARN = False
    print("WARNING: Fairlearn not available. Install with: pip install fairlearn")

from sklearn.calibration import calibration_curve
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from utils import setup_logging, save_results

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logger = setup_logging(LOGS_DIR, 'ablation_real')


class AblationStudyReal:
    """Ablation study usando execução REAL do DeepBridge e baseline"""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        logger.info("Initialized Real Ablation Study")

    def load_adult_dataset(self):
        """Carrega Adult Income dataset REAL"""
        logger.info("Loading Adult Income dataset...")

        data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df = data.frame.dropna()

        # Preparar features
        X = df.drop('class', axis=1)
        y = df['class']

        # Encoding
        le = LabelEncoder()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str)).astype(int)

        # Garantir todas as colunas são numéricas
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str)).astype(int)

        y = le.fit_transform(y).astype(int)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        logger.info(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Treina modelo XGBoost REAL"""
        logger.info("Training XGBoost model...")

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.seed,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_train, y_train)
        logger.info(f"Model trained - Accuracy: {accuracy:.4f}")

        return model

    def run_deepbridge_full(self, X_test, y_test, model):
        """Executa DeepBridge COMPLETO (configuração full)"""
        logger.info("\n=== Running DeepBridge FULL ===")

        # Criar DataFrame para DBDataset
        test_df = X_test.copy()
        test_df['target'] = y_test
        test_df = test_df.reset_index(drop=True)

        # Identificar protected attributes
        protected_attrs = []
        for attr in ['sex', 'race', 'age']:
            if attr in test_df.columns:
                protected_attrs.append(attr)

        logger.info(f"Protected attributes: {protected_attrs}")

        # Criar DBDataset
        dataset = DBDataset(
            data=test_df,
            target_column='target',
            model=model
        )

        # Criar Experiment
        exp = Experiment(
            dataset=dataset,
            experiment_type='binary_classification',
            protected_attributes=protected_attrs,
            tests=['fairness', 'robustness', 'uncertainty', 'resilience']
        )

        # Executar testes
        start = time.time()
        results = exp.run_tests(config_name='full')
        elapsed = time.time() - start

        logger.info(f"DeepBridge FULL completed in {elapsed:.2f}s")

        return elapsed

    def run_baseline_fragmented(self, X_train, X_test, y_train, y_test, model):
        """Executa baseline FRAGMENTADO (múltiplas bibliotecas)"""
        logger.info("\n=== Running Baseline FRAGMENTED ===")

        times = {}

        # 1. Fairness (AIF360 + Fairlearn)
        if HAVE_AIF360 and HAVE_FAIRLEARN:
            start = time.time()

            # Preparar dados
            df = X_test.copy()
            df['target'] = y_test
            df['prediction'] = model.predict(X_test)

            # AIF360
            if 'sex' in df.columns and 'race' in df.columns:
                for attr in ['sex', 'race']:
                    sex_map = {v: k for k, v in enumerate(df[attr].unique())}
                    df_encoded = df.copy()
                    df_encoded[attr] = df[attr].map(sex_map)

                    aif_dataset = BinaryLabelDataset(
                        df=df_encoded,
                        label_names=['target'],
                        protected_attribute_names=[attr]
                    )

                    metric = BinaryLabelDatasetMetric(
                        aif_dataset,
                        privileged_groups=[{attr: 1}],
                        unprivileged_groups=[{attr: 0}]
                    )
                    di = metric.disparate_impact()

                # Fairlearn
                for attr in ['sex', 'race']:
                    if attr in df.columns:
                        dpd = demographic_parity_difference(
                            df['target'],
                            df['prediction'],
                            sensitive_features=df[attr]
                        )
                        eod = equalized_odds_difference(
                            df['target'],
                            df['prediction'],
                            sensitive_features=df[attr]
                        )

            times['fairness'] = time.time() - start
        else:
            logger.warning("Skipping fairness (missing libraries)")
            times['fairness'] = 0.0

        # 2. Robustness (sklearn + NumPy)
        start = time.time()
        X_numeric = X_test.select_dtypes(include=[np.number])

        for noise_level in [0.01, 0.05, 0.1]:
            noise = np.random.normal(0, noise_level, X_numeric.shape)
            X_perturbed = X_numeric + noise
            y_pred_perturbed = model.predict(X_perturbed)

        times['robustness'] = time.time() - start

        # 3. Uncertainty (sklearn calibration)
        start = time.time()
        y_proba = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
        ece = np.abs(fraction_pos - mean_pred).mean()
        times['uncertainty'] = time.time() - start

        # 4. Resilience (scipy Wasserstein)
        start = time.time()
        X_train_numeric = X_train.select_dtypes(include=[np.number])
        X_test_numeric = X_test.select_dtypes(include=[np.number])

        for col in X_train_numeric.columns[:6]:  # Primeiras 6 colunas
            wd = wasserstein_distance(
                X_train_numeric[col].values,
                X_test_numeric[col].values
            )
        times['resilience'] = time.time() - start

        # 5. Report (matplotlib)
        start = time.time()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].hist(y_proba, bins=20)
        axes[0, 1].plot(fraction_pos, mean_pred, 'o-')
        plt.tight_layout()
        report_path = RESULTS_DIR / 'baseline_report.png'
        plt.savefig(report_path, dpi=100)
        plt.close()
        times['report'] = time.time() - start

        # Total
        times['total'] = sum(times.values())

        logger.info(f"Baseline FRAGMENTED completed in {times['total']:.2f}s")
        logger.info(f"  Fairness: {times['fairness']:.2f}s")
        logger.info(f"  Robustness: {times['robustness']:.2f}s")
        logger.info(f"  Uncertainty: {times['uncertainty']:.2f}s")
        logger.info(f"  Resilience: {times['resilience']:.2f}s")
        logger.info(f"  Report: {times['report']:.2f}s")

        return times['total'], times

    def run_ablation_study(self, num_runs=10):
        """Executa ablation study completo"""
        logger.info(f"\n{'='*60}")
        logger.info(f"REAL Ablation Study - {num_runs} runs")
        logger.info(f"{'='*60}\n")

        results = {
            'deepbridge_full': {'times': [], 'config': 'all_components_enabled'},
            'baseline_fragmented': {'times': [], 'times_breakdown': [], 'config': 'manual_fragmented_workflow'}
        }

        for run in range(num_runs):
            logger.info(f"\n--- Run {run + 1}/{num_runs} ---")

            # Carregar dados
            X_train, X_test, y_train, y_test = self.load_adult_dataset()
            model = self.train_model(X_train, y_train)

            # Test 1: DeepBridge FULL
            try:
                time_full = self.run_deepbridge_full(X_test, y_test, model)
                results['deepbridge_full']['times'].append(time_full)
            except Exception as e:
                logger.error(f"DeepBridge FULL failed: {e}")
                import traceback
                traceback.print_exc()

            # Test 2: Baseline FRAGMENTED
            try:
                time_baseline, breakdown = self.run_baseline_fragmented(
                    X_train, X_test, y_train, y_test, model
                )
                results['baseline_fragmented']['times'].append(time_baseline)
                results['baseline_fragmented']['times_breakdown'].append(breakdown)
            except Exception as e:
                logger.error(f"Baseline FRAGMENTED failed: {e}")
                import traceback
                traceback.print_exc()

        # Calcular estatísticas
        stats = {}

        for config_name, config_data in results.items():
            if len(config_data['times']) > 0:
                times_array = np.array(config_data['times'])

                stats[config_name] = {
                    'mean_seconds': float(np.mean(times_array)),
                    'std_seconds': float(np.std(times_array)),
                    'min_seconds': float(np.min(times_array)),
                    'max_seconds': float(np.max(times_array)),
                    'mean_minutes': float(np.mean(times_array) / 60),
                    'std_minutes': float(np.std(times_array) / 60),
                    'all_times_seconds': config_data['times'],
                    'num_runs': len(config_data['times']),
                    'config': config_data['config']
                }

                # Breakdown para baseline
                if 'times_breakdown' in config_data and len(config_data['times_breakdown']) > 0:
                    breakdown_stats = {}
                    breakdown_list = config_data['times_breakdown']

                    for component in breakdown_list[0].keys():
                        component_times = [bd[component] for bd in breakdown_list]
                        breakdown_stats[component] = {
                            'mean_seconds': float(np.mean(component_times)),
                            'std_seconds': float(np.std(component_times))
                        }

                    stats[config_name]['breakdown'] = breakdown_stats
            else:
                logger.warning(f"No successful runs for {config_name}")
                stats[config_name] = {
                    'mean_seconds': 0.0,
                    'num_runs': 0,
                    'status': 'failed'
                }

        # Calcular speedup
        if ('deepbridge_full' in stats and 'baseline_fragmented' in stats and
            stats['deepbridge_full']['num_runs'] > 0 and
            stats['baseline_fragmented']['num_runs'] > 0):

            deepbridge_time = stats['deepbridge_full']['mean_seconds']
            baseline_time = stats['baseline_fragmented']['mean_seconds']

            stats['comparison'] = {
                'deepbridge_mean_seconds': deepbridge_time,
                'baseline_mean_seconds': baseline_time,
                'speedup': baseline_time / deepbridge_time if deepbridge_time > 0 else 0,
                'interpretation': 'baseline_faster' if baseline_time < deepbridge_time else 'deepbridge_faster'
            }

        # Salvar resultados
        output_path = RESULTS_DIR / 'ablation_study_REAL'
        with open(f'{output_path}.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("Ablation Study COMPLETED!")
        logger.info(f"{'='*60}\n")

        # Print summary
        logger.info("Summary Statistics:")
        for config_name, config_stats in stats.items():
            if config_name != 'comparison' and config_stats['num_runs'] > 0:
                logger.info(f"\n{config_name}:")
                logger.info(f"  Mean: {config_stats['mean_seconds']:.2f}s ({config_stats['mean_minutes']:.2f} min)")
                logger.info(f"  Std:  {config_stats['std_seconds']:.2f}s")
                logger.info(f"  Runs: {config_stats['num_runs']}")

        if 'comparison' in stats:
            logger.info(f"\nComparison:")
            logger.info(f"  DeepBridge: {stats['comparison']['deepbridge_mean_seconds']:.2f}s")
            logger.info(f"  Baseline:   {stats['comparison']['baseline_mean_seconds']:.2f}s")
            logger.info(f"  Speedup:    {stats['comparison']['speedup']:.2f}×")
            logger.info(f"  Winner:     {stats['comparison']['interpretation']}")

        return stats


def main():
    """Executa ablation study REAL"""
    logger.info("="*70)
    logger.info("REAL ABLATION STUDY (não simulação!)")
    logger.info("="*70)

    ablation = AblationStudyReal(seed=42)
    results = ablation.run_ablation_study(num_runs=10)

    logger.info("\nResults saved successfully!")
    logger.info(f"Check: {RESULTS_DIR}/ablation_study_REAL.json")

    return results


if __name__ == "__main__":
    main()
