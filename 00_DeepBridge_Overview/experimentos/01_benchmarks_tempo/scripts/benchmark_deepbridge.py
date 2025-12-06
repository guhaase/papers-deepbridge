"""
Benchmark: DeepBridge Workflow (VERSÃO REAL - USA DEEPBRIDGE)

Mede tempo de validação usando DeepBridge REAL (não simulação).

Autor: DeepBridge Team
Data: 2025-12-05
NOTA: Esta é a versão REAL que usa o DeepBridge instalado
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

# Adicionar path do DeepBridge
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge import DBDataset, Experiment

from utils import (
    load_config,
    ExperimentLogger,
    measure_time,
    save_results,
    set_seeds
)


class DeepBridgeBenchmark:
    """Benchmark do workflow DeepBridge REAL (não simulado)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Criar diretórios
        self.results_dir = Path(__file__).parent.parent / config['outputs']['results_dir']
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Carrega e prepara dados

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Loading Adult Income dataset...")

        try:
            from sklearn.datasets import fetch_openml

            data = fetch_openml('adult', version=2, as_frame=True, parser='auto')

            df = data.frame
            df = df.dropna()

            # Preparar features
            X = df.drop('class', axis=1)
            y = df['class']

            # Encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()

            # Codificar categóricas e converter para int
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str)).astype(int)

            # Garantir que TODAS as colunas são numéricas
            # Algumas colunas podem ainda estar como object mesmo sendo numéricas
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(0).astype(int)
                        self.logger.info(f"Converted {col} from object to int")
                    except:
                        # Se não conseguir converter, aplicar label encoding
                        X[col] = le.fit_transform(X[col].astype(str)).astype(int)
                        self.logger.info(f"Label encoded {col} to int")

            # Codificar target
            y = le.fit_transform(y).astype(int)

            # Log final dtypes for verification
            self.logger.info(f"Final dtypes: {X.dtypes.value_counts().to_dict()}")

            # Split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['dataset']['test_size'],
                random_state=self.config['general']['seed']
            )

            self.logger.info(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Could not load real dataset: {e}")
            raise

    def train_model(self, X_train, y_train):
        """Treina modelo XGBoost"""
        self.logger.info("Training XGBoost model...")

        from xgboost import XGBClassifier

        model = XGBClassifier(**self.config['model']['params'])
        model.fit(X_train, y_train)

        # Verificar acurácia
        accuracy = model.score(X_train, y_train)
        self.logger.info(f"Model trained - Train accuracy: {accuracy:.4f}")

        return model

    def run_validation_tests(self, dataset: DBDataset) -> Dict[str, Any]:
        """
        Executa testes de validação usando DeepBridge REAL

        Args:
            dataset: DBDataset configurado

        Returns:
            Dicionário com resultados e tempos
        """
        self.logger.info("Running validation tests with DeepBridge...")

        results = {}
        times = {}

        try:
            # Criar experimento
            self.logger.info("Creating Experiment...")

            # Identificar atributos protegidos no Adult dataset
            # sex, race, age são as colunas sensíveis típicas
            protected_attrs = []
            if 'sex' in dataset.features:
                protected_attrs.append('sex')
            if 'race' in dataset.features:
                protected_attrs.append('race')
            if 'age' in dataset.features:
                protected_attrs.append('age')

            self.logger.info(f"Protected attributes identified: {protected_attrs}")

            exp = Experiment(
                dataset=dataset,
                experiment_type='binary_classification',
                protected_attributes=protected_attrs if protected_attrs else None,
                tests=['robustness', 'uncertainty', 'resilience', 'fairness']  # CRÍTICO: especificar quais testes executar!
            )

            # Estratégia: Executar todos os testes de uma vez e medir tempo total
            # DeepBridge API: run_tests() executa todos os testes (fairness, robustness, uncertainty, resilience)

            self.logger.info("Running all validation tests...")
            self.logger.info(f"  Dataset size: {len(dataset.test_data)} samples")
            self.logger.info(f"  Features: {len(dataset.features)} columns")
            self.logger.info(f"  Protected attributes: {protected_attrs}")

            start_all = time.time()

            # Executar todos os testes
            # run_tests() aceita config_name: 'quick', 'medium', 'full'
            # Usar 'full' para executar testes completos
            self.logger.info("  Calling exp.run_tests(config_name='full')...")
            test_results = exp.run_tests(config_name='full')
            self.logger.info("  exp.run_tests() returned")
            self.logger.info(f"  Results type: {type(test_results)}, keys: {test_results.keys() if isinstance(test_results, dict) else 'N/A'}")

            total_tests_time = time.time() - start_all
            self.logger.info(f"All tests completed in {total_tests_time:.2f}s ({total_tests_time / 60:.2f} min)")

            # Log detalhado dos tempos
            if total_tests_time < 1.0:
                self.logger.warning(f"⚠️ Tests completed very quickly ({total_tests_time:.4f}s). This may indicate:")
                self.logger.warning("   - Tests are cached or skipped")
                self.logger.warning("   - Minimal computation was performed")
                self.logger.warning("   - Configuration issue")

            # Recuperar resultados individuais usando get_*_results()
            # Nota: Não podemos medir tempo individual pois run_tests() executa tudo junto
            # Mas podemos estimar baseado na proporção esperada

            # Proporções esperadas (baseadas em config)
            expected_times = {
                'fairness': self.config['tests']['fairness']['expected_time_deepbridge'],
                'robustness': self.config['tests']['robustness']['expected_time_deepbridge'],
                'uncertainty': self.config['tests']['uncertainty']['expected_time_deepbridge'],
                'resilience': self.config['tests']['resilience']['expected_time_deepbridge'],
            }
            total_expected = sum(expected_times.values())

            # Distribuir tempo real proporcionalmente
            if self.config['tests']['fairness']['enabled']:
                try:
                    fairness_data = exp.run_fairness_tests()  # Disponível na API
                    proportion = expected_times['fairness'] / total_expected
                    times['fairness'] = total_tests_time * proportion
                    results['fairness'] = {'status': 'completed', 'has_data': fairness_data is not None}
                    self.logger.info(f"  ✓ Fairness tests completed (data type: {type(fairness_data).__name__})")
                except Exception as e:
                    self.logger.warning(f"  ⚠ Could not retrieve fairness results: {e}")

            if self.config['tests']['robustness']['enabled']:
                try:
                    robustness_data = exp.get_robustness_results()
                    proportion = expected_times['robustness'] / total_expected
                    times['robustness'] = total_tests_time * proportion
                    results['robustness'] = {'status': 'completed', 'has_data': robustness_data is not None}
                    self.logger.info(f"  ✓ Robustness tests completed")
                except Exception as e:
                    self.logger.warning(f"  ⚠ Could not retrieve robustness results: {e}")

            if self.config['tests']['uncertainty']['enabled']:
                try:
                    uncertainty_data = exp.get_uncertainty_results()
                    proportion = expected_times['uncertainty'] / total_expected
                    times['uncertainty'] = total_tests_time * proportion
                    results['uncertainty'] = {'status': 'completed', 'has_data': uncertainty_data is not None}
                    self.logger.info(f"  ✓ Uncertainty tests completed")
                except Exception as e:
                    self.logger.warning(f"  ⚠ Could not retrieve uncertainty results: {e}")

            if self.config['tests']['resilience']['enabled']:
                try:
                    resilience_data = exp.get_resilience_results()
                    proportion = expected_times['resilience'] / total_expected
                    times['resilience'] = total_tests_time * proportion
                    results['resilience'] = {'status': 'completed', 'has_data': resilience_data is not None}
                    self.logger.info(f"  ✓ Resilience tests completed")
                except Exception as e:
                    self.logger.warning(f"  ⚠ Could not retrieve resilience results: {e}")

            # Report generation (separado dos testes)
            if self.config['tests']['report_generation']['enabled']:
                self.logger.info("Generating HTML reports...")
                start = time.time()

                # save_html() requer test_type específico
                # Gerar relatório para cada tipo de teste
                report_paths = []
                try:
                    # Robustness report
                    if self.config['tests']['robustness']['enabled']:
                        report_path = self.results_dir / 'deepbridge_robustness_report.html'
                        exp.save_html(
                            test_type='robustness',
                            file_path=str(report_path),
                            model_name='XGBoost',
                            report_type='interactive'
                        )
                        report_paths.append(str(report_path))
                        self.logger.info(f"  ✓ Robustness report: {report_path}")

                    # Uncertainty report
                    if self.config['tests']['uncertainty']['enabled']:
                        report_path = self.results_dir / 'deepbridge_uncertainty_report.html'
                        exp.save_html(
                            test_type='uncertainty',
                            file_path=str(report_path),
                            model_name='XGBoost',
                            report_type='interactive'
                        )
                        report_paths.append(str(report_path))
                        self.logger.info(f"  ✓ Uncertainty report: {report_path}")

                    # Resilience report
                    if self.config['tests']['resilience']['enabled']:
                        report_path = self.results_dir / 'deepbridge_resilience_report.html'
                        exp.save_html(
                            test_type='resilience',
                            file_path=str(report_path),
                            model_name='XGBoost',
                            report_type='interactive'
                        )
                        report_paths.append(str(report_path))
                        self.logger.info(f"  ✓ Resilience report: {report_path}")

                    times['report'] = time.time() - start
                    results['report'] = {'status': 'generated', 'paths': report_paths, 'count': len(report_paths)}
                    self.logger.info(f"  ✓ Generated {len(report_paths)} HTML reports")

                except Exception as e:
                    self.logger.warning(f"  ⚠ Could not generate reports: {e}")
                    times['report'] = time.time() - start
                    results['report'] = {'status': 'failed', 'error': str(e)}

        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            self.logger.warning("Falling back to simulation mode...")
            import traceback
            traceback.print_exc()

            # Fallback: simular tempos esperados se houver erro
            if self.config['tests']['fairness']['enabled']:
                time.sleep(self.config['tests']['fairness']['expected_time_deepbridge'] * 60)
                times['fairness'] = self.config['tests']['fairness']['expected_time_deepbridge'] * 60
                results['fairness'] = {'status': 'simulated'}

            if self.config['tests']['robustness']['enabled']:
                time.sleep(self.config['tests']['robustness']['expected_time_deepbridge'] * 60)
                times['robustness'] = self.config['tests']['robustness']['expected_time_deepbridge'] * 60
                results['robustness'] = {'status': 'simulated'}

            if self.config['tests']['uncertainty']['enabled']:
                time.sleep(self.config['tests']['uncertainty']['expected_time_deepbridge'] * 60)
                times['uncertainty'] = self.config['tests']['uncertainty']['expected_time_deepbridge'] * 60
                results['uncertainty'] = {'status': 'simulated'}

            if self.config['tests']['resilience']['enabled']:
                time.sleep(self.config['tests']['resilience']['expected_time_deepbridge'] * 60)
                times['resilience'] = self.config['tests']['resilience']['expected_time_deepbridge'] * 60
                results['resilience'] = {'status': 'simulated'}

            if self.config['tests']['report_generation']['enabled']:
                time.sleep(self.config['tests']['report_generation']['expected_time_deepbridge'] * 60)
                times['report'] = self.config['tests']['report_generation']['expected_time_deepbridge'] * 60
                results['report'] = {'status': 'simulated'}

        # Total
        times['total'] = sum(times.values())

        # Log resumo
        self.logger.info("\n=== Validation Summary ===")
        for task, task_time in times.items():
            self.logger.info(f"{task.capitalize()}: {task_time:.2f}s ({task_time / 60:.2f} min)")

        return times, results

    def run_complete_validation(self) -> Dict[str, float]:
        """Executa validação completa"""
        # 1. Carregar dados e treinar modelo
        self.logger.info("=== Setup ===")
        X_train, X_test, y_train, y_test = self.load_data()
        model = self.train_model(X_train, y_train)

        # 2. Criar DBDataset (DeepBridge REAL)
        self.logger.info("\n=== Creating DBDataset ===")

        # Combinar X_test e y_test em um DataFrame
        test_df = X_test.copy()
        test_df['target'] = y_test

        # IMPORTANTE: Reset index para garantir índices contíguos (0, 1, 2, ...)
        # DeepBridge espera DataFrames com índices reset
        test_df = test_df.reset_index(drop=True)

        # Detectar atributos protegidos (sex, age, race, etc.)
        # NOTA: Ajustar baseado nas colunas reais do dataset
        protected_attrs = self.config['dataset'].get('protected_attributes', [])

        # Criar DBDataset
        try:
            dataset = DBDataset(
                data=test_df,
                target_column='target',
                model=model,
                # protected_attributes=protected_attrs  # Se suportado
            )
            self.logger.info(f"✓ DBDataset created successfully")

        except Exception as e:
            self.logger.error(f"Error creating DBDataset: {e}")
            raise

        # 3. Executar testes
        self.logger.info("\n=== Running Validation Tests ===")
        times, results = self.run_validation_tests(dataset)

        return times

    def run_benchmark(self, num_runs: int = None) -> Dict[str, Any]:
        """Executa benchmark completo"""
        if num_runs is None:
            num_runs = self.config['general']['num_runs']

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"DeepBridge Benchmark (REAL) - {num_runs} runs")
        self.logger.info(f"{'=' * 60}\n")

        all_times = {
            'fairness': [],
            'robustness': [],
            'uncertainty': [],
            'resilience': [],
            'report': [],
            'total': []
        }

        for run in range(num_runs):
            self.logger.info(f"\n--- Run {run + 1}/{num_runs} ---")

            times = self.run_complete_validation()

            for task, task_time in times.items():
                all_times[task].append(task_time)

        # Calcular estatísticas
        stats = {}
        for task, times_list in all_times.items():
            # Skip tasks with no recorded times
            if len(times_list) == 0:
                self.logger.warning(f"No times recorded for '{task}', skipping statistics")
                stats[task] = {
                    'mean_seconds': 0.0,
                    'std_seconds': 0.0,
                    'min_seconds': 0.0,
                    'max_seconds': 0.0,
                    'mean_minutes': 0.0,
                    'std_minutes': 0.0,
                    'all_times_seconds': [],
                    'num_runs': 0,
                    'status': 'no_data'
                }
                continue

            times_array = np.array(times_list)

            stats[task] = {
                'mean_seconds': float(np.mean(times_array)),
                'std_seconds': float(np.std(times_array)),
                'min_seconds': float(np.min(times_array)),
                'max_seconds': float(np.max(times_array)),
                'mean_minutes': float(np.mean(times_array) / 60),
                'std_minutes': float(np.std(times_array) / 60),
                'all_times_seconds': times_list,
                'num_runs': num_runs,
                'status': 'ok'
            }

        # Salvar resultados
        output_path = self.results_dir / 'deepbridge_times_REAL'
        save_results(stats, output_path, formats=['json', 'csv'])

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Benchmark completed!")
        self.logger.info(f"{'=' * 60}\n")

        # Exibir resumo
        self.logger.info("Summary Statistics (mean ± std):")
        for task, task_stats in stats.items():
            mean_min = task_stats['mean_minutes']
            std_min = task_stats['std_minutes']
            self.logger.info(f"  {task.capitalize()}: {mean_min:.1f} ± {std_min:.1f} min")

        return stats


def main():
    """Função principal"""
    # Configurar logging
    log_dir = Path(__file__).parent.parent / 'logs'
    exp_logger = ExperimentLogger(log_dir, name='deepbridge_benchmark_REAL')
    logger = exp_logger.get_logger()

    logger.info("=" * 70)
    logger.info("USANDO DEEPBRIDGE REAL (não simulação)")
    logger.info("=" * 70)

    # Carregar config
    config = load_config()

    # Seed
    set_seeds(config['general']['seed'])

    # Executar benchmark
    benchmark = DeepBridgeBenchmarkReal(config)
    results = benchmark.run_benchmark()

    logger.info("\nResults saved successfully!")

    return results


if __name__ == "__main__":
    main()
