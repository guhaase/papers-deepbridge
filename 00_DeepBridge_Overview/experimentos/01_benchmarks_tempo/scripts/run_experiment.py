"""
Script Principal: Orquestrador do Experimento 01

Executa todo o pipeline do experimento de benchmarks de tempo.

Uso:
    python run_experiment.py --all          # Executa tudo
    python run_experiment.py --deepbridge   # Apenas DeepBridge
    python run_experiment.py --fragmented   # Apenas Fragmentado
    python run_experiment.py --analyze      # Apenas análise
    python run_experiment.py --figures      # Apenas figuras

Autor: DeepBridge Team
Data: 2025-12-05
"""

import argparse
import logging
import sys
from pathlib import Path

# Adicionar scripts ao path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config,
    ExperimentLogger,
    get_system_info
)
from benchmark_deepbridge import DeepBridgeBenchmark
from benchmark_fragmented import FragmentedWorkflowBenchmark
from compare_and_analyze import BenchmarkAnalysis
from generate_figures import FigureGenerator


class ExperimentOrchestrator:
    """Orquestrador do experimento completo"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_all(self):
        """Executa pipeline completo"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENTO 01: BENCHMARKS DE TEMPO")
        self.logger.info("=" * 70 + "\n")

        # Informações do sistema
        self.logger.info("System Information:")
        sys_info = get_system_info()
        for key, value in sys_info.items():
            self.logger.info(f"  {key}: {value}")

        self.logger.info("\n" + "-" * 70)
        self.logger.info("PIPELINE:")
        self.logger.info("  1. DeepBridge Benchmark")
        self.logger.info("  2. Fragmented Workflow Benchmark")
        self.logger.info("  3. Statistical Analysis")
        self.logger.info("  4. Figure Generation")
        self.logger.info("-" * 70 + "\n")

        # Confirmar execução
        if not self.config['general'].get('auto_confirm', False):
            response = input("Proceed with full experiment? [y/N]: ")
            if response.lower() != 'y':
                self.logger.info("Experiment cancelled by user")
                return

        # 1. DeepBridge Benchmark
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 1/4: DEEPBRIDGE BENCHMARK")
        self.logger.info("=" * 70 + "\n")

        try:
            db_benchmark = DeepBridgeBenchmark(self.config)
            db_results = db_benchmark.run_benchmark()
            self.logger.info("✓ DeepBridge benchmark completed")
        except Exception as e:
            self.logger.error(f"✗ DeepBridge benchmark failed: {e}")
            raise

        # 2. Fragmented Workflow Benchmark
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 2/4: FRAGMENTED WORKFLOW BENCHMARK")
        self.logger.info("=" * 70 + "\n")

        try:
            frag_benchmark = FragmentedWorkflowBenchmark(self.config)
            frag_results = frag_benchmark.run_benchmark()
            self.logger.info("✓ Fragmented benchmark completed")
        except Exception as e:
            self.logger.error(f"✗ Fragmented benchmark failed: {e}")
            raise

        # 3. Statistical Analysis
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 3/4: STATISTICAL ANALYSIS")
        self.logger.info("=" * 70 + "\n")

        try:
            analysis = BenchmarkAnalysis(self.config)
            analysis_results = analysis.run_analysis()
            self.logger.info("✓ Analysis completed")
        except Exception as e:
            self.logger.error(f"✗ Analysis failed: {e}")
            raise

        # 4. Figure Generation
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 4/4: FIGURE GENERATION")
        self.logger.info("=" * 70 + "\n")

        try:
            fig_generator = FigureGenerator(self.config)
            fig_generator.generate_all_figures()
            self.logger.info("✓ Figures generated")
        except Exception as e:
            self.logger.error(f"✗ Figure generation failed: {e}")
            raise

        # Final summary
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 70 + "\n")

        self.logger.info("Output locations:")
        base_dir = Path(__file__).parent.parent
        self.logger.info(f"  Results: {base_dir / 'results'}")
        self.logger.info(f"  Figures: {base_dir / 'figures'}")
        self.logger.info(f"  Tables: {base_dir / 'tables'}")
        self.logger.info(f"  Logs: {base_dir / 'logs'}")

        return {
            'deepbridge': db_results,
            'fragmented': frag_results,
            'analysis': analysis_results
        }

    def run_deepbridge_only(self):
        """Executa apenas benchmark DeepBridge"""
        self.logger.info("\nRunning DeepBridge benchmark only...")

        db_benchmark = DeepBridgeBenchmark(self.config)
        return db_benchmark.run_benchmark()

    def run_fragmented_only(self):
        """Executa apenas benchmark Fragmentado"""
        self.logger.info("\nRunning Fragmented workflow benchmark only...")

        frag_benchmark = FragmentedWorkflowBenchmark(self.config)
        return frag_benchmark.run_benchmark()

    def run_analysis_only(self):
        """Executa apenas análise"""
        self.logger.info("\nRunning analysis only...")

        analysis = BenchmarkAnalysis(self.config)
        return analysis.run_analysis()

    def run_figures_only(self):
        """Gera apenas figuras"""
        self.logger.info("\nGenerating figures only...")

        fig_generator = FigureGenerator(self.config)
        fig_generator.generate_all_figures()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Experimento 01: Benchmarks de Tempo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --all          # Run complete experiment
  python run_experiment.py --deepbridge   # Run DeepBridge benchmark only
  python run_experiment.py --fragmented   # Run Fragmented benchmark only
  python run_experiment.py --analyze      # Run analysis only
  python run_experiment.py --figures      # Generate figures only
  python run_experiment.py --quick        # Quick test (1 run)
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete experiment pipeline'
    )

    parser.add_argument(
        '--deepbridge',
        action='store_true',
        help='Run DeepBridge benchmark only'
    )

    parser.add_argument(
        '--fragmented',
        action='store_true',
        help='Run Fragmented workflow benchmark only'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run statistical analysis only'
    )

    parser.add_argument(
        '--figures',
        action='store_true',
        help='Generate figures only'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (1 run instead of 10)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Se nenhuma opção, executar --all
    if not any([args.all, args.deepbridge, args.fragmented, args.analyze, args.figures]):
        args.all = True

    return args


def main():
    """Função principal"""
    args = parse_args()

    # Configurar logging
    log_dir = Path(__file__).parent.parent / 'logs'
    exp_logger = ExperimentLogger(log_dir, name='experiment_orchestrator')
    logger = exp_logger.get_logger()

    # Carregar config
    config = load_config(args.config)

    # Modo rápido
    if args.quick:
        logger.info("QUICK MODE: Setting num_runs=1")
        config['general']['num_runs'] = 1
        config['general']['auto_confirm'] = True

    # Criar orchestrator
    orchestrator = ExperimentOrchestrator(config)

    try:
        # Executar ação solicitada
        if args.all:
            results = orchestrator.run_all()

        elif args.deepbridge:
            results = orchestrator.run_deepbridge_only()

        elif args.fragmented:
            results = orchestrator.run_fragmented_only()

        elif args.analyze:
            results = orchestrator.run_analysis_only()

        elif args.figures:
            orchestrator.run_figures_only()
            results = None

        logger.info("\n✓ All tasks completed successfully!")

        return results

    except KeyboardInterrupt:
        logger.info("\n\nExperiment interrupted by user (Ctrl+C)")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
