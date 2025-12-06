#!/usr/bin/env python3
"""
Teste rápido do benchmark_deepbridge_REAL.py

Executa apenas 1 run para verificar se a API está funcionando corretamente.

Uso:
    python3 test_benchmark_real.py
"""

import sys
import logging
from pathlib import Path

# Configurar path do DeepBridge
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from utils import load_config, ExperimentLogger, set_seeds
from benchmark_deepbridge_REAL import DeepBridgeBenchmarkReal


def main():
    """Teste rápido com 1 run"""

    print("=" * 70)
    print("TESTE RÁPIDO: DeepBridge Benchmark REAL (1 run)")
    print("=" * 70)
    print()

    # Configurar logging
    log_dir = Path(__file__).parent.parent / 'logs'
    exp_logger = ExperimentLogger(log_dir, name='test_benchmark_real')
    logger = exp_logger.get_logger()

    # Carregar config
    config = load_config()

    # Seed
    set_seeds(config['general']['seed'])

    # Criar benchmark
    benchmark = DeepBridgeBenchmarkReal(config)

    logger.info("Starting single run test...")

    try:
        # Executar apenas 1 run
        results = benchmark.run_benchmark(num_runs=1)

        logger.info("\n" + "=" * 70)
        logger.info("✓ TESTE CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 70)

        # Exibir resultados
        print("\nResultados do teste:")
        print("-" * 70)
        for task, stats in results.items():
            mean_min = stats['mean_minutes']
            print(f"  {task.capitalize()}: {mean_min:.2f} min")

        print("\n✓ O script está funcionando corretamente com a API real do DeepBridge!")
        print("✓ Você pode executar o experimento completo com 10 runs.")
        print()

        return True

    except Exception as e:
        logger.error(f"\n✗ TESTE FALHOU: {e}")
        import traceback
        traceback.print_exc()

        print("\n✗ Erro ao executar teste.")
        print("Verifique os logs para mais detalhes.")
        print()

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
