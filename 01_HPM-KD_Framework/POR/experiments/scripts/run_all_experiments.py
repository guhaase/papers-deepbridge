#!/usr/bin/env python3
"""
Script Mestre: Executar Todos os Experimentos HPM-KD

Executa sequencialmente os 4 experimentos principais:
    1. Compression Efficiency (RQ1)
    2. Ablation Studies (RQ2)
    3. Generalization (RQ3)
    4. Computational Efficiency (RQ4)

Features:
    - Execução automática em sequência
    - Captura de logs e saídas
    - Relatório consolidado final
    - Tratamento de erros
    - Estimativa de tempo total
    - Compatível com Google Colab

Tempo estimado total:
    - Quick Mode: ~3-4 horas
    - Full Mode: ~8-10 horas

Requerimentos:
    - DeepBridge library instalada (pip install deepbridge)
    - PyTorch + torchvision
    - GPU recomendada

Uso no Google Colab:
    !python run_all_experiments.py --mode quick --dataset MNIST
    !python run_all_experiments.py --mode full --dataset CIFAR10 --gpu 0

Uso local:
    python run_all_experiments.py --mode quick --dataset MNIST
    python run_all_experiments.py --mode full --datasets MNIST CIFAR10 --gpu 0
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENTS = [
    {
        'id': 1,
        'name': 'Compression Efficiency',
        'script': '01_compression_efficiency.py',
        'rq': 'RQ1',
        'description': 'Compara HPM-KD vs 5 baselines em taxas de compressão',
        'time_quick': 45,  # minutos
        'time_full': 240,  # minutos
        'supports_multiple_datasets': True,
    },
    {
        'id': 2,
        'name': 'Ablation Studies',
        'script': '02_ablation_studies.py',
        'rq': 'RQ2',
        'description': 'Analisa contribuição individual de cada componente HPM-KD',
        'time_quick': 60,
        'time_full': 120,
        'supports_multiple_datasets': False,
    },
    {
        'id': 3,
        'name': 'Generalization',
        'script': '03_generalization.py',
        'rq': 'RQ3',
        'description': 'Testa robustez em class imbalance, label noise e t-SNE',
        'time_quick': 90,
        'time_full': 180,
        'supports_multiple_datasets': False,
    },
    {
        'id': 4,
        'name': 'Computational Efficiency',
        'script': '04_computational_efficiency.py',
        'rq': 'RQ4',
        'description': 'Mede overhead computacional e tempo de treinamento',
        'time_quick': 30,
        'time_full': 60,
        'supports_multiple_datasets': False,
    },
]

# ============================================================================
# Setup Logging
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console"""
    log_file = output_dir / 'run_all_experiments.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# Helper Functions
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def estimate_total_time(mode: str) -> int:
    """Estimate total execution time in minutes"""
    if mode == 'quick':
        return sum(exp['time_quick'] for exp in EXPERIMENTS)
    else:
        return sum(exp['time_full'] for exp in EXPERIMENTS)


def print_banner(text: str, char: str = "="):
    """Print a banner"""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def check_deepbridge():
    """Check if DeepBridge is installed"""
    try:
        import deepbridge
        return True
    except ImportError:
        return False


# ============================================================================
# Execution Functions
# ============================================================================

def run_experiment(exp: Dict, args: argparse.Namespace,
                   output_dir: Path, logger: logging.Logger) -> Dict:
    """Run a single experiment"""

    print_banner(f"Experimento {exp['id']}: {exp['name']} ({exp['rq']})", "=")

    logger.info(f"Iniciando: {exp['name']}")
    logger.info(f"Descrição: {exp['description']}")
    logger.info(f"Script: {exp['script']}")

    # Build command
    script_path = Path(__file__).parent / exp['script']
    cmd = [sys.executable, str(script_path)]

    # Add mode
    cmd.extend(['--mode', args.mode])

    # Add dataset(s)
    if exp['supports_multiple_datasets']:
        cmd.extend(['--datasets'] + args.datasets)
    else:
        cmd.extend(['--dataset', args.datasets[0]])

    # Add GPU
    if args.gpu is not None:
        cmd.extend(['--gpu', str(args.gpu)])

    # Add output dir
    exp_output = output_dir / f"exp_{exp['id']:02d}_{exp['script'].replace('.py', '')}"
    exp_output.mkdir(parents=True, exist_ok=True)
    cmd.extend(['--output', str(exp_output)])

    logger.info(f"Comando: {' '.join(cmd)}")

    # Estimate time
    time_estimate = exp[f'time_{args.mode}']
    logger.info(f"Tempo estimado: {time_estimate} minutos")

    # Run experiment
    start_time = time.time()

    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output
        log_file = exp_output / f"{exp['script'].replace('.py', '')}.log"
        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(line, end='')
                f.write(line)

        process.wait()

        elapsed_time = time.time() - start_time

        if process.returncode == 0:
            logger.info(f"✅ Experimento {exp['id']} concluído com sucesso!")
            logger.info(f"Tempo real: {format_time(elapsed_time)}")
            status = 'success'
            error = None
        else:
            logger.error(f"❌ Experimento {exp['id']} falhou com código {process.returncode}")
            status = 'failed'
            error = f"Exit code: {process.returncode}"

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Erro ao executar experimento {exp['id']}: {e}")
        status = 'error'
        error = str(e)

    result = {
        'experiment_id': exp['id'],
        'name': exp['name'],
        'script': exp['script'],
        'status': status,
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_formatted': format_time(elapsed_time),
        'estimated_time_minutes': time_estimate,
        'output_dir': str(exp_output),
        'error': error,
    }

    return result


def generate_final_report(results: List[Dict], output_dir: Path,
                          args: argparse.Namespace, total_time: float):
    """Generate consolidated final report"""

    report_path = output_dir / 'RELATORIO_FINAL.md'

    report = f"""# Relatório Final - Experimentos HPM-KD

**Data de execução:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Modo:** {args.mode.upper()}
**Datasets:** {', '.join(args.datasets)}
**GPU:** {args.gpu if args.gpu is not None else 'CPU'}
**Tempo total:** {format_time(total_time)}

---

## Resumo de Execução

"""

    # Summary table
    report += "| # | Experimento | Status | Tempo Real | Tempo Estimado |\n"
    report += "|---|-------------|--------|------------|----------------|\n"

    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        report += f"| {result['experiment_id']} | {result['name']} | {status_icon} {result['status']} | "
        report += f"{result['elapsed_time_formatted']} | {result['estimated_time_minutes']}min |\n"

    # Statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    report += f"\n**Total:** {len(results)} experimentos  \n"
    report += f"**Sucesso:** {successful} ✅  \n"
    report += f"**Falhas:** {failed} ❌  \n"

    report += "\n---\n\n## Detalhes dos Experimentos\n\n"

    for result in results:
        report += f"### Experimento {result['experiment_id']}: {result['name']}\n\n"
        report += f"- **Script:** `{result['script']}`\n"
        report += f"- **Status:** {result['status']}\n"
        report += f"- **Tempo:** {result['elapsed_time_formatted']}\n"
        report += f"- **Output:** `{result['output_dir']}`\n"

        if result['error']:
            report += f"- **Erro:** {result['error']}\n"

        report += "\n"

    report += "---\n\n## Estrutura de Saída\n\n"
    report += "```\n"
    report += f"{output_dir.name}/\n"
    for result in results:
        exp_dir = Path(result['output_dir']).name
        report += f"├── {exp_dir}/\n"
        report += f"│   ├── results/\n"
        report += f"│   ├── figures/\n"
        report += f"│   ├── models/\n"
        report += f"│   └── report.md\n"
    report += f"├── run_all_experiments.log\n"
    report += f"└── RELATORIO_FINAL.md\n"
    report += "```\n\n"

    report += "---\n\n## Próximos Passos\n\n"

    if successful == len(results):
        report += "✅ **Todos os experimentos foram concluídos com sucesso!**\n\n"
        report += "Você pode agora:\n\n"
        report += "1. Analisar os relatórios individuais em cada pasta `exp_XX/report.md`\n"
        report += "2. Visualizar as figuras geradas em `exp_XX/figures/`\n"
        report += "3. Usar os modelos salvos em `exp_XX/models/`\n"
        report += "4. Compilar os resultados para o paper\n"
    else:
        report += f"⚠️ **{failed} experimento(s) falharam.**\n\n"
        report += "Verifique os logs individuais para detalhes dos erros.\n"

    report += f"\n---\n\n*Gerado automaticamente por `run_all_experiments.py`*\n"

    with open(report_path, 'w') as f:
        f.write(report)

    return report_path


def save_results_json(results: List[Dict], output_dir: Path):
    """Save results as JSON"""
    json_path = output_dir / 'results.json'

    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'experiments': results
        }, f, indent=2)

    return json_path


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Executar todos os experimentos HPM-KD em sequência',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--mode', type=str, choices=['quick', 'full'], default='quick',
                        help='Modo de execução: quick (rápido) ou full (completo)')

    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['MNIST'],
                        help='Datasets para usar (MNIST, FashionMNIST, CIFAR10, CIFAR100)')

    parser.add_argument('--dataset', type=str,
                        help='Dataset único (alias para --datasets)')

    parser.add_argument('--gpu', type=int, default=None,
                        help='ID da GPU (None para CPU)')

    parser.add_argument('--output', type=str, default=None,
                        help='Diretório de saída (padrão: results_YYYYMMDD_HHMMSS)')

    parser.add_argument('--skip', type=int, nargs='+', default=[],
                        help='Pular experimentos específicos (ex: --skip 1 3)')

    parser.add_argument('--only', type=int, nargs='+', default=[],
                        help='Executar apenas experimentos específicos (ex: --only 1 2)')

    args = parser.parse_args()

    # Handle dataset alias
    if args.dataset:
        args.datasets = [args.dataset]

    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{args.mode}_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)

    # Print header
    print_banner("EXECUÇÃO DE TODOS OS EXPERIMENTOS HPM-KD", "=")

    logger.info(f"Modo: {args.mode.upper()}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"GPU: {args.gpu if args.gpu is not None else 'CPU'}")
    logger.info(f"Output: {output_dir}")

    # Check DeepBridge
    if not check_deepbridge():
        logger.error("❌ DeepBridge não está instalado!")
        logger.error("Execute: pip install deepbridge")
        sys.exit(1)
    else:
        logger.info("✅ DeepBridge instalado")

    # Estimate total time
    total_estimate = estimate_total_time(args.mode)
    logger.info(f"\n⏱️  Tempo estimado total: {total_estimate} minutos (~{total_estimate/60:.1f} horas)")

    estimated_finish = datetime.now() + timedelta(minutes=total_estimate)
    logger.info(f"📅 Previsão de término: {estimated_finish.strftime('%H:%M:%S')}")

    # Filter experiments
    experiments_to_run = EXPERIMENTS.copy()

    if args.only:
        experiments_to_run = [exp for exp in experiments_to_run if exp['id'] in args.only]
        logger.info(f"Executando apenas experimentos: {args.only}")

    if args.skip:
        experiments_to_run = [exp for exp in experiments_to_run if exp['id'] not in args.skip]
        logger.info(f"Pulando experimentos: {args.skip}")

    logger.info(f"\nTotal de experimentos a executar: {len(experiments_to_run)}\n")

    # Run experiments
    results = []
    overall_start = time.time()

    for i, exp in enumerate(experiments_to_run, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"PROGRESSO: {i}/{len(experiments_to_run)}")
        logger.info(f"{'='*80}\n")

        result = run_experiment(exp, args, output_dir, logger)
        results.append(result)

        # Break on critical failure (optional)
        # if result['status'] != 'success':
        #     logger.warning("Experimento falhou. Continuando...")

    overall_time = time.time() - overall_start

    # Generate reports
    print_banner("GERANDO RELATÓRIOS FINAIS", "=")

    logger.info("Salvando resultados JSON...")
    json_path = save_results_json(results, output_dir)
    logger.info(f"✅ Salvo em: {json_path}")

    logger.info("Gerando relatório consolidado...")
    report_path = generate_final_report(results, output_dir, args, overall_time)
    logger.info(f"✅ Relatório salvo em: {report_path}")

    # Final summary
    print_banner("EXECUÇÃO CONCLUÍDA", "=")

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    logger.info(f"✅ Experimentos concluídos: {successful}/{len(results)}")
    logger.info(f"❌ Experimentos falhados: {failed}/{len(results)}")
    logger.info(f"⏱️  Tempo total: {format_time(overall_time)}")
    logger.info(f"📊 Relatório: {report_path}")
    logger.info(f"📁 Outputs: {output_dir}")

    if successful == len(results):
        logger.info("\n🎉 Todos os experimentos concluídos com sucesso!")
        return 0
    else:
        logger.warning(f"\n⚠️  {failed} experimento(s) falharam. Verifique os logs.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
