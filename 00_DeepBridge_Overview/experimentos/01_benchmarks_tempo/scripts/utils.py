"""
Utilidades comuns para o experimento 01: Benchmarks de Tempo

Autor: DeepBridge Team
Data: 2025-12-05
"""

import logging
import time
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Callable
from functools import wraps
from datetime import datetime


class ExperimentLogger:
    """Logger configurado para o experimento"""

    def __init__(self, log_dir: Path, name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(name)

    def get_logger(self):
        return self.logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carrega arquivo de configuração YAML

    Args:
        config_path: Caminho para arquivo de configuração

    Returns:
        Dicionário com configurações
    """
    config_file = Path(__file__).parent.parent / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def timer(func: Callable) -> Callable:
    """
    Decorator para medir tempo de execução de uma função

    Args:
        func: Função a ser medida

    Returns:
        Função decorada que retorna (resultado, tempo_segundos)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    return wrapper


def measure_time(func: Callable, *args, **kwargs) -> tuple:
    """
    Mede tempo de execução de uma função

    Args:
        func: Função a executar
        *args: Argumentos posicionais
        **kwargs: Argumentos nomeados

    Returns:
        (resultado, tempo_segundos)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time

    return result, elapsed_time


def run_multiple_times(
    func: Callable,
    num_runs: int = 10,
    verbose: bool = True,
    **func_kwargs
) -> Dict[str, Any]:
    """
    Executa função múltiplas vezes e coleta estatísticas de tempo

    Args:
        func: Função a executar
        num_runs: Número de execuções
        verbose: Se True, exibe progresso
        **func_kwargs: Argumentos para a função

    Returns:
        Dict com estatísticas (mean, std, min, max, all_times)
    """
    times = []
    results = []

    logger = logging.getLogger(__name__)

    for run in range(num_runs):
        if verbose:
            logger.info(f"Run {run + 1}/{num_runs}...")

        result, elapsed = measure_time(func, **func_kwargs)

        times.append(elapsed)
        results.append(result)

        if verbose:
            logger.info(f"  Time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")

    times_array = np.array(times)

    stats = {
        'mean_seconds': float(np.mean(times_array)),
        'std_seconds': float(np.std(times_array)),
        'min_seconds': float(np.min(times_array)),
        'max_seconds': float(np.max(times_array)),
        'mean_minutes': float(np.mean(times_array) / 60),
        'std_minutes': float(np.std(times_array) / 60),
        'all_times_seconds': times,
        'num_runs': num_runs
    }

    return stats, results


def save_results(
    results: Dict[str, Any],
    output_path: Path,
    formats: List[str] = ['json', 'csv']
):
    """
    Salva resultados em múltiplos formatos

    Args:
        results: Dicionário com resultados
        output_path: Caminho base (sem extensão)
        formats: Lista de formatos ('json', 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    if 'json' in formats:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")

    if 'csv' in formats:
        csv_path = output_path.with_suffix('.csv')

        # Converter para DataFrame
        if isinstance(results, dict):
            # Se é dict de dicts (ex: resultados por tarefa)
            if all(isinstance(v, dict) for v in results.values()):
                df = pd.DataFrame(results).T
            else:
                df = pd.DataFrame([results])
        elif isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            raise ValueError("Results must be dict or list")

        df.to_csv(csv_path, index=True)
        logger.info(f"Saved CSV: {csv_path}")


def create_results_summary(
    deepbridge_times: Dict[str, Any],
    fragmented_times: Dict[str, Any]
) -> pd.DataFrame:
    """
    Cria DataFrame resumo comparando DeepBridge vs. Fragmentado

    Args:
        deepbridge_times: Tempos do DeepBridge
        fragmented_times: Tempos do workflow fragmentado

    Returns:
        DataFrame com comparação
    """
    summary = []

    for task in deepbridge_times.keys():
        if task == 'total':
            continue

        db_mean = deepbridge_times[task]['mean_minutes']
        frag_mean = fragmented_times[task]['mean_minutes']

        speedup = frag_mean / db_mean
        reduction_pct = ((frag_mean - db_mean) / frag_mean) * 100

        summary.append({
            'Task': task.capitalize(),
            'DeepBridge (min)': f"{db_mean:.1f}",
            'Fragmented (min)': f"{frag_mean:.1f}",
            'Speedup': f"{speedup:.1f}x",
            'Reduction (%)': f"{reduction_pct:.1f}%"
        })

    # Total
    db_total = deepbridge_times['total']['mean_minutes']
    frag_total = fragmented_times['total']['mean_minutes']
    speedup_total = frag_total / db_total
    reduction_total = ((frag_total - db_total) / frag_total) * 100

    summary.append({
        'Task': 'TOTAL',
        'DeepBridge (min)': f"{db_total:.1f}",
        'Fragmented (min)': f"{frag_total:.1f}",
        'Speedup': f"{speedup_total:.1f}x",
        'Reduction (%)': f"{reduction_total:.1f}%"
    })

    return pd.DataFrame(summary)


def set_seeds(seed: int = 42):
    """
    Define seeds para reprodutibilidade

    Args:
        seed: Seed para random, numpy, etc.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class ProgressTracker:
    """Rastreador de progresso para experimentos longos"""

    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)

    def update(self, step: int = 1, message: str = ""):
        """Atualiza progresso"""
        self.current_step += step
        elapsed = time.time() - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100

        eta_seconds = (elapsed / self.current_step) * (self.total_steps - self.current_step)
        eta_minutes = eta_seconds / 60

        self.logger.info(
            f"{self.description}: {self.current_step}/{self.total_steps} "
            f"({progress_pct:.1f}%) - ETA: {eta_minutes:.1f} min"
        )

        if message:
            self.logger.info(f"  {message}")

    def finish(self):
        """Finaliza rastreamento"""
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"{self.description}: Completed in {elapsed / 60:.2f} minutes"
        )


def format_time(seconds: float) -> str:
    """
    Formata tempo em segundos para string legível

    Args:
        seconds: Tempo em segundos

    Returns:
        String formatada (ex: "2m 30s")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    if minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_system_info() -> Dict[str, Any]:
    """
    Coleta informações do sistema

    Returns:
        Dict com informações (CPU, RAM, OS, etc.)
    """
    import platform
    import psutil

    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'python_version': platform.python_version()
    }

    return info


if __name__ == "__main__":
    # Teste
    config = load_config()
    print("Config loaded successfully:")
    print(json.dumps(config, indent=2))

    # Teste de timer
    @timer
    def slow_function():
        time.sleep(1)
        return "done"

    result, elapsed = slow_function()
    print(f"\nTimer test: {result} in {elapsed:.2f}s")

    # Teste de sistema
    print("\nSystem info:")
    print(json.dumps(get_system_info(), indent=2))
