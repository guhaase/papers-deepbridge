"""
Utilidades comuns para todos os experimentos.
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation"):
    """Context manager para medir tempo de execução."""
    start = time.time()
    print(f"⏱️  {name}...")
    yield
    elapsed = time.time() - start
    print(f"✅ {name} concluído em {elapsed:.2f}s ({elapsed/60:.1f}min)")


def save_json(data: Dict, filepath: Path, indent: int = 2):
    """Salva dados em JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"💾 Salvo: {filepath}")


def load_json(filepath: Path) -> Dict:
    """Carrega dados de JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, filepath: Path):
    """Salva DataFrame em CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"💾 Salvo: {filepath}")


def create_synthetic_binary_classification_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_sensitive: int = 2,
    bias_strength: float = 0.2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Cria dataset sintético para classificação binária com bias.

    Args:
        n_samples: Número de amostras
        n_features: Número de features totais
        n_sensitive: Número de atributos sensíveis
        bias_strength: Força do bias (0=sem bias, 1=bias máximo)
        seed: Random seed

    Returns:
        DataFrame com features e target
    """
    np.random.seed(seed)

    # Create base features
    X = np.random.randn(n_samples, n_features)

    # Create sensitive attributes
    sensitive_attrs = {}
    for i in range(n_sensitive):
        if i == 0:
            # Binary sensitive attribute (e.g., gender)
            sensitive_attrs[f'sensitive_{i}'] = np.random.choice(['A', 'B'], n_samples)
        else:
            # Categorical sensitive attribute (e.g., race)
            sensitive_attrs[f'sensitive_{i}'] = np.random.choice(
                ['Group1', 'Group2', 'Group3'], n_samples
            )

    # Create target with bias
    y_base = (X.sum(axis=1) > 0).astype(int)

    # Add bias based on first sensitive attribute
    y = y_base.copy()
    if bias_strength > 0 and n_sensitive > 0:
        mask = sensitive_attrs['sensitive_0'] == 'A'
        flip_prob = bias_strength * 0.3
        flip_mask = np.random.random(mask.sum()) < flip_prob
        y[mask] = (y[mask] + flip_mask.astype(int)) % 2

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])

    for name, values in sensitive_attrs.items():
        df[name] = values

    df['target'] = y

    return df


def calculate_metrics_summary(results: List[Dict]) -> Dict:
    """Calcula resumo de métricas a partir de lista de resultados."""

    if not results:
        return {}

    # Extract numeric values
    metrics = {}
    for key in results[0].keys():
        values = [r.get(key) for r in results if isinstance(r.get(key), (int, float))]
        if values:
            metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

    return metrics


def validate_claim(
    actual: float,
    target: float,
    tolerance: float = 0.1,
    direction: str = 'equal'
) -> bool:
    """
    Valida se um valor está dentro da tolerância esperada.

    Args:
        actual: Valor obtido
        target: Valor esperado
        tolerance: Tolerância (±10% por padrão)
        direction: 'equal', 'greater', 'less', 'greater_equal', 'less_equal'

    Returns:
        True se claim validada
    """
    if direction == 'equal':
        return abs(actual - target) / target <= tolerance
    elif direction == 'greater':
        return actual > target * (1 - tolerance)
    elif direction == 'less':
        return actual < target * (1 + tolerance)
    elif direction == 'greater_equal':
        return actual >= target * (1 - tolerance)
    elif direction == 'less_equal':
        return actual <= target * (1 + tolerance)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def format_claim_validation(
    claim_name: str,
    actual: float,
    target: float,
    unit: str = '',
    validated: bool = None,
    tolerance: float = 0.1
) -> str:
    """Formata string de validação de claim."""

    if validated is None:
        validated = validate_claim(actual, target, tolerance)

    symbol = '✅' if validated else '❌'
    status = 'PASS' if validated else 'FAIL'

    return f"   {claim_name}: {symbol} {status} (target: {target}{unit}, actual: {actual:.2f}{unit})"


class ExperimentLogger:
    """Logger para experimentos com suporte a seções."""

    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / f"{name}_log.txt"
        self.start_time = time.time()

        self._write_header()

    def _write_header(self):
        """Escreve header do log."""
        header = f"""
{'='*60}
EXPERIMENTO: {self.name}
Data: {pd.Timestamp.now()}
{'='*60}
"""
        self._write(header)

    def _write(self, message: str):
        """Escreve mensagem no log e stdout."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def section(self, title: str):
        """Inicia nova seção."""
        message = f"\n{'='*60}\n{title}\n{'='*60}"
        self._write(message)

    def info(self, message: str):
        """Log de informação."""
        self._write(f"ℹ️  {message}")

    def success(self, message: str):
        """Log de sucesso."""
        self._write(f"✅ {message}")

    def warning(self, message: str):
        """Log de warning."""
        self._write(f"⚠️  {message}")

    def error(self, message: str):
        """Log de erro."""
        self._write(f"❌ {message}")

    def metric(self, name: str, value: Any, unit: str = ''):
        """Log de métrica."""
        self._write(f"   {name}: {value}{unit}")

    def validation(self, claim: str, passed: bool, details: str = ''):
        """Log de validação de claim."""
        symbol = '✅' if passed else '❌'
        status = 'PASS' if passed else 'FAIL'
        message = f"   {claim}: {symbol} {status}"
        if details:
            message += f" ({details})"
        self._write(message)

    def finalize(self):
        """Finaliza log."""
        elapsed = time.time() - self.start_time
        footer = f"""
{'='*60}
EXPERIMENTO CONCLUÍDO
Tempo Total: {elapsed:.2f}s ({elapsed/60:.1f}min)
Log salvo em: {self.log_file}
{'='*60}
"""
        self._write(footer)


def generate_experiment_report(
    experiment_name: str,
    results: Dict,
    output_dir: Path,
    claims: Dict[str, Dict] = None
):
    """
    Gera relatório consolidado de um experimento.

    Args:
        experiment_name: Nome do experimento
        results: Dicionário com resultados
        output_dir: Diretório de saída
        claims: Dicionário com claims a validar
            {claim_name: {'target': value, 'actual': value, 'unit': str}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create markdown report
    report = f"# Relatório: {experiment_name}\n\n"
    report += f"**Data**: {pd.Timestamp.now()}\n\n"

    # Results summary
    report += "## Resultados\n\n"
    report += "```json\n"
    report += json.dumps(results, indent=2, default=str)
    report += "\n```\n\n"

    # Claims validation
    if claims:
        report += "## Validação de Claims\n\n"
        report += "| Claim | Target | Actual | Status |\n"
        report += "|-------|--------|--------|--------|\n"

        for claim_name, claim_data in claims.items():
            target = claim_data.get('target', 0)
            actual = claim_data.get('actual', 0)
            unit = claim_data.get('unit', '')
            validated = claim_data.get('validated', validate_claim(actual, target))

            status = '✅ PASS' if validated else '❌ FAIL'
            report += f"| {claim_name} | {target}{unit} | {actual:.2f}{unit} | {status} |\n"

        report += "\n"

    # Save report
    report_file = output_dir / f"{experiment_name}_report.md"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"📄 Relatório gerado: {report_file}")

    return report_file


def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""

    required = {
        'deepbridge': 'DeepBridge Fairness',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
    }

    missing = []

    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NÃO INSTALADO")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Instale os pacotes faltantes:")
        print(f"   pip install {' '.join(missing)}")
        return False

    print("\n✅ Todas as dependências estão instaladas!")
    return True


if __name__ == '__main__':
    # Test utilities
    print("🧪 Testando utilidades...\n")

    # Test timer
    with timer("Operação de teste"):
        time.sleep(0.1)

    # Test synthetic dataset creation
    print("\n📊 Criando dataset sintético...")
    df = create_synthetic_binary_classification_dataset(n_samples=100)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # Test validation
    print("\n✅ Testando validação de claims...")
    print(format_claim_validation("Teste 1", 0.90, 0.85, tolerance=0.1))
    print(format_claim_validation("Teste 2", 0.70, 0.85, tolerance=0.1))

    # Test dependencies
    print("\n🔍 Verificando dependências...")
    check_dependencies()

    print("\n✅ Testes concluídos!")
