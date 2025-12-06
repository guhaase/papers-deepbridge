#!/usr/bin/env python3
"""
Teste mínimo para verificar API do DeepBridge

Este script testa se o DeepBridge está instalado corretamente e
lista os métodos disponíveis no Experiment.

Uso:
    python3 test_deepbridge_api.py
"""

import sys
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge import DBDataset, Experiment
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

def print_section(title):
    """Helper para imprimir seções"""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}\n")

def main():
    print_section("TESTE DA API DEEPBRIDGE")

    # 1. Gerar dados sintéticos
    print("1. Gerando dataset sintético...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    print(f"   ✓ Dataset criado: {df.shape}")
    print(f"   ✓ Classes: {np.unique(y, return_counts=True)}")

    # 2. Treinar modelo
    print("\n2. Treinando modelo XGBoost...")
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(df.drop('target', axis=1), df['target'])
    accuracy = model.score(df.drop('target', axis=1), df['target'])
    print(f"   ✓ Modelo treinado")
    print(f"   ✓ Acurácia: {accuracy:.4f}")

    # 3. Criar DBDataset
    print("\n3. Criando DBDataset...")
    try:
        dataset = DBDataset(
            data=df,
            target_column='target',
            model=model
        )
        print(f"   ✓ DBDataset criado com sucesso")
        print(f"   ✓ Tipo: {type(dataset).__name__}")

        # Listar atributos públicos
        public_attrs = [a for a in dir(dataset) if not a.startswith('_')]
        print(f"   ✓ Atributos disponíveis ({len(public_attrs)}):")
        for attr in public_attrs[:15]:  # Primeiros 15
            print(f"      - {attr}")
        if len(public_attrs) > 15:
            print(f"      ... e mais {len(public_attrs) - 15}")

    except Exception as e:
        print(f"   ✗ Erro ao criar DBDataset:")
        print(f"      {type(e).__name__}: {e}")
        return False

    # 4. Criar Experiment
    print("\n4. Criando Experiment...")
    try:
        exp = Experiment(
            dataset=dataset,
            experiment_type='binary_classification'
        )
        print(f"   ✓ Experiment criado com sucesso")
        print(f"   ✓ Tipo: {type(exp).__name__}")

    except Exception as e:
        print(f"   ✗ Erro ao criar Experiment:")
        print(f"      {type(e).__name__}: {e}")
        return False

    # 5. Listar TODOS os métodos
    print_section("MÉTODOS DISPONÍVEIS NO EXPERIMENT")

    all_methods = [m for m in dir(exp) if not m.startswith('_') and callable(getattr(exp, m))]

    print(f"Total de métodos públicos: {len(all_methods)}\n")

    # Categorizar métodos
    categories = {
        'Testes (test/run/validate)': [],
        'Relatórios (report/save/generate)': [],
        'Análise (analyze/evaluate)': [],
        'Outros': []
    }

    for method in all_methods:
        method_lower = method.lower()
        if any(kw in method_lower for kw in ['test', 'run', 'validate', 'check']):
            categories['Testes (test/run/validate)'].append(method)
        elif any(kw in method_lower for kw in ['report', 'save', 'generate', 'export']):
            categories['Relatórios (report/save/generate)'].append(method)
        elif any(kw in method_lower for kw in ['analyze', 'evaluate', 'calculate', 'compute']):
            categories['Análise (analyze/evaluate)'].append(method)
        else:
            categories['Outros'].append(method)

    for category, methods in categories.items():
        if methods:
            print(f"{category} ({len(methods)}):")
            for method in sorted(methods):
                print(f"  - {method}")
            print()

    # 6. Métodos mais importantes para benchmarks
    print_section("MÉTODOS RELEVANTES PARA BENCHMARKS")

    important_patterns = [
        'fairness',
        'robust',
        'uncertain',
        'resilien',
        'drift',
        'report',
        'pdf',
        'html',
        'save',
        'export'
    ]

    relevant_methods = []
    for method in all_methods:
        if any(pattern in method.lower() for pattern in important_patterns):
            relevant_methods.append(method)

    if relevant_methods:
        print("Métodos que podem ser úteis para benchmarks:\n")
        for method in sorted(relevant_methods):
            # Tentar pegar docstring
            try:
                doc = getattr(exp, method).__doc__
                first_line = doc.split('\n')[0].strip() if doc else "Sem documentação"
                print(f"  • {method}()")
                print(f"    {first_line}\n")
            except:
                print(f"  • {method}()\n")
    else:
        print("Nenhum método específico encontrado.")
        print("Use a lista completa acima.\n")

    # 7. Teste de execução simples
    print_section("TESTE DE EXECUÇÃO")

    print("Tentando executar método de teste (se disponível)...\n")

    # Tentar métodos comuns
    test_methods_to_try = [
        'run_tests',
        'run_all_tests',
        'validate',
        'run_validation',
        'run_fairness',
        'run_fairness_tests'
    ]

    executed = False
    for method_name in test_methods_to_try:
        if hasattr(exp, method_name):
            print(f"Tentando executar: {method_name}()...")
            try:
                method = getattr(exp, method_name)
                # Não executar por enquanto, apenas mostrar que existe
                print(f"   ✓ Método {method_name}() existe e pode ser chamado")
                print(f"   ℹ Para executar: results = exp.{method_name}()")
                executed = True
                break
            except Exception as e:
                print(f"   ⚠ Erro ao tentar executar {method_name}(): {e}")

    if not executed:
        print("   ℹ Nenhum método de teste padrão encontrado.")
        print("   ℹ Consulte a documentação do DeepBridge para uso correto.")

    # Resumo final
    print_section("RESUMO")

    print("✓ DeepBridge está instalado e funcional")
    print(f"✓ DBDataset pode ser criado")
    print(f"✓ Experiment pode ser criado")
    print(f"✓ {len(all_methods)} métodos públicos disponíveis")
    print(f"\nPara usar nos benchmarks, consulte os métodos listados acima")
    print(f"e atualize benchmark_deepbridge_REAL.py com as chamadas corretas.")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
