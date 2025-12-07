#!/usr/bin/env python3
"""
Calcula concordância entre anotadores (Cohen's Kappa).

Uso:
    python calculate_inter_rater_agreement.py \
        --reviewer1 ../data/annotations_reviewer1.csv \
        --reviewer2 ../data/annotations_reviewer2.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def parse_sensitive_attributes(attr_str: str) -> set:
    """Parse string de atributos separados por vírgula."""
    if pd.isna(attr_str) or attr_str == '':
        return set()
    return set(attr.strip() for attr in str(attr_str).split(','))


def calculate_agreement(df1: pd.DataFrame, df2: pd.DataFrame):
    """Calcula concordância entre dois conjuntos de anotações."""

    print("\n" + "="*60)
    print("📊 ANÁLISE DE CONCORDÂNCIA ENTRE ANOTADORES")
    print("="*60)

    # Merge datasets
    merged = df1.merge(
        df2,
        on='dataset_name',
        suffixes=('_r1', '_r2'),
        how='outer'
    )

    print(f"\n📋 Total de datasets: {len(merged)}")
    print(f"   Anotados por Revisor 1: {df1['dataset_name'].nunique()}")
    print(f"   Anotados por Revisor 2: {df2['dataset_name'].nunique()}")

    # Check for missing annotations
    missing_r1 = merged[merged['sensitive_attributes_r1'].isna()]
    missing_r2 = merged[merged['sensitive_attributes_r2'].isna()]

    if len(missing_r1) > 0:
        print(f"\n⚠️  Datasets faltando em Revisor 1: {len(missing_r1)}")
        print(f"   {list(missing_r1['dataset_name'])}")

    if len(missing_r2) > 0:
        print(f"\n⚠️  Datasets faltando em Revisor 2: {len(missing_r2)}")
        print(f"   {list(missing_r2['dataset_name'])}")

    # Remove datasets with missing annotations
    merged = merged.dropna(subset=['sensitive_attributes_r1', 'sensitive_attributes_r2'])

    print(f"\n✅ Datasets com anotações completas: {len(merged)}")

    # Calculate agreement per dataset
    results = []

    for _, row in merged.iterrows():
        attrs_r1 = parse_sensitive_attributes(row['sensitive_attributes_r1'])
        attrs_r2 = parse_sensitive_attributes(row['sensitive_attributes_r2'])

        # Exact match
        exact_match = attrs_r1 == attrs_r2

        # Jaccard similarity
        if len(attrs_r1) == 0 and len(attrs_r2) == 0:
            jaccard = 1.0
        elif len(attrs_r1) == 0 or len(attrs_r2) == 0:
            jaccard = 0.0
        else:
            intersection = len(attrs_r1 & attrs_r2)
            union = len(attrs_r1 | attrs_r2)
            jaccard = intersection / union if union > 0 else 0.0

        results.append({
            'dataset': row['dataset_name'],
            'attrs_r1': sorted(attrs_r1),
            'attrs_r2': sorted(attrs_r2),
            'exact_match': exact_match,
            'jaccard': jaccard
        })

    df_results = pd.DataFrame(results)

    # Overall statistics
    exact_match_rate = df_results['exact_match'].mean()
    avg_jaccard = df_results['jaccard'].mean()

    print("\n📈 Estatísticas Gerais:")
    print(f"   Taxa de concordância exata: {exact_match_rate*100:.1f}%")
    print(f"   Jaccard médio: {avg_jaccard:.3f}")

    # Show disagreements
    disagreements = df_results[~df_results['exact_match']]

    if len(disagreements) > 0:
        print(f"\n⚠️  Discordâncias ({len(disagreements)} datasets):")

        for _, row in disagreements.head(10).iterrows():
            print(f"\n   Dataset: {row['dataset']}")
            print(f"      Revisor 1: {row['attrs_r1']}")
            print(f"      Revisor 2: {row['attrs_r2']}")
            print(f"      Jaccard: {row['jaccard']:.3f}")

        if len(disagreements) > 10:
            print(f"\n   ... e mais {len(disagreements) - 10} discordâncias")

    # Cohen's Kappa (per attribute type)
    print("\n📊 Cohen's Kappa por tipo de atributo:")

    # Common attribute types
    attribute_types = [
        'gender', 'sex',
        'race', 'ethnicity',
        'age',
        'religion',
        'disability',
        'nationality', 'country',
        'marital', 'marriage'
    ]

    kappa_results = []

    for attr_type in attribute_types:
        # Check if each dataset has this attribute type
        y1 = []
        y2 = []

        for _, row in df_results.iterrows():
            has_r1 = any(attr_type in str(attr).lower() for attr in row['attrs_r1'])
            has_r2 = any(attr_type in str(attr).lower() for attr in row['attrs_r2'])

            y1.append(1 if has_r1 else 0)
            y2.append(1 if has_r2 else 0)

        if sum(y1) > 0 or sum(y2) > 0:  # Only if attribute appears
            kappa = cohen_kappa_score(y1, y2)
            kappa_results.append({
                'attribute_type': attr_type,
                'kappa': kappa,
                'n_datasets_r1': sum(y1),
                'n_datasets_r2': sum(y2)
            })

            print(f"   {attr_type}: κ={kappa:.3f} (R1:{sum(y1)}, R2:{sum(y2)})")

    # Overall Kappa (treating each attribute as separate annotation)
    # Flatten all attributes
    all_attrs_r1 = []
    all_attrs_r2 = []

    for _, row in df_results.iterrows():
        attrs_r1 = set(row['attrs_r1'])
        attrs_r2 = set(row['attrs_r2'])

        # All unique attributes mentioned by either reviewer
        all_attrs = sorted(attrs_r1 | attrs_r2)

        for attr in all_attrs:
            all_attrs_r1.append(1 if attr in attrs_r1 else 0)
            all_attrs_r2.append(1 if attr in attrs_r2 else 0)

    if len(all_attrs_r1) > 0:
        overall_kappa = cohen_kappa_score(all_attrs_r1, all_attrs_r2)
    else:
        overall_kappa = 0.0

    print(f"\n🎯 Cohen's Kappa Geral: {overall_kappa:.3f}")

    # Interpretation
    if overall_kappa >= 0.85:
        interpretation = "Excelente (≥0.85) ✅"
    elif overall_kappa >= 0.70:
        interpretation = "Boa (0.70-0.84) ⚠️"
    elif overall_kappa >= 0.50:
        interpretation = "Moderada (0.50-0.69) ⚠️"
    else:
        interpretation = "Fraca (<0.50) ❌"

    print(f"   Interpretação: {interpretation}")

    # Validation for paper
    print("\n✅ Validação para o Paper:")

    if overall_kappa >= 0.85:
        print("   ✅ PASS: Kappa ≥ 0.85 (requisito atendido)")
    else:
        print(f"   ❌ FAIL: Kappa {overall_kappa:.3f} < 0.85")
        print("   ⚠️  Ação necessária:")
        print("      1. Revisar discordâncias")
        print("      2. Clarificar guidelines de anotação")
        print("      3. Re-anotar datasets com discordância")

    # Save results
    output_dir = Path("../results/inter_rater_agreement")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_dir / "agreement_per_dataset.csv", index=False)

    if kappa_results:
        pd.DataFrame(kappa_results).to_csv(output_dir / "kappa_by_attribute.csv", index=False)

    summary = {
        'n_datasets': len(df_results),
        'exact_match_rate': exact_match_rate,
        'avg_jaccard': avg_jaccard,
        'overall_kappa': overall_kappa,
        'interpretation': interpretation,
        'validation_passed': overall_kappa >= 0.85
    }

    import json
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Resultados salvos em: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Calcula concordância entre anotadores')
    parser.add_argument('--reviewer1', type=str, required=True, help='CSV com anotações do revisor 1')
    parser.add_argument('--reviewer2', type=str, required=True, help='CSV com anotações do revisor 2')

    args = parser.parse_args()

    # Load annotations
    print("📂 Carregando anotações...")

    df1 = pd.read_csv(args.reviewer1)
    df2 = pd.read_csv(args.reviewer2)

    print(f"   Revisor 1: {len(df1)} datasets")
    print(f"   Revisor 2: {len(df2)} datasets")

    # Calculate agreement
    summary = calculate_agreement(df1, df2)

    # Exit status
    if summary['validation_passed']:
        print("\n✅ Validação PASSOU!")
        exit(0)
    else:
        print("\n❌ Validação FALHOU!")
        exit(1)


if __name__ == '__main__':
    main()
