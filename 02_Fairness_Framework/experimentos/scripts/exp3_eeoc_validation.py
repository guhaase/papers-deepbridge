#!/usr/bin/env python3
"""
Experimento 3: Verificação EEOC/ECOA

Valida as seguintes claims do paper:
- 100% precisão na detecção de violações EEOC/ECOA
- 0 falsos positivos
- Regra 80%, Question 21, Adverse Actions

Uso:
    python exp3_eeoc_validation.py
    python exp3_eeoc_validation.py --dataset compas
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

try:
    from deepbridge import DBDataset, FairnessTestManager
except ImportError:
    print("⚠️  DeepBridge não instalado. Instale com: pip install deepbridge")
    exit(1)


class EEOCValidationExperiment:
    """Experimento de validação EEOC/ECOA."""

    def __init__(self):
        self.results = []
        self.output_dir = Path("../results/eeoc_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_80_percent_rule(self):
        """Testa a regra 80% da EEOC com casos controlados."""

        print("\n" + "="*60)
        print("🔍 TESTE 1: REGRA 80% EEOC")
        print("="*60)

        test_cases = [
            # (sr_protected, sr_reference, expected_violation, description)
            (0.40, 0.50, False, "DI=0.80 - BOUNDARY CASE (exatamente 80%)"),
            (0.39, 0.50, True,  "DI=0.78 - VIOLATION (abaixo de 80%)"),
            (0.41, 0.50, False, "DI=0.82 - OK (acima de 80%)"),
            (0.70, 0.80, False, "DI=0.875 - OK"),
            (0.50, 0.70, True,  "DI=0.714 - VIOLATION"),
            (0.60, 0.75, True,  "DI=0.80 - BOUNDARY EXACT"),
            (0.20, 0.30, False, "DI=0.667 - OK (low rates)"),
            (0.10, 0.15, False, "DI=0.667 - OK (very low rates)"),
            (0.80, 1.00, True,  "DI=0.80 - BOUNDARY"),
            (0.48, 0.60, True,  "DI=0.80 - BOUNDARY"),
        ]

        results = []
        errors = 0

        for sr_p, sr_r, expected_violation, description in test_cases:
            di = sr_p / sr_r if sr_r > 0 else 0

            # Criar dataset sintético com as taxas especificadas
            df = self._create_synthetic_dataset(sr_p, sr_r)

            try:
                dataset = DBDataset(data=df, target_column='outcome')
                ftm = FairnessTestManager(dataset)

                compliance = ftm.check_eeoc_compliance()

                # Verificar se a regra 80% foi violada
                is_violation = not compliance.get('eeoc_80_rule', True)

                # Comparar com expectativa
                is_correct = (is_violation == expected_violation)

                if not is_correct:
                    errors += 1
                    print(f"   ❌ ERRO: {description}")
                    print(f"      Esperado: {'VIOLATION' if expected_violation else 'OK'}")
                    print(f"      Obtido: {'VIOLATION' if is_violation else 'OK'}")
                    print(f"      DI: {di:.3f}")
                else:
                    print(f"   ✅ PASS: {description} (DI={di:.3f})")

                results.append({
                    'sr_protected': sr_p,
                    'sr_reference': sr_r,
                    'disparate_impact': di,
                    'expected_violation': expected_violation,
                    'detected_violation': is_violation,
                    'correct': is_correct,
                    'description': description
                })

            except Exception as e:
                errors += 1
                print(f"   ❌ ERRO EXCEPTION: {description}")
                print(f"      Exception: {str(e)}")
                results.append({
                    'description': description,
                    'error': str(e)
                })

        # Resumo
        print(f"\n📊 Resumo:")
        print(f"   Total de casos: {len(test_cases)}")
        print(f"   ✅ Corretos: {len(test_cases) - errors}")
        print(f"   ❌ Erros: {errors}")
        print(f"   📈 Acurácia: {(len(test_cases) - errors) / len(test_cases) * 100:.1f}%")

        # Validação da claim
        claim_validated = (errors == 0)
        print(f"\n✅ Claim '100% precisão': {'VALIDATED ✅' if claim_validated else 'FAILED ❌'}")

        # Salvar resultados
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.output_dir / "eeoc_80_rule_validation.csv", index=False)

        return {
            'total_cases': len(test_cases),
            'correct': len(test_cases) - errors,
            'errors': errors,
            'accuracy': (len(test_cases) - errors) / len(test_cases),
            'claim_validated': claim_validated
        }

    def test_question_21(self):
        """Testa EEOC Question 21 (representação mínima 2%)."""

        print("\n" + "="*60)
        print("🔍 TESTE 2: EEOC QUESTION 21")
        print("="*60)

        test_cases = [
            # (group_representation, expected_valid, description)
            (0.025, True,  "2.5% - OK"),
            (0.020, True,  "2.0% - BOUNDARY (exatamente 2%)"),
            (0.019, False, "1.9% - VIOLATION (abaixo de 2%)"),
            (0.015, False, "1.5% - VIOLATION"),
            (0.001, False, "0.1% - SEVERE VIOLATION"),
            (0.100, True,  "10% - OK (bem acima)"),
            (0.500, True,  "50% - OK (maioria)"),
        ]

        results = []
        errors = 0

        for representation, expected_valid, description in test_cases:
            # Criar dataset sintético
            n_total = 1000
            n_group = int(representation * n_total)

            df = pd.DataFrame({
                'group': ['protected'] * n_group + ['reference'] * (n_total - n_group),
                'feature1': np.random.randn(n_total),
                'outcome': np.random.choice([0, 1], n_total)
            })

            try:
                dataset = DBDataset(data=df, target_column='outcome')
                dataset.protected_attributes = ['group']

                ftm = FairnessTestManager(dataset)
                compliance = ftm.check_eeoc_compliance()

                # Verificar Question 21
                is_valid = compliance.get('eeoc_question_21', False)

                is_correct = (is_valid == expected_valid)

                if not is_correct:
                    errors += 1
                    print(f"   ❌ ERRO: {description}")
                    print(f"      Esperado: {'VALID' if expected_valid else 'VIOLATION'}")
                    print(f"      Obtido: {'VALID' if is_valid else 'VIOLATION'}")
                else:
                    print(f"   ✅ PASS: {description}")

                results.append({
                    'representation': representation,
                    'expected_valid': expected_valid,
                    'detected_valid': is_valid,
                    'correct': is_correct,
                    'description': description
                })

            except Exception as e:
                errors += 1
                print(f"   ❌ ERRO EXCEPTION: {description}")
                print(f"      Exception: {str(e)}")
                results.append({
                    'description': description,
                    'error': str(e)
                })

        # Resumo
        print(f"\n📊 Resumo:")
        print(f"   Total de casos: {len(test_cases)}")
        print(f"   ✅ Corretos: {len(test_cases) - errors}")
        print(f"   ❌ Erros: {errors}")
        print(f"   📈 Acurácia: {(len(test_cases) - errors) / len(test_cases) * 100:.1f}%")

        # Validação da claim
        claim_validated = (errors == 0)
        print(f"\n✅ Claim '100% precisão': {'VALIDATED ✅' if claim_validated else 'FAILED ❌'}")

        # Salvar resultados
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.output_dir / "eeoc_question_21_validation.csv", index=False)

        return {
            'total_cases': len(test_cases),
            'correct': len(test_cases) - errors,
            'errors': errors,
            'accuracy': (len(test_cases) - errors) / len(test_cases),
            'claim_validated': claim_validated
        }

    def test_adverse_actions(self):
        """Testa geração de ECOA Adverse Action Notices."""

        print("\n" + "="*60)
        print("🔍 TESTE 3: ECOA ADVERSE ACTION NOTICES")
        print("="*60)

        # Criar dataset de crédito sintético
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'age': np.random.randint(18, 70, n),
            'income': np.random.randint(20000, 120000, n),
            'debt_ratio': np.random.uniform(0.1, 0.7, n),
            'credit_score': np.random.randint(300, 850, n),
            'approved': np.random.choice([0, 1], n, p=[0.3, 0.7])
        })

        try:
            dataset = DBDataset(data=df, target_column='approved')
            ftm = FairnessTestManager(dataset)

            # Gerar adverse action notices para negações
            denied_indices = df[df['approved'] == 0].index[:10]

            notices = []
            for idx in denied_indices:
                notice = ftm.generate_adverse_action_notice(idx)
                notices.append(notice)

            # Validar conteúdo dos notices
            print(f"\n📋 Validando {len(notices)} Adverse Action Notices:")

            valid_count = 0

            for i, notice in enumerate(notices, 1):
                has_reasons = 'reasons' in notice or 'primary_reasons' in notice
                has_scores = 'score' in notice or 'metrics' in notice
                has_threshold = 'threshold' in notice or 'decision_boundary' in notice

                is_valid = has_reasons  # Pelo menos razões devem estar presentes

                if is_valid:
                    valid_count += 1
                    print(f"   ✅ Notice {i}: Valid")
                else:
                    print(f"   ❌ Notice {i}: Missing required fields")

                # Mostrar exemplo do primeiro notice
                if i == 1:
                    print(f"\n   📄 Exemplo de Notice:")
                    print(f"      {json.dumps(notice, indent=6)}")

            # Resumo
            print(f"\n📊 Resumo:")
            print(f"   Total de notices: {len(notices)}")
            print(f"   ✅ Válidos: {valid_count}")
            print(f"   ❌ Inválidos: {len(notices) - valid_count}")
            print(f"   📈 Taxa de validade: {valid_count / len(notices) * 100:.1f}%")

            # Validação da claim (90% dos notices devem ser válidos)
            claim_validated = (valid_count / len(notices) >= 0.90)
            print(f"\n✅ Claim '90%+ notices válidos': {'VALIDATED ✅' if claim_validated else 'FAILED ❌'}")

            # Salvar notices
            with open(self.output_dir / "adverse_action_notices_sample.json", 'w') as f:
                json.dump(notices, f, indent=2)

            return {
                'total_notices': len(notices),
                'valid': valid_count,
                'invalid': len(notices) - valid_count,
                'validity_rate': valid_count / len(notices),
                'claim_validated': claim_validated
            }

        except Exception as e:
            print(f"   ❌ ERRO: {str(e)}")
            return {'error': str(e)}

    def _create_synthetic_dataset(self, sr_protected: float, sr_reference: float, n: int = 1000) -> pd.DataFrame:
        """Cria dataset sintético com selection rates específicas."""

        # Criar grupos balanceados
        n_protected = n // 2
        n_reference = n - n_protected

        # Calcular quantos devem ser selecionados em cada grupo
        n_selected_protected = int(sr_protected * n_protected)
        n_selected_reference = int(sr_reference * n_reference)

        # Criar labels
        y_protected = [1] * n_selected_protected + [0] * (n_protected - n_selected_protected)
        y_reference = [1] * n_selected_reference + [0] * (n_reference - n_selected_reference)

        # Shuffle
        np.random.shuffle(y_protected)
        np.random.shuffle(y_reference)

        # Criar dataframe
        df = pd.DataFrame({
            'group': ['protected'] * n_protected + ['reference'] * n_reference,
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'outcome': y_protected + y_reference,
            'prediction': y_protected + y_reference  # Para este teste, prediction = outcome
        })

        return df

    def run_all(self):
        """Executa todos os testes de validação."""

        print("="*60)
        print("🔬 EXPERIMENTO 3: VERIFICAÇÃO EEOC/ECOA")
        print("="*60)
        print(f"📁 Resultados serão salvos em: {self.output_dir}")

        # Teste 1: Regra 80%
        result_80 = self.test_80_percent_rule()

        # Teste 2: Question 21
        result_q21 = self.test_question_21()

        # Teste 3: Adverse Actions
        result_aa = self.test_adverse_actions()

        # Consolidar resultados
        print("\n" + "="*60)
        print("📊 RESULTADOS CONSOLIDADOS")
        print("="*60)

        all_validated = (
            result_80.get('claim_validated', False) and
            result_q21.get('claim_validated', False) and
            result_aa.get('claim_validated', False)
        )

        print(f"\n✅ Regra 80%: {'PASS' if result_80.get('claim_validated') else 'FAIL'}")
        print(f"   - Acurácia: {result_80.get('accuracy', 0)*100:.1f}%")
        print(f"   - Erros: {result_80.get('errors', 0)}/{result_80.get('total_cases', 0)}")

        print(f"\n✅ Question 21: {'PASS' if result_q21.get('claim_validated') else 'FAIL'}")
        print(f"   - Acurácia: {result_q21.get('accuracy', 0)*100:.1f}%")
        print(f"   - Erros: {result_q21.get('errors', 0)}/{result_q21.get('total_cases', 0)}")

        print(f"\n✅ Adverse Actions: {'PASS' if result_aa.get('claim_validated') else 'FAIL'}")
        print(f"   - Validade: {result_aa.get('validity_rate', 0)*100:.1f}%")
        print(f"   - Válidos: {result_aa.get('valid', 0)}/{result_aa.get('total_notices', 0)}")

        print(f"\n{'='*60}")
        print(f"🎯 VALIDAÇÃO FINAL: {'✅ TODOS TESTES PASSARAM' if all_validated else '❌ ALGUNS TESTES FALHARAM'}")
        print(f"{'='*60}")

        # Salvar resumo
        summary = {
            'rule_80_percent': result_80,
            'question_21': result_q21,
            'adverse_actions': result_aa,
            'all_validated': all_validated
        }

        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Resumo salvo em: {self.output_dir / 'summary.json'}")

        return summary


def main():
    parser = argparse.ArgumentParser(description='Experimento 3: Verificação EEOC/ECOA')
    parser.add_argument('--dataset', type=str, help='Dataset específico para testar (compas, credit, etc)')

    args = parser.parse_args()

    experiment = EEOCValidationExperiment()

    if args.dataset:
        print(f"⚠️  Teste em dataset específico '{args.dataset}' ainda não implementado")
        print("   Use sem argumentos para executar testes controlados")
        return

    # Executar todos os testes
    results = experiment.run_all()

    # Verificar se passou
    if results['all_validated']:
        print("\n✅ EXPERIMENTO CONCLUÍDO COM SUCESSO!")
        exit(0)
    else:
        print("\n❌ EXPERIMENTO FALHOU. Revise os resultados acima.")
        exit(1)


if __name__ == '__main__':
    main()
