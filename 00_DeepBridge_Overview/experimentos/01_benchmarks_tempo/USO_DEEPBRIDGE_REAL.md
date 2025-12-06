# Usando DeepBridge REAL vs. Simulação

## Status: DeepBridge está PRONTO ✅

O DeepBridge **JÁ ESTÁ IMPLEMENTADO** e funcionando (versão 0.1.59) em:
```
/home/guhaase/projetos/DeepBridge/deepbridge/
```

## Dois Modos de Execução

### Modo 1: SIMULAÇÃO (para testes rápidos)

**Arquivo**: `benchmark_deepbridge.py`

- ✅ Usa `time.sleep()` para simular tempos esperados
- ✅ Rápido para testar a estrutura do experimento
- ✅ Não requer DeepBridge totalmente funcional
- ⚠️ **NÃO gera resultados reais para o paper**

**Quando usar**:
- Testar estrutura dos scripts
- Validar pipeline de análise
- Debug rápido

### Modo 2: DEEPBRIDGE REAL (para resultados do paper)

**Arquivo**: `benchmark_deepbridge_REAL.py`

- ✅ Usa `from deepbridge import DBDataset, Experiment`
- ✅ Executa testes reais de validação
- ✅ Gera resultados verdadeiros para o paper
- ✅ Mede tempos reais de execução

**Quando usar**:
- Coletar dados para o paper
- Resultados finais
- Benchmarks de produção

## Como Usar o DeepBridge Real

### 1. Verificar Instalação

```bash
cd /home/guhaase/projetos/DeepBridge
python3 -c "from deepbridge import DBDataset, Experiment; print('✓ OK')"
```

### 2. Executar Benchmark REAL

```bash
cd experimentos/01_benchmarks_tempo/scripts

# Teste rápido (1 run)
python3 benchmark_deepbridge_REAL.py

# Experimento completo (10 runs)
# Editar config.yaml: num_runs = 10
python3 benchmark_deepbridge_REAL.py
```

### 3. Estrutura do Código Real (API Verificada)

```python
from deepbridge import DBDataset, Experiment

# 1. Criar DBDataset
dataset = DBDataset(
    data=test_df,           # DataFrame com dados de teste
    target_column='target',  # Nome da coluna target
    model=xgb_model          # Modelo treinado
)

# 2. Criar Experiment
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification'
)

# 3. Executar testes (API VERIFICADA - v0.1.59)
# Opção A: Executar todos os testes de uma vez
exp.run_tests()  # Executa fairness, robustness, uncertainty, resilience

# Opção B: Executar testes individuais (apenas fairness disponível)
fairness_results = exp.run_fairness_tests()

# Recuperar resultados (após run_tests())
robustness_results = exp.get_robustness_results()
uncertainty_results = exp.get_uncertainty_results()
resilience_results = exp.get_resilience_results()

# Obter todos os resultados de uma vez
comprehensive_results = exp.get_comprehensive_results()

# 4. Gerar relatório
exp.save_html('validation_report.html')  # Gera relatório HTML
```

## API do DeepBridge - Verificar Métodos Disponíveis

Para saber exatamente quais métodos o `Experiment` oferece:

```bash
cd /home/guhaase/projetos/DeepBridge

# Ver documentação do Experiment
python3 -c "from deepbridge import Experiment; help(Experiment)"

# Ver métodos disponíveis
python3 -c "from deepbridge import Experiment; import inspect; print([m for m in dir(Experiment) if not m.startswith('_')])"

# Ver código fonte
cat deepbridge/core/experiment/experiment.py
```

## Checklist para Usar DeepBridge Real

- [ ] Verificar que DeepBridge está instalado
- [ ] Consultar API do `Experiment` para métodos disponíveis
- [ ] Identificar quais testes estão implementados:
  - [ ] Fairness
  - [ ] Robustness
  - [ ] Uncertainty
  - [ ] Resilience
  - [ ] Report generation
- [ ] Atualizar `benchmark_deepbridge_REAL.py` com chamadas corretas
- [ ] Testar com 1 run primeiro
- [ ] Executar experimento completo (10 runs)

## Exemplo Mínimo de Teste

Crie um arquivo `test_deepbridge_api.py`:

```python
#!/usr/bin/env python3
"""
Teste mínimo para verificar API do DeepBridge
"""

import sys
sys.path.insert(0, '/home/guhaase/projetos/DeepBridge')

from deepbridge import DBDataset, Experiment
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

print("=" * 60)
print("Teste da API DeepBridge")
print("=" * 60)

# 1. Gerar dados sintéticos
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

print(f"\n✓ Dataset criado: {df.shape}")

# 2. Treinar modelo
model = XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(df.drop('target', axis=1), df['target'])
print(f"✓ Modelo treinado: acurácia = {model.score(df.drop('target', axis=1), df['target']):.4f}")

# 3. Criar DBDataset
try:
    dataset = DBDataset(
        data=df,
        target_column='target',
        model=model
    )
    print(f"✓ DBDataset criado com sucesso")
    print(f"  - Tipo: {type(dataset)}")
    print(f"  - Atributos disponíveis: {[a for a in dir(dataset) if not a.startswith('_')][:10]}")
except Exception as e:
    print(f"✗ Erro ao criar DBDataset: {e}")
    sys.exit(1)

# 4. Criar Experiment
try:
    exp = Experiment(
        dataset=dataset,
        experiment_type='binary_classification'
    )
    print(f"✓ Experiment criado com sucesso")
    print(f"  - Tipo: {type(exp)}")
    print(f"  - Métodos disponíveis: {[m for m in dir(exp) if not m.startswith('_') and callable(getattr(exp, m))][:15]}")
except Exception as e:
    print(f"✗ Erro ao criar Experiment: {e}")
    sys.exit(1)

# 5. Listar métodos de teste disponíveis
print(f"\n{'=' * 60}")
print("Métodos de teste disponíveis:")
print(f"{'=' * 60}")

test_methods = [m for m in dir(exp) if 'test' in m.lower() or 'run' in m.lower()]
for method in test_methods:
    print(f"  - {method}")

print(f"\n✓ Teste concluído! DeepBridge está funcionando corretamente.")
```

Execute:
```bash
cd experimentos/01_benchmarks_tempo/scripts
chmod +x test_deepbridge_api.py
python3 test_deepbridge_api.py
```

## Atualização Necessária

Uma vez que você tenha a lista exata de métodos do `Experiment`, atualize `benchmark_deepbridge_REAL.py` substituindo os comentários `# NOTA:` pelas chamadas reais.

Por exemplo, se o método for `exp.run_fairness()`:

```python
# De:
# fairness_results = exp.run_fairness_tests()  # NOTA: verificar

# Para:
fairness_results = exp.run_fairness()  # Método real
```

## Resumo

| Aspecto | Simulação | Real |
|---------|-----------|------|
| Arquivo | `benchmark_deepbridge.py` | `benchmark_deepbridge_REAL.py` |
| DeepBridge | Não usa | Usa `import deepbridge` |
| Tempo | `time.sleep()` | Tempo real de execução |
| Resultados | Simulados | Reais para o paper |
| Quando usar | Testes, debug | Coleta final de dados |

**Próximo passo**: Execute o teste mínimo acima para descobrir os métodos exatos do `Experiment` e atualize `benchmark_deepbridge_REAL.py`.

---

**Data**: 2025-12-05
**Status**: DeepBridge v0.1.59 instalado e funcional ✅

---

## Métodos Disponíveis na API (Verificado em 2025-12-05)

### Métodos de Teste

```python
exp.run_tests()           # Executa todos os testes (fairness, robustness, uncertainty, resilience)
exp.run_test()            # Executa teste individual
exp.run_fairness_tests()  # Executa apenas testes de fairness
```

### Métodos de Recuperação de Resultados

```python
exp.get_robustness_results()      # Recupera resultados de robustness
exp.get_uncertainty_results()     # Recupera resultados de uncertainty
exp.get_resilience_results()      # Recupera resultados de resilience
exp.get_comprehensive_results()   # Recupera todos os resultados
```

### Métodos de Relatório

```python
exp.save_html('report.html')  # Gera relatório HTML
```

### Outros Métodos Úteis

```python
exp.calculate_metrics()           # Calcula métricas
exp.get_feature_importance()      # Obtém importância das features
exp.detect_sensitive_attributes() # Detecta atributos sensíveis automaticamente
exp.compare_all_models()          # Compara modelos
exp.get_hyperparameter_results()  # Obtém resultados de hiperparâmetros
```

### Exemplo Completo de Uso

```python
from deepbridge import DBDataset, Experiment
import pandas as pd
from xgboost import XGBClassifier

# 1. Preparar dados
test_df = pd.read_csv('data.csv')
model = XGBClassifier()
model.fit(X_train, y_train)

# 2. Criar DBDataset
dataset = DBDataset(
    data=test_df,
    target_column='target',
    model=model
)

# 3. Criar Experiment
exp = Experiment(
    dataset=dataset,
    experiment_type='binary_classification'
)

# 4. Executar validação completa
exp.run_tests()

# 5. Recuperar resultados
fairness = exp.run_fairness_tests()
robustness = exp.get_robustness_results()
uncertainty = exp.get_uncertainty_results()
resilience = exp.get_resilience_results()
all_results = exp.get_comprehensive_results()

# 6. Gerar relatório
exp.save_html('validation_report.html')

print("Validação completa!")
```
